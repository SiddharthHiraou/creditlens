"""Application state assembled once at startup.

Everything expensive — the booster, the ONNX session, the TreeExplainer, the
feature spec — is built during the lifespan hook, never per request. A
TreeExplainer construction is ~1s; doing it inside the request path would put
the p99 target out of reach on its own.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
from fastapi import Request

from src.api.cache import FeatureCache
from src.api.settings import Settings
from src.api.store import AuditStore
from src.config import ARTIFACTS
from src.explainability.reason_codes import ReasonCodeMapper
from src.explainability.shap_service import ShapService
from src.features.spec import FeatureSpec
from src.models.decision import DecisionPolicy, pd_to_score


@dataclass
class AppState:
    settings: Settings
    spec: FeatureSpec
    model: Any
    native_model: Any
    shap_service: Any
    mapper: ReasonCodeMapper
    policy: DecisionPolicy
    cache: FeatureCache
    store: AuditStore
    serving_backend: str
    model_version: str
    metrics: dict
    # Training-fold PD, not score. The drift endpoint compares PDs; the
    # policy converts to score first. See _baseline_pd.
    baseline_pd: np.ndarray | None = None
    holdout: dict | None = None


def _load_metrics() -> dict:
    path = ARTIFACTS / "phase2_metrics.json"
    return json.loads(path.read_text()) if path.exists() else {}


def build_state(settings: Settings) -> AppState:
    spec = FeatureSpec.load()
    # The wrapper pickle carries the booster, so this import needs CatBoost.
    # Only the SHAP path requires it; see `enable_shap`.
    native = joblib.load(ARTIFACTS / "champion_model.joblib")

    serving_model: Any = native
    backend = "native"
    if settings.use_onnx:
        from src.models.onnx_export import ONNX_PATH, OnnxScorer, export

        if not ONNX_PATH.exists():
            export(native)
        serving_model = OnnxScorer.load(feature_names=spec.features, calibrator=native.calibrator)
        backend = "onnx"

    # SHAP always runs against the native booster: TreeSHAP needs the tree
    # structure, which the ONNX graph does not expose. Without it the API still
    # serves decisions, just no reason codes -- so it is a deployment choice,
    # not a silent degradation.
    shap_service = ShapService.from_model(native, spec.features) if settings.enable_shap else None

    cache = FeatureCache.connect(settings.redis_url)
    if settings.warm_cache_on_startup and cache.backend.startswith("memory"):
        from src.api.warm_cache import warm_into

        warm_into(cache, spec)

    metrics = _load_metrics()
    baseline_pd = _baseline_pd()

    # Cutoffs are quantiles of the *score* distribution, not the PD one. These
    # are different scales -- score runs 300-850, PD runs 0-1 -- and mixing them
    # puts the cutoff near 0.05, which every score clears, so the API approves
    # every applicant while looking entirely healthy.
    policy = (
        DecisionPolicy.from_approval_rate(
            pd_to_score(baseline_pd), approve_rate=0.60, refer_rate=0.10
        )
        if baseline_pd is not None
        else DecisionPolicy()
    )

    return AppState(
        settings=settings,
        spec=spec,
        model=serving_model,
        native_model=native,
        shap_service=shap_service,
        mapper=ReasonCodeMapper.load(),
        policy=policy,
        cache=cache,
        store=AuditStore.connect(settings.database_url),
        serving_backend=backend,
        model_version=str(metrics.get("champion", "unknown")),
        metrics=metrics,
        baseline_pd=baseline_pd,
        holdout=_load_holdout(),
    )


def _baseline_pd() -> np.ndarray | None:
    """Training-fold PD distribution.

    Serves two distinct purposes and the units matter for both: drift is PSI
    over PDs, while policy cutoffs are quantiles of the derived *score*.
    """
    path = ARTIFACTS / "serving_holdout.npz"
    if path.exists():
        return np.load(path)["train_pd"]
    return None


def _load_holdout() -> dict | None:
    """Out-of-time fold with known outcomes, used by the cutoff simulator.

    The simulator has to answer "what would the bad rate be" for cutoffs nobody
    has served yet, which needs realised outcomes. The out-of-time fold is the
    only population where those are both known and not what the model trained on.
    """
    path = ARTIFACTS / "serving_holdout.npz"
    if not path.exists():
        return None
    data = np.load(path)
    return {"score": data["score"], "y": data["y"], "exposure": data["exposure"]}


def get_state(request: Request) -> AppState:
    return request.app.state.creditlens


def get_scoring_service(request: Request):
    return request.app.state.scoring
