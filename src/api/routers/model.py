"""Model metadata and monitoring."""

from __future__ import annotations

import datetime as dt

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, status

from src.api.deps import AppState, get_state
from src.api.schemas import DriftOut, ModelMetadata
from src.api.security import require_api_key
from src.evaluation.psi import psi
from src.explainability.reason_codes import ReasonCodeMapper

router = APIRouter(prefix="/v1", tags=["model"], dependencies=[Depends(require_api_key)])


@router.get("/model/metadata", response_model=ModelMetadata)
def model_metadata(state: AppState = Depends(get_state)) -> ModelMetadata:
    """What is actually serving right now, and how it performed at validation."""
    metrics = state.metrics
    champion = metrics.get("champion_calibrated_test", {})
    return ModelMetadata(
        model_version=state.model_version,
        model_type=str(metrics.get("champion", "unknown")),
        trained_at=state.spec.created_at,
        feature_spec_version=state.spec.version,
        feature_spec_fingerprint=state.spec.fingerprint,
        n_features=state.spec.n_features,
        n_monotonic_constraints=len(state.spec.monotonic),
        features=state.spec.features,
        performance={
            k: round(float(v), 6) for k, v in champion.items() if isinstance(v, (int, float))
        },
        calibration={
            k: round(float(v), 6)
            for k, v in metrics.get("calibration", {}).items()
            if isinstance(v, (int, float))
        },
        serving_backend=state.serving_backend,
        reason_code_mapping_version=ReasonCodeMapper.load().version,
    )


@router.get("/monitoring/drift", response_model=DriftOut)
def drift(state: AppState = Depends(get_state)) -> DriftOut:
    """PSI of the live score distribution against the training baseline.

    Computed over decisions actually served, not a fixture. PSI answers "has
    the population moved" — which arrives months before "has performance
    degraded", because performance needs outcomes and outcomes take 12 months.
    """
    if state.baseline_pd is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No training baseline available; run `make train` to write it.",
        )

    from sqlalchemy import select

    from src.api.store import decisions as decisions_table

    with state.store.engine.connect() as conn:
        served = np.array(
            [r[0] for r in conn.execute(select(decisions_table.c.pd)).all()], dtype=float
        )

    if served.size < 30:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Only {served.size} decisions served; need at least 30 for a stable PSI.",
        )

    result = psi(state.baseline_pd, served)
    worst = (
        result.table.sort("contribution", descending=True)
        .head(5)
        .select("bin", "lower", "upper", "expected_share", "actual_share", "contribution")
        .to_dicts()
    )
    return DriftOut(
        score_psi=round(result.psi, 6),
        verdict=result.verdict,
        is_alarm=result.is_alarm,
        baseline_mean_pd=float(state.baseline_pd.mean()),
        current_mean_pd=float(served.mean()),
        n_baseline=int(state.baseline_pd.size),
        n_current=int(served.size),
        worst_features=worst,
        computed_at=dt.datetime.now(dt.UTC).replace(tzinfo=None),
    )
