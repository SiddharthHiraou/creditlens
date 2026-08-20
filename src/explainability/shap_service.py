"""SHAP explanations for the champion model.

TreeSHAP, not LIME. For a tree ensemble TreeSHAP is exact and fast; LIME is a
local surrogate whose explanations are unstable across reruns, which is
disqualifying when the output is a regulatory disclosure an applicant can
challenge.

**Two things about these values that must not be misread.**

*They are attributions, not causes.* A SHAP value says how much a feature moved
this model's output relative to a baseline, given the other features. It does
not say that changing the feature would change the applicant's real-world risk.
The counterfactual module exists precisely because "what would move the
decision" is a different question from "what drove the score".

*They are in raw margin space.* The champion is isotonic-calibrated, and
calibration is a monotone post-transform applied after the booster. SHAP
explains the booster. Because the transform is monotone the *ordering and sign*
of contributions carry over to the calibrated PD unchanged, which is all the
reason-code layer needs; the magnitudes do not, so they are never presented as
"this feature added N points of PD".
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from src.config import ARTIFACTS

GLOBAL_CACHE = ARTIFACTS / "shap_global.parquet"
BASELINE_CACHE = ARTIFACTS / "shap_baseline.json"


@dataclass
class ShapService:
    """Wraps a TreeExplainer over the champion's underlying booster."""

    explainer: object
    feature_names: list[str]

    @property
    def expected_value(self) -> float:
        """The explainer's base value, read live rather than cached.

        This is deliberately a property. SHAP *mutates* ``expected_value`` on
        the explainer after the first ``shap_values`` call -- for this CatBoost
        champion it moves from 0.0261 to 0.0157 -- so a value snapshotted at
        construction is stale by exactly that amount, and every reconstructed
        margin is off by it. The additivity check catches this; caching hides it.
        """
        raw = self.explainer.expected_value
        if isinstance(raw, (list, np.ndarray)):
            return float(np.asarray(raw).ravel()[-1])
        return float(raw)

    @classmethod
    def from_model(cls, model, feature_names: list[str] | None = None) -> ShapService:
        """Build from either a calibrated wrapper or a bare GbdtModel."""
        import shap

        base = getattr(model, "base", model)  # unwrap CalibratedModel
        booster = getattr(base, "booster", base)  # unwrap GbdtModel
        names = feature_names or getattr(base, "feature_names", None) or []
        return cls(explainer=shap.TreeExplainer(booster), feature_names=list(names))

    def values(self, x: np.ndarray) -> np.ndarray:
        """SHAP values, shape (n_rows, n_features), for the positive class."""
        raw = self.explainer.shap_values(x)
        arr = np.asarray(raw)
        if arr.ndim == 3:
            # (rows, features, classes) or (classes, rows, features) depending
            # on the booster; take the positive class either way.
            arr = arr[..., -1] if arr.shape[-1] <= 2 else arr[-1]
        return arr

    def explain_one(self, x_row: np.ndarray, *, top_k: int = 10) -> pl.DataFrame:
        """Signed contributions for a single application, largest first.

        Positive pushes toward default, negative toward repayment.
        """
        row = np.asarray(x_row, dtype=np.float32).reshape(1, -1)
        v = self.values(row)[0]
        return (
            pl.DataFrame(
                {
                    "feature": self.feature_names,
                    "value": row[0].astype(float),
                    "shap": v.astype(float),
                }
            )
            .with_columns(abs_shap=pl.col("shap").abs())
            .sort("abs_shap", descending=True)
            .head(top_k)
        )

    def global_importance(self, x: np.ndarray, *, sample: int = 5000) -> pl.DataFrame:
        """Mean |SHAP| per feature — the global summary, computed once.

        Also reports the mean *signed* value, which distinguishes a feature that
        consistently pushes one direction from one that pushes both ways
        depending on its value.
        """
        x = np.asarray(x, dtype=np.float32)
        if len(x) > sample:
            rng = np.random.default_rng(0)
            x = x[rng.choice(len(x), sample, replace=False)]
        v = self.values(x)
        return (
            pl.DataFrame(
                {
                    "feature": self.feature_names,
                    "mean_abs_shap": np.abs(v).mean(axis=0).astype(float),
                    "mean_shap": v.mean(axis=0).astype(float),
                    "std_shap": v.std(axis=0).astype(float),
                }
            )
            .with_columns(
                share=pl.col("mean_abs_shap") / pl.col("mean_abs_shap").sum(),
            )
            .sort("mean_abs_shap", descending=True)
        )

    def dependence(self, x: np.ndarray, feature: str, *, bins: int = 20) -> pl.DataFrame:
        """Binned dependence: how the contribution varies across the feature's range."""
        if feature not in self.feature_names:
            raise KeyError(f"{feature!r} is not in the model's feature list")
        idx = self.feature_names.index(feature)
        x = np.asarray(x, dtype=np.float32)
        v = self.values(x)[:, idx]
        col = x[:, idx].astype(float)
        mask = np.isfinite(col)
        if mask.sum() == 0:
            return pl.DataFrame(schema={"bin": pl.Int32, "n": pl.UInt32})
        ranks = np.argsort(np.argsort(col[mask], kind="mergesort"), kind="mergesort")
        b = np.minimum((ranks * bins) // max(mask.sum(), 1), bins - 1)
        return (
            pl.DataFrame({"bin": b.astype(np.int32), "value": col[mask], "shap": v[mask]})
            .group_by("bin")
            .agg(
                n=pl.len(),
                value_lo=pl.col("value").min(),
                value_hi=pl.col("value").max(),
                mean_shap=pl.col("shap").mean(),
            )
            .sort("bin")
        )

    def cache_global(self, x: np.ndarray, path: Path = GLOBAL_CACHE) -> Path:
        """Persist the global summary so serving never recomputes it."""
        path.parent.mkdir(parents=True, exist_ok=True)
        self.global_importance(x).write_parquet(path)
        BASELINE_CACHE.write_text(
            json.dumps(
                {"expected_value": self.expected_value, "n_features": len(self.feature_names)}
            )
        )
        return path


def additivity_error(service: ShapService, model, x: np.ndarray) -> float:
    """Max deviation from the SHAP additivity property, over the given rows.

    TreeSHAP guarantees ``expected_value + sum(shap) == raw model output``.
    Checking it is a cheap way to catch a wrong explainer, a wrong class index,
    or a column-order mismatch between the matrix and the model -- all of which
    otherwise produce plausible-looking but wrong reason codes.
    """
    x = np.asarray(x, dtype=np.float32)
    v = service.values(x)
    reconstructed = service.expected_value + v.sum(axis=1)

    base = getattr(model, "base", model)
    booster = getattr(base, "booster", base)
    actual = _raw_margin(booster, x)
    return float(np.abs(reconstructed - actual).max())


def _raw_margin(booster, x: np.ndarray) -> np.ndarray:
    """Model output in margin space, read directly where the API offers it.

    Dispatch is on the concrete class, deliberately, rather than duck-typed
    with try/except. ``LGBMClassifier.predict`` forwards unknown keyword
    arguments to the underlying booster instead of raising ``TypeError``, so a
    CatBoost-shaped call against LightGBM silently returns *class labels* and
    the additivity check then reports an error of ~6 that looks like a SHAP
    bug and is not one.
    """
    cls = type(booster).__name__

    if cls.startswith("CatBoost"):
        return np.asarray(booster.predict(x, prediction_type="RawFormulaVal"), dtype=float)

    if cls.startswith("LGBM"):
        return np.asarray(booster.booster_.predict(x, raw_score=True), dtype=float)

    if cls.startswith("XGB"):
        import xgboost as xgb

        return np.asarray(
            booster.get_booster().predict(xgb.DMatrix(x), output_margin=True), dtype=float
        )

    p = np.clip(booster.predict_proba(x)[:, 1], 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))
