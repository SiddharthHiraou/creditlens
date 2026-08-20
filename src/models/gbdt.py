"""GBDT champion and challengers behind one interface.

All three consume the same numpy matrix, the same monotonic constraint vector
and the same imbalance handling, so the comparison in the model card is
genuinely like-for-like rather than three differently-tuned pipelines.

Imbalance is handled with ``scale_pos_weight``, never resampling. It reweights
the gradient without inventing borrowers, and it leaves the ranking intact for
the isotonic calibration that follows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import polars as pl

from src.config import RANDOM_SEED


class Fitted(Protocol):
    def predict_proba(self, x: np.ndarray) -> np.ndarray: ...


@dataclass
class GbdtModel:
    """A trained booster plus everything needed to reproduce and serve it."""

    name: str
    booster: Any
    feature_names: list[str]
    params: dict[str, Any]
    best_iteration: int | None = None

    def predict_pd(self, x: np.ndarray | pl.DataFrame) -> np.ndarray:
        if isinstance(x, pl.DataFrame):
            x = to_matrix(x, self.feature_names)
        return np.asarray(self.booster.predict_proba(x))[:, 1]

    def gain_importance(self) -> pl.DataFrame:
        """Gain-based importance, normalised to sum to 1."""
        if hasattr(self.booster, "booster_"):  # LightGBM
            gain = self.booster.booster_.feature_importance("gain")
        elif hasattr(self.booster, "feature_importances_"):
            gain = np.asarray(self.booster.feature_importances_, dtype=float)
        else:
            gain = np.zeros(len(self.feature_names))
        total = gain.sum() or 1.0
        return pl.DataFrame({"feature": self.feature_names, "gain": gain / total}).sort(
            "gain", descending=True
        )


def to_matrix(df: pl.DataFrame, features: list[str]) -> np.ndarray:
    """Spec-ordered float32 matrix. Nulls stay as NaN: every booster here
    handles missing natively, and imputing would discard the informative
    missingness that thin-file applicants carry."""
    return df.select(features).to_numpy().astype(np.float32)


def scale_pos_weight(y: np.ndarray) -> float:
    """negatives / positives — the standard imbalance correction."""
    pos = float(y.sum())
    return (len(y) - pos) / max(pos, 1.0)


def fit_lightgbm(
    x: np.ndarray,
    y: np.ndarray,
    features: list[str],
    *,
    params: dict[str, Any] | None = None,
    monotone: list[int] | None = None,
    eval_set: tuple[np.ndarray, np.ndarray] | None = None,
    early_stopping_rounds: int = 100,
) -> GbdtModel:
    import lightgbm as lgb

    p: dict[str, Any] = {
        "objective": "binary",
        "metric": "auc",
        "n_estimators": 2000,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 50,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "scale_pos_weight": scale_pos_weight(y),
        "verbose": -1,
        "n_jobs": -1,
        "random_state": RANDOM_SEED,
    }
    p.update(params or {})
    if monotone is not None and any(monotone):
        p["monotone_constraints"] = monotone
        # 'basic', not 'advanced'. LightGBM documents 'advanced' as the less
        # conservative method, but in testing on this feature set it admitted
        # a real monotonicity violation (PD fell as a +1-constrained feature
        # rose) while 'basic' held exactly -- and 'basic' scored marginally
        # better besides. A constraint that is not guaranteed is worthless for
        # the regulatory purpose it exists to serve.
        # Pinned by tests/integration/test_phase2.py::test_monotonic_constraints_hold.
        p["monotone_constraints_method"] = "basic"

    model = lgb.LGBMClassifier(**p)
    callbacks = []
    fit_kwargs: dict[str, Any] = {}
    if eval_set is not None:
        # LightGBM 4.7 deprecated `eval_set` in the sklearn API.
        fit_kwargs["eval_X"], fit_kwargs["eval_y"] = eval_set
        fit_kwargs["eval_metric"] = "auc"
        callbacks = [
            lgb.early_stopping(early_stopping_rounds, verbose=False),
            lgb.log_evaluation(0),
        ]
    if callbacks:
        fit_kwargs["callbacks"] = callbacks

    model.fit(x, y, **fit_kwargs)
    return GbdtModel(
        name="lightgbm",
        booster=model,
        feature_names=features,
        params=p,
        best_iteration=getattr(model, "best_iteration_", None),
    )


def fit_xgboost(
    x: np.ndarray,
    y: np.ndarray,
    features: list[str],
    *,
    params: dict[str, Any] | None = None,
    monotone: list[int] | None = None,
    eval_set: tuple[np.ndarray, np.ndarray] | None = None,
    early_stopping_rounds: int = 100,
) -> GbdtModel:
    import xgboost as xgb

    p: dict[str, Any] = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "n_estimators": 2000,
        "learning_rate": 0.03,
        "max_depth": 6,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "scale_pos_weight": scale_pos_weight(y),
        "tree_method": "hist",
        "n_jobs": -1,
        "random_state": RANDOM_SEED,
    }
    p.update(params or {})
    if monotone is not None and any(monotone):
        p["monotone_constraints"] = tuple(monotone)
    if eval_set is not None:
        p["early_stopping_rounds"] = early_stopping_rounds

    model = xgb.XGBClassifier(**p)
    model.fit(x, y, eval_set=[eval_set] if eval_set else None, verbose=False)
    return GbdtModel(
        name="xgboost",
        booster=model,
        feature_names=features,
        params=p,
        best_iteration=getattr(model, "best_iteration", None),
    )


def fit_catboost(
    x: np.ndarray,
    y: np.ndarray,
    features: list[str],
    *,
    params: dict[str, Any] | None = None,
    monotone: list[int] | None = None,
    eval_set: tuple[np.ndarray, np.ndarray] | None = None,
    early_stopping_rounds: int = 100,
) -> GbdtModel:
    from catboost import CatBoostClassifier

    p: dict[str, Any] = {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "iterations": 2000,
        "learning_rate": 0.03,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "scale_pos_weight": scale_pos_weight(y),
        "random_seed": RANDOM_SEED,
        "verbose": 0,
        "allow_writing_files": False,
    }
    p.update(params or {})
    if monotone is not None and any(monotone):
        p["monotone_constraints"] = list(monotone)

    model = CatBoostClassifier(**p)
    model.fit(
        x,
        y,
        eval_set=eval_set,
        early_stopping_rounds=early_stopping_rounds if eval_set else None,
        verbose=False,
    )
    return GbdtModel(
        name="catboost",
        booster=model,
        feature_names=features,
        params=p,
        best_iteration=model.get_best_iteration(),
    )


FITTERS = {"lightgbm": fit_lightgbm, "xgboost": fit_xgboost, "catboost": fit_catboost}
