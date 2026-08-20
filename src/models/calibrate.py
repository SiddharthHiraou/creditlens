"""Probability calibration.

A GBDT trained with ``scale_pos_weight`` produces a good *ranking* and a bad
*probability*. That distinction is the whole reason this module exists: the
decision layer computes expected loss as ``EL = PD × LGD × EAD``, and an
uncalibrated PD makes that arithmetic wrong in a way no AUC check would reveal.

Isotonic regression is used rather than Platt scaling because it is
non-parametric: it fits any monotone distortion, and the distortion introduced
by class reweighting is not sigmoidal. It needs more data than Platt, which the
dedicated calibration fold provides.

The calibration fold is the **tail of train**, never the validation fold used
for early stopping or hyperparameter search. Fitting calibration on data the
model already tuned against would report a calibration quality the model does
not have in production.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss


@dataclass
class CalibratedModel:
    """Wraps a fitted model with an isotonic post-processor."""

    base: object
    calibrator: IsotonicRegression
    feature_names: list[str]

    def predict_pd(self, x) -> np.ndarray:
        raw = self.base.predict_pd(x)  # type: ignore[attr-defined]
        return np.clip(self.calibrator.predict(raw), 1e-6, 1 - 1e-6)

    def predict_pd_uncalibrated(self, x) -> np.ndarray:
        return self.base.predict_pd(x)  # type: ignore[attr-defined]


def fit_isotonic(y_true: np.ndarray, y_score: np.ndarray) -> IsotonicRegression:
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(y_score, y_true)
    return iso


def calibrate(base, x_cal, y_cal: np.ndarray) -> CalibratedModel:
    raw = base.predict_pd(x_cal)
    return CalibratedModel(
        base=base, calibrator=fit_isotonic(y_cal, raw), feature_names=base.feature_names
    )


def calibration_table(y_true: np.ndarray, y_score: np.ndarray, *, n_bins: int = 10) -> pl.DataFrame:
    """Predicted vs observed default rate per score band.

    A well-calibrated model sits on the diagonal: among applicants predicted at
    5% PD, about 5% should actually default. Systematic deviation means the
    expected-loss numbers are wrong even when the ranking is right.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    ranks = np.argsort(np.argsort(y_score, kind="mergesort"), kind="mergesort")
    bins = np.minimum((ranks * n_bins) // max(len(y_score), 1), n_bins - 1)

    return (
        pl.DataFrame({"bin": bins.astype(np.int32), "y": y_true, "pd": y_score})
        .group_by("bin")
        .agg(
            n=pl.len(),
            mean_predicted=pl.col("pd").mean(),
            observed=pl.col("y").mean(),
            n_bad=pl.col("y").sum(),
        )
        .sort("bin")
        .with_columns(gap=pl.col("mean_predicted") - pl.col("observed"))
    )


def expected_calibration_error(
    y_true: np.ndarray, y_score: np.ndarray, *, n_bins: int = 10
) -> float:
    """Sample-weighted mean absolute gap between predicted and observed."""
    tab = calibration_table(y_true, y_score, n_bins=n_bins)
    w = tab["n"].to_numpy() / tab["n"].sum()
    return float(np.sum(w * np.abs(tab["gap"].to_numpy())))


def calibration_report(y_true: np.ndarray, raw: np.ndarray, cal: np.ndarray) -> dict[str, float]:
    """Before/after summary — the evidence that calibration actually helped."""
    return {
        "brier_raw": float(brier_score_loss(y_true, raw)),
        "brier_calibrated": float(brier_score_loss(y_true, cal)),
        "ece_raw": expected_calibration_error(y_true, raw),
        "ece_calibrated": expected_calibration_error(y_true, cal),
        "mean_pd_raw": float(raw.mean()),
        "mean_pd_calibrated": float(cal.mean()),
        "actual_bad_rate": float(y_true.mean()),
    }
