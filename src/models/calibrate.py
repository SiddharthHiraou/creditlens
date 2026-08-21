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


class SmoothedIsotonic:
    """Isotonic regression with within-block ranking restored.

    Plain isotonic is a step function. On this champion it collapsed 16,091
    distinct raw scores into 120 calibrated values, with a single value shared
    by 1,800 applicants -- 11% of the book assigned an identical PD. Three
    things break as a result:

    * **Cutoffs become coarse.** Nudging the approval threshold jumps whole
      blocks of applicants at once, so a target approval rate cannot be hit.
    * **Counterfactuals stop working.** A feasible change moves the raw score
      but not the calibrated one, so the search has no surface to climb.
    * **AUC drops.** Ties are pure ranking loss.

    The fix keeps the isotonic fit and spreads each flat block across the gap
    to its neighbours, positioning each point by where its *raw* score sits
    inside the block. Monotonicity is preserved because the bands are ordered
    and non-overlapping, and the block's centre is unchanged, so calibration
    is retained while the ranking comes back.

    This is data-limited rather than wrong: 29 knots is what a 4,724-row
    calibration fold supports. The smoothing recovers resolution the isotonic
    fit had to pool away; it does not invent information about the level.
    """

    def __init__(self, iso: IsotonicRegression):
        self.iso = iso
        x = np.asarray(iso.X_thresholds_, dtype=float)
        y = np.asarray(iso.y_thresholds_, dtype=float)

        # Collapse the fit into blocks of constant calibrated value.
        blocks: list[tuple[float, float, float]] = []  # (x_lo, x_hi, y)
        start = 0
        for i in range(1, len(y) + 1):
            if i == len(y) or y[i] != y[start]:
                blocks.append((float(x[start]), float(x[i - 1]), float(y[start])))
                start = i
        self.blocks = blocks

        # Each block gets a band reaching halfway to its neighbours, so adjacent
        # bands meet exactly and the sequence is non-decreasing.
        bands: list[tuple[float, float]] = []
        for j, (_, _, yv) in enumerate(blocks):
            prev_y = blocks[j - 1][2] if j > 0 else yv
            next_y = blocks[j + 1][2] if j + 1 < len(blocks) else yv
            bands.append(((yv + prev_y) / 2.0, (yv + next_y) / 2.0))
        self.bands = bands

        # Build one monotone piecewise-linear curve over the whole real line.
        #
        # The earlier implementation spread each block in place and left the
        # *gaps between* blocks to the raw isotonic step. Those gaps are where
        # it broke: a value just below a block's start still carried the higher
        # step value, then dropped to the block's band floor on entry. Measured
        # on the champion that was 55 monotonicity violations with a worst drop
        # of 0.11 — a higher raw risk mapping to a materially lower calibrated
        # PD, which is the one property a calibrator must never lose.
        #
        # Interpolating over the concatenated block edges removes the gaps
        # entirely: knots are non-decreasing in x and in y by construction, so
        # np.interp cannot produce an inversion.
        knots_x: list[float] = []
        knots_y: list[float] = []
        for (x_lo, x_hi, _), (band_lo, band_hi) in zip(blocks, bands, strict=True):
            knots_x.extend([x_lo, x_hi])
            knots_y.extend([band_lo, band_hi])

        # np.interp needs strictly increasing x. Nudge exact ties forward by an
        # ulp-scale step rather than dropping them, so a single-point block
        # still contributes its band.
        xs = np.asarray(knots_x, dtype=float)
        ys = np.maximum.accumulate(np.asarray(knots_y, dtype=float))
        for i in range(1, xs.size):
            if xs[i] <= xs[i - 1]:
                xs[i] = np.nextafter(xs[i - 1], np.inf)
        self._knots_x = xs
        self._knots_y = ys

    def predict(self, raw: np.ndarray) -> np.ndarray:
        r = np.asarray(raw, dtype=float)
        if self._knots_x.size < 2:
            return np.clip(np.asarray(self.iso.predict(r), dtype=float), 0.0, 1.0)
        # Outside the fitted range np.interp clamps to the end knots, which is
        # the same out-of-bounds behaviour the isotonic fit was built with.
        return np.clip(np.interp(r, self._knots_x, self._knots_y), 0.0, 1.0)


@dataclass
class CalibratedModel:
    """Wraps a fitted model with an isotonic post-processor."""

    base: object
    calibrator: IsotonicRegression | SmoothedIsotonic
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


def calibrate(base, x_cal, y_cal: np.ndarray, *, smooth: bool = True) -> CalibratedModel:
    """Fit the calibrator on a held-out fold.

    ``smooth=True`` restores within-block ranking; see :class:`SmoothedIsotonic`
    for why the plain step function is not usable as a decisioning score.
    """
    raw = base.predict_pd(x_cal)
    iso = fit_isotonic(y_cal, raw)
    calibrator = SmoothedIsotonic(iso) if smooth else iso
    return CalibratedModel(base=base, calibrator=calibrator, feature_names=base.feature_names)


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
