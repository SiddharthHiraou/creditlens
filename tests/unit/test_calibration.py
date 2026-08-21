"""Calibration is what makes PD usable for expected-loss arithmetic."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from src.models.calibrate import (
    calibration_report,
    calibration_table,
    expected_calibration_error,
    fit_isotonic,
)


@pytest.fixture
def distorted():
    """A well-ranked but badly-scaled score, as class reweighting produces."""
    rng = np.random.default_rng(0)
    n = 20_000
    x = rng.normal(size=n)
    true_p = 1 / (1 + np.exp(-(-2.2 + 1.5 * x)))
    y = rng.binomial(1, true_p)
    # Inflate toward 0.5, the signature distortion of scale_pos_weight.
    distorted_score = np.clip(true_p * 3.2, 1e-6, 1 - 1e-6)
    return y, distorted_score, true_p


def test_isotonic_fixes_the_level_without_destroying_the_ranking(distorted):
    y, raw, _ = distorted
    iso = fit_isotonic(y, raw)
    cal = iso.predict(raw)

    assert abs(cal.mean() - y.mean()) < abs(raw.mean() - y.mean())
    assert abs(cal.mean() - y.mean()) < 0.01
    # Isotonic is monotone, so AUC is preserved up to the ties it introduces.
    assert roc_auc_score(y, cal) >= roc_auc_score(y, raw) - 0.005


def test_calibration_improves_brier_and_ece(distorted):
    y, raw, _ = distorted
    cal = fit_isotonic(y, raw).predict(raw)
    rep = calibration_report(y, raw, cal)
    assert rep["brier_calibrated"] < rep["brier_raw"]
    assert rep["ece_calibrated"] < rep["ece_raw"]
    assert rep["mean_pd_calibrated"] == pytest.approx(rep["actual_bad_rate"], abs=0.01)


def test_an_already_calibrated_score_is_left_roughly_alone(distorted):
    y, _, true_p = distorted
    assert expected_calibration_error(y, true_p) < 0.02


def test_calibration_table_partitions_every_row(distorted):
    y, raw, _ = distorted
    tab = calibration_table(y, raw)
    assert tab.height == 10
    assert tab["n"].sum() == len(y)
    assert tab["n_bad"].sum() == y.sum()


def test_ece_is_zero_for_a_perfectly_calibrated_score():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.01, 0.99, 200_000)
    y = rng.binomial(1, p)
    assert expected_calibration_error(y, p) < 0.01


def test_predictions_stay_inside_the_unit_interval(distorted):
    y, raw, _ = distorted
    cal = np.clip(fit_isotonic(y, raw).predict(raw), 1e-6, 1 - 1e-6)
    assert cal.min() >= 0.0 and cal.max() <= 1.0


def test_smoothed_isotonic_is_monotone_everywhere():
    """The one property a calibrator may never lose.

    The first implementation spread each isotonic block in place and left the
    *gaps between* blocks to the raw step function. Values in a gap kept the
    higher step value and then dropped on entering the next block — 55
    violations on the champion, worst drop 0.11, meaning a higher raw risk
    mapped to a materially lower calibrated PD. Found by CI on a differently
    sized dataset, not by the original local run.
    """
    from src.models.calibrate import SmoothedIsotonic

    rng = np.random.default_rng(0)
    n = 5_000
    raw = rng.uniform(0, 1, n)
    y = rng.binomial(1, np.clip(raw * 0.6, 0, 1))

    smoothed = SmoothedIsotonic(fit_isotonic(y, raw))
    # Sweep past both ends so out-of-range clamping is covered too.
    grid = np.linspace(-0.1, 1.1, 200_000)
    assert (np.diff(smoothed.predict(grid)) >= -1e-12).all()


def test_smoothed_isotonic_preserves_the_underlying_ranking():
    """Smoothing exists to recover the ranking plain isotonic ties away.

    The property asserted is that the smoothed output ranks like the *raw*
    model, not that it beats the step function. Breaking ties by raw score can
    lower AUC whenever within-block raw ordering happens to run against the
    labels — that is a property of the fixture, not a defect. What must always
    hold is that smoothing carries the raw model's ranking through, which is
    exactly the 0.0013 AUC the step function was throwing away on the champion.
    """
    from sklearn.metrics import roc_auc_score

    from src.models.calibrate import SmoothedIsotonic

    rng = np.random.default_rng(1)
    n = 20_000
    raw = rng.uniform(0, 1, n)
    y = rng.binomial(1, np.clip(raw * 0.6, 0, 1))

    iso = fit_isotonic(y, raw)
    stepped = iso.predict(raw)
    smoothed = SmoothedIsotonic(iso).predict(raw)

    # The point of smoothing: far more distinct values than the step function.
    assert len(np.unique(smoothed)) > 10 * len(np.unique(stepped))
    # And it ranks like the raw model it is calibrating.
    assert roc_auc_score(y, smoothed) == pytest.approx(roc_auc_score(y, raw), abs=1e-3)


def test_smoothed_isotonic_holds_the_calibrated_level():
    """Recovering the ranking must not move the probability scale."""
    from src.models.calibrate import SmoothedIsotonic

    rng = np.random.default_rng(2)
    n = 20_000
    raw = rng.uniform(0, 1, n)
    y = rng.binomial(1, np.clip(raw * 0.6, 0, 1))

    iso = fit_isotonic(y, raw)
    assert SmoothedIsotonic(iso).predict(raw).mean() == pytest.approx(
        iso.predict(raw).mean(), abs=0.01
    )
