"""Fairness metrics, cross-checked against Fairlearn's implementations."""

from __future__ import annotations

import numpy as np
import pytest

from src.fairness.report import (
    FOUR_FIFTHS,
    age_band,
    calibration_by_group,
    fairlearn_metrics,
    fairness_report,
    group_metrics,
)


@pytest.fixture
def biased():
    """Group B approved at half the rate of group A."""
    rng = np.random.default_rng(0)
    n = 8_000
    g = np.where(rng.random(n) < 0.5, "A", "B")
    y = rng.binomial(1, 0.15, n)
    approved = np.where(g == "A", rng.random(n) < 0.80, rng.random(n) < 0.40)
    return y, approved, g


def test_perfect_parity_gives_disparate_impact_one():
    rng = np.random.default_rng(1)
    n = 10_000
    g = np.where(rng.random(n) < 0.5, "A", "B")
    y = rng.binomial(1, 0.15, n)
    approved = rng.random(n) < 0.6  # independent of group
    rep = fairness_report(y, approved, g, attribute="g")
    assert rep.disparate_impact == pytest.approx(1.0, abs=0.05)
    assert rep.passes_four_fifths


def test_four_fifths_rule_fires_on_a_clear_disparity(biased):
    y, approved, g = biased
    rep = fairness_report(y, approved, g, attribute="g")
    assert rep.disparate_impact == pytest.approx(0.5, abs=0.05)
    assert not rep.passes_four_fifths
    assert rep.disparate_impact < FOUR_FIFTHS
    assert rep.worst_group == "B"
    assert rep.reference_group == "A"


def test_our_metrics_agree_with_fairlearn(biased):
    """Two independent implementations must agree, or one of them is wrong."""
    y, approved, g = biased
    rep = fairness_report(y, approved, g, attribute="g")
    ref = fairlearn_metrics(y, approved, g)
    assert rep.disparate_impact == pytest.approx(ref["demographic_parity_ratio"], abs=1e-9)
    assert rep.selection_rate_difference == pytest.approx(
        ref["demographic_parity_difference"], abs=1e-9
    )
    assert rep.equalized_odds_difference == pytest.approx(
        ref["equalized_odds_difference"], abs=1e-9
    )


def test_group_metrics_partitions_the_population(biased):
    y, approved, g = biased
    out = group_metrics(y, approved, g)
    assert out["n"].sum() == len(y)
    assert out["share"].sum() == pytest.approx(1.0)


def test_equal_opportunity_gap_is_zero_when_good_applicants_fare_equally():
    n = 6_000
    rng = np.random.default_rng(2)
    g = np.where(rng.random(n) < 0.5, "A", "B")
    y = rng.binomial(1, 0.2, n)
    # Approve every good applicant regardless of group.
    approved = y == 0
    rep = fairness_report(y, approved, g, attribute="g")
    assert rep.equal_opportunity_difference == pytest.approx(0.0, abs=1e-9)


def test_calibration_gap_is_reported_per_group():
    n = 4_000
    rng = np.random.default_rng(3)
    g = np.where(rng.random(n) < 0.5, "A", "B")
    y = rng.binomial(1, 0.2, n)
    pd_hat = np.full(n, 0.2)
    out = group_metrics(y, y == 0, g, pd_hat)
    assert "calibration_gap" in out.columns
    assert out["calibration_gap"].abs().max() < 0.05


def test_calibration_by_group_covers_every_group_and_band():
    n = 5_000
    rng = np.random.default_rng(4)
    g = np.where(rng.random(n) < 0.5, "A", "B")
    p = rng.uniform(0.01, 0.6, n)
    y = rng.binomial(1, p)
    out = calibration_by_group(y, p, g, n_bins=5)
    assert out["n"].sum() == n
    assert set(out["group"].unique()) == {"A", "B"}
    assert out["band"].n_unique() == 5


@pytest.mark.parametrize(
    ("days", "expected"),
    [(-20 * 365.25, "18-24"), (-30 * 365.25, "25-34"), (-70 * 365.25, "65+")],
)
def test_age_bands_map_from_negative_day_offsets(days, expected):
    assert age_band(np.array([days]))[0] == expected


def test_single_group_is_trivially_parity():
    y = np.array([0, 1, 0, 1])
    rep = fairness_report(
        y, np.array([True, False, True, False]), np.array(["A"] * 4), attribute="g"
    )
    assert rep.disparate_impact == pytest.approx(1.0)
