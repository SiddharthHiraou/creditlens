"""PSI / CSI — the metric a credit risk team actually pages on."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from src.evaluation.psi import PSI_ALARM, csi, psi


def test_identical_distributions_give_psi_near_zero():
    rng = np.random.default_rng(0)
    base = rng.normal(size=50_000)
    assert psi(base, rng.normal(size=50_000)).psi < 0.01


def test_psi_grows_monotonically_with_the_size_of_the_shift():
    rng = np.random.default_rng(1)
    base = rng.normal(size=50_000)
    values = [psi(base, rng.normal(shift, 1, 30_000)).psi for shift in (0.0, 0.3, 0.6, 1.0)]
    assert values == sorted(values)


def test_a_large_shift_trips_the_alarm_threshold():
    rng = np.random.default_rng(2)
    base = rng.normal(size=50_000)
    result = psi(base, rng.normal(1.0, 1, 30_000))
    assert result.psi >= PSI_ALARM
    assert result.is_alarm
    assert result.verdict == "significant shift"


def test_verdict_bands_match_the_documented_thresholds():
    rng = np.random.default_rng(3)
    base = rng.normal(size=50_000)
    assert psi(base, rng.normal(size=30_000)).verdict == "stable"


def test_an_empty_bin_yields_a_finite_contribution():
    """Without the epsilon floor this is log(0) and PSI becomes inf."""
    base = np.concatenate([np.zeros(1000), np.ones(1000)])
    shifted = np.zeros(1000)  # the upper bin is now completely empty
    assert np.isfinite(psi(base, shifted).psi)


def test_psi_shares_sum_to_one():
    rng = np.random.default_rng(4)
    tab = psi(rng.normal(size=10_000), rng.normal(size=5_000)).table
    assert tab["expected_share"].sum() == pytest.approx(1.0, abs=1e-4)
    assert tab["actual_share"].sum() == pytest.approx(1.0, abs=1e-4)


def test_csi_flags_the_feature_that_actually_moved():
    rng = np.random.default_rng(5)
    n = 20_000
    expected = pl.DataFrame({"stable": rng.normal(size=n), "drifted": rng.normal(size=n)})
    actual = pl.DataFrame({"stable": rng.normal(size=n), "drifted": rng.normal(1.2, 1, size=n)})
    out = csi(expected, actual, ["stable", "drifted"])
    assert out["feature"][0] == "drifted"
    assert out.filter(pl.col("feature") == "stable")["verdict"].item() == "stable"


def test_csi_handles_categorical_columns():
    expected = pl.DataFrame({"c": ["a"] * 800 + ["b"] * 200})
    actual = pl.DataFrame({"c": ["a"] * 200 + ["b"] * 800})
    out = csi(expected, actual, ["c"])
    assert out["csi"][0] > PSI_ALARM
