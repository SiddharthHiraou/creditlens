"""Metric behaviour, including the cases that catch a silently broken model."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from src.evaluation.metrics import (
    decile_table,
    discrimination,
    is_rank_ordered,
    is_strictly_rank_ordered,
    ks_statistic,
    rank_order_violations,
)


@pytest.fixture
def separable():
    y = np.array([0] * 500 + [1] * 500)
    s = np.concatenate([np.full(500, 0.1), np.full(500, 0.9)])
    return y, s


def test_perfect_separation_gives_auc_one_and_ks_one(separable):
    y, s = separable
    rep = discrimination(y, s)
    assert rep.auc == pytest.approx(1.0)
    assert rep.gini == pytest.approx(1.0)
    assert ks_statistic(y, s)[0] == pytest.approx(1.0)


def test_random_score_gives_auc_half_and_gini_zero():
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.1, 20_000)
    rep = discrimination(y, rng.random(20_000))
    assert rep.auc == pytest.approx(0.5, abs=0.02)
    assert rep.gini == pytest.approx(0.0, abs=0.04)


def test_gini_is_exactly_two_auc_minus_one():
    rng = np.random.default_rng(1)
    y = rng.binomial(1, 0.2, 5_000)
    s = rng.random(5_000)
    rep = discrimination(y, s)
    assert rep.gini == pytest.approx(2 * rep.auc - 1)


def test_ks_is_undefined_for_a_single_class():
    y = np.zeros(100, dtype=int)
    assert np.isnan(ks_statistic(y, np.random.random(100))[0])


def test_pr_auc_baseline_is_the_prevalence():
    """With a random score, PR-AUC should sit at the positive-class rate."""
    rng = np.random.default_rng(2)
    y = rng.binomial(1, 0.1, 50_000)
    rep = discrimination(y, rng.random(50_000))
    assert rep.pr_auc == pytest.approx(0.1, abs=0.015)


def test_decile_table_partitions_every_row():
    rng = np.random.default_rng(3)
    n = 10_000
    y = rng.binomial(1, 0.12, n)
    s = rng.random(n)
    d = decile_table(y, s)
    assert d.height == 10
    assert d["n"].sum() == n
    assert d["n_bad"].sum() == y.sum()
    assert d["cum_bad_capture"].to_list()[-1] == pytest.approx(1.0)


def test_decile_one_is_the_riskiest_bin():
    rng = np.random.default_rng(4)
    n = 20_000
    x = rng.normal(size=n)
    p = 1 / (1 + np.exp(-(-2.0 + 1.5 * x)))
    y = rng.binomial(1, p)
    d = decile_table(y, p)
    assert d["mean_pd"][0] > d["mean_pd"][9]
    assert is_rank_ordered(d)


def test_rank_ordering_check_rejects_a_non_monotonic_model():
    broken = pl.DataFrame(
        {"decile": [1, 2, 3], "n": [5_000, 5_000, 5_000], "bad_rate": [0.10, 0.25, 0.05]}
    )
    assert not is_rank_ordered(broken)
    assert not is_strictly_rank_ordered(broken)


def test_noise_scale_inversion_is_not_a_rank_order_failure():
    """A two-loan swing between small adjacent bins is noise, not a broken model."""
    noisy = pl.DataFrame(
        {"decile": [1, 2, 3], "n": [260, 260, 260], "bad_rate": [0.30, 0.0923, 0.10]}
    )
    assert not is_strictly_rank_ordered(noisy)  # strict test trips
    assert is_rank_ordered(noisy)  # noise-aware test does not
    assert rank_order_violations(noisy).height == 0


def test_large_inversion_is_flagged_even_in_big_bins():
    broken = pl.DataFrame(
        {"decile": [1, 2, 3], "n": [5_000, 5_000, 5_000], "bad_rate": [0.10, 0.30, 0.05]}
    )
    assert not is_rank_ordered(broken)
    v = rank_order_violations(broken)
    assert v.height == 1
    assert v["from_decile"][0] == 1 and v["to_decile"][0] == 2


def test_the_same_inversion_becomes_significant_as_n_grows():
    """Sample size, not the gap, is what turns noise into evidence."""
    rates = [0.20, 0.24]
    small = pl.DataFrame({"decile": [1, 2], "n": [80, 80], "bad_rate": rates})
    large = pl.DataFrame({"decile": [1, 2], "n": [100_000, 100_000], "bad_rate": rates})
    assert is_rank_ordered(small)
    assert not is_rank_ordered(large)


def test_perfectly_ordered_table_has_no_violations():
    clean = pl.DataFrame(
        {
            "decile": list(range(1, 11)),
            "n": [1000] * 10,
            "bad_rate": [0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.12, 0.08, 0.05, 0.02],
        }
    )
    assert is_rank_ordered(clean)
    assert is_strictly_rank_ordered(clean)
    assert rank_order_violations(clean).is_empty()
