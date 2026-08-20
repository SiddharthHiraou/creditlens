"""Mitigation experiments and the tradeoff curves they produce."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from src.fairness.mitigation import (
    _ScorePassthrough,
    cutoff_tradeoff_curve,
    threshold_optimizer_frontier,
)


@pytest.fixture
def disparate():
    """Group B genuinely riskier, so a group-blind cutoff creates disparity."""
    rng = np.random.default_rng(0)
    n = 6_000
    g = np.where(rng.random(n) < 0.5, "A", "B")
    latent = rng.normal(np.where(g == "A", -0.5, 0.5), 1.0)
    y = rng.binomial(1, 1 / (1 + np.exp(-(-1.5 + latent))))
    score = 600 - 40 * latent + rng.normal(0, 5, n)  # higher score = safer
    return y, score, g


def test_score_passthrough_returns_the_score_unchanged():
    est = _ScorePassthrough().fit(np.zeros((3, 1)))
    assert est.predict(np.array([[1.0], [2.0], [3.0]])) == pytest.approx([1.0, 2.0, 3.0])


def test_score_passthrough_satisfies_sklearn_is_fitted():
    """Fairlearn calls check_is_fitted; a bare object fails with AttributeError."""
    from sklearn.utils.validation import check_is_fitted

    check_is_fitted(_ScorePassthrough())


def test_cutoff_curve_covers_every_requested_approval_rate(disparate):
    y, score, g = disparate
    curve = cutoff_tradeoff_curve(y, score, g, approval_rates=(0.4, 0.6, 0.8))
    assert curve.height == 3
    assert curve["target_approval_rate"].to_list() == [0.4, 0.6, 0.8]


def test_looser_cutoffs_raise_both_approval_rate_and_bad_rate(disparate):
    y, score, g = disparate
    curve = cutoff_tradeoff_curve(y, score, g, approval_rates=(0.4, 0.6, 0.8, 0.9))
    rates = curve["approval_rate"].to_list()
    bad = curve["bad_rate_among_approved"].to_list()
    assert rates == sorted(rates)
    assert bad == sorted(bad), "approving more must admit more bads"


def test_looser_cutoffs_reduce_measured_disparity(disparate):
    """Most of the disparity is a property of where the cutoff sits."""
    y, score, g = disparate
    curve = cutoff_tradeoff_curve(y, score, g, approval_rates=(0.4, 0.9))
    di = curve["disparate_impact"].to_list()
    assert di[1] > di[0]


def test_cutoff_curve_reports_the_actual_threshold(disparate):
    y, score, g = disparate
    curve = cutoff_tradeoff_curve(y, score, g, approval_rates=(0.5,))
    cutoff = curve["cutoff"][0]
    assert (score >= cutoff).mean() == pytest.approx(0.5, abs=0.01)


def test_a_group_blind_cutoff_costs_no_auc(disparate):
    """Moving a single cutoff changes who is approved, never the ranking."""
    y, score, g = disparate
    curve = cutoff_tradeoff_curve(y, score, g, approval_rates=(0.4, 0.6, 0.8))
    assert curve["auc_cost"].abs().max() == pytest.approx(0.0, abs=1e-12)


def test_threshold_optimizer_reduces_disparity_below_the_blind_cutoff(disparate):
    y, score, g = disparate
    blind = cutoff_tradeoff_curve(y, score, g, approval_rates=(0.6,))["disparate_impact"][0]
    frontier = threshold_optimizer_frontier(y, score, g, constraints=("demographic_parity",))
    assert frontier.height == 1
    row = frontier.row(0, named=True)
    assert not str(row["note"]).startswith("failed"), row["note"]
    assert row["disparate_impact"] > blind


def test_threshold_optimizer_output_is_not_degenerate(disparate):
    """A DummyClassifier estimator silently approves everyone, which looks
    like perfect parity and means nothing. Guard against that regression."""
    y, score, g = disparate
    row = threshold_optimizer_frontier(y, score, g, constraints=("demographic_parity",)).row(
        0, named=True
    )
    assert 0.05 < row["approval_rate"] < 0.999


def test_threshold_optimizer_labels_itself_as_not_deployable(disparate):
    y, score, g = disparate
    note = threshold_optimizer_frontier(y, score, g, constraints=("demographic_parity",))["note"][0]
    assert "unlawful" in note


def test_a_failing_constraint_is_reported_not_raised(disparate):
    y, score, g = disparate
    out = threshold_optimizer_frontier(y, score, g, constraints=("not_a_constraint",))
    assert out.height == 1
    assert str(out["note"][0]).startswith("failed")
    assert isinstance(out, pl.DataFrame)
