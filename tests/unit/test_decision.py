"""PD -> score -> decision. Policy, not modelling, and tested as such."""

from __future__ import annotations

import numpy as np
import pytest

from src.models.decision import (
    BASE_ODDS,
    BASE_SCORE,
    SCORE_MAX,
    SCORE_MIN,
    Decision,
    DecisionPolicy,
    decide,
    expected_loss,
    pd_to_score,
    portfolio_summary,
    score_to_pd,
)
from src.models.scorecard import PDO


def test_score_and_pd_round_trip():
    p = np.array([0.01, 0.05, 0.15, 0.4, 0.7])
    assert score_to_pd(pd_to_score(p)) == pytest.approx(p, abs=1e-6)


def test_higher_pd_always_means_lower_score():
    p = np.linspace(0.01, 0.9, 50)
    s = pd_to_score(p)
    assert np.all(np.diff(s) < 0)


def test_base_odds_land_on_the_base_score():
    assert pd_to_score(1 / (1 + BASE_ODDS)) == pytest.approx(BASE_SCORE, abs=1e-6)


def test_every_pdo_points_doubles_the_odds():
    p1 = 1 / (1 + BASE_ODDS)
    p2 = 1 / (1 + BASE_ODDS * 2)
    assert pd_to_score(p2) - pd_to_score(p1) == pytest.approx(PDO, abs=1e-6)


def test_scores_are_clipped_to_the_industry_range():
    s = pd_to_score(np.array([1e-12, 1 - 1e-12]))
    assert s.max() <= SCORE_MAX and s.min() >= SCORE_MIN


def test_three_way_decision_bands():
    policy = DecisionPolicy(approve_at=600, refer_at=550)
    out = decide(np.array([610.0, 600.0, 575.0, 550.0, 549.0]), policy)
    assert list(out) == [
        Decision.APPROVE,
        Decision.APPROVE,
        Decision.REFER,
        Decision.REFER,
        Decision.DECLINE,
    ]


def test_policy_rejects_inverted_cutoffs():
    with pytest.raises(ValueError, match="must not exceed"):
        DecisionPolicy(approve_at=550, refer_at=600)


def test_policy_from_approval_rate_hits_the_target():
    rng = np.random.default_rng(0)
    scores = rng.normal(550, 30, 20_000)
    policy = DecisionPolicy.from_approval_rate(scores, approve_rate=0.6, refer_rate=0.1)
    decisions = decide(scores, policy)
    assert (decisions == Decision.APPROVE).mean() == pytest.approx(0.6, abs=0.01)
    assert (decisions == Decision.REFER).mean() == pytest.approx(0.1, abs=0.01)


def test_policy_from_approval_rate_validates_its_inputs():
    scores = np.linspace(400, 700, 100)
    with pytest.raises(ValueError, match="approve_rate"):
        DecisionPolicy.from_approval_rate(scores, approve_rate=1.5)
    with pytest.raises(ValueError, match="refer_rate"):
        DecisionPolicy.from_approval_rate(scores, approve_rate=0.9, refer_rate=0.2)


def test_expected_loss_is_pd_times_lgd_times_exposure():
    policy = DecisionPolicy(lgd=0.65)
    el = expected_loss(np.array([0.10]), np.array([10_000.0]), policy)
    assert el[0] == pytest.approx(0.10 * 0.65 * 10_000)


def test_portfolio_bad_rate_falls_from_decline_to_approve():
    rng = np.random.default_rng(1)
    n = 20_000
    p = rng.beta(2, 8, n)
    y = rng.binomial(1, p)
    policy = DecisionPolicy.from_approval_rate(pd_to_score(p), approve_rate=0.6, refer_rate=0.1)
    summary = portfolio_summary(p, y, policy)
    rates = summary["observed_bad_rate"].to_list()  # decline, refer, approve
    assert rates[0] > rates[1] > rates[2]
