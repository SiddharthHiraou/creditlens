"""The logistic scorecard track and its PDO scaling."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from src.models.scorecard import BASE_ODDS, BASE_SCORE, PDO, fit_scorecard


@pytest.fixture(scope="module")
def fitted():
    rng = np.random.default_rng(0)
    n = 12_000
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = rng.binomial(1, 1 / (1 + np.exp(-(-2.0 + 1.4 * x1 + 0.6 * x2))))
    df = pl.DataFrame({"x1": x1, "x2": x2, "label": y})
    return fit_scorecard(df, ["x1", "x2"]), df


def test_pdo_scaling_constants_are_the_industry_convention(fitted):
    sc, _ = fitted
    assert sc.factor == pytest.approx(PDO / np.log(2))
    assert sc.offset == pytest.approx(BASE_SCORE - sc.factor * np.log(BASE_ODDS))


def test_every_pdo_points_doubles_the_odds(fitted):
    """The defining property of the scaling: +20 points = 2x odds of good."""
    sc, _ = fitted
    odds = 50.0
    s1 = sc.offset + sc.factor * np.log(odds)
    s2 = sc.offset + sc.factor * np.log(odds * 2)
    assert (s2 - s1) == pytest.approx(PDO)


def test_base_odds_map_to_the_base_score(fitted):
    sc, _ = fitted
    assert sc.offset + sc.factor * np.log(BASE_ODDS) == pytest.approx(BASE_SCORE)


def test_score_moves_opposite_to_pd(fitted):
    """Higher score must mean lower risk, matching consumer convention."""
    sc, df = fitted
    pd_hat = sc.predict_pd(df)
    score = sc.score(df)
    assert np.corrcoef(pd_hat, score)[0, 1] < -0.95


def test_scores_stay_inside_the_300_850_range(fitted):
    sc, df = fitted
    s = sc.score(df)
    assert s.min() >= 300 and s.max() <= 850


def test_scorecard_discriminates(fitted):
    from sklearn.metrics import roc_auc_score

    sc, df = fitted
    assert roc_auc_score(df["label"].to_numpy(), sc.predict_pd(df)) > 0.70


def test_points_table_covers_every_feature_and_bin(fitted):
    sc, _ = fitted
    tab = sc.points_table()
    assert set(tab["feature"].unique()) == {"x1", "x2"}
    assert tab["points"].is_finite().all()


def test_stronger_feature_has_the_wider_points_range(fitted):
    sc, _ = fitted
    contrib = sc.feature_contributions()
    assert contrib["feature"][0] == "x1"
