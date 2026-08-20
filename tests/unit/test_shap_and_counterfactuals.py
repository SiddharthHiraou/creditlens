"""SHAP wiring and counterfactual lever mechanics."""

from __future__ import annotations

import numpy as np
import pytest

from src.explainability.counterfactuals import (
    ACTIONABLE,
    IMMUTABLE,
    LEVERS,
    BorrowLess,
    ClearArrears,
    CounterfactualSearch,
    PayDownRevolving,
)
from src.explainability.shap_service import ShapService, additivity_error


@pytest.fixture(scope="module")
def tiny_model():
    import lightgbm as lgb

    from src.models.gbdt import GbdtModel

    rng = np.random.default_rng(0)
    n, names = 4_000, ["RATIO_annuity_to_income", "AMT_INCOME_TOTAL", "CC_util_mean", "EXT_mean"]
    x = np.column_stack(
        [
            rng.uniform(0.01, 0.5, n),
            rng.uniform(50_000, 300_000, n),
            rng.uniform(0.0, 1.2, n),
            rng.uniform(0.0, 1.0, n),
        ]
    ).astype(np.float32)
    logit = -2 + 12 * x[:, 0] + 1.5 * x[:, 2] - 3 * x[:, 3]
    y = rng.binomial(1, 1 / (1 + np.exp(-logit)))
    # Constrained, mirroring production: without constraints an unconstrained
    # GBDT is locally non-monotone and the lever assertions below would be
    # testing noise rather than behaviour.
    booster = lgb.LGBMClassifier(
        n_estimators=120,
        verbose=-1,
        random_state=0,
        monotone_constraints=[1, 0, 1, -1],
        monotone_constraints_method="basic",
    ).fit(x, y)
    return GbdtModel(name="lightgbm", booster=booster, feature_names=names, params={}), x, names


def test_shap_satisfies_additivity(tiny_model):
    """expected_value + sum(shap) must equal the raw margin. A larger error
    means a wrong class index, a stale base value or a column mismatch."""
    model, x, _ = tiny_model
    service = ShapService.from_model(model)
    assert additivity_error(service, model, x[:400]) < 1e-6


def test_expected_value_is_read_live_not_cached(tiny_model):
    """SHAP mutates expected_value after the first shap_values call; a cached
    copy is stale by exactly the additivity error."""
    model, x, _ = tiny_model
    service = ShapService.from_model(model)
    service.values(x[:50])
    assert service.expected_value == pytest.approx(float(service.explainer.expected_value))


def test_explain_one_returns_signed_contributions(tiny_model):
    model, x, names = tiny_model
    service = ShapService.from_model(model)
    out = service.explain_one(x[0], top_k=3)
    assert out.height == 3
    assert set(out.columns) >= {"feature", "value", "shap"}
    assert out["abs_shap"].to_list() == sorted(out["abs_shap"].to_list(), reverse=True)


def test_global_importance_shares_sum_to_one(tiny_model):
    model, x, _ = tiny_model
    service = ShapService.from_model(model)
    out = service.global_importance(x)
    assert out["share"].sum() == pytest.approx(1.0)
    # The strongest true driver should top the ranking.
    assert out["feature"][0] == "RATIO_annuity_to_income"


def test_borrow_less_moves_every_amount_ratio_together():
    """The bug this prevents: borrowing less while repaying the same."""
    names = ["RATIO_annuity_to_income", "RATIO_credit_to_income", "AMT_INCOME_TOTAL"]
    row = np.array([0.30, 3.0, 100_000.0])
    out = BorrowLess().apply(row, names, 0.5)
    assert out[0] == pytest.approx(0.15)
    assert out[1] == pytest.approx(1.5)
    assert out[2] == pytest.approx(100_000.0)  # income is not a lever


def test_borrow_less_raises_residual_income_by_the_annuity_saved():
    names = ["RATIO_annuity_to_income", "AMT_INCOME_TOTAL", "RATIO_residual_income"]
    row = np.array([0.20, 100_000.0, 40_000.0])
    out = BorrowLess().apply(row, names, 0.5)
    # annuity = 0.20 * 100000 = 20000; halving saves 12 * 10000 = 120000 a year
    assert out[2] == pytest.approx(40_000.0 + 0.5 * 12 * 20_000.0)


def test_pay_down_revolving_scales_all_utilisation_measures():
    names = ["CC_util_mean", "CC_util_max", "CC_balance_mean", "EXT_mean"]
    row = np.array([0.8, 1.0, 5_000.0, 0.5])
    out = PayDownRevolving().apply(row, names, 0.5)
    assert out[:3] == pytest.approx([0.4, 0.5, 2_500.0])
    assert out[3] == pytest.approx(0.5)  # untouched


def test_clear_arrears_drives_overdue_to_zero_at_full_magnitude():
    names = ["BURO_overdue_total", "BURO_n_overdue_lines"]
    out = ClearArrears().apply(np.array([5_000.0, 3.0]), names, 1.0)
    assert out == pytest.approx([0.0, 0.0])


def test_levers_preserve_nan_rather_than_inventing_history():
    names = ["CC_util_mean", "CC_balance_mean"]
    out = PayDownRevolving().apply(np.array([np.nan, np.nan]), names, 0.5)
    assert np.isnan(out).all()


def test_no_lever_ever_touches_an_immutable_feature():
    names = sorted(IMMUTABLE)
    row = np.arange(len(names), dtype=float) + 1.0
    for lever in LEVERS:
        assert lever.apply(row, names, 1.0) == pytest.approx(row), (
            f"{lever.key} moved a protected feature"
        )


def test_actionable_set_and_immutable_set_do_not_overlap():
    assert not (set(ACTIONABLE) & IMMUTABLE)


def test_proposals_are_clipped_to_the_observed_range(tiny_model):
    model, x, names = tiny_model
    search = CounterfactualSearch.from_reference(model, names, x)
    extreme = x[0].copy()
    extreme[0] = 10.0  # far outside the training range
    clipped = search._clip(extreme.reshape(1, -1))[0]
    assert clipped[0] <= search.upper[0] + 1e-6


def test_lever_proposals_never_increase_predicted_pd(tiny_model):
    model, x, names = tiny_model
    search = CounterfactualSearch.from_reference(model, names, x)
    for row in x[:15]:
        for proposal in search.propose_levers(row, target_score=900.0):
            assert proposal.pd_after <= proposal.pd_before + 1e-9
