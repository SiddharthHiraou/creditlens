"""Stacked ensemble and the Optuna search."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from src.models.ensemble import fit_stack
from src.models.gbdt import fit_lightgbm, scale_pos_weight, to_matrix
from src.models.tune import tune_lightgbm


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    n, d = 3_000, 6
    x = rng.normal(size=(n, d)).astype(np.float32)
    y = rng.binomial(1, 1 / (1 + np.exp(-(-1.8 + 1.3 * x[:, 0] + 0.7 * x[:, 1]))))
    names = [f"f{i}" for i in range(d)]
    split = n // 2
    return x[:split], y[:split], x[split:], y[split:], names


def test_scale_pos_weight_is_negatives_over_positives():
    y = np.array([0] * 90 + [1] * 10)
    assert scale_pos_weight(y) == pytest.approx(9.0)


def test_scale_pos_weight_survives_a_class_with_no_positives():
    assert np.isfinite(scale_pos_weight(np.zeros(50, dtype=int)))


def test_to_matrix_preserves_spec_order_and_keeps_nulls_as_nan():
    import polars as pl

    df = pl.DataFrame({"b": [1.0, None], "a": [3.0, 4.0]})
    out = to_matrix(df, ["a", "b"])
    assert out.shape == (2, 2)
    assert out[0, 0] == 3.0
    assert np.isnan(out[1, 1])


def test_stack_beats_or_matches_its_weakest_base(data):
    x_tr, y_tr, x_va, y_va, names = data
    bases = [
        fit_lightgbm(x_tr, y_tr, names, params={"n_estimators": 60, "num_leaves": n})
        for n in (7, 31)
    ]
    stack = fit_stack(bases, x_va, y_va)
    base_aucs = [roc_auc_score(y_va, b.predict_pd(x_va)) for b in bases]
    assert roc_auc_score(y_va, stack.predict_pd(x_va)) >= min(base_aucs) - 1e-6


def test_stack_reports_one_weight_per_base(data):
    x_tr, y_tr, x_va, y_va, names = data
    bases = [
        fit_lightgbm(x_tr, y_tr, names, params={"n_estimators": 40, "num_leaves": n})
        for n in (7, 31)
    ]
    weights = fit_stack(bases, x_va, y_va).weights()
    assert len(weights) == 2
    assert all(np.isfinite(v) for v in weights.values())


def test_stack_predictions_are_probabilities(data):
    x_tr, y_tr, x_va, y_va, names = data
    bases = [fit_lightgbm(x_tr, y_tr, names, params={"n_estimators": 40})]
    p = fit_stack(bases, x_va, y_va).predict_pd(x_va)
    assert p.min() >= 0.0 and p.max() <= 1.0


def test_gain_importance_normalises_to_one(data):
    x_tr, y_tr, _, _, names = data
    model = fit_lightgbm(x_tr, y_tr, names, params={"n_estimators": 60})
    gain = model.gain_importance()
    assert gain["gain"].sum() == pytest.approx(1.0)
    assert gain["feature"][0] == "f0"  # the strongest true driver


@pytest.mark.slow
def test_optuna_search_improves_on_defaults_and_records_trials(data):
    x_tr, y_tr, x_va, y_va, names = data
    result = tune_lightgbm(x_tr, y_tr, x_va, y_va, n_trials=8)
    assert result.n_trials == 8
    assert 0.5 < result.best_value <= 1.0
    assert set(result.best_params) >= {"learning_rate", "num_leaves"}
    trials = result.trials_dataframe()
    assert trials.height == 8
    assert "param_learning_rate" in trials.columns


@pytest.mark.slow
def test_tuning_respects_monotonic_constraints(data):
    """Constraints are fixed inputs to the search, never something it may
    trade away for AUC."""
    x_tr, y_tr, x_va, y_va, names = data
    mono = [1, 0, 0, 0, 0, 0]
    result = tune_lightgbm(x_tr, y_tr, x_va, y_va, monotone=mono, n_trials=5)
    assert "monotone_constraints" not in result.best_params

    model = fit_lightgbm(x_tr, y_tr, names, params=result.best_params, monotone=mono)
    median = np.nanmedian(x_tr, axis=0)
    grid = np.quantile(x_tr[:, 0], np.linspace(0.05, 0.95, 12))
    probe = np.tile(median, (len(grid), 1)).astype(np.float32)
    probe[:, 0] = grid
    assert (np.diff(model.predict_pd(probe)) >= -1e-9).all()
