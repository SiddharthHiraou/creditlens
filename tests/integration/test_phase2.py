"""Phase 2 end-to-end: features -> selection -> GBDT -> calibration.

Small synthetic draw so this stays inside a CI budget. The full-scale numbers
come from `make train`.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.config import SYNTHETIC_SPLIT, SYNTHETIC_TARGET
from src.evaluation.metrics import discrimination
from src.evaluation.psi import psi
from src.features.build import build
from src.features.monotonic import constraints_for
from src.features.selection import select
from src.features.spec import FeatureSpec
from src.ingestion.splits import split_by_time
from src.ingestion.synthetic import SyntheticConfig, generate, write
from src.ingestion.target import assign_labels_from_dpd, modelling_population
from src.models.baseline import fit as fit_baseline
from src.models.baseline import prepare
from src.models.calibrate import calibrate, calibration_report
from src.models.gbdt import fit_lightgbm, to_matrix


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    """Generate a small dataset into a temp dir and build the full matrix."""
    out = tmp_path_factory.mktemp("syn")
    write(generate(SyntheticConfig(n_applicants=12_000)), out_dir=out)

    from src.ingestion import loaders

    original = loaders.SYNTHETIC
    loaders.SYNTHETIC = out
    try:
        fm = build(validate=False)
        pop = modelling_population(assign_labels_from_dpd(fm.frame, SYNTHETIC_TARGET))
        splits = split_by_time(pop, SYNTHETIC_SPLIT)
        frames = {
            "train": splits.train.collect(),
            "cal": splits.calibration.collect(),
            "valid": splits.valid.collect(),
            "test": splits.test.collect(),
        }
        yield fm, frames
    finally:
        loaders.SYNTHETIC = original


def test_feature_count_lands_in_the_intended_range(prepared):
    fm, _ = prepared
    assert 150 <= fm.n_features <= 300, f"built {fm.n_features}, brief targets 150-300"


def test_selection_reduces_the_matrix_and_records_every_reason(prepared):
    fm, frames = prepared
    report = select(frames["train"], fm.feature_names, use_null_importance=False)
    assert 0 < len(report.kept) < fm.n_features
    assert set(fm.feature_names) - set(report.kept) == set(report.dropped)


def test_champion_beats_the_phase_1_baseline_out_of_time(prepared):
    """The claim Phase 2 exists to make. If this fails, the feature work did
    not earn its complexity and the scorecard should ship instead."""
    fm, frames = prepared
    train, valid, test = frames["train"], frames["valid"], frames["test"]
    y_test = test["label"].to_numpy().astype(int)

    baseline = fit_baseline(prepare(train.lazy()).collect())
    baseline_auc = discrimination(y_test, baseline.predict_pd(prepare(test.lazy()).collect())).auc

    report = select(train, fm.feature_names, use_null_importance=False)
    numeric = [f for f in report.kept if train[f].dtype.is_numeric()]
    model = fit_lightgbm(
        to_matrix(train, numeric),
        train["label"].to_numpy().astype(int),
        numeric,
        monotone=constraints_for(numeric),
        eval_set=(to_matrix(valid, numeric), valid["label"].to_numpy().astype(int)),
    )
    champion_auc = discrimination(y_test, model.predict_pd(to_matrix(test, numeric))).auc

    assert champion_auc > baseline_auc, (
        f"champion {champion_auc:.4f} did not beat baseline {baseline_auc:.4f}"
    )


def test_monotonic_constraints_hold_in_the_fitted_model(prepared):
    """A constrained feature must never make the model *less* risky as it rises.

    Checked empirically by sweeping one feature across its range with all
    others held at their median -- a partial dependence probe.
    """
    fm, frames = prepared
    train, valid = frames["train"], frames["valid"]
    report = select(train, fm.feature_names, use_null_importance=False)
    numeric = [f for f in report.kept if train[f].dtype.is_numeric()]
    mono = constraints_for(numeric)
    if not any(mono):
        pytest.skip("no constrained features survived selection")

    model = fit_lightgbm(
        to_matrix(train, numeric),
        train["label"].to_numpy().astype(int),
        numeric,
        monotone=mono,
        eval_set=(to_matrix(valid, numeric), valid["label"].to_numpy().astype(int)),
    )

    x = to_matrix(train, numeric)
    median = np.nanmedian(x, axis=0)
    for i, direction in enumerate(mono):
        if direction == 0:
            continue
        col = x[:, i]
        col = col[np.isfinite(col)]
        if col.size < 100 or np.unique(col).size < 5:
            continue
        grid = np.quantile(col, np.linspace(0.05, 0.95, 12))
        probe = np.tile(median, (len(grid), 1)).astype(np.float32)
        probe[:, i] = grid
        preds = model.predict_pd(probe)
        diffs = np.diff(preds)
        tol = 1e-6
        if direction > 0:
            assert (diffs >= -tol).all(), f"{numeric[i]} is constrained +1 but PD fell"
        else:
            assert (diffs <= tol).all(), f"{numeric[i]} is constrained -1 but PD rose"


def test_calibration_moves_mean_pd_onto_the_observed_bad_rate(prepared):
    fm, frames = prepared
    train, cal, valid, test = (frames[k] for k in ("train", "cal", "valid", "test"))
    report = select(train, fm.feature_names, use_null_importance=False)
    numeric = [f for f in report.kept if train[f].dtype.is_numeric()]

    model = fit_lightgbm(
        to_matrix(train, numeric),
        train["label"].to_numpy().astype(int),
        numeric,
        eval_set=(to_matrix(valid, numeric), valid["label"].to_numpy().astype(int)),
    )
    calibrated = calibrate(model, to_matrix(cal, numeric), cal["label"].to_numpy().astype(int))

    y_test = test["label"].to_numpy().astype(int)
    x_test = to_matrix(test, numeric)
    rep = calibration_report(
        y_test, calibrated.predict_pd_uncalibrated(x_test), calibrated.predict_pd(x_test)
    )

    assert rep["brier_calibrated"] < rep["brier_raw"]
    assert abs(rep["mean_pd_calibrated"] - rep["actual_bad_rate"]) < abs(
        rep["mean_pd_raw"] - rep["actual_bad_rate"]
    )


def test_score_is_stable_between_train_and_out_of_time(prepared):
    """PSI on the model's own score. A large value here means the population
    moved, which is the earliest warning available before outcomes arrive."""
    fm, frames = prepared
    train, valid, test = frames["train"], frames["valid"], frames["test"]
    report = select(train, fm.feature_names, use_null_importance=False)
    numeric = [f for f in report.kept if train[f].dtype.is_numeric()]
    model = fit_lightgbm(
        to_matrix(train, numeric),
        train["label"].to_numpy().astype(int),
        numeric,
        eval_set=(to_matrix(valid, numeric), valid["label"].to_numpy().astype(int)),
    )
    result = psi(
        model.predict_pd(to_matrix(train, numeric)), model.predict_pd(to_matrix(test, numeric))
    )
    assert np.isfinite(result.psi)
    assert result.psi < 0.25, f"score PSI {result.psi:.4f} is at alarm level on stable data"


def test_feature_spec_round_trips_and_pins_order(prepared, tmp_path):
    fm, frames = prepared
    train = frames["train"]
    report = select(train, fm.feature_names, use_null_importance=False)
    numeric = [f for f in report.kept if train[f].dtype.is_numeric()]
    spec = FeatureSpec.build(train, numeric, fm.categorical_names, dropped=report.dropped)
    loaded = FeatureSpec.load(spec.save(tmp_path / "spec.yaml"))
    assert loaded.features == numeric
    assert loaded.matrix(train).columns == numeric
