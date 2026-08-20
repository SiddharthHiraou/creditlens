"""Prefect flows, and the promotion gate in particular.

The gate is the MLOps centerpiece, so it is tested in both directions: it must
pass a model that is fine and refuse one that is not. A gate only ever tested
against a good candidate is a gate nobody knows works.
"""

from __future__ import annotations

import json

import joblib
import pytest

from flows.monitoring import compute_drift, ingest_and_validate
from flows.retraining import (
    MAX_AUC_REGRESSION,
    drift_triggered_retrain,
    run_gate,
    validate_and_promote,
)
from src.config import ARTIFACTS

pytestmark = pytest.mark.skipif(
    not (ARTIFACTS / "champion_model.joblib").exists(),
    reason="champion not trained; run `make train` first",
)


def test_ingest_flow_validates_every_source_table():
    result = ingest_and_validate()
    assert len(result["tables"]) == 7
    assert all(t["valid"] for t in result["tables"])
    assert result["total_rows"] > 1_000_000


def test_drift_flow_computes_psi_and_records_history():
    result = compute_drift()
    assert result["score_psi"] >= 0
    assert result["verdict"] in {"stable", "moderate shift", "significant shift"}
    assert len(result["worst_features"]) > 0
    assert (ARTIFACTS / "monitoring_history.jsonl").exists()


def test_drift_flow_raises_an_alert_only_above_the_threshold(tmp_path, monkeypatch):
    from flows import monitoring

    alert = tmp_path / "drift_alert.json"
    monkeypatch.setattr(monitoring, "DRIFT_ALERT", alert)
    monkeypatch.setattr(monitoring, "MONITORING_LOG", tmp_path / "history.jsonl")

    # Real PSI here is ~0.017, so a normal run must not alert...
    assert compute_drift()["is_alarm"] is False
    assert not alert.exists()

    # ...and an impossibly low threshold must.
    assert compute_drift(alarm_threshold=0.0001)["is_alarm"] is True
    assert alert.exists()


def test_retrain_does_not_fire_without_an_alert(tmp_path, monkeypatch):
    from flows import retraining

    monkeypatch.setattr(retraining, "DRIFT_ALERT", tmp_path / "absent.json")
    result = drift_triggered_retrain()
    assert result["triggered"] is False


def test_gate_passes_a_candidate_identical_to_the_incumbent(tmp_path):
    champion = ARTIFACTS / "champion_model.joblib"
    candidate = tmp_path / "candidate_same.joblib"
    joblib.dump(joblib.load(champion), candidate)

    checks = run_gate(champion, candidate)
    assert len(checks) == 5
    assert all(c.passed for c in checks), [c.render() for c in checks if not c.passed]


@pytest.fixture(scope="module")
def crippled(tmp_path_factory):
    """A real but deliberately weak model: six decision stumps.

    Not a mock — a genuinely worse artifact of the kind a bad config or a
    truncated training run actually produces.
    """

    from src.config import SYNTHETIC_SPLIT, SYNTHETIC_TARGET
    from src.features.build import build
    from src.features.spec import FeatureSpec
    from src.ingestion.splits import split_by_time
    from src.ingestion.target import assign_labels_from_dpd, modelling_population
    from src.models.calibrate import calibrate
    from src.models.gbdt import fit_lightgbm, to_matrix

    spec = FeatureSpec.load()
    pop = modelling_population(assign_labels_from_dpd(build().frame, SYNTHETIC_TARGET))
    splits = split_by_time(pop, SYNTHETIC_SPLIT)
    train, cal = splits.train.collect(), splits.calibration.collect()

    model = fit_lightgbm(
        to_matrix(train, spec.features),
        train["label"].to_numpy().astype(int),
        spec.features,
        params={"n_estimators": 6, "num_leaves": 2, "max_depth": 1, "learning_rate": 0.05},
    )
    calibrated = calibrate(
        model, to_matrix(cal, spec.features), cal["label"].to_numpy().astype(int)
    )
    path = tmp_path_factory.mktemp("cand") / "candidate_weak.joblib"
    joblib.dump(calibrated, path)
    return path


def test_gate_refuses_a_materially_worse_candidate(crippled):
    checks = run_gate(ARTIFACTS / "champion_model.joblib", crippled)
    discrimination = next(c for c in checks if c.name == "discrimination")
    assert not discrimination.passed
    assert discrimination.candidate < discrimination.incumbent * (1 - MAX_AUC_REGRESSION)


def test_a_failed_gate_writes_an_issue_and_does_not_promote(crippled, tmp_path, monkeypatch):
    from flows import retraining

    monkeypatch.setattr(retraining, "ISSUE_DIR", tmp_path / "issues")
    monkeypatch.setattr(retraining, "GATE_REPORT", tmp_path / "gate.json")

    report = validate_and_promote(candidate_path=str(crippled))
    assert report["promoted"] is False
    assert "issue" in report

    issue = json.loads((tmp_path / "gate.json").read_text())
    assert issue["promoted"] is False

    body = (tmp_path / "issues").glob("promotion-gate-*.md")
    text = next(body).read_text()
    assert "checks failed" in text
    assert "The incumbent remains in production" in text


def test_gate_covers_every_dimension_the_policy_requires():
    """docs/credit_policy.md §8.5 names five conditions. All five must be gated."""
    champion = ARTIFACTS / "champion_model.joblib"
    names = {c.name for c in run_gate(champion, champion)}
    assert names == {"discrimination", "calibration", "stability", "rank_ordering", "fairness"}
