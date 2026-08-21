"""The generated governance documents.

These are generated rather than hand-written because a governance document whose
numbers disagree with the model is worse than none — it will be believed. That
only holds if the generator is correct, so it is tested: the figures must match
the artifacts, and no placeholder may survive into the output.
"""

from __future__ import annotations

import json
import re

import pytest

from src.config import ARTIFACTS, DOCS
from src.features.dictionary import classify
from src.features.dictionary import generate as generate_dictionary
from src.features.spec import FeatureSpec
from src.governance import model_card, validation_report

pytestmark = pytest.mark.skipif(
    not (ARTIFACTS / "phase3_report.json").exists(),
    reason="artifacts absent; run `make train && make audit` first",
)


@pytest.fixture(scope="module")
def metrics() -> dict:
    return json.loads((ARTIFACTS / "phase2_metrics.json").read_text())


@pytest.fixture(scope="module")
def card() -> str:
    return model_card()


@pytest.fixture(scope="module")
def report() -> str:
    return validation_report()


@pytest.fixture(scope="module")
def dictionary() -> str:
    return generate_dictionary()


@pytest.mark.parametrize("doc", ["card", "report", "dictionary"])
def test_no_unrendered_placeholders(doc, request):
    """A broken f-string leaves a literal brace and a wrong document."""
    text = request.getfixturevalue(doc)
    leftovers = re.findall(r"\{[a-z_][a-z0-9_\[\]\"']*\}", text)
    assert not leftovers, f"unrendered placeholders: {leftovers[:5]}"


def test_model_card_reports_the_actual_champion_metrics(card, metrics):
    champ = metrics["champion_calibrated_test"]
    assert f"{champ['auc']:.4f}" in card
    assert f"{champ['gini']:.4f}" in card
    assert f"{champ['ks']:.4f}" in card
    assert metrics["champion"] in card


def test_model_card_states_the_synthetic_data_limitation(card):
    """The most important caveat must not be quietly dropped by an edit."""
    assert "synthetic" in card.lower()
    assert "not on real borrowers" in card.lower() or "not real borrowers" in card.lower()


def test_model_card_reports_the_fairness_failure(card):
    """A model card that omits a four-fifths failure is not a model card."""
    fairness = json.loads((ARTIFACTS / "phase3_report.json").read_text())["fairness"]
    assert f"{fairness['age_band']['disparate_impact']:.4f}" in card
    assert "fails" in card.lower()


def test_model_card_refuses_to_report_accuracy(card):
    """Accuracy on an imbalanced target is meaningless and appears nowhere."""
    assert "Accuracy is reported nowhere" in card


def test_model_card_lists_out_of_scope_uses(card):
    lowered = card.lower()
    assert "out of scope" in lowered
    for use in ("mortgage", "commercial", "pricing"):
        assert use in lowered


def test_validation_report_follows_sr_11_7(report):
    """The three pillars the guidance is organised around."""
    lowered = report.lower()
    for pillar in ("conceptual soundness", "outcomes analysis", "ongoing monitoring"):
        assert pillar in lowered
    assert "sr 11-7" in lowered


def test_validation_report_states_conditions_not_just_praise(report):
    assert "conditional approval" in report.lower()
    assert "Conditions" in report
    # The two blocking conditions must both be present.
    assert "real" in report.lower() and "fair-lending review" in report.lower()


def test_validation_report_credits_self_identified_defects(report):
    """Effective challenge means saying where the assurance came from."""
    assert "effective challenge" in report.lower()
    assert "monotonicity" in report.lower()


def test_data_dictionary_covers_every_built_feature(dictionary):
    from src.features.build import build

    for feature in build().feature_names:
        assert f"`{feature}`" in dictionary, f"{feature} missing from the dictionary"


def test_data_dictionary_marks_selected_features(dictionary):
    spec = FeatureSpec.load()
    assert f"**{len(spec.features)}** selected" in dictionary
    assert f"**{len(spec.monotonic)}** carry a monotonic constraint" in dictionary


def test_data_dictionary_lists_suppressed_features(dictionary):
    from src.explainability.reason_codes import ReasonCodeMapper

    for feature in ReasonCodeMapper.load().suppressed:
        assert f"`{feature}`" in dictionary


@pytest.mark.parametrize(
    ("feature", "table"),
    [
        ("BURO_n_lines", "bureau"),
        ("BB_n_late_total", "bureau_balance"),
        ("INST_late_share_total", "installments_payments"),
        ("CC_util_mean", "credit_card_balance"),
        ("POS_dpd_max", "POS_CASH_balance"),
        ("PREV_n_refused", "previous_application"),
        ("RATIO_annuity_to_income", "derived (application)"),
        ("XSRC_total_debt_to_income", "derived (cross-source)"),
    ],
)
def test_features_are_attributed_to_the_right_source_table(feature, table):
    """BB_ must resolve before BURO_ — prefix order matters."""
    assert classify(feature)[0] == table


def test_generated_docs_are_written_to_disk(dictionary):
    for name in ("model_card.md", "model_validation_report.md", "data_dictionary.md"):
        assert (DOCS / name).exists()
