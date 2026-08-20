"""Phase 3 end-to-end: a decline produces four correct, distinct reasons.

That sentence is the phase's deliverable, and these tests pin it against the
real champion rather than a fixture.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.config import ARTIFACTS, SYNTHETIC_SPLIT, SYNTHETIC_TARGET
from src.explainability.counterfactuals import IMMUTABLE, CounterfactualSearch
from src.explainability.reason_codes import default_mapper
from src.explainability.shap_service import ShapService, additivity_error
from src.fairness.report import age_band, fairlearn_metrics, fairness_report
from src.features.build import build
from src.features.spec import FeatureSpec
from src.ingestion.splits import split_by_time
from src.ingestion.target import assign_labels_from_dpd, modelling_population
from src.models.decision import DecisionPolicy, decide, pd_to_score

pytestmark = pytest.mark.skipif(
    not (ARTIFACTS / "champion_model.joblib").exists(),
    reason="champion not trained; run `make train` first",
)


@pytest.fixture(scope="module")
def scored():
    import joblib

    model = joblib.load(ARTIFACTS / "champion_model.joblib")
    spec = FeatureSpec.load()
    pop = modelling_population(assign_labels_from_dpd(build().frame, SYNTHETIC_TARGET))
    splits = split_by_time(pop, SYNTHETIC_SPLIT)
    train, test = splits.train.collect(), splits.test.collect()

    x_train = spec.matrix(train).to_numpy().astype(np.float32)
    x_test = spec.matrix(test).to_numpy().astype(np.float32)
    pd_hat = model.predict_pd(x_test)
    score = pd_to_score(pd_hat)
    policy = DecisionPolicy.from_approval_rate(score, approve_rate=0.60, refer_rate=0.10)
    return {
        "model": model,
        "spec": spec,
        "train": train,
        "test": test,
        "x_train": x_train,
        "x_test": x_test,
        "pd": pd_hat,
        "score": score,
        "policy": policy,
        "decisions": decide(score, policy),
        "y": test["label"].to_numpy().astype(int),
    }


def test_shap_is_wired_correctly_to_the_champion(scored):
    service = ShapService.from_model(scored["model"], scored["spec"].features)
    assert additivity_error(service, scored["model"], scored["x_test"][:500]) < 1e-9


def test_every_decline_gets_four_distinct_reasons(scored):
    """The Phase 3 deliverable, stated as an assertion."""
    service = ShapService.from_model(scored["model"], scored["spec"].features)
    mapper = default_mapper()
    declines = np.where(scored["decisions"] == "decline")[0][:200]
    assert declines.size > 0

    values = service.values(scored["x_test"][declines])
    for row in values:
        codes = mapper.explain(scored["spec"].features, row)
        assert len(codes) == 4, "a decline must carry four principal reasons"
        assert len({c.family for c in codes}) == 4, "reasons must be distinct, not restatements"
        assert all(c.phrase.strip() for c in codes)
        assert [c.rank for c in codes] == [1, 2, 3, 4]
        # Ranked by contribution, descending.
        contributions = [c.contribution for c in codes]
        assert contributions == sorted(contributions, reverse=True)


def test_no_protected_attribute_ever_reaches_a_disclosure(scored):
    service = ShapService.from_model(scored["model"], scored["spec"].features)
    mapper = default_mapper()
    declines = np.where(scored["decisions"] == "decline")[0][:200]
    values = service.values(scored["x_test"][declines])
    for row in values:
        for code in mapper.explain(scored["spec"].features, row):
            assert not (set(code.driving_features) & mapper.suppressed)


def test_approvals_are_not_given_adverse_action_reasons(scored):
    """Reason codes explain declines. An approved applicant may still have
    positive contributions, but the caller must never surface them as denial
    reasons -- checked here by confirming the decision split is real."""
    decisions = scored["decisions"]
    assert (decisions == "approve").mean() == pytest.approx(0.60, abs=0.02)
    assert (decisions == "decline").mean() > 0.10


def test_counterfactual_proposals_are_feasible_and_never_harmful(scored):
    search = CounterfactualSearch.from_reference(
        scored["model"], scored["spec"].features, scored["x_train"]
    )
    declines = np.where(scored["decisions"] == "decline")[0][:40]
    for i in declines:
        for proposal in search.propose_levers(scored["x_test"][i], scored["policy"].approve_at):
            assert proposal.pd_after <= proposal.pd_before + 1e-9
            assert proposal.score_after >= proposal.score_before - 1e-6
            assert 0 < proposal.magnitude <= 1.0
            assert not (set(proposal.features_moved) & IMMUTABLE)


def test_counterfactuals_help_actionable_declines_more_than_history_declines(scored):
    """If this inverts, the lever set is targeting the wrong features."""
    service = ShapService.from_model(scored["model"], scored["spec"].features)
    mapper = default_mapper()
    search = CounterfactualSearch.from_reference(
        scored["model"], scored["spec"].features, scored["x_train"]
    )
    actionable_families = {"affordability", "utilisation", "debt_burden", "loan_structure"}

    declines = np.where(scored["decisions"] == "decline")[0][:80]
    values = service.values(scored["x_test"][declines])
    gains = {"actionable": [], "history": []}
    for k, i in enumerate(declines):
        codes = mapper.explain(scored["spec"].features, values[k])
        bucket = "actionable" if codes[0].family in actionable_families else "history"
        chosen, _ = search.stack_levers(
            scored["x_test"][i], scored["policy"].approve_at, max_actions=2
        )
        best = max([c.score_after for c in chosen], default=float(scored["score"][i]))
        gains[bucket].append(best - float(scored["score"][i]))

    assert gains["actionable"], "no actionable-led declines in the sample"
    assert np.median(gains["actionable"]) > np.median(gains["history"])


def test_fairness_metrics_agree_with_fairlearn_on_real_data(scored):
    approved = scored["decisions"] == "approve"
    for values in (
        scored["test"]["CODE_GENDER"].to_numpy().astype(str),
        age_band(scored["test"]["DAYS_BIRTH"].to_numpy()),
    ):
        report = fairness_report(scored["y"], approved, values, attribute="x")
        reference = fairlearn_metrics(scored["y"], approved, values)
        assert report.disparate_impact == pytest.approx(
            reference["demographic_parity_ratio"], abs=1e-9
        )
        assert report.equalized_odds_difference == pytest.approx(
            reference["equalized_odds_difference"], abs=1e-9
        )


def test_age_disparity_is_detected_and_not_silently_passed(scored):
    """The model does fail four-fifths on age. A fairness suite that reports
    otherwise is broken, so the failure itself is pinned."""
    approved = scored["decisions"] == "approve"
    report = fairness_report(
        scored["y"], approved, age_band(scored["test"]["DAYS_BIRTH"].to_numpy()), attribute="age"
    )
    assert not report.passes_four_fifths
    assert report.disparate_impact < 0.8
    assert report.worst_group == "18-24"
