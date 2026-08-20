"""The GenAI layer's guardrails, which are the part that must not regress."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.llm.client import PRICING, cost_per_1000_decisions, cost_usd, prompt_hash
from src.llm.memo_agent import MAX_REASONS, generate, validate_grounding
from src.llm.schemas import AdverseActionMemo, MemoInput
from src.llm.tools import (
    PORTFOLIO_QUERIES,
    UnknownQueryError,
    get_model_metrics,
    query_portfolio_stats,
    search_credit_policy,
)

REASONS = [
    {
        "rank": 1,
        "family": "affordability",
        "label": "Debt-to-income",
        "phrase": "Proposed monthly payment is high relative to stated income",
        "actionable": True,
    },
    {
        "rank": 2,
        "family": "repayment_history",
        "label": "Repayment record",
        "phrase": "Record of late or partial payments on prior obligations",
        "actionable": True,
    },
    {
        "rank": 3,
        "family": "bureau_delinquency",
        "label": "Delinquency on credit file",
        "phrase": "Credit file shows past-due amounts",
        "actionable": True,
    },
    {
        "rank": 4,
        "family": "external_score",
        "label": "Credit bureau score",
        "phrase": "Agency score is below the level required",
        "actionable": False,
    },
]


@pytest.fixture
def payload() -> MemoInput:
    return MemoInput(
        decision_id="d-1",
        decision="decline",
        score=512.0,
        reason_codes=REASONS,
        requested_amount=616_300,
        stated_income_band="75,000-100,000",
    )


# -- cost accounting ---------------------------------------------------------


def test_cost_is_input_and_output_priced_separately():
    assert cost_usd("claude-haiku-4-5", 1_000_000, 0) == pytest.approx(1.00)
    assert cost_usd("claude-haiku-4-5", 0, 1_000_000) == pytest.approx(5.00)


def test_memo_tier_is_cheaper_than_copilot_tier():
    """The tiering only makes sense if the volume workload is on the cheap tier."""
    assert PRICING["claude-haiku-4-5"] < PRICING["claude-sonnet-5"]


def test_unit_economics_only_charges_for_non_approvals():
    econ = cost_per_1000_decisions(memo_rate=0.4)
    assert econ["memos_per_1000_decisions"] == 400
    assert econ["cost_per_1000_decisions_usd"] == pytest.approx(
        400 * econ["cost_per_memo_usd"], rel=1e-6
    )


def test_prompt_hash_is_stable_and_order_sensitive():
    assert prompt_hash("a", "b") == prompt_hash("a", "b")
    assert prompt_hash("a", "b") != prompt_hash("b", "a")
    # Concatenation must not collide with separate parts.
    assert prompt_hash("ab", "c") != prompt_hash("a", "bc")


# -- memo guardrails ---------------------------------------------------------


def test_memo_rejects_a_family_it_was_not_given(payload):
    """The single most important check: the model may not invent a reason."""
    memo = AdverseActionMemo(
        summary="x" * 50,
        detail="y" * 70,
        reason_families_cited=["affordability", "criminal_record"],
        next_steps="z" * 25,
    )
    with pytest.raises(ValueError, match="not given"):
        validate_grounding(memo, payload)


def test_memo_accepts_a_subset_of_the_given_families(payload):
    memo = AdverseActionMemo(
        summary="x" * 50,
        detail="y" * 70,
        reason_families_cited=["affordability"],
        next_steps="z" * 25,
    )
    validate_grounding(memo, payload)  # must not raise


@pytest.mark.parametrize(
    "text",
    [
        "I approve this application because the record is strong enough overall here.",
        "On balance I recommend approving the applicant for the requested amount today.",
        "We should decline this one outright given the delinquency on the credit file.",
    ],
)
def test_memo_rejects_decision_language(text):
    """A memo explains a decision; it must never read as making one."""
    with pytest.raises(ValidationError):
        AdverseActionMemo(
            summary=text,
            detail="y" * 70,
            reason_families_cited=["affordability"],
            next_steps="z" * 25,
        )


def test_memo_input_forbids_unexpected_fields():
    """The prompt receives a narrow, known set of fields — never a whole applicant."""
    with pytest.raises(ValidationError):
        MemoInput(
            decision_id="d",
            decision="decline",
            score=500,
            reason_codes=REASONS,
            requested_amount=1000,
            stated_income_band="a",
            date_of_birth="1990-01-01",
        )


def test_offline_memo_is_grounded_and_flagged(payload):
    result = generate(payload)
    assert result.offline is True
    assert result.model == "offline-template"
    assert set(result.memo.reason_families_cited) <= {r["family"] for r in REASONS}
    validate_grounding(result.memo, payload)


def test_offline_memo_is_deterministic(payload):
    """Identical inputs must produce an identical memo."""
    assert generate(payload).memo == generate(payload).memo


def test_memo_uses_at_most_four_reasons(payload):
    many = MemoInput(
        decision_id="d-2",
        decision="decline",
        score=500,
        reason_codes=REASONS
        + [
            {
                "rank": 5,
                "family": "utilisation",
                "label": "Revolving",
                "phrase": "Balances are high",
                "actionable": True,
            }
        ],
        requested_amount=1000,
        stated_income_band="a",
    )
    assert len(generate(many).memo.reason_families_cited) <= MAX_REASONS


def test_memo_never_contains_the_raw_income(payload):
    """Only a band reaches the prompt, so only a band can reach the output."""
    result = generate(payload)
    blob = result.memo.model_dump_json()
    assert "75000" not in blob.replace(",", "")


# -- tool guardrails ---------------------------------------------------------


def test_the_model_cannot_supply_sql():
    """The whitelist is the boundary — a prompt is not one."""
    for attempt in ("'; DROP TABLE decisions;--", "SELECT * FROM decisions", "decision_mix; --"):
        with pytest.raises(UnknownQueryError):
            query_portfolio_stats(attempt)


def test_every_whitelisted_query_actually_runs():
    """A documented query that errors is worse than one that does not exist."""
    for name in PORTFOLIO_QUERIES:
        result = query_portfolio_stats(name)
        assert isinstance(result, dict) and result


def test_cutoff_query_responds_to_its_parameter():
    strict = query_portfolio_stats("approval_rate_at_cutoff", score_cutoff=600)
    loose = query_portfolio_stats("approval_rate_at_cutoff", score_cutoff=500)
    assert strict["approval_rate"] < loose["approval_rate"]
    assert strict["bad_rate_among_approved"] < loose["bad_rate_among_approved"]


def test_model_metrics_are_read_not_invented():
    out = get_model_metrics(["oot_auc", "oot_gini"])
    assert out["source"] in {"mlflow", "artifacts"}
    assert 0.5 < out["metrics"]["oot_auc"] < 1.0


def test_metric_filter_is_respected():
    assert set(get_model_metrics(["oot_auc"])["metrics"]) == {"oot_auc"}


@pytest.mark.parametrize(
    ("question", "expected_heading"),
    [
        ("What does the policy say about thin bureau files?", "3.2 Thin bureau file"),
        ("When must we retrain the model?", "8.1 Stability"),
        ("What justification is required for an override?", "7.2 Justification"),
    ],
)
def test_retrieval_finds_the_governing_section(question, expected_heading):
    headings = [p["heading"] for p in search_credit_policy(question, top_k=3)["passages"]]
    assert expected_heading in headings, f"got {headings}"


def test_retrieval_returns_nothing_for_an_unrelated_question():
    """Better to return nothing than a confidently irrelevant passage."""
    assert search_credit_policy("zzzz qqqq xxxx")["passages"] == []


def test_retrieval_cites_its_source():
    for passage in search_credit_policy("promotion gate")["passages"]:
        assert passage["source"].endswith(".md")
        assert passage["heading"]
