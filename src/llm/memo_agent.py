"""Adverse action memo generation.

Regulatory context: under ECOA and Regulation B a declined applicant must
receive the specific principal reasons for the denial. Those reasons are
produced deterministically by the SHAP-driven reason-code mapper. This module
does one narrow thing — turns four terse reason phrases into two readable
paragraphs.

**The LLM narrates; it never decides and never adds a reason.** That is enforced
three ways rather than asked for once:

* The prompt states it.
* The output is parsed into a Pydantic model that rejects decision language.
* :func:`validate_grounding` checks that every family the memo cites was in the
  input. A memo citing a family it was not given is discarded, not repaired.

**Cheap tier, deliberately.** Memos are high volume and templated, so they run on
Claude Haiku. ``temperature=0`` is set because the memo should not vary between
identical inputs — note that this parameter only exists on pre-4.6 models, and
sending it to the copilot's Sonnet model would be a 400.

**Offline fallback.** Where no credentials are configured the module produces a
deterministic template memo instead, flagged ``offline=True``. Tests and the
demo therefore work without an API key, and nothing silently pretends a model
wrote something a template did.
"""

from __future__ import annotations

import datetime as dt
import json
import time
from pathlib import Path

from src.llm.client import (
    MEMO_MODEL,
    CredentialsUnavailableError,
    Usage,
    cost_usd,
    credentials_available,
    get_client,
    log_call,
    prompt_hash,
)
from src.llm.schemas import AdverseActionMemo, MemoInput, MemoResult

PROMPT_VERSION = "memo-v1"
SYSTEM_PROMPT = (Path(__file__).parent / "prompts" / "memo_system.txt").read_text()

# Rejecting a whole memo for one unexpected family is the correct failure mode:
# a disclosure with an invented reason is worse than no disclosure.
MAX_REASONS = 4


def _user_prompt(payload: MemoInput) -> str:
    reasons = [
        {
            "rank": rc["rank"],
            "family": rc["family"],
            "label": rc["label"],
            "phrase": rc["phrase"],
            "actionable": rc.get("actionable", False),
        }
        for rc in payload.reason_codes[:MAX_REASONS]
    ]
    return json.dumps(
        {
            "decision": payload.decision,
            "credit_score": round(payload.score),
            "requested_amount": round(payload.requested_amount),
            "stated_income_band": payload.stated_income_band,
            "principal_reasons": reasons,
        },
        indent=2,
    )


def validate_grounding(memo: AdverseActionMemo, payload: MemoInput) -> None:
    """Reject a memo that cites a reason family it was not given.

    This is the check that makes the "narrates, never invents" claim testable
    rather than aspirational.
    """
    allowed = {rc["family"] for rc in payload.reason_codes[:MAX_REASONS]}
    cited = set(memo.reason_families_cited)
    invented = cited - allowed
    if invented:
        raise ValueError(
            f"Memo cites reason families it was not given: {sorted(invented)}. "
            f"Allowed: {sorted(allowed)}"
        )


def generate(payload: MemoInput, *, allow_offline: bool = True) -> MemoResult:
    """Produce a validated adverse action memo."""
    started = time.perf_counter()
    user_prompt = _user_prompt(payload)
    digest = prompt_hash(SYSTEM_PROMPT, user_prompt, PROMPT_VERSION)

    if not credentials_available():
        if not allow_offline:
            raise CredentialsUnavailableError("No Anthropic credentials and allow_offline=False.")
        memo = _offline_memo(payload)
        validate_grounding(memo, payload)
        result = MemoResult(
            decision_id=payload.decision_id,
            memo=memo,
            model="offline-template",
            prompt_hash=digest,
            prompt_version=PROMPT_VERSION,
            input_tokens=0,
            output_tokens=0,
            cost_usd=0.0,
            latency_ms=round((time.perf_counter() - started) * 1000, 3),
            generated_at=dt.datetime.now(dt.UTC).replace(tzinfo=None),
            offline=True,
        )
        _log(result, payload)
        return result

    client = get_client()
    response = client.messages.parse(
        model=MEMO_MODEL,
        max_tokens=1200,
        # Identical inputs must produce an identical memo. Available on Haiku
        # 4.5; this parameter is rejected with a 400 on Sonnet 5 and later.
        temperature=0,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
        output_config={"format": AdverseActionMemo},
    )

    memo = response.parsed_output
    if memo is None:
        raise ValueError("Model returned no parseable memo.")
    validate_grounding(memo, payload)

    usage = Usage.from_response(response)
    result = MemoResult(
        decision_id=payload.decision_id,
        memo=memo,
        model=MEMO_MODEL,
        prompt_hash=digest,
        prompt_version=PROMPT_VERSION,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cost_usd=cost_usd(MEMO_MODEL, usage.input_tokens, usage.output_tokens),
        latency_ms=round((time.perf_counter() - started) * 1000, 3),
        generated_at=dt.datetime.now(dt.UTC).replace(tzinfo=None),
        offline=False,
    )
    _log(result, payload)
    return result


def _log(result: MemoResult, payload: MemoInput) -> None:
    log_call(
        {
            "kind": "memo",
            "decision_id": result.decision_id,
            "decision": payload.decision,
            "model": result.model,
            "prompt_hash": result.prompt_hash,
            "prompt_version": result.prompt_version,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "cost_usd": result.cost_usd,
            "latency_ms": result.latency_ms,
            "offline": result.offline,
            "families": result.memo.reason_families_cited,
        }
    )


def _offline_memo(payload: MemoInput) -> AdverseActionMemo:
    """Deterministic template used when no credentials are configured.

    Grounded in exactly the same inputs and subject to the same validation, so
    the downstream contract is identical. It reads like a template because it is
    one — that is preferable to a fluent paragraph nobody can attribute.
    """
    reasons = payload.reason_codes[:MAX_REASONS]
    families = [rc["family"] for rc in reasons]
    bullets = " ".join(f"({rc['rank']}) {rc['phrase']}." for rc in reasons)
    actionable = [rc for rc in reasons if rc.get("actionable")]

    verb = {
        "decline": "was not approved",
        "refer": "has been referred for manual review",
        "approve": "was approved",
    }[payload.decision]

    summary = (
        f"The application for {payload.requested_amount:,.0f} {verb}. "
        f"The principal reasons recorded were: {bullets}"
    )
    detail = (
        f"This decision reflects {len(reasons)} principal reasons drawn from the "
        f"information in the application and the credit file. {bullets} "
        "Each reason is listed separately because each contributed independently; "
        "no single item determined the outcome on its own."
    )
    if actionable:
        next_steps = (
            "Some of the reasons above can change over time: "
            + " ".join(rc["label"].lower() for rc in actionable)
            + ". An applicant may request a copy of the credit report used and may "
            "reapply once circumstances have changed."
        )
    else:
        next_steps = (
            "An applicant may request a copy of the credit report used in this "
            "decision and may ask for the reasons to be reviewed by an underwriter."
        )

    return AdverseActionMemo(
        summary=summary[:800],
        detail=detail[:1600],
        reason_families_cited=families,
        next_steps=next_steps[:600],
    )
