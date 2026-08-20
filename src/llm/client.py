"""Model tiering, cost accounting, and the audit trail for every LLM call.

**Two tiers, chosen for the workload rather than by default.** Adverse action
memos are high volume and templated, so they run on the cheap fast tier.
The analyst copilot answers open questions over portfolio statistics and policy
text and needs real reasoning, so it runs on the mid tier. Both are specified in
the project brief; neither is a guess.

**Every call is logged** with the model, the prompt hash, token counts and the
computed cost. A memo that reaches an applicant must be reconstructible, and
"which prompt produced this" is the first question anyone will ask.

**Nothing here can make a decision.** The scoring path never calls this module.
The memo generator receives a decision that has already been made and writes
prose about it.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any

from src.config import ARTIFACTS

# Model IDs. Do not append date suffixes -- these strings are complete.
MEMO_MODEL = "claude-haiku-4-5"
COPILOT_MODEL = "claude-sonnet-5"

# USD per million tokens, as published 2026-06-24. Sonnet 5 carries introductory
# pricing through 2026-08-31; the standard rate is $3.00 / $15.00.
PRICING: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-opus-5": (5.00, 25.00),
}

LLM_LOG = ARTIFACTS / "llm_audit.jsonl"


class CredentialsUnavailableError(RuntimeError):
    """Raised when a live call is required but no credentials are configured."""


def credentials_available() -> bool:
    """Whether a live API call can be made.

    The SDK resolves an API key, an auth token, or an `ant auth login` profile.
    An unset ``ANTHROPIC_API_KEY`` alone does not mean there are no credentials,
    so the profile directory is checked too.
    """
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return True
    profile_dir = os.path.expanduser("~/.config/anthropic")
    return os.path.isdir(profile_dir) and bool(os.listdir(profile_dir))


def get_client():
    """Construct the SDK client, letting it resolve credentials itself."""
    import anthropic

    if not credentials_available():
        raise CredentialsUnavailableError(
            "No Anthropic credentials found. Set ANTHROPIC_API_KEY or run `ant auth login`."
        )
    return anthropic.Anthropic()


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Cost of a single call in USD."""
    rate_in, rate_out = PRICING.get(model, (0.0, 0.0))
    return (input_tokens * rate_in + output_tokens * rate_out) / 1_000_000


def prompt_hash(*parts: str) -> str:
    """Stable hash of the exact prompt text sent.

    Logged with every output so a memo produced months ago can be traced to the
    prompt that produced it, even after the prompt has been rewritten.
    """
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()[:16]


@dataclass(frozen=True)
class Usage:
    input_tokens: int
    output_tokens: int

    @classmethod
    def from_response(cls, response: Any) -> Usage:
        usage = getattr(response, "usage", None)
        return cls(
            input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
        )


def log_call(record: dict) -> None:
    """Append one JSON line per LLM call.

    Never contains the applicant's raw attributes -- only the decision id, the
    prompt hash, and the accounting. A log that quietly accumulates income and
    dates of birth is a privacy incident waiting to be found.
    """
    LLM_LOG.parent.mkdir(parents=True, exist_ok=True)
    payload = {"ts": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"), **record}
    with LLM_LOG.open("a") as handle:
        handle.write(json.dumps(payload, default=str) + "\n")


def cost_per_1000_decisions(
    *,
    memo_rate: float = 0.30,
    memo_input_tokens: int = 900,
    memo_output_tokens: int = 320,
) -> dict[str, float]:
    """Unit economics for the README.

    Only non-approvals get a memo, so the memo rate is the share of decisions
    that are referred or declined -- roughly 40% at a 60% approval policy.
    Copilot usage is analyst-driven and does not scale with decision volume, so
    it is excluded here and priced separately.
    """
    memos = 1000 * memo_rate
    per_memo = cost_usd(MEMO_MODEL, memo_input_tokens, memo_output_tokens)
    return {
        "memo_model": MEMO_MODEL,
        "memos_per_1000_decisions": memos,
        "cost_per_memo_usd": round(per_memo, 6),
        "cost_per_1000_decisions_usd": round(memos * per_memo, 4),
    }
