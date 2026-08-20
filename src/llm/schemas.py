"""Contracts for the GenAI layer.

The memo output is validated against a Pydantic model rather than accepted as
free text. That is the enforcement point for the rule that matters: the LLM
narrates reason codes it was handed and may not introduce a new one.
"""

from __future__ import annotations

import datetime as dt
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class MemoInput(BaseModel):
    """Structured input to the memo generator.

    Deliberately narrow. The LLM sees the decision, the reason codes and a few
    rounded figures — not the applicant's full feature vector, not their date of
    birth, not anything that is not needed to write two paragraphs.
    """

    model_config = ConfigDict(extra="forbid")

    decision_id: str
    decision: Literal["approve", "refer", "decline"]
    score: float = Field(ge=300, le=850)
    reason_codes: list[dict] = Field(min_length=1)
    requested_amount: float = Field(gt=0)
    # Rounded to the nearest thousand before it ever reaches a prompt.
    stated_income_band: str


class AdverseActionMemo(BaseModel):
    """Validated memo. Anything the model returns that does not fit is rejected."""

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(min_length=40, max_length=800)
    detail: str = Field(min_length=60, max_length=1600)
    reason_families_cited: list[str] = Field(min_length=1, max_length=4)
    next_steps: str = Field(min_length=20, max_length=600)

    @field_validator("summary", "detail", "next_steps")
    @classmethod
    def no_decision_language(cls, v: str) -> str:
        """The memo explains a decision; it must not appear to make one.

        Phrases that read as the model deciding are rejected outright rather
        than softened, because an underwriter skimming a memo should never see
        a sentence that sounds like the LLM approved anything.
        """
        banned = (
            "i approve",
            "i have approved",
            "i decline",
            "i have declined",
            "i recommend approving",
            "i recommend declining",
            "we should approve",
            "we should decline",
            "overturn",
        )
        lowered = v.lower()
        for phrase in banned:
            if phrase in lowered:
                raise ValueError(f"Memo contains decision language: {phrase!r}")
        return v


class MemoResult(BaseModel):
    """A memo plus everything an audit needs to reconstruct how it was produced."""

    model_config = ConfigDict(extra="forbid")

    decision_id: str
    memo: AdverseActionMemo
    model: str
    prompt_hash: str
    prompt_version: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float
    generated_at: dt.datetime
    offline: bool = Field(
        default=False,
        description="True when produced by the deterministic fallback, not the API.",
    )


class CopilotAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str
    answer: str
    tools_called: list[dict]
    model: str
    prompt_hash: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float
    generated_at: dt.datetime
    offline: bool = False
