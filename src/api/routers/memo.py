"""Adverse action memo generation from a recorded decision."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict

from src.api.deps import AppState, get_state
from src.api.security import require_api_key
from src.llm.memo_agent import generate
from src.llm.schemas import MemoInput, MemoResult

router = APIRouter(prefix="/v1", tags=["memo"], dependencies=[Depends(require_api_key)])


class MemoRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision_id: str
    allow_offline: bool = True


@router.post("/memo", response_model=MemoResult)
def create_memo(payload: MemoRequest, state: AppState = Depends(get_state)) -> MemoResult:
    """Draft an adverse action memo for a decision already on record.

    The memo is generated *from the stored decision*, never from a payload the
    caller supplies. That is the whole point: a caller cannot hand the model a
    reason the decision did not actually carry, because the reasons are read
    back out of the audit log.

    Approvals are refused. An adverse action notice explains a denial; there is
    nothing adverse about an approval, and generating one would be meaningless
    at best and misleading in a letter at worst.
    """
    record = state.store.get_decision(payload.decision_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No decision {payload.decision_id}.",
        )
    if record["decision"] == "approve":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Adverse action memos apply to declines and referrals, not approvals.",
        )
    if not record["reason_codes"]:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "This decision carries no reason codes; it was scored with "
                "explain=never. Rescore with explanations before requesting a memo."
            ),
        )

    features = record["features"] or {}
    income = features.get("AMT_INCOME_TOTAL") or 0.0
    amount = record.get("exposure")
    if not amount:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Decision record carries no requested amount; cannot draft a memo.",
        )
    memo_input = MemoInput(
        decision_id=record["decision_id"],
        decision=record["decision"],
        score=record["score"],
        reason_codes=record["reason_codes"],
        requested_amount=float(amount),
        # Banded before it reaches a prompt. The model has no use for an exact
        # income and no business seeing one.
        stated_income_band=_band(income),
    )

    try:
        return generate(memo_input, allow_offline=payload.allow_offline)
    except ValueError as exc:
        # A memo that failed grounding validation is discarded, not returned
        # with a warning attached.
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Memo failed validation and was discarded: {exc}",
        ) from exc


def _band(income: float) -> str:
    """Round income into a 25k band."""
    if income <= 0:
        return "not stated"
    lower = int(income // 25_000) * 25_000
    return f"{lower:,}-{lower + 25_000:,}"
