"""Audit retrieval and underwriter overrides."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from src.api.deps import AppState, get_state
from src.api.schemas import DecisionRecord, OverrideIn
from src.api.security import require_api_key

router = APIRouter(
    prefix="/v1/decisions", tags=["decisions"], dependencies=[Depends(require_api_key)]
)


@router.get("/{decision_id}", response_model=DecisionRecord)
def get_decision(decision_id: str, state: AppState = Depends(get_state)) -> DecisionRecord:
    """The full audit record: inputs, model version, features, output, overrides.

    Everything needed to reconstruct the decision months later, including the
    feature vector as scored — a reason code cannot be re-derived from a
    probability alone.
    """
    record = state.store.get_decision(decision_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"No decision {decision_id}."
        )
    return DecisionRecord(**record)


@router.post("/{decision_id}/override", response_model=DecisionRecord)
def override_decision(
    decision_id: str, payload: OverrideIn, state: AppState = Depends(get_state)
) -> DecisionRecord:
    """Record an underwriter override against a decision.

    The original decision is never mutated. An override is an *additional*
    record, so the audit trail shows what the model said, what the human said,
    and why — which is what a validation team asks for. Overwriting the model's
    output would destroy exactly the evidence that makes overrides reviewable.
    """
    record = state.store.get_decision(decision_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"No decision {decision_id}."
        )
    if payload.new_decision == record["decision"]:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Decision is already '{record['decision']}'; nothing to override.",
        )

    state.store.record_override(
        decision_id=decision_id,
        original=record["decision"],
        new=payload.new_decision,
        justification=payload.justification,
        underwriter_id=payload.underwriter_id,
    )
    return DecisionRecord(**state.store.get_decision(decision_id))
