"""Cutoff simulation — the interactive demo the dashboard is built around.

Answers the question a credit committee actually asks: if we move the cutoff,
what happens to volume, losses and profit? Evaluated against the held-out
out-of-time fold, which is the only population where the outcomes are both
known and not the ones the model trained on.
"""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, status

from src.api.deps import AppState, get_state
from src.api.schemas import CutoffSimulationIn, CutoffSimulationOut
from src.api.security import require_api_key
from src.models.decision import DecisionPolicy, decide, score_to_pd

router = APIRouter(
    prefix="/v1/simulate", tags=["simulate"], dependencies=[Depends(require_api_key)]
)


@router.post("/cutoff", response_model=CutoffSimulationOut)
def simulate_cutoff(
    payload: CutoffSimulationIn, state: AppState = Depends(get_state)
) -> CutoffSimulationOut:
    holdout = state.holdout if hasattr(state, "holdout") else None
    if holdout is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No evaluation holdout loaded; run `make train` to write it.",
        )

    scores, y, exposure = holdout["score"], holdout["y"], holdout["exposure"]

    if payload.approve_at is not None:
        approve_at = float(payload.approve_at)
    else:
        approve_at = float(np.quantile(scores, 1 - payload.approve_rate))
    refer_at = float(
        np.quantile(scores, max(0.0, _rate_at(scores, approve_at) - payload.refer_band))
    )
    refer_at = min(refer_at, approve_at)

    policy = DecisionPolicy(approve_at=approve_at, refer_at=refer_at, lgd=payload.lgd)
    bands = decide(scores, policy)
    approved = bands == "approve"

    pd_hat = score_to_pd(scores)
    el = float((pd_hat[approved] * payload.lgd * exposure[approved]).sum())
    # Margin is earned only on the loans that perform; defaulted principal is
    # already captured by expected loss.
    income = float((exposure[approved] * payload.interest_margin * (1 - pd_hat[approved])).sum())

    band_rows = []
    for name in ("decline", "refer", "approve"):
        mask = bands == name
        if not mask.any():
            band_rows.append({"decision": name, "n": 0})
            continue
        band_rows.append(
            {
                "decision": name,
                "n": int(mask.sum()),
                "share": float(mask.mean()),
                "observed_bad_rate": float(y[mask].mean()),
                "mean_pd": float(pd_hat[mask].mean()),
                "expected_loss": float((pd_hat[mask] * payload.lgd * exposure[mask]).sum()),
            }
        )

    return CutoffSimulationOut(
        approve_at=approve_at,
        refer_at=refer_at,
        approval_rate=float(approved.mean()),
        refer_rate=float((bands == "refer").mean()),
        bad_rate_among_approved=float(y[approved].mean()) if approved.any() else 0.0,
        expected_loss=el,
        interest_income=income,
        estimated_profit=income - el,
        n_evaluated=int(scores.size),
        bands=band_rows,
    )


@router.get("/cutoff/curve")
def cutoff_curve(
    lgd: float = 0.65, interest_margin: float = 0.12, state: AppState = Depends(get_state)
) -> dict:
    """Profit across the whole cutoff range, and the profit-maximising point."""
    holdout = state.holdout if hasattr(state, "holdout") else None
    if holdout is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="No evaluation holdout loaded."
        )

    scores, y, exposure = holdout["score"], holdout["y"], holdout["exposure"]
    pd_hat = score_to_pd(scores)

    points = []
    for rate in np.linspace(0.05, 0.98, 40):
        cutoff = float(np.quantile(scores, 1 - rate))
        approved = scores >= cutoff
        if not approved.any():
            continue
        el = float((pd_hat[approved] * lgd * exposure[approved]).sum())
        income = float((exposure[approved] * interest_margin * (1 - pd_hat[approved])).sum())
        points.append(
            {
                "approval_rate": float(approved.mean()),
                "cutoff": cutoff,
                "bad_rate_among_approved": float(y[approved].mean()),
                "expected_loss": el,
                "interest_income": income,
                "profit": income - el,
            }
        )

    best = max(points, key=lambda p: p["profit"]) if points else None
    return {
        "points": points,
        "profit_maximising": best,
        "lgd": lgd,
        "interest_margin": interest_margin,
    }


def _rate_at(scores: np.ndarray, cutoff: float) -> float:
    return float((scores >= cutoff).mean())
