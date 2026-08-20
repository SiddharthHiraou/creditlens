"""Scoring endpoints: single and batch."""

from __future__ import annotations

import datetime as dt
from typing import Literal

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request, status

from src.api.deps import AppState, get_state
from src.api.schemas import ApplicationIn, BatchAccepted, BatchIn, BatchStatus, ScoreOut
from src.api.security import require_api_key
from src.api.store import new_id

router = APIRouter(prefix="/v1", tags=["scoring"], dependencies=[Depends(require_api_key)])


@router.post("/score", response_model=ScoreOut, status_code=status.HTTP_200_OK)
def score(
    application: ApplicationIn,
    request: Request,
    explain: Literal["auto", "always", "never"] = Query(
        default="auto",
        description=(
            "SHAP is ~92% of a request's cost. 'auto' computes it only for "
            "non-approvals, which is all that needs adverse action reasons."
        ),
    ),
    state: AppState = Depends(get_state),
) -> ScoreOut:
    """Score one application and write the decision to the audit log."""
    service = request.app.state.scoring
    decision_id = new_id()
    response, audit_row = service.score(application, decision_id=decision_id, explain=explain)

    audit_row["api_key_id"] = getattr(request.state, "api_key_id", None)
    audit_row["request_id"] = getattr(request.state, "request_id", None)
    state.store.record_decision(audit_row)
    return response


@router.post("/score/batch", response_model=BatchAccepted, status_code=status.HTTP_202_ACCEPTED)
def score_batch(
    payload: BatchIn,
    background: BackgroundTasks,
    request: Request,
    state: AppState = Depends(get_state),
) -> BatchAccepted:
    """Accept up to ``batch_max_rows`` applications and score them in the background.

    Returns 202 with a job id rather than blocking. A synchronous batch of 1000
    would hold a worker for seconds and blow the p99 on every concurrent single
    score sharing that worker.
    """
    limit = state.settings.batch_max_rows
    if len(payload.applications) > limit:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"Batch of {len(payload.applications)} exceeds the limit of {limit}.",
        )

    job_id = new_id()
    state.store.create_job(job_id, len(payload.applications))
    background.add_task(
        _run_batch,
        state,
        request.app.state.scoring,
        job_id,
        payload,
        getattr(request.state, "api_key_id", None),
        getattr(request.state, "request_id", None),
    )
    return BatchAccepted(
        job_id=job_id,
        n_submitted=len(payload.applications),
        status="queued",
        submitted_at=dt.datetime.now(dt.UTC).replace(tzinfo=None),
    )


@router.get("/score/batch/{job_id}", response_model=BatchStatus)
def batch_status(job_id: str, state: AppState = Depends(get_state)) -> BatchStatus:
    job = state.store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"No job {job_id}.")
    return BatchStatus(**job)


def _run_batch(state, service, job_id, payload, api_key_id, request_id) -> None:
    state.update = getattr(state, "update", None)
    state.store.update_job(job_id, status="running")
    try:
        responses, rows = [], []
        for application in payload.applications:
            response, audit_row = service.score(application, decision_id=new_id())
            audit_row["api_key_id"] = api_key_id
            audit_row["request_id"] = request_id
            responses.append(response.model_dump(mode="json"))
            rows.append(audit_row)
        state.store.record_decisions(rows)
        state.store.update_job(
            job_id,
            status="complete",
            n_scored=len(rows),
            completed_at=dt.datetime.now(dt.UTC).replace(tzinfo=None),
            results=responses,
        )
    except Exception as exc:  # noqa: BLE001 - surface on the job, never crash the worker
        state.store.update_job(
            job_id,
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
            completed_at=dt.datetime.now(dt.UTC).replace(tzinfo=None),
        )
