"""Liveness and readiness.

They answer different questions and are deliberately not the same endpoint.
``/health`` says the process is up; a container orchestrator restarts on it.
``/ready`` says this instance can actually serve a request — model loaded,
database reachable, cache reporting its true backend. An instance whose Redis
has gone should be visibly degraded, not silently slower.

Neither requires an API key: a probe that needs a secret is a probe that fails
for the wrong reason.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Response, status

from src.api.deps import AppState, get_state
from src.api.schemas import HealthOut, ReadyOut

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthOut)
def health(state: AppState = Depends(get_state)) -> HealthOut:
    return HealthOut(
        status="ok", version=state.settings.version, environment=state.settings.environment
    )


@router.get("/ready", response_model=ReadyOut)
def ready(response: Response, state: AppState = Depends(get_state)) -> ReadyOut:
    database = state.store.healthy()
    model_loaded = state.model is not None
    spec_loaded = bool(state.spec.features)

    ok = database and model_loaded and spec_loaded
    if not ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    detail = None
    if not ok:
        missing = [
            name
            for name, present in (
                ("model", model_loaded),
                ("feature_spec", spec_loaded),
                ("database", database),
            )
            if not present
        ]
        detail = f"unavailable: {', '.join(missing)}"

    return ReadyOut(
        status="ready" if ok else "not_ready",
        model_loaded=model_loaded,
        feature_spec_loaded=spec_loaded,
        database=database,
        cache=state.cache.backend,
        detail=detail,
    )
