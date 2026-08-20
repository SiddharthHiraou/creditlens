"""FastAPI application factory.

Model, ONNX session, TreeExplainer and feature spec are built once in the
lifespan hook. Anything expensive that happens per request is a latency bug.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.api.deps import build_state
from src.api.logging_setup import RequestContextMiddleware, configure_logging
from src.api.routers import decisions, health, memo, model, scoring, simulate
from src.api.scoring import ScoringService
from src.api.security import RateLimiter
from src.api.settings import Settings, get_settings

DESCRIPTION = """
Credit decisioning API. Submit an application, receive a probability of default,
a mapped 300-850 score, an approve/refer/decline decision, ECOA-compliant reason
codes and the SHAP contributions behind them.

Authenticate with an `X-API-Key` header. Every response carries an
`X-Request-ID`; quote it when reporting a problem.
"""


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    logger = configure_logging()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state = build_state(settings)
        app.state.creditlens = state
        app.state.settings = settings
        app.state.rate_limiter = RateLimiter(limit_per_minute=settings.rate_limit_per_minute)
        app.state.scoring = ScoringService(
            model=state.model,
            spec=state.spec,
            mapper=state.mapper,
            policy=state.policy,
            cache=state.cache,
            shap_service=state.shap_service,
            model_version=state.model_version,
            shap_top_k=settings.shap_top_k,
        )
        logger.info(
            "startup",
            extra={
                "extra_fields": {
                    "backend": state.serving_backend,
                    "model_version": state.model_version,
                    "n_features": state.spec.n_features,
                    "feature_spec_fingerprint": state.spec.fingerprint,
                    "cache": state.cache.backend,
                    "approve_at": state.policy.approve_at,
                }
            },
        )
        yield
        logger.info("shutdown", extra={"extra_fields": {}})

    app = FastAPI(
        title=settings.app_name,
        version=settings.version,
        description=DESCRIPTION,
        lifespan=lifespan,
    )
    app.add_middleware(RequestContextMiddleware, logger=logger)

    for router in (
        health.router,
        scoring.router,
        decisions.router,
        memo.router,
        model.router,
        simulate.router,
    ):
        app.include_router(router)

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception) -> JSONResponse:
        """Never leak a stack trace to a caller.

        The traceback goes to the structured log with the request id; the caller
        gets the id and nothing else. An exception message can carry feature
        values, file paths, or connection strings.
        """
        return JSONResponse(
            status_code=500,
            content={
                "detail": "Internal error.",
                "request_id": getattr(request.state, "request_id", None),
            },
        )

    return app


app = create_app()
