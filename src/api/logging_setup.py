"""Structured JSON logging with a request id on every line.

One line per request, machine-parseable, carrying the request id, the hashed
API key, the route, the status and the latency. The request id is echoed back in
the ``X-Request-ID`` response header so a caller reporting a problem can hand
over the one string needed to find their request.

Never logged: the raw API key, and no field from the application payload beyond
the applicant id. A decision log that quietly accumulates income and date of
birth in plaintext is a privacy incident waiting to be discovered.
"""

from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

REQUEST_ID_HEADER = "X-Request-ID"


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        payload.update(getattr(record, "extra_fields", {}) or {})
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> logging.Logger:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)
    # uvicorn's own access log duplicates what the middleware records, in a
    # different shape. One structured line per request is the point.
    for noisy in ("uvicorn.access",):
        logging.getLogger(noisy).handlers = []
        logging.getLogger(noisy).propagate = False
    return logging.getLogger("creditlens.api")


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Assign a request id, time the request, emit one structured line."""

    def __init__(self, app, logger: logging.Logger):
        super().__init__(app)
        self.logger = logger

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get(REQUEST_ID_HEADER) or str(uuid.uuid4())
        request.state.request_id = request_id
        started = time.perf_counter()

        try:
            response = await call_next(request)
        except Exception:
            self.logger.exception(
                "request failed",
                extra={
                    "extra_fields": {
                        "request_id": request_id,
                        "method": request.method,
                        "path": request.url.path,
                        "latency_ms": round((time.perf_counter() - started) * 1000, 3),
                    }
                },
            )
            raise

        latency_ms = round((time.perf_counter() - started) * 1000, 3)
        response.headers[REQUEST_ID_HEADER] = request_id
        remaining = getattr(request.state, "rate_limit_remaining", None)
        if remaining is not None:
            response.headers["X-RateLimit-Remaining"] = str(remaining)

        self.logger.info(
            "request",
            extra={
                "extra_fields": {
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status": response.status_code,
                    "latency_ms": latency_ms,
                    "api_key_id": getattr(request.state, "api_key_id", None),
                }
            },
        )
        return response
