"""API key authentication and per-key rate limiting.

Deliberately modest: a header-based key check and a fixed-window counter. Both
are the right size for this system and both are honest about their limits.

**Keys are compared with `secrets.compare_digest`**, not `==`. String equality
short-circuits on the first differing byte, which leaks key material through
timing. It costs nothing to do correctly.

**The rate limiter is in-process.** With more than one worker each holds its own
counter, so the effective limit multiplies by the worker count. That is fine for
a demo and wrong for production, where the counter belongs in Redis. It is
called out in the README rather than left for someone to discover.
"""

from __future__ import annotations

import hashlib
import secrets
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field

from fastapi import Header, HTTPException, Request, status

API_KEY_HEADER = "X-API-Key"


def key_id(api_key: str) -> str:
    """Stable, non-reversible identifier for logs and the audit trail.

    The raw key must never reach a log line or a database row.
    """
    return hashlib.sha256(api_key.encode()).hexdigest()[:16]


@dataclass
class RateLimiter:
    """Sliding-window limiter keyed by API key."""

    limit_per_minute: int
    window_seconds: int = 60
    hits: dict[str, deque[float]] = field(default_factory=lambda: defaultdict(deque))

    def check(self, identity: str) -> tuple[bool, int, float]:
        """Returns (allowed, remaining, retry_after_seconds)."""
        now = time.monotonic()
        window = self.hits[identity]
        cutoff = now - self.window_seconds
        while window and window[0] < cutoff:
            window.popleft()

        if len(window) >= self.limit_per_minute:
            return False, 0, max(0.0, window[0] + self.window_seconds - now)

        window.append(now)
        return True, self.limit_per_minute - len(window), 0.0

    def reset(self) -> None:
        self.hits.clear()


def require_api_key(
    request: Request,
    x_api_key: str | None = Header(default=None, alias=API_KEY_HEADER),
) -> str:
    """Authenticate and rate-limit. Returns the hashed key id."""
    settings = request.app.state.settings
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Missing {API_KEY_HEADER} header.",
            headers={"WWW-Authenticate": API_KEY_HEADER},
        )

    # Compare against every known key so the time taken does not reveal how
    # far through the set a near-match got.
    matched = False
    for known in settings.key_set:
        if secrets.compare_digest(x_api_key, known):
            matched = True
    if not matched:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key.")

    identity = key_id(x_api_key)
    allowed, remaining, retry_after = request.app.state.rate_limiter.check(identity)
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit of {settings.rate_limit_per_minute}/min exceeded.",
            headers={"Retry-After": str(int(retry_after) + 1)},
        )

    request.state.api_key_id = identity
    request.state.rate_limit_remaining = remaining
    return identity
