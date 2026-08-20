"""Load test for the scoring path.

Run against a live server, not TestClient — TestClient bypasses the ASGI
server, the event loop and the network stack, so its timings flatter the real
p99 by exactly the parts that break under concurrency.

    make serve            # in one shell
    make loadtest         # in another

Payloads are replayed from `artifacts/sample_payloads.json`, drawn from the
out-of-time fold, so the cache-hit path is exercised. Scoring random strangers
would measure the thin-file path and understate the feature lookup.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from locust import HttpUser, between, events, task

ROOT = Path(__file__).resolve().parents[1]
PAYLOAD_PATH = ROOT / "artifacts" / "sample_payloads.json"
API_KEY = "demo-key-underwriter"

_PAYLOADS: list[dict] = []


@events.init.add_listener
def _load_payloads(environment, **_kwargs) -> None:
    global _PAYLOADS
    if not PAYLOAD_PATH.exists():
        raise SystemExit(
            f"No payloads at {PAYLOAD_PATH}. Run `make warm-cache` first."
        )
    _PAYLOADS = json.loads(PAYLOAD_PATH.read_text())


class Underwriter(HttpUser):
    """Single-application scoring, the latency-critical path."""

    wait_time = between(0.01, 0.05)

    @property
    def headers(self) -> dict[str, str]:
        return {"X-API-Key": API_KEY, "Content-Type": "application/json"}

    @task(10)
    def score(self) -> None:
        payload = random.choice(_PAYLOADS)
        with self.client.post(
            "/v1/score",
            json=payload,
            headers=self.headers,
            name="POST /v1/score",
            catch_response=True,
        ) as response:
            if response.status_code == 429:
                # Throttled requests are counted under their own name, never
                # folded into the success bucket. A 429 returns in ~1ms, so
                # mixing them in makes a saturated service look fast: a run
                # that is 95% throttled reports a 3ms p50 that measures the
                # rate limiter, not the model. Raise the limit before
                # benchmarking.
                response.failure("throttled (429) — raise the rate limit to benchmark")
            elif response.status_code != 200:
                response.failure(f"status {response.status_code}")

    @task(2)
    def score_without_explanation(self) -> None:
        payload = random.choice(_PAYLOADS)
        self.client.post(
            "/v1/score?explain=never",
            json=payload,
            headers=self.headers,
            name="POST /v1/score?explain=never",
        )

    @task(1)
    def metadata(self) -> None:
        self.client.get("/v1/model/metadata", headers=self.headers, name="GET /v1/model/metadata")

    @task(1)
    def health(self) -> None:
        self.client.get("/health", name="GET /health")
