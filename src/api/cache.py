"""Applicant history feature cache.

At inference the caller supplies application-level fields only. The bureau and
repayment aggregations — 161 of the 221 features — are precomputed and looked up
by ``SK_ID_CURR``. Two reasons that split is the right one:

* **Latency.** Recomputing them means scanning 9.6M child rows. Precomputed,
  it is one key lookup.
* **Trust.** The client does not have the applicant's bureau history and must
  not be able to assert it. Sending 161 features over the wire would let a
  caller declare their own credit record.

Redis is the production path. When no ``redis_url`` is configured the cache
falls back to an in-process dict, so the API runs with nothing else installed
and the tests need no service. The fallback is reported by ``backend`` and shows
up in ``/ready`` — a silent downgrade would be worse than no cache at all.

A cache **miss is not an error**. It means a thin-file applicant with no
history, which is a real and risk-relevant segment; those features go to the
model as nulls, exactly as they did in training.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

KEY_PREFIX = "creditlens:history:"


@dataclass
class FeatureCache:
    client: Any | None = None
    local: dict[int, dict[str, float | None]] = field(default_factory=dict)
    backend: str = "memory"

    @classmethod
    def connect(cls, redis_url: str | None) -> FeatureCache:
        if not redis_url:
            return cls(backend="memory")
        try:
            import redis

            client = redis.Redis.from_url(redis_url, decode_responses=True, socket_timeout=1.0)
            client.ping()
            return cls(client=client, backend="redis")
        except Exception:  # noqa: BLE001 - degrade visibly, never fail the app
            return cls(backend="memory (redis unreachable)")

    def get(self, sk_id_curr: int) -> dict[str, float | None] | None:
        if self.client is not None:
            raw = self.client.get(f"{KEY_PREFIX}{sk_id_curr}")
            return json.loads(raw) if raw else None
        return self.local.get(sk_id_curr)

    def put(self, sk_id_curr: int, features: dict[str, float | None]) -> None:
        if self.client is not None:
            self.client.set(f"{KEY_PREFIX}{sk_id_curr}", json.dumps(features))
        else:
            self.local[sk_id_curr] = features

    def put_many(self, rows: dict[int, dict[str, float | None]]) -> int:
        if self.client is not None:
            pipe = self.client.pipeline()
            for key, value in rows.items():
                pipe.set(f"{KEY_PREFIX}{key}", json.dumps(value))
            pipe.execute()
        else:
            self.local.update(rows)
        return len(rows)

    def size(self) -> int:
        if self.client is not None:
            return sum(1 for _ in self.client.scan_iter(f"{KEY_PREFIX}*", count=1000))
        return len(self.local)

    def healthy(self) -> bool:
        if self.client is None:
            return True  # the in-memory fallback is always available
        try:
            return bool(self.client.ping())
        except Exception:  # noqa: BLE001
            return False
