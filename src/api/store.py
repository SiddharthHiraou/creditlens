"""Audit log. Every decision must be reconstructible.

The regulatory requirement this satisfies: a decision, months later, must be
explainable in terms of the exact inputs, model version and policy that produced
it. Storing the score alone is not enough — you cannot re-derive a reason code
from a probability.

So each row carries the full feature vector, the reason codes as issued, the
model version and the feature-spec fingerprint. If the champion is retrained,
old decisions still reconstruct against the model that actually made them.

SQLAlchemy Core rather than the ORM: the schema is three tables and the queries
are trivial, so the ORM's mapping layer would be overhead with no payoff.
Postgres in deployment, SQLite when nothing is running.
"""

from __future__ import annotations

import datetime as dt
import json
import uuid
from dataclasses import dataclass
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    select,
)
from sqlalchemy.engine import Engine

metadata = MetaData()

decisions = Table(
    "decisions",
    metadata,
    Column("decision_id", String(36), primary_key=True),
    Column("sk_id_curr", Integer, nullable=False, index=True),
    Column("pd", Float, nullable=False),
    Column("score", Float, nullable=False),
    Column("decision", String(16), nullable=False),
    # The requested amount is part of the decision, not just an input to it.
    # It is stored explicitly because feature selection dropped AMT_CREDIT from
    # the spec, so the feature vector alone cannot reconstruct it.
    Column("exposure", Float),
    Column("expected_loss", Float),
    Column("model_version", String(64), nullable=False),
    Column("feature_spec_fingerprint", String(32), nullable=False),
    Column("policy_version", String(32), nullable=False),
    Column("reason_codes", JSON, nullable=False),
    Column("features", JSON, nullable=False),
    Column("history_found", Boolean, nullable=False, default=False),
    Column("latency_ms", Float),
    Column("api_key_id", String(64)),
    Column("request_id", String(36)),
    Column("scored_at", DateTime, nullable=False),
)

overrides = Table(
    "overrides",
    metadata,
    Column("override_id", String(36), primary_key=True),
    Column("decision_id", String(36), nullable=False, index=True),
    Column("original_decision", String(16), nullable=False),
    Column("new_decision", String(16), nullable=False),
    Column("justification", Text, nullable=False),
    Column("underwriter_id", String(64), nullable=False),
    Column("created_at", DateTime, nullable=False),
)

batch_jobs = Table(
    "batch_jobs",
    metadata,
    Column("job_id", String(36), primary_key=True),
    Column("status", String(16), nullable=False),
    Column("n_submitted", Integer, nullable=False),
    Column("n_scored", Integer, nullable=False, default=0),
    Column("submitted_at", DateTime, nullable=False),
    Column("completed_at", DateTime),
    Column("error", Text),
    Column("results", JSON),
)


def new_id() -> str:
    return str(uuid.uuid4())


def _now() -> dt.datetime:
    return dt.datetime.now(dt.UTC).replace(tzinfo=None)


@dataclass
class AuditStore:
    engine: Engine

    @classmethod
    def connect(cls, url: str) -> AuditStore:
        # check_same_thread is a SQLite-only concern; the API serves the same
        # connection across worker threads.
        kwargs: dict[str, Any] = {"future": True}
        if url.startswith("sqlite"):
            kwargs["connect_args"] = {"check_same_thread": False}
        else:
            kwargs["pool_pre_ping"] = True
        engine = create_engine(url, **kwargs)
        # create_all() creates *missing tables*; it never alters an existing
        # one. Adding a column to a table that already exists is silently a
        # no-op, and the first query referencing it fails at runtime. There is
        # no migration tool in this project — Alembic is the right answer and is
        # named in the README's known gaps rather than half-built — so the
        # missing-column case is detected and reported instead of surfacing as
        # an opaque ProgrammingError on the first request.
        metadata.create_all(engine)
        cls._assert_schema_current(engine)
        return cls(engine=engine)

    @staticmethod
    def _assert_schema_current(engine: Engine) -> None:
        from sqlalchemy import inspect

        inspector = inspect(engine)
        for table in metadata.sorted_tables:
            if not inspector.has_table(table.name):
                continue
            existing = {c["name"] for c in inspector.get_columns(table.name)}
            missing = {c.name for c in table.columns} - existing
            if missing:
                raise RuntimeError(
                    f"Table {table.name!r} is missing column(s) {sorted(missing)}. "
                    "The schema predates this build and create_all() does not alter "
                    "existing tables. Run a migration, or for a disposable "
                    "environment: docker compose down -v && docker compose up."
                )

    def healthy(self) -> bool:
        try:
            with self.engine.connect() as conn:
                conn.execute(select(1))
            return True
        except Exception:  # noqa: BLE001
            return False

    # -- decisions ----------------------------------------------------------

    def record_decision(self, row: dict) -> str:
        row = {**row, "scored_at": row.get("scored_at") or _now()}
        with self.engine.begin() as conn:
            conn.execute(decisions.insert().values(**row))
        return row["decision_id"]

    def record_decisions(self, rows: list[dict]) -> int:
        if not rows:
            return 0
        rows = [{**r, "scored_at": r.get("scored_at") or _now()} for r in rows]
        with self.engine.begin() as conn:
            conn.execute(decisions.insert(), rows)
        return len(rows)

    def get_decision(self, decision_id: str) -> dict | None:
        with self.engine.connect() as conn:
            row = (
                conn.execute(select(decisions).where(decisions.c.decision_id == decision_id))
                .mappings()
                .first()
            )
            if row is None:
                return None
            record = dict(row)
            override = (
                conn.execute(
                    select(overrides)
                    .where(overrides.c.decision_id == decision_id)
                    .order_by(overrides.c.created_at.desc())
                )
                .mappings()
                .first()
            )

        record["reason_codes"] = _as_json(record["reason_codes"])
        record["features"] = _as_json(record["features"])
        record["overridden"] = override is not None
        if override is not None:
            record["override_decision"] = override["new_decision"]
            record["override_justification"] = override["justification"]
            record["override_by"] = override["underwriter_id"]
            record["overridden_at"] = override["created_at"]
        return record

    def record_override(
        self, decision_id: str, original: str, new: str, justification: str, underwriter_id: str
    ) -> str:
        override_id = new_id()
        with self.engine.begin() as conn:
            conn.execute(
                overrides.insert().values(
                    override_id=override_id,
                    decision_id=decision_id,
                    original_decision=original,
                    new_decision=new,
                    justification=justification,
                    underwriter_id=underwriter_id,
                    created_at=_now(),
                )
            )
        return override_id

    # -- batch jobs ----------------------------------------------------------

    def create_job(self, job_id: str, n_submitted: int) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                batch_jobs.insert().values(
                    job_id=job_id,
                    status="queued",
                    n_submitted=n_submitted,
                    n_scored=0,
                    submitted_at=_now(),
                )
            )

    def update_job(self, job_id: str, **values: Any) -> None:
        with self.engine.begin() as conn:
            conn.execute(batch_jobs.update().where(batch_jobs.c.job_id == job_id).values(**values))

    def get_job(self, job_id: str) -> dict | None:
        with self.engine.connect() as conn:
            row = (
                conn.execute(select(batch_jobs).where(batch_jobs.c.job_id == job_id))
                .mappings()
                .first()
            )
        if row is None:
            return None
        record = dict(row)
        record["results"] = _as_json(record["results"]) if record["results"] else None
        return record


def _as_json(value: Any) -> Any:
    """SQLite hands back JSON columns as text on some driver versions."""
    return json.loads(value) if isinstance(value, str) else value
