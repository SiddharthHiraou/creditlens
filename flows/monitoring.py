"""Daily flows: ingestion validation and drift computation.

Prefect over Airflow because the whole orchestration layer here is four flows
and a schedule. Airflow's scheduler, webserver and metadata database are a lot
of moving parts to run four Python functions on a cron.
"""

from __future__ import annotations

import datetime as dt
import json
from typing import Any

import numpy as np
import polars as pl
from prefect import flow, get_run_logger, task

from src.config import ARTIFACTS, SYNTHETIC_SPLIT, SYNTHETIC_TARGET
from src.evaluation.psi import PSI_ALARM, PSI_INVESTIGATE, csi, psi
from src.features.build import build
from src.features.spec import FeatureSpec
from src.ingestion.loaders import HOME_CREDIT_FILES, load
from src.ingestion.splits import split_by_time
from src.ingestion.target import assign_labels_from_dpd, modelling_population

MONITORING_LOG = ARTIFACTS / "monitoring_history.jsonl"
DRIFT_ALERT = ARTIFACTS / "drift_alert.json"


@task(retries=2, retry_delay_seconds=30)
def validate_table(name: str) -> dict[str, Any]:
    """Enforce one table's Pandera contract.

    Retried because a transient read failure is worth retrying and a contract
    violation is not — the schema error raises on every attempt and surfaces
    after the retries are exhausted, which is the correct outcome.
    """
    logger = get_run_logger()
    frame = load(name, validate=True)
    rows = int(frame.select(pl.len()).collect().item())
    logger.info("validated %s: %s rows", name, f"{rows:,}")
    return {"table": name, "rows": rows, "valid": True}


@flow(name="ingest-and-validate", log_prints=True)
def ingest_and_validate() -> dict[str, Any]:
    """Daily: validate every source table against its schema contract.

    Fails the run on a contract violation rather than landing bad data. A
    silently-accepted schema change is the failure mode this exists to prevent.
    """
    logger = get_run_logger()
    results = [validate_table(name) for name in HOME_CREDIT_FILES]
    total = sum(r["rows"] for r in results)
    logger.info("all %d tables valid, %s rows total", len(results), f"{total:,}")
    return {
        "tables": results,
        "total_rows": total,
        "ran_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
    }


@task
def score_population() -> dict[str, np.ndarray]:
    import joblib

    model = joblib.load(ARTIFACTS / "champion_model.joblib")
    spec = FeatureSpec.load()
    pop = modelling_population(assign_labels_from_dpd(build().frame, SYNTHETIC_TARGET))
    splits = split_by_time(pop, SYNTHETIC_SPLIT)
    train, test = splits.train.collect(), splits.test.collect()
    return {
        "baseline_pd": model.predict_pd(spec.matrix(train).to_numpy().astype(np.float32)),
        "current_pd": model.predict_pd(spec.matrix(test).to_numpy().astype(np.float32)),
        "train": train,
        "test": test,
        "features": spec.features,
    }


@flow(name="compute-drift", log_prints=True)
def compute_drift(*, alarm_threshold: float = PSI_ALARM) -> dict[str, Any]:
    """Daily: PSI on the score, CSI per feature, alert above the alarm threshold.

    Writes ``drift_alert.json`` when PSI crosses the threshold. The retraining
    flow reads that file rather than being called directly, so a drift alert and
    a scheduled retrain arrive through the same door.
    """
    logger = get_run_logger()
    data = score_population()

    score_psi = psi(data["baseline_pd"], data["current_pd"])
    numeric = [f for f in data["features"] if f in data["train"].columns]
    feature_csi = csi(data["train"], data["test"], numeric)
    worst = feature_csi.head(10).to_dicts()

    record = {
        "ran_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "score_psi": round(score_psi.psi, 6),
        "verdict": score_psi.verdict,
        "is_alarm": bool(score_psi.psi >= alarm_threshold),
        "needs_investigation": bool(score_psi.psi >= PSI_INVESTIGATE),
        "n_baseline": int(data["baseline_pd"].size),
        "n_current": int(data["current_pd"].size),
        "worst_features": worst,
    }

    MONITORING_LOG.parent.mkdir(parents=True, exist_ok=True)
    with MONITORING_LOG.open("a") as handle:
        handle.write(json.dumps(record, default=str) + "\n")

    if record["is_alarm"]:
        DRIFT_ALERT.write_text(json.dumps(record, indent=2, default=str))
        logger.error(
            "PSI %.4f is at or above the alarm threshold %.2f — retraining candidate raised",
            score_psi.psi,
            alarm_threshold,
        )
    elif record["needs_investigation"]:
        logger.warning("PSI %.4f warrants investigation", score_psi.psi)
    else:
        logger.info("PSI %.4f — stable", score_psi.psi)

    return record
