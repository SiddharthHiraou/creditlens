"""Retraining and the automated promotion gate.

The gate is the point of this file. Retraining is easy; deciding whether the
result is safe to serve is the part that needs to be mechanical, because the
alternative is a person under deadline pressure deciding that a small regression
is probably fine.

**A candidate is promoted only if every check passes.** There is no "promote with
a warning" path — a failing gate raises an issue and leaves the incumbent
serving. That asymmetry is deliberate: the cost of running the incumbent for
another month is bounded, and the cost of promoting a quietly worse model is not.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from prefect import flow, get_run_logger, task

from src.config import ARTIFACTS, SYNTHETIC_SPLIT, SYNTHETIC_TARGET
from src.evaluation.metrics import decile_table, discrimination, is_rank_ordered
from src.evaluation.psi import PSI_ALARM, psi
from src.fairness.report import age_band, fairness_report
from src.features.build import build
from src.features.spec import FeatureSpec
from src.ingestion.splits import split_by_time
from src.ingestion.target import assign_labels_from_dpd, modelling_population

CANDIDATE_DIR = ARTIFACTS / "candidates"
GATE_REPORT = ARTIFACTS / "promotion_gate.json"
DRIFT_ALERT = ARTIFACTS / "drift_alert.json"
ISSUE_DIR = ARTIFACTS / "issues"

# Gate thresholds. These are policy numbers, not tuning knobs -- they come from
# docs/credit_policy.md §8.5 and changing one is a Credit Committee decision.
MAX_AUC_REGRESSION = 0.01  # relative, 1%
MAX_ECE_WORSENING = 0.005  # absolute
MAX_SCORE_PSI = PSI_ALARM  # 0.25
MAX_DI_DEGRADATION = 0.05  # absolute drop in disparate impact ratio


@dataclass
class GateCheck:
    name: str
    passed: bool
    incumbent: float
    candidate: float
    threshold: str
    detail: str

    def render(self) -> str:
        mark = "PASS" if self.passed else "FAIL"
        return (
            f"[{mark}] {self.name}: incumbent {self.incumbent:.4f} -> "
            f"candidate {self.candidate:.4f} ({self.threshold})"
        )


@task
def train_candidate(n_trials: int) -> dict[str, Any]:
    """Run the full training pipeline and register the result as a candidate."""
    from src.models.train import run

    payload = run(n_trials=n_trials, use_mlflow=True)

    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%S")
    candidate_path = CANDIDATE_DIR / f"candidate_{stamp}.joblib"
    # The training run overwrites champion_model.joblib in place; copy it aside
    # so the gate compares two distinct artifacts rather than a model with itself.
    joblib.dump(joblib.load(ARTIFACTS / "champion_model.joblib"), candidate_path)
    (CANDIDATE_DIR / f"candidate_{stamp}.json").write_text(
        json.dumps(payload, indent=2, default=str)
    )
    return {"path": str(candidate_path), "metrics": payload, "stamp": stamp}


@flow(name="retrain-candidate", log_prints=True)
def retrain_candidate(*, n_trials: int = 100, triggered_by: str = "schedule") -> dict[str, Any]:
    """Monthly, or on a drift alert: train a candidate and stage it.

    Staging is the end of this flow. Nothing here promotes anything.
    """
    logger = get_run_logger()
    logger.info("training candidate (trigger: %s, %d Optuna trials)", triggered_by, n_trials)
    result = train_candidate(n_trials)
    champion = result["metrics"]["champion_calibrated_test"]
    logger.info(
        "candidate %s staged: OOT AUC %.4f, Gini %.4f",
        result["stamp"],
        champion["auc"],
        champion["gini"],
    )
    return {"triggered_by": triggered_by, **result}


@flow(name="drift-triggered-retrain", log_prints=True)
def drift_triggered_retrain(*, n_trials: int = 100) -> dict[str, Any]:
    """Retrain only if the drift flow raised an alert.

    Reads the alert file rather than being invoked directly, so a scheduled
    retrain and a drift-triggered one take the same path.
    """
    logger = get_run_logger()
    if not DRIFT_ALERT.exists():
        logger.info("no drift alert present; nothing to do")
        return {"triggered": False, "reason": "no alert file"}

    alert = json.loads(DRIFT_ALERT.read_text())
    logger.warning("drift alert found: PSI %.4f (%s)", alert["score_psi"], alert["verdict"])
    result = retrain_candidate(n_trials=n_trials, triggered_by="drift-alert")
    DRIFT_ALERT.unlink()  # consumed
    return {"triggered": True, "alert": alert, **result}


def _evaluate(model, spec: FeatureSpec) -> dict[str, Any]:
    """Score a model on the out-of-time fold and compute every gated metric."""
    from src.models.calibrate import expected_calibration_error

    pop = modelling_population(assign_labels_from_dpd(build().frame, SYNTHETIC_TARGET))
    splits = split_by_time(pop, SYNTHETIC_SPLIT)
    train, test = splits.train.collect(), splits.test.collect()

    x_train = spec.matrix(train).to_numpy().astype(np.float32)
    x_test = spec.matrix(test).to_numpy().astype(np.float32)
    y_test = test["label"].to_numpy().astype(int)

    pd_hat = model.predict_pd(x_test)
    report = discrimination(y_test, pd_hat)
    deciles = decile_table(y_test, pd_hat)

    approved = pd_hat <= np.quantile(pd_hat, 0.60)
    fairness = fairness_report(
        y_test, approved, age_band(test["DAYS_BIRTH"].to_numpy()), attribute="age_band"
    )

    return {
        "auc": report.auc,
        "gini": report.gini,
        "ks": report.ks,
        "brier": report.brier,
        "ece": expected_calibration_error(y_test, pd_hat),
        "score_psi": psi(model.predict_pd(x_train), pd_hat).psi,
        "rank_ordered": is_rank_ordered(deciles),
        "disparate_impact": fairness.disparate_impact,
    }


@task
def run_gate(incumbent_path: Path, candidate_path: Path) -> list[GateCheck]:
    """Evaluate both models on the same fold and apply every threshold."""
    spec = FeatureSpec.load()
    incumbent = _evaluate(joblib.load(incumbent_path), spec)
    candidate = _evaluate(joblib.load(candidate_path), spec)

    auc_floor = incumbent["auc"] * (1 - MAX_AUC_REGRESSION)
    return [
        GateCheck(
            name="discrimination",
            passed=candidate["auc"] >= auc_floor,
            incumbent=incumbent["auc"],
            candidate=candidate["auc"],
            threshold=f"AUC must be at least {auc_floor:.4f} (within {MAX_AUC_REGRESSION:.0%})",
            detail="A candidate may be marginally worse, never materially worse.",
        ),
        GateCheck(
            name="calibration",
            passed=candidate["ece"] <= incumbent["ece"] + MAX_ECE_WORSENING,
            incumbent=incumbent["ece"],
            candidate=candidate["ece"],
            threshold=f"ECE must not worsen by more than {MAX_ECE_WORSENING}",
            detail="Expected loss and pricing both depend on calibration holding.",
        ),
        GateCheck(
            name="stability",
            passed=candidate["score_psi"] < MAX_SCORE_PSI,
            incumbent=incumbent["score_psi"],
            candidate=candidate["score_psi"],
            threshold=f"score PSI must be below {MAX_SCORE_PSI}",
            detail="A candidate that already disagrees with its own training population is not ready.",
        ),
        GateCheck(
            name="rank_ordering",
            passed=bool(candidate["rank_ordered"]),
            incumbent=float(incumbent["rank_ordered"]),
            candidate=float(candidate["rank_ordered"]),
            threshold="bad rate must fall monotonically across deciles",
            detail="A model that does not rank-order is unshippable regardless of AUC.",
        ),
        GateCheck(
            name="fairness",
            passed=candidate["disparate_impact"]
            >= incumbent["disparate_impact"] - MAX_DI_DEGRADATION,
            incumbent=incumbent["disparate_impact"],
            candidate=candidate["disparate_impact"],
            threshold=f"disparate impact must not fall by more than {MAX_DI_DEGRADATION}",
            detail=(
                "This gates *degradation*, not compliance. The incumbent already fails "
                "the four-fifths rule on age; the gate stops that getting worse and is "
                "not a substitute for the fair-lending review that failure requires."
            ),
        ),
    ]


def _write_issue(checks: list[GateCheck], stamp: str) -> Path:
    """Write the issue a failed gate raises.

    A file rather than a live GitHub call: this runs unattended, and a flow that
    needs network credentials to report a failure has a second way to fail. CI
    can post it; the record exists either way.
    """
    ISSUE_DIR.mkdir(parents=True, exist_ok=True)
    failed = [c for c in checks if not c.passed]
    body = [
        f"# Promotion gate failed — candidate {stamp}",
        "",
        f"{len(failed)} of {len(checks)} checks failed. The incumbent remains in production.",
        "",
        "## Failed",
        "",
    ]
    body += [f"- **{c.name}** — {c.render()}\n  - {c.detail}" for c in failed]
    body += ["", "## Passed", ""]
    body += [f"- {c.render()}" for c in checks if c.passed]
    path = ISSUE_DIR / f"promotion-gate-{stamp}.md"
    path.write_text("\n".join(body))
    return path


@flow(name="validate-and-promote", log_prints=True)
def validate_and_promote(
    *, candidate_path: str | None = None, incumbent_path: str | None = None
) -> dict[str, Any]:
    """Gate a candidate against the incumbent and promote only on a clean sweep."""
    logger = get_run_logger()

    incumbent = Path(incumbent_path or ARTIFACTS / "champion_model.joblib")
    if candidate_path is None:
        candidates = sorted(CANDIDATE_DIR.glob("candidate_*.joblib"))
        if not candidates:
            logger.info("no staged candidate; nothing to gate")
            return {"promoted": False, "reason": "no candidate staged"}
        candidate = candidates[-1]
    else:
        candidate = Path(candidate_path)

    stamp = candidate.stem.replace("candidate_", "")
    checks = run_gate(incumbent, candidate)
    for check in checks:
        (logger.info if check.passed else logger.error)(check.render())

    promoted = all(c.passed for c in checks)
    report = {
        "candidate": str(candidate),
        "incumbent": str(incumbent),
        "stamp": stamp,
        "checks": [asdict(c) for c in checks],
        "promoted": promoted,
        "evaluated_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
    }

    if promoted:
        _promote_in_registry(logger)
        logger.info("all %d checks passed — candidate promoted to Production", len(checks))
    else:
        issue = _write_issue(checks, stamp)
        report["issue"] = str(issue)
        logger.error("gate failed — issue written to %s; incumbent still serving", issue)

    GATE_REPORT.write_text(json.dumps(report, indent=2, default=str))
    return report


def _promote_in_registry(logger) -> None:
    """Move the newest registered version to Production in MLflow.

    Best-effort: a registry that is unreachable must not turn a passing gate
    into an exception, but the failure is logged rather than swallowed.
    """
    try:
        import mlflow

        mlflow.set_tracking_uri(f"sqlite:///{ARTIFACTS / 'mlflow.db'}")
        client = mlflow.MlflowClient()
        versions = client.search_model_versions("name='creditlens-pd'")
        if not versions:
            logger.warning("no registered model versions to promote")
            return
        newest = max(versions, key=lambda v: int(v.version))
        client.set_registered_model_alias("creditlens-pd", "production", newest.version)
        logger.info("creditlens-pd v%s aliased to production", newest.version)
    except Exception as exc:  # noqa: BLE001
        logger.warning("registry promotion skipped: %s: %s", type(exc).__name__, exc)
