"""Export static demo data for the frontend.

The frontend must work as a public link with no backend running — a portfolio
site that 500s because a container is asleep is worse than no link. So every
page is driven by JSON snapshots written here at build time, and the interactive
pieces (the cutoff simulator especially) compute client-side from a seeded
sample rather than round-tripping to an API.

Where a live API *is* available, the scoring page calls it; otherwise it falls
back to these snapshots and says so.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl

from src.config import ARTIFACTS, ROOT, SYNTHETIC_SPLIT, SYNTHETIC_TARGET
from src.evaluation.metrics import decile_table
from src.evaluation.psi import csi, psi
from src.explainability.reason_codes import default_mapper
from src.explainability.shap_service import ShapService
from src.fairness.mitigation import cutoff_tradeoff_curve, threshold_optimizer_frontier
from src.fairness.report import age_band, fairness_report
from src.features.build import build
from src.features.spec import FeatureSpec
from src.ingestion.splits import split_by_time, vintage_column
from src.ingestion.target import assign_labels_from_dpd, modelling_population
from src.models.decision import DecisionPolicy, decide, pd_to_score

OUT = ROOT / "frontend" / "public" / "data"

# The simulator recomputes bad rate and profit on every slider tick. 4,000 rows
# is enough for the curve to be smooth and small enough to ship in the page.
SIMULATOR_SAMPLE = 4_000
DEMO_APPLICANTS = 12


def _write(name: str, payload: Any) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.json"
    path.write_text(json.dumps(payload, separators=(",", ":"), default=_default))
    return path


def _default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _round(values: np.ndarray, places: int = 4) -> list[float]:
    return [round(float(v), places) for v in values]


def export() -> dict[str, int]:
    model = joblib.load(ARTIFACTS / "champion_model.joblib")
    spec = FeatureSpec.load()
    metrics = json.loads((ARTIFACTS / "phase2_metrics.json").read_text())
    phase3 = json.loads((ARTIFACTS / "phase3_report.json").read_text())

    fm = build()
    labelled = vintage_column(assign_labels_from_dpd(fm.frame, SYNTHETIC_TARGET))
    pop = modelling_population(labelled)
    splits = split_by_time(pop, SYNTHETIC_SPLIT)
    train, test = splits.train.collect(), splits.test.collect()

    x_train = spec.matrix(train).to_numpy().astype(np.float32)
    x_test = spec.matrix(test).to_numpy().astype(np.float32)
    y_test = test["label"].to_numpy().astype(int)
    pd_hat = model.predict_pd(x_test)
    score = pd_to_score(pd_hat)
    exposure = test["AMT_CREDIT"].to_numpy().astype(float)

    policy = DecisionPolicy.from_approval_rate(score, approve_rate=0.60, refer_rate=0.10)
    decisions = decide(score, policy)

    written: dict[str, int] = {}

    # -- headline ------------------------------------------------------------
    champion = metrics["champion_calibrated_test"]
    summary = {
        "champion": metrics["champion"],
        "generatedAt": spec.created_at,
        "modelVersion": metrics["champion"],
        "featureSpecFingerprint": spec.fingerprint,
        "nFeaturesBuilt": metrics["n_features_built"],
        "nFeaturesSelected": metrics["n_features_selected"],
        "nMonotonicConstraints": metrics["n_monotonic_constraints"],
        "headline": {
            "auc": round(champion["auc"], 4),
            "gini": round(champion["gini"], 4),
            "ks": round(champion["ks"], 4),
            "brier": round(champion["brier"], 5),
            "prAuc": round(champion["pr_auc"], 4),
            "badRate": round(champion["bad_rate"], 4),
            "n": int(champion["n"]),
            "psi": round(metrics["score_psi_train_vs_oot"], 4),
        },
        "baseline": {"auc": 0.7624, "gini": 0.5249, "ks": 0.3818, "brier": 0.2066},
        "calibration": {k: round(float(v), 5) for k, v in metrics["calibration"].items()},
        "tracks": [
            {
                "name": name,
                "validAuc": round(r["valid"]["auc"], 4),
                "ootAuc": round(r["test_oot"]["auc"], 4),
                "ootGini": round(r["test_oot"]["gini"], 4),
                "ootKs": round(r["test_oot"]["ks"], 4),
                "isChampion": name == metrics["champion"],
            }
            for name, r in metrics["metrics"].items()
        ],
        "latency": {
            "onnxP99Ms": 0.169,
            "nativeP99Ms": 0.472,
            "apiP99Ms": 46,
            "apiUsers": 16,
        },
        "policy": {
            "approveAt": round(policy.approve_at, 1),
            "referAt": round(policy.refer_at, 1),
            "lgd": policy.lgd,
        },
    }
    written["summary"] = len(_write("summary", summary).read_bytes())

    # -- portfolio -----------------------------------------------------------
    deciles = decile_table(y_test, pd_hat)
    hist, edges = np.histogram(score, bins=40)
    by_vintage = (
        test.with_columns(pl.Series("score", score), pl.Series("pd", pd_hat))
        .group_by("vintage")
        .agg(
            n=pl.len(),
            badRate=pl.col("label").mean(),
            meanPd=pl.col("pd").mean(),
            meanScore=pl.col("score").mean(),
        )
        .sort("vintage")
    )
    bands = []
    for name in ("decline", "refer", "approve"):
        mask = decisions == name
        bands.append(
            {
                "decision": name,
                "n": int(mask.sum()),
                "share": round(float(mask.mean()), 4),
                "badRate": round(float(y_test[mask].mean()), 4),
                "meanPd": round(float(pd_hat[mask].mean()), 4),
                "expectedLoss": round(float((pd_hat[mask] * policy.lgd * exposure[mask]).sum()), 2),
            }
        )

    portfolio = {
        "bands": bands,
        "scoreDistribution": [
            {"score": round(float((edges[i] + edges[i + 1]) / 2), 1), "count": int(hist[i])}
            for i in range(len(hist))
        ],
        "deciles": [
            {
                "decile": int(r["decile"]),
                "n": int(r["n"]),
                "badRate": round(float(r["bad_rate"]), 4),
                "meanPd": round(float(r["mean_pd"]), 4),
                "lift": round(float(r["lift"]), 3),
                "cumBadCapture": round(float(r["cum_bad_capture"]), 4),
            }
            for r in deciles.iter_rows(named=True)
        ],
        "vintages": [
            {
                "vintage": r["vintage"],
                "n": int(r["n"]),
                "badRate": round(float(r["badRate"]), 4),
                "meanPd": round(float(r["meanPd"]), 4),
                "meanScore": round(float(r["meanScore"]), 1),
            }
            for r in by_vintage.iter_rows(named=True)
        ],
    }
    written["portfolio"] = len(_write("portfolio", portfolio).read_bytes())

    # -- simulator sample ----------------------------------------------------
    # A stratified-by-score sample so the tails survive; a uniform draw would
    # thin out exactly the region where the cutoff actually moves.
    order = np.argsort(score)
    step = max(len(order) // SIMULATOR_SAMPLE, 1)
    idx = order[::step][:SIMULATOR_SAMPLE]
    written["simulator"] = len(
        _write(
            "simulator",
            {
                "score": _round(score[idx], 2),
                "pd": _round(pd_hat[idx], 5),
                "y": [int(v) for v in y_test[idx]],
                "exposure": [round(float(v), 2) for v in exposure[idx]],
                "sampledFrom": int(len(score)),
                "lgd": policy.lgd,
            },
        ).read_bytes()
    )

    # -- monitoring ----------------------------------------------------------
    feature_csi = csi(train, test, [f for f in spec.features if f in train.columns])
    score_psi = psi(model.predict_pd(x_train), pd_hat)
    monitoring = {
        "scorePsi": round(score_psi.psi, 5),
        "verdict": score_psi.verdict,
        "isAlarm": score_psi.is_alarm,
        "thresholds": {"investigate": 0.10, "alarm": 0.25},
        "psiBins": [
            {
                "bin": int(r["bin"]),
                "expected": round(float(r["expected_share"]), 5),
                "actual": round(float(r["actual_share"]), 5),
                "contribution": round(float(r["contribution"]), 6),
            }
            for r in score_psi.table.iter_rows(named=True)
        ],
        "featureCsi": [
            {"feature": r["feature"], "csi": round(float(r["csi"]), 5), "verdict": r["verdict"]}
            for r in feature_csi.head(25).iter_rows(named=True)
        ],
        "vintagePsi": _vintage_psi(train, test, model, spec),
        "challengers": [
            {
                "name": name,
                "ootAuc": round(r["test_oot"]["auc"], 4),
                "ootGini": round(r["test_oot"]["gini"], 4),
                "isChampion": name == metrics["champion"],
            }
            for name, r in metrics["metrics"].items()
        ],
    }
    written["monitoring"] = len(_write("monitoring", monitoring).read_bytes())

    # -- fairness ------------------------------------------------------------
    approved = decisions == "approve"
    attributes = {
        "gender": test["CODE_GENDER"].to_numpy().astype(str),
        "ageBand": age_band(test["DAYS_BIRTH"].to_numpy()),
    }
    groups = {}
    for name, values in attributes.items():
        report = fairness_report(y_test, approved, values, attribute=name, pd_hat=pd_hat)
        groups[name] = {
            **report.summary(),
            "byGroup": [
                {
                    "group": r["group"],
                    "n": int(r["n"]),
                    "selectionRate": round(float(r["selection_rate"]), 4),
                    "observedBadRate": round(float(r["observed_bad_rate"]), 4),
                    "meanPredictedPd": round(float(r["mean_predicted_pd"]), 4),
                    "calibrationGap": round(float(r["calibration_gap"]), 4),
                    "tprGoodApproved": round(float(r["tpr_good_approved"]), 4),
                }
                for r in report.by_group.iter_rows(named=True)
            ],
        }

    curve = cutoff_tradeoff_curve(y_test, score, attributes["ageBand"])
    frontier = threshold_optimizer_frontier(y_test, score, attributes["ageBand"])
    fairness = {
        "fourFifths": 0.80,
        "groups": groups,
        "cutoffCurve": [
            {
                "approvalRate": round(float(r["target_approval_rate"]), 3),
                "disparateImpact": round(float(r["disparate_impact"]), 4),
                "equalOpportunityDifference": round(float(r["equal_opportunity_difference"]), 4),
                "badRateAmongApproved": round(float(r["bad_rate_among_approved"]), 4),
                "passesFourFifths": bool(r["passes_four_fifths"]),
            }
            for r in curve.iter_rows(named=True)
        ],
        "thresholdOptimizer": [
            {
                "strategy": r["strategy"],
                "approvalRate": round(float(r["approval_rate"]), 4),
                "disparateImpact": round(float(r["disparate_impact"]), 4),
                "equalOpportunityDifference": round(float(r["equal_opportunity_difference"]), 4),
                "badRateAmongApproved": round(float(r["bad_rate_among_approved"]), 4),
                "note": r["note"],
            }
            for r in frontier.iter_rows(named=True)
        ],
    }
    written["fairness"] = len(_write("fairness", fairness).read_bytes())

    # -- demo applicants for the scoring page --------------------------------
    written["applicants"] = len(
        _write(
            "applicants",
            _demo_applicants(model, spec, test, x_test, pd_hat, score, decisions, exposure),
        ).read_bytes()
    )

    # -- explainability / model card -----------------------------------------
    written["explainability"] = len(
        _write(
            "explainability",
            {
                "globalShap": phase3["shap"]["global_top20"],
                "additivityError": phase3["shap"]["additivity_max_error"],
                "reasonCodes": phase3["reason_codes"],
                "counterfactuals": phase3["counterfactuals"],
                "informationValues": _information_values(),
            },
        ).read_bytes()
    )

    return written


def _vintage_psi(train, test, model, spec) -> list[dict]:
    """PSI of each out-of-time vintage against the training baseline.

    A single PSI number hides when the drift started. Per-vintage shows the
    shape, which is what a monitoring page is for.
    """
    baseline = model.predict_pd(spec.matrix(train).to_numpy().astype(np.float32))
    out = []
    for vintage in sorted(test["vintage"].unique().to_list()):
        subset = test.filter(pl.col("vintage") == vintage)
        if subset.height < 100:
            continue
        current = model.predict_pd(spec.matrix(subset).to_numpy().astype(np.float32))
        result = psi(baseline, current)
        out.append(
            {
                "vintage": vintage,
                "n": int(subset.height),
                "psi": round(result.psi, 5),
                "verdict": result.verdict,
            }
        )
    return out


def _information_values() -> list[dict]:
    path = ARTIFACTS / "information_values.parquet"
    if not path.exists():
        return []
    df = pl.read_parquet(path).head(25)
    return [
        {"feature": r["feature"], "iv": round(float(r["iv"]), 4), "strength": r["strength"]}
        for r in df.iter_rows(named=True)
    ]


def _demo_applicants(model, spec, test, x_test, pd_hat, score, decisions, exposure) -> list[dict]:
    """A spread of scored applicants with full explanations.

    Deliberately mixed across decision bands: a scoring page that only ever
    shows declines does not demonstrate the product.
    """
    shap_service = ShapService.from_model(model, spec.features)
    mapper = default_mapper()

    chosen: list[int] = []
    for band in ("decline", "refer", "approve"):
        band_idx = np.where(decisions == band)[0]
        if band_idx.size:
            chosen.extend(band_idx[:: max(band_idx.size // 4, 1)][:4].tolist())

    values = shap_service.values(x_test[chosen])
    out = []
    for k, i in enumerate(chosen):
        row = test.row(int(i), named=True)
        contributions = values[k]
        order = np.argsort(-np.abs(contributions))[:12]
        out.append(
            {
                "skIdCurr": int(row["SK_ID_CURR"]),
                "pd": round(float(pd_hat[i]), 4),
                "score": round(float(score[i]), 1),
                "decision": str(decisions[i]),
                "exposure": round(float(exposure[i]), 2),
                "expectedLoss": round(float(pd_hat[i] * 0.65 * exposure[i]), 2),
                "profile": {
                    "income": round(float(row["AMT_INCOME_TOTAL"]), 0),
                    "credit": round(float(row["AMT_CREDIT"]), 0),
                    "annuity": round(float(row["AMT_ANNUITY"]), 0),
                    "ageYears": round(-float(row["DAYS_BIRTH"]) / 365.25, 1),
                    "employedYears": round(-float(row["DAYS_EMPLOYED"]) / 365.25, 1),
                    "education": row["NAME_EDUCATION_TYPE"],
                    "occupation": row["OCCUPATION_TYPE"],
                    "contractType": row["NAME_CONTRACT_TYPE"],
                },
                "reasonCodes": [rc.as_dict() for rc in mapper.explain(spec.features, contributions)]
                if str(decisions[i]) != "approve"
                else [],
                "shap": [
                    {
                        "feature": spec.features[j],
                        "value": None
                        if not np.isfinite(x_test[i, j])
                        else round(float(x_test[i, j]), 4),
                        "shap": round(float(contributions[j]), 5),
                    }
                    for j in order
                ],
                "baseValue": round(float(shap_service.expected_value), 5),
            }
        )
    return out


if __name__ == "__main__":
    for name, size in export().items():
        print(f"  {name:16s} {size / 1024:8.1f} KB")
