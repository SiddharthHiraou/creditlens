"""Phase 2 end-to-end: features -> selection -> four model tracks -> champion.

Run with ``make train``. Produces a calibrated, registered champion and the
full evaluation suite on out-of-time data.

Track order is deliberate. The scorecard runs first and its number is the one
every later model must justify itself against, because a GBDT that beats a
scorecard by 0.005 AUC is not worth the explainability cost.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import asdict
from typing import Any

import joblib
import numpy as np
import polars as pl
from rich.console import Console

from src.config import ARTIFACTS, SYNTHETIC_SPLIT, SYNTHETIC_TARGET
from src.evaluation.metrics import (
    decile_table,
    discrimination,
    is_rank_ordered,
    is_strictly_rank_ordered,
)
from src.evaluation.psi import csi, psi
from src.features.build import build
from src.features.monotonic import constraints_for
from src.features.selection import select
from src.features.spec import FeatureSpec
from src.ingestion.loaders import Source
from src.ingestion.splits import assert_no_temporal_leakage, split_by_time, vintage_column
from src.ingestion.target import assign_labels_from_dpd, modelling_population
from src.models.calibrate import calibrate, calibration_report, calibration_table
from src.models.ensemble import fit_stack
from src.models.gbdt import fit_catboost, fit_lightgbm, fit_xgboost, to_matrix
from src.models.scorecard import fit_scorecard
from src.models.tune import tune_lightgbm

warnings.filterwarnings("ignore", category=UserWarning)
console = Console()


def _rich(df: pl.DataFrame, title: str, limit: int = 15) -> None:
    from rich.table import Table

    t = Table(title=title, header_style="bold")
    for c in df.columns:
        t.add_column(c, justify="right" if df[c].dtype.is_numeric() else "left")
    for row in df.head(limit).iter_rows():
        t.add_row(
            *[
                f"{v:,.4f}" if isinstance(v, float) else f"{v:,}" if isinstance(v, int) else str(v)
                for v in row
            ]
        )
    console.print(t)


def run(
    source: Source = Source.AUTO,
    *,
    n_trials: int = 100,
    use_mlflow: bool = True,
    fast: bool = False,
) -> dict[str, Any]:
    if fast:
        n_trials = 12

    console.rule("[bold]CreditLens Phase 2 — feature pipeline and champion selection")

    # ---- data ------------------------------------------------------------
    fm = build(source=source)
    labelled = vintage_column(assign_labels_from_dpd(fm.frame, SYNTHETIC_TARGET))
    pop = modelling_population(labelled)
    splits = split_by_time(pop, SYNTHETIC_SPLIT)
    assert_no_temporal_leakage(splits)

    train = splits.train.collect()
    cal = splits.calibration.collect()
    valid = splits.valid.collect()
    test = splits.test.collect()
    console.print(
        f"[bold]splits[/bold] train={train.height:,} cal={cal.height:,} "
        f"valid={valid.height:,} test={test.height:,}   "
        f"features built={fm.n_features}"
    )

    # ---- selection (train only) -------------------------------------------
    console.print("Selecting features on training data only...")
    report = select(train, fm.feature_names)
    numeric = [f for f in report.kept if train[f].dtype.is_numeric()]
    console.print(
        f"  kept [bold]{len(report.kept)}[/bold] of {fm.n_features} "
        f"({len(numeric)} numeric usable by the GBDTs)"
    )
    _rich(report.summary(), "Why features were dropped", limit=8)

    spec = FeatureSpec.build(
        train, numeric, fm.categorical_names, dropped=report.dropped, iv=report.iv_table
    )
    spec.save()
    console.print(f"  feature spec v{spec.version} fingerprint={spec.fingerprint}")

    mono = constraints_for(numeric)
    n_constrained = sum(1 for m in mono if m != 0)
    console.print(f"  monotonic constraints on {n_constrained}/{len(numeric)} features")

    x = {
        name: to_matrix(df, numeric)
        for name, df in (("train", train), ("cal", cal), ("valid", valid), ("test", test))
    }
    y = {
        name: df["label"].to_numpy().astype(int)
        for name, df in (("train", train), ("cal", cal), ("valid", valid), ("test", test))
    }

    models: dict[str, Any] = {}
    results: dict[str, dict] = {}

    def evaluate(name: str, pd_valid: np.ndarray, pd_test: np.ndarray) -> None:
        results[name] = {
            "valid": discrimination(y["valid"], pd_valid).as_dict(),
            "test_oot": discrimination(y["test"], pd_test).as_dict(),
        }
        v, t = results[name]["valid"]["auc"], results[name]["test_oot"]["auc"]
        console.print(f"  [cyan]{name:22s}[/cyan] valid AUC {v:.4f}   OOT AUC {t:.4f}")

    # ---- track 1: logistic scorecard on WOE --------------------------------
    console.rule("[bold]Track 1 — logistic scorecard (WOE + PDO)")
    scorecard = fit_scorecard(train, numeric)
    models["scorecard"] = scorecard
    evaluate("scorecard", scorecard.predict_pd(valid), scorecard.predict_pd(test))
    _rich(scorecard.feature_contributions(), "Scorecard points range per feature", limit=10)

    # ---- track 2: LightGBM + Optuna ---------------------------------------
    console.rule(f"[bold]Track 2 — LightGBM with Optuna ({n_trials} trials)")
    tuning = tune_lightgbm(
        x["train"], y["train"], x["valid"], y["valid"], monotone=mono, n_trials=n_trials
    )
    console.print(
        f"  best valid AUC {tuning.best_value:.4f} over {tuning.n_trials} trials "
        f"({tuning.n_pruned} pruned early)"
    )
    console.print(f"  best params: {json.dumps(tuning.best_params, default=str)}")
    lgbm = fit_lightgbm(
        x["train"],
        y["train"],
        numeric,
        params=tuning.best_params,
        monotone=mono,
        eval_set=(x["valid"], y["valid"]),
    )
    models["lightgbm"] = lgbm
    evaluate("lightgbm", lgbm.predict_pd(x["valid"]), lgbm.predict_pd(x["test"]))

    # ---- track 3: challengers ---------------------------------------------
    console.rule("[bold]Track 3 — challengers")
    for name, fitter in (("xgboost", fit_xgboost), ("catboost", fit_catboost)):
        m = fitter(
            x["train"], y["train"], numeric, monotone=mono, eval_set=(x["valid"], y["valid"])
        )
        models[name] = m
        evaluate(name, m.predict_pd(x["valid"]), m.predict_pd(x["test"]))

    # ---- track 4: stacked ensemble ----------------------------------------
    console.rule("[bold]Track 4 — stacked ensemble (ceiling reference)")
    bases = [models["lightgbm"], models["xgboost"], models["catboost"]]
    stack = fit_stack(bases, x["valid"], y["valid"])
    models["stack"] = stack
    evaluate("stack", stack.predict_pd(x["valid"]), stack.predict_pd(x["test"]))
    console.print(
        f"  meta weights: {json.dumps({k: round(v, 3) for k, v in stack.weights().items()})}"
    )

    # ---- champion selection ------------------------------------------------
    # Chosen on validation. The stack is excluded: it is fitted on validation
    # predictions, so its validation score is optimistic by construction and
    # it cannot be served without all three bases.
    candidates = {k: v for k, v in results.items() if k != "stack"}
    champion_name = max(candidates, key=lambda k: candidates[k]["valid"]["auc"])
    champion = models[champion_name]
    console.rule(f"[bold green]Champion: {champion_name}")

    # ---- calibration --------------------------------------------------------
    is_gbdt = champion_name in ("lightgbm", "xgboost", "catboost")
    cal_input = x["cal"] if is_gbdt else cal
    test_input = x["test"] if is_gbdt else test

    calibrated = calibrate(champion, cal_input, y["cal"])
    raw_test = calibrated.predict_pd_uncalibrated(test_input)
    cal_test = calibrated.predict_pd(test_input)
    cal_report = calibration_report(y["test"], raw_test, cal_test)

    console.print(
        f"  Brier {cal_report['brier_raw']:.5f} -> [green]{cal_report['brier_calibrated']:.5f}[/green]   "
        f"ECE {cal_report['ece_raw']:.5f} -> [green]{cal_report['ece_calibrated']:.5f}[/green]"
    )
    console.print(
        f"  mean PD {cal_report['mean_pd_raw']:.4f} -> {cal_report['mean_pd_calibrated']:.4f} "
        f"vs actual bad rate {cal_report['actual_bad_rate']:.4f}"
    )
    _rich(calibration_table(y["test"], cal_test), "Calibration, out-of-time test", limit=10)

    champion_metrics = discrimination(y["test"], cal_test)
    console.print(f"\n[bold]Champion out-of-time (calibrated)[/bold]\n{champion_metrics.render()}")

    # ---- rank ordering and stability ---------------------------------------
    deciles = decile_table(y["test"], cal_test)
    ordered = is_rank_ordered(deciles)
    _rich(deciles, "Rank ordering, out-of-time test", limit=10)
    console.print(
        f"[bold {'green' if ordered else 'red'}]Rank ordering (noise-aware): {ordered}[/]"
        f"   strictly monotonic: {is_strictly_rank_ordered(deciles)}"
    )

    score_psi = psi(calibrated.predict_pd(x["train"] if is_gbdt else train), cal_test)
    console.print(f"[bold]Score PSI train vs OOT: {score_psi.psi:.4f}[/bold] ({score_psi.verdict})")
    feature_csi = csi(train, test, numeric)
    _rich(feature_csi.head(10), "Least stable features (CSI)", limit=10)

    # ---- persist -------------------------------------------------------------
    payload = {
        "phase": 2,
        "champion": champion_name,
        "n_features_built": fm.n_features,
        "n_features_selected": len(numeric),
        "n_monotonic_constraints": n_constrained,
        "feature_spec_fingerprint": spec.fingerprint,
        "tuning": {
            "n_trials": tuning.n_trials,
            "n_pruned": tuning.n_pruned,
            "best_valid_auc": tuning.best_value,
            "best_params": tuning.best_params,
        },
        "metrics": results,
        "champion_calibrated_test": champion_metrics.as_dict(),
        "calibration": cal_report,
        "rank_ordered_oot": ordered,
        "score_psi_train_vs_oot": score_psi.psi,
        "score_psi_verdict": score_psi.verdict,
        "worst_csi": feature_csi.head(10).to_dicts(),
        "decile_table_oot": deciles.to_dicts(),
        "stack_weights": stack.weights(),
    }
    (ARTIFACTS / "phase2_metrics.json").write_text(json.dumps(payload, indent=2, default=str))
    joblib.dump(calibrated, ARTIFACTS / "champion_model.joblib")
    feature_csi.write_parquet(ARTIFACTS / "feature_csi.parquet")
    if report.iv_table is not None:
        report.iv_table.write_parquet(ARTIFACTS / "information_values.parquet")
    scorecard.points_table().write_parquet(ARTIFACTS / "scorecard_points.parquet")

    if use_mlflow:
        _log_mlflow(payload, champion_name, calibrated, spec)

    console.print(f"\n[green]Wrote[/green] {ARTIFACTS / 'phase2_metrics.json'}")
    return payload


def _log_mlflow(payload: dict, champion_name: str, model, spec: FeatureSpec) -> None:
    """Track the run and register the champion in the MLflow model registry.

    The registry is what gives this project SR 11-7 lifecycle language for free:
    a model moves Staging -> Production only through the validation gate built
    in Phase 6, and every promotion is recorded with the metrics that justified it.
    """
    import mlflow

    # MLflow 3.x rejects the filesystem store, and the model registry needs a
    # database backend for stage transitions regardless -- which is what the
    # Phase 6 promotion gate will move models through.
    (ARTIFACTS / "mlruns").mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{ARTIFACTS / 'mlflow.db'}")
    mlflow.set_registry_uri(f"sqlite:///{ARTIFACTS / 'mlflow.db'}")
    mlflow.set_experiment("creditlens")

    with mlflow.start_run(run_name=f"phase2-{champion_name}"):
        mlflow.log_params(
            {
                "champion": champion_name,
                "n_features_built": payload["n_features_built"],
                "n_features_selected": payload["n_features_selected"],
                "n_monotonic_constraints": payload["n_monotonic_constraints"],
                "feature_spec_fingerprint": payload["feature_spec_fingerprint"],
                "optuna_trials": payload["tuning"]["n_trials"],
                **{f"hp_{k}": v for k, v in payload["tuning"]["best_params"].items()},
            }
        )
        champ = payload["champion_calibrated_test"]
        mlflow.log_metrics(
            {
                "oot_auc": champ["auc"],
                "oot_gini": champ["gini"],
                "oot_ks": champ["ks"],
                "oot_pr_auc": champ["pr_auc"],
                "oot_brier": champ["brier"],
                "brier_raw": payload["calibration"]["brier_raw"],
                "brier_calibrated": payload["calibration"]["brier_calibrated"],
                "ece_calibrated": payload["calibration"]["ece_calibrated"],
                "score_psi": payload["score_psi_train_vs_oot"],
                **{
                    f"{name}_oot_auc": r["test_oot"]["auc"]
                    for name, r in payload["metrics"].items()
                },
            }
        )
        mlflow.log_dict(asdict(spec), "feature_spec.json")
        mlflow.log_dict(payload, "phase2_metrics.json")
        # MLflow's skops serializer refuses unknown classes by default. These
        # are this project's own wrapper types plus the booster, so they are
        # declared trusted explicitly rather than by disabling the check.
        mlflow.sklearn.log_model(
            model,
            name="model",
            registered_model_name="creditlens-pd",
            skops_trusted_types=[
                "src.models.calibrate.CalibratedModel",
                "src.models.calibrate.SmoothedIsotonic",
                "src.models.gbdt.GbdtModel",
                "src.models.scorecard.Scorecard",
                "catboost.core.CatBoostClassifier",
                "lightgbm.sklearn.LGBMClassifier",
                "xgboost.sklearn.XGBClassifier",
            ],
        )
    console.print(f"  [green]logged to MLflow[/green] at {ARTIFACTS / 'mlflow.db'}")


if __name__ == "__main__":
    run()
