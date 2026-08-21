"""Phase 3 orchestrator: explainability and fairness audit of the champion.

Run with ``make audit``. Produces ``artifacts/phase3_report.json`` plus the
cached global SHAP summary that serving will read instead of recomputing.

The deliverable this exists to demonstrate: a decline that produces four
correct, distinct, human-readable reasons — and an honest account of what the
applicant could do about it and who the model treats differently.
"""

from __future__ import annotations

import json
import warnings
from typing import Any

import joblib
import numpy as np
import polars as pl
from rich.console import Console
from rich.table import Table

from src.config import ARTIFACTS, SYNTHETIC_SPLIT, SYNTHETIC_TARGET
from src.explainability.counterfactuals import CounterfactualSearch, levers_to_frame
from src.explainability.reason_codes import default_mapper
from src.explainability.shap_service import ShapService, additivity_error
from src.fairness.mitigation import cutoff_tradeoff_curve, threshold_optimizer_frontier
from src.fairness.report import (
    age_band,
    calibration_by_group,
    fairlearn_metrics,
    fairness_report,
)
from src.features.build import build
from src.features.spec import FeatureSpec
from src.ingestion.splits import split_by_time
from src.ingestion.target import assign_labels_from_dpd, modelling_population
from src.models.decision import DecisionPolicy, decide, pd_to_score, portfolio_summary

warnings.filterwarnings("ignore", category=UserWarning)
console = Console()

ACTIONABLE_FAMILIES = frozenset({"affordability", "utilisation", "debt_burden", "loan_structure"})


def _rich(df: pl.DataFrame, title: str, limit: int = 12) -> None:
    t = Table(title=title, header_style="bold")
    for c in df.columns:
        t.add_column(c, justify="right" if df[c].dtype.is_numeric() else "left")
    for row in df.head(limit).iter_rows():
        t.add_row(
            *[
                f"{v:,.4f}"
                if isinstance(v, float)
                else f"{v:,}"
                if isinstance(v, int)
                else str(v)[:44]
                for v in row
            ]
        )
    console.print(t)


def run(*, approve_rate: float = 0.60, n_counterfactual: int = 120) -> dict[str, Any]:
    console.rule("[bold]CreditLens Phase 3 — explainability and fairness audit")

    model = joblib.load(ARTIFACTS / "champion_model.joblib")
    spec = FeatureSpec.load()

    fm = build()
    pop = modelling_population(assign_labels_from_dpd(fm.frame, SYNTHETIC_TARGET))
    splits = split_by_time(pop, SYNTHETIC_SPLIT)
    train, test = splits.train.collect(), splits.test.collect()

    x_train = spec.matrix(train).to_numpy().astype(np.float32)
    x_test = spec.matrix(test).to_numpy().astype(np.float32)
    y_test = test["label"].to_numpy().astype(int)

    pd_hat = model.predict_pd(x_test)
    score = pd_to_score(pd_hat)
    policy = DecisionPolicy.from_approval_rate(score, approve_rate=approve_rate, refer_rate=0.10)
    decisions = decide(score, policy)

    console.print(
        f"[bold]policy[/bold] approve>={policy.approve_at:.1f} refer>={policy.refer_at:.1f} "
        f"(target {approve_rate:.0%} approval)"
    )
    _rich(
        portfolio_summary(pd_hat, y_test, policy, exposure=test["AMT_CREDIT"].to_numpy()),
        "Portfolio by decision band",
    )

    # ---- SHAP -------------------------------------------------------------
    console.rule("[bold]Explainability")
    shap_service = ShapService.from_model(model, spec.features)
    err = additivity_error(shap_service, model, x_test[:2000])
    console.print(f"SHAP additivity max error: {err:.2e} (must be ~1e-14; larger means miswired)")
    global_shap = shap_service.global_importance(x_test)
    shap_service.cache_global(x_test)
    _rich(global_shap.head(10), "Global SHAP importance", limit=10)

    # ---- reason codes -----------------------------------------------------
    mapper = default_mapper()
    unmapped = mapper.unmapped(spec.features)
    if unmapped:
        console.print(f"[red]{len(unmapped)} features have no reason-code family:[/red] {unmapped}")
    else:
        console.print(
            "[green]Every model feature maps to a reason-code family or is suppressed.[/green]"
        )

    declines = np.where(decisions == "decline")[0]
    shap_declines = shap_service.values(x_test[declines[:n_counterfactual]])

    console.print(f"\n[bold]Sample adverse action reasons[/bold] ({len(declines):,} declines)")
    examples = []
    for k, i in enumerate(declines[:3]):
        codes = mapper.explain(spec.features, shap_declines[k])
        console.print(
            f"\n  applicant {int(test['SK_ID_CURR'][int(i)])} — "
            f"PD {pd_hat[i]:.3f}, score {score[i]:.0f}, declined"
        )
        for rc in codes:
            console.print(f"    {rc.rank}. [{rc.label}] {rc.phrase}")
        examples.append(
            {
                "applicant": int(test["SK_ID_CURR"][int(i)]),
                "pd": float(pd_hat[i]),
                "score": float(score[i]),
                "reasons": [rc.as_dict() for rc in codes],
            }
        )

    # Distinctness is the compliance property: four reasons must be four
    # different families, not four spellings of one.
    counts = [len({rc.family for rc in mapper.explain(spec.features, v)}) for v in shap_declines]
    console.print(
        f"\n  distinct families per decline: mean {np.mean(counts):.2f}, "
        f"min {int(np.min(counts))}, share with 4 = {np.mean(np.array(counts) == 4):.1%}"
    )

    # ---- counterfactuals ---------------------------------------------------
    console.rule("[bold]Counterfactuals")
    search = CounterfactualSearch.from_reference(model, spec.features, x_train)
    sample = declines[:n_counterfactual]
    single, stacked, gains, by_segment = 0, 0, [], {"actionable-led": [], "history-led": []}
    for k, i in enumerate(sample):
        proposals = search.propose_levers(x_test[i], policy.approve_at)
        chosen, flipped = search.stack_levers(x_test[i], policy.approve_at, max_actions=3)
        single += any(p.flips_decision for p in proposals)
        stacked += flipped
        best = max([c.score_after for c in chosen], default=float(score[i]))
        gains.append(best - float(score[i]))
        codes = mapper.explain(spec.features, shap_declines[k])
        led = (
            "actionable-led" if codes and codes[0].family in ACTIONABLE_FAMILIES else "history-led"
        )
        by_segment[led].append(best - float(score[i]))

    console.print(
        f"of {len(sample)} declines: {single} flip on a single action, {stacked} flip on up to three"
    )
    console.print(f"median achievable score gain: {np.median(gains):.1f} points")
    for seg, values in by_segment.items():
        if values:
            console.print(
                f"  {seg:16s} n={len(values):3d}  median gain {np.median(values):5.1f}  "
                f"max {np.max(values):5.1f}"
            )
    if len(declines):
        _rich(
            levers_to_frame(search.propose_levers(x_test[declines[0]], policy.approve_at)).select(
                "lever", "magnitude", "score_before", "score_after", "flips_decision"
            ),
            f"Levers for applicant {int(test['SK_ID_CURR'][int(declines[0])])}",
        )

    # ---- fairness ----------------------------------------------------------
    console.rule("[bold]Fairness")
    approved = decisions == "approve"
    attributes = {
        "CODE_GENDER": test["CODE_GENDER"].to_numpy().astype(str),
        "age_band": age_band(test["DAYS_BIRTH"].to_numpy()),
    }

    fairness: dict[str, Any] = {}
    for name, values in attributes.items():
        report = fairness_report(y_test, approved, values, attribute=name, pd_hat=pd_hat)
        cross = fairlearn_metrics(y_test, approved, values)
        _rich(report.by_group, f"Group metrics — {name}")
        verdict = "PASSES" if report.passes_four_fifths else "FAILS"
        colour = "green" if report.passes_four_fifths else "red"
        console.print(
            f"[bold {colour}]{name}: disparate impact {report.disparate_impact:.4f} "
            f"— {verdict} the four-fifths rule[/]  "
            f"(worst {report.worst_group} vs {report.reference_group}, "
            f"equal-opportunity gap {report.equal_opportunity_difference:.4f})"
        )
        # Two independent implementations must agree or one of them is wrong.
        drift = abs(cross["demographic_parity_ratio"] - report.disparate_impact)
        console.print(f"  fairlearn cross-check delta: {drift:.2e}")
        fairness[name] = {
            **report.summary(),
            "by_group": report.by_group.to_dicts(),
            "fairlearn": cross,
            "cross_check_delta": drift,
            "calibration_by_group": calibration_by_group(y_test, pd_hat, values).to_dicts(),
        }

    # ---- mitigation --------------------------------------------------------
    console.rule("[bold]Mitigation tradeoff")
    ages = attributes["age_band"]
    curve = cutoff_tradeoff_curve(y_test, score, ages)
    _rich(
        curve.select(
            "target_approval_rate",
            "disparate_impact",
            "equal_opportunity_difference",
            "bad_rate_among_approved",
            "passes_four_fifths",
        ),
        "Single group-blind cutoff — the only legally deployable lever",
    )
    frontier = threshold_optimizer_frontier(y_test, score, ages)
    _rich(
        frontier.select(
            "strategy",
            "approval_rate",
            "disparate_impact",
            "equal_opportunity_difference",
            "bad_rate_among_approved",
        ),
        "Group-specific thresholds: analytical only, unlawful to deploy",
    )
    console.print(
        "[yellow]Parity here is bought by approving almost everyone. Read the bad rate "
        "among approved alongside the disparity numbers, never on its own.[/yellow]"
    )

    payload = {
        "phase": 3,
        "policy": {
            "approve_at": policy.approve_at,
            "refer_at": policy.refer_at,
            "target_approval_rate": approve_rate,
            "version": policy.version,
        },
        "shap": {
            "additivity_max_error": err,
            "global_top20": global_shap.head(20).to_dicts(),
        },
        "reason_codes": {
            "mapping_version": mapper.version,
            "unmapped_features": unmapped,
            "mean_distinct_families": float(np.mean(counts)),
            "share_with_four_reasons": float(np.mean(np.array(counts) == 4)),
            "examples": examples,
        },
        "counterfactuals": {
            "n_evaluated": len(sample),
            "flip_single_action": int(single),
            "flip_stacked_three": int(stacked),
            "median_score_gain": float(np.median(gains)),
            "by_segment": {k: float(np.median(v)) for k, v in by_segment.items() if v},
        },
        "fairness": fairness,
        "mitigation": {
            "cutoff_curve": curve.to_dicts(),
            "threshold_optimizer": frontier.to_dicts(),
        },
    }
    (ARTIFACTS / "phase3_report.json").write_text(json.dumps(payload, indent=2, default=str))
    global_shap.write_parquet(ARTIFACTS / "shap_global.parquet")
    console.print(f"\n[green]Wrote[/green] {ARTIFACTS / 'phase3_report.json'}")
    return payload


if __name__ == "__main__":
    run()
