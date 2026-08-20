"""Phase 1 end-to-end run: load -> label -> split out-of-time -> fit -> report.

Run with ``make baseline``. Writes ``artifacts/baseline_metrics.json`` and
``artifacts/baseline_model.joblib``.
"""

from __future__ import annotations

import json

import joblib
import polars as pl
from rich.console import Console
from rich.table import Table

from src.config import ARTIFACTS, SYNTHETIC_SPLIT, SYNTHETIC_TARGET
from src.evaluation.metrics import (
    decile_table,
    discrimination,
    is_rank_ordered,
    is_strictly_rank_ordered,
    rank_order_violations,
)
from src.ingestion.loaders import Source, load
from src.ingestion.splits import assert_no_temporal_leakage, split_by_time, vintage_column
from src.ingestion.target import (
    assign_labels_from_dpd,
    label_summary,
    modelling_population,
)
from src.models.baseline import fit, prepare

console = Console()


def _rich_table(df: pl.DataFrame, title: str) -> Table:
    t = Table(title=title, header_style="bold")
    for c in df.columns:
        t.add_column(c, justify="right" if df[c].dtype.is_numeric() else "left")
    for row in df.iter_rows():
        t.add_row(
            *[
                f"{v:,.4f}" if isinstance(v, float) else f"{v:,}" if isinstance(v, int) else str(v)
                for v in row
            ]
        )
    return t


def run(source: Source = Source.AUTO, *, n_bins: int = 10) -> dict:
    console.rule("[bold]CreditLens Phase 1 baseline")

    lf = load("application", source)
    lf = assign_labels_from_dpd(lf, SYNTHETIC_TARGET)
    lf = vintage_column(lf)

    console.print(_rich_table(label_summary(lf), "Target distribution (all rows)"))

    model_pop = modelling_population(lf)
    model_pop = prepare(model_pop)

    splits = split_by_time(model_pop, SYNTHETIC_SPLIT)
    assert_no_temporal_leakage(splits)
    counts = splits.counts()
    console.print(f"[bold]Out-of-time splits:[/bold] {counts}")

    train_df = splits.train.collect()
    frames = {
        "calibration": splits.calibration.collect(),
        "valid": splits.valid.collect(),
        "test_oot": splits.test.collect(),
    }

    console.print(f"Fitting logistic baseline on {train_df.height:,} rows...")
    model = fit(train_df)

    results: dict[str, dict] = {}
    for name, df in {"train": train_df, **frames}.items():
        if df.height == 0:
            continue
        pd_hat = model.predict_pd(df)
        rep = discrimination(df["label"].to_numpy(), pd_hat)
        results[name] = rep.as_dict()
        console.print(f"\n[bold cyan]{name}[/bold cyan]\n{rep.render()}")

    oot = frames["test_oot"]
    oot_pd = model.predict_pd(oot)
    deciles = decile_table(oot["label"].to_numpy(), oot_pd, n_bins=n_bins)
    ordered = is_rank_ordered(deciles)
    strict = is_strictly_rank_ordered(deciles)
    violations = rank_order_violations(deciles)
    console.print(_rich_table(deciles, "Rank ordering, out-of-time test"))
    console.print(
        f"[bold {'green' if ordered else 'red'}]"
        f"Rank ordering (noise-aware, 2 SE): {ordered}[/]"
        f"   strictly monotonic: {strict}"
    )
    if violations.height:
        console.print(_rich_table(violations, "Significant rank-order inversions"))

    console.print(_rich_table(model.coefficients().head(12), "Top standardised coefficients"))

    payload = {
        "phase": 1,
        "model": "logistic_regression_baseline",
        "source": str(source),
        "split_config": {k: str(v) for k, v in SYNTHETIC_SPLIT.model_dump().items()},
        "target_config": {k: str(v) for k, v in SYNTHETIC_TARGET.model_dump().items()},
        "split_counts": counts,
        "metrics": results,
        "rank_ordered_oot": ordered,
        "strictly_monotonic_oot": strict,
        "rank_order_violations_oot": violations.to_dicts(),
        "decile_table_oot": deciles.to_dicts(),
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "baseline_metrics.json").write_text(json.dumps(payload, indent=2, default=str))
    joblib.dump(model, ARTIFACTS / "baseline_model.joblib")
    console.print(f"\n[green]Wrote[/green] {ARTIFACTS / 'baseline_metrics.json'}")
    return payload


if __name__ == "__main__":
    run()
