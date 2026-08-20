"""CreditLens command line. ``creditlens --help`` after an editable install."""

from __future__ import annotations

import typer
from rich.console import Console

app = typer.Typer(add_completion=False, help="CreditLens: credit decisioning and model risk.")
console = Console()


@app.command()
def data(
    n: int = typer.Option(50_000, help="Number of synthetic applicants."),
    seed: int = typer.Option(20260820, help="Generator seed."),
) -> None:
    """Generate the synthetic Home-Credit-shaped dataset."""
    from src.ingestion.synthetic import SyntheticConfig, generate, write

    tables = generate(SyntheticConfig(n_applicants=n, seed=seed))
    for name, summary in write(tables).items():
        console.print(f"  {name:26s} {summary}")


@app.command()
def validate() -> None:
    """Run every Pandera contract against whatever source resolves."""
    from src.ingestion.loaders import describe_sources, load

    console.print(describe_sources())
    for name in describe_sources()["table"]:
        try:
            load(name)
            console.print(f"  [green]PASS[/green] {name}")
        except Exception as exc:  # noqa: BLE001 - report, do not swallow
            console.print(f"  [red]FAIL[/red] {name}: {str(exc)[:200]}")
            raise typer.Exit(1) from exc


@app.command()
def baseline() -> None:
    """Train and evaluate the Phase 1 logistic baseline."""
    from src.models.train_baseline import run

    run()


@app.command()
def train(
    trials: int = typer.Option(100, help="Optuna trials for the LightGBM search."),
    fast: bool = typer.Option(False, help="12 trials — for smoke tests, not for reporting."),
    mlflow: bool = typer.Option(True, help="Log the run and register the champion."),
) -> None:
    """Phase 2: features, selection, four model tracks, calibrated champion."""
    from src.models.train import run

    run(n_trials=trials, use_mlflow=mlflow, fast=fast)


@app.command()
def features() -> None:
    """Build the feature matrix and report its shape."""
    from src.features.build import build

    fm = build()
    df = fm.frame.collect()
    console.print(
        f"{df.height:,} applicants x [bold]{fm.n_features}[/bold] features "
        f"({len(fm.categorical_names)} categorical)"
    )


@app.command()
def sources() -> None:
    """Show which file each table will be read from."""
    from src.ingestion.loaders import describe_sources

    console.print(describe_sources())


if __name__ == "__main__":
    app()
