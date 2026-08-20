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
def audit(
    approve_rate: float = typer.Option(0.60, help="Target approval rate for the policy cutoff."),
) -> None:
    """Phase 3: SHAP, ECOA reason codes, counterfactuals, fairness, mitigation."""
    from src.models.audit import run

    run(approve_rate=approve_rate)


@app.command()
def explain(
    applicant: int = typer.Option(
        None, help="SK_ID_CURR to explain; default is the first decline."
    ),
) -> None:
    """Print the adverse action reasons and counterfactual levers for one applicant."""
    import joblib
    import numpy as np

    from src.config import ARTIFACTS, SYNTHETIC_SPLIT, SYNTHETIC_TARGET
    from src.explainability.counterfactuals import CounterfactualSearch, levers_to_frame
    from src.explainability.reason_codes import default_mapper
    from src.explainability.shap_service import ShapService
    from src.features.build import build
    from src.features.spec import FeatureSpec
    from src.ingestion.splits import split_by_time
    from src.ingestion.target import assign_labels_from_dpd, modelling_population
    from src.models.decision import DecisionPolicy, decide, pd_to_score

    model = joblib.load(ARTIFACTS / "champion_model.joblib")
    spec = FeatureSpec.load()
    pop = modelling_population(assign_labels_from_dpd(build().frame, SYNTHETIC_TARGET))
    splits = split_by_time(pop, SYNTHETIC_SPLIT)
    train, test = splits.train.collect(), splits.test.collect()

    x_train = spec.matrix(train).to_numpy().astype("float32")
    x_test = spec.matrix(test).to_numpy().astype("float32")
    pd_hat = model.predict_pd(x_test)
    score = pd_to_score(pd_hat)
    policy = DecisionPolicy.from_approval_rate(score, approve_rate=0.60, refer_rate=0.10)
    decisions = decide(score, policy)

    if applicant is None:
        idx = int(np.where(decisions == "decline")[0][0])
    else:
        matches = np.where(test["SK_ID_CURR"].to_numpy() == applicant)[0]
        if not matches.size:
            console.print(f"[red]No applicant {applicant} in the out-of-time test fold.[/red]")
            raise typer.Exit(1)
        idx = int(matches[0])

    console.print(
        f"[bold]applicant {int(test['SK_ID_CURR'][idx])}[/bold]  "
        f"PD {pd_hat[idx]:.4f}  score {score[idx]:.0f}  -> [bold]{decisions[idx]}[/bold]  "
        f"(approve at {policy.approve_at:.0f})"
    )

    shap_values = ShapService.from_model(model, spec.features).values(x_test[idx : idx + 1])[0]
    for code in default_mapper().explain(spec.features, shap_values):
        console.print(f"  {code.rank}. [{code.label}] {code.phrase}")

    if decisions[idx] != "approve":
        search = CounterfactualSearch.from_reference(model, spec.features, x_train)
        console.print(
            levers_to_frame(search.propose_levers(x_test[idx], policy.approve_at)).select(
                "lever", "magnitude", "score_after", "flips_decision"
            )
        )


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
def serve(
    host: str = typer.Option("0.0.0.0", help="Bind address."),
    port: int = typer.Option(8000, help="Port."),
    reload: bool = typer.Option(False, help="Auto-reload on source changes."),
) -> None:
    """Run the scoring API."""
    import uvicorn

    uvicorn.run("src.api.main:app", host=host, port=port, reload=reload, log_config=None)


@app.command(name="warm-cache")
def warm_cache_cmd() -> None:
    """Precompute applicant history features into the serving cache."""
    from src.api.warm_cache import warm, write_sample_payloads

    written, n_features, backend = warm()
    console.print(f"cached {written:,} applicants x {n_features} history features -> {backend}")
    console.print(f"wrote {write_sample_payloads()} sample payloads for the load test")


@app.command()
def sources() -> None:
    """Show which file each table will be read from."""
    from src.ingestion.loaders import describe_sources

    console.print(describe_sources())


if __name__ == "__main__":
    app()
