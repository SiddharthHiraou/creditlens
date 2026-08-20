# CreditLens

A credit decisioning and model risk platform for consumer lending. A lender
submits an application; CreditLens returns a probability of default, a mapped
credit score, an approve/decline/refer decision, ECOA-compliant reason codes, and
an LLM-drafted underwriter memo — with the monitoring, fairness testing and
governance artifacts a bank model validation team would ask for.

> **Build status: Phase 1 of 7 complete.** Data layer, target definition,
> out-of-time splitting and the logistic baseline are done, tested and
> reproducible. The API, frontend, GBDT champion, SHAP layer and MLOps flows are
> not built yet. Everything claimed below is reproducible today with two commands.

---

## Phase 1 headline numbers

Logistic regression on **application-level fields only** — no bureau or
repayment history. This is the floor Phase 2 has to beat, kept deliberately
honest rather than pre-tuned.

| Metric | Out-of-time test | Validation | Train |
|---|---|---|---|
| AUC | **0.7746** | 0.7723 | 0.7797 |
| Gini | **0.5493** | 0.5445 | 0.5594 |
| KS | **0.4081** | 0.4063 | 0.4099 |
| PR-AUC | 0.4367 | 0.4549 | 0.4229 |
| Brier | 0.2009 | 0.2101 | 0.1902 |
| Bad rate | 15.67% | 17.70% | 14.38% |
| n | 16,364 | 7,258 | 18,763 |

**Rank ordering across all 10 deciles: monotonic** — bad rate falls 49.2% → 2.0%
from riskiest to safest decile, a 3.14x lift in decile 1. A model that fails this
is not shippable regardless of AUC: a cutoff drawn near an inversion would admit
worse loans than the band below it.

The check is deliberately noise-aware. A bad rate per bin is a binomial
proportion, so at small n a two-loan swing between adjacent deciles flips strict
monotonicity on sampling noise alone. `is_rank_ordered` only counts an inversion
that exceeds two standard errors of the difference; `is_strictly_rank_ordered`
reports the raw version alongside it, and both are written to the metrics JSON.

Two things in that table are load-bearing:

- **Train→OOT AUC drops only 0.005.** The model is not overfit, and the
  out-of-time split is doing its job rather than accidentally reproducing train.
- **Brier is poor (0.20) and is supposed to be.** `class_weight="balanced"`
  deliberately distorts the probability scale to fix discrimination first.
  Expected-loss arithmetic needs a *real* probability, so Phase 2 calibrates with
  isotonic regression on the held-out calibration slice. Using these PDs for
  `EL = PD × LGD × EAD` today would be wrong.

The validation fold carries a higher bad rate (17.7%) than train (14.4%) because
the data contains a deliberate 2022 vintage deterioration — the drift that Phase 6's
PSI monitoring is built to catch.

Reproduce: `make data && make baseline` → writes `artifacts/baseline_metrics.json`.

## Quickstart

```bash
make setup && make data && make baseline
```

```bash
make test
```

46 tests, 80% coverage on `src/`.

## Getting the data

The pipeline runs today on a **synthetic generator** that emits the real Home
Credit table and column names, wired to a latent per-applicant risk factor so the
relational aggregations in Phase 2 carry genuine signal rather than noise.

Both Kaggle sources need an authenticated download, and Home Credit additionally
requires accepting the competition rules in a browser, so neither can be fetched
non-interactively:

```bash
pip install kaggle   # then place kaggle.json in ~/.kaggle/
kaggle competitions download -c home-credit-default-risk -p data/raw && unzip -o 'data/raw/*.zip' -d data/raw
```

Drop the CSVs in `data/raw/` and **the whole pipeline switches over with no code
change** — `src/ingestion/loaders.py` prefers real files and falls back to
synthetic. Check what will be read with `make validate` or:

```bash
.venv/bin/python -m src.cli sources
```

Synthetic scale today: 50,000 applicants across 7 tables, **9.6M rows total**.

## The target

**Bad = 90+ days past due within 12 months of origination.** Full rationale,
including the indeterminate band and the right-censoring rule, in
[`docs/target_definition.md`](docs/target_definition.md).

The definition is enforced in code, not just documented:

- 30–89 DPD is **indeterminate** — excluded from training, scored at evaluation.
- Loans whose 12-month window has not closed are **censored and dropped**, even
  if already 200 DPD. Keeping them would manufacture clean records in exactly the
  recent vintages the model is judged on.
- Splits are **out-of-time**, and `assert_no_temporal_leakage` fails the run if
  any fold's date range overlaps a later one.

### The Home Credit caveat, stated up front

**Home Credit has no application date.** It is a single snapshot with a prebuilt
binary `TARGET`, so a genuine out-of-time split on it is impossible. Rather than
invent a date, this project splits the roles: Home Credit supplies the
**relational feature engineering**, Lending Club (which has a real `issue_d`)
supplies **out-of-time evaluation, vintage analysis and PSI**, and the synthetic
generator carries a real origination calendar so the OOT machinery is exercisable
now. The real-data loader writes a **null** date sentinel, never a guess.

## Architecture (Phase 1 slice)

```
data/raw/*.csv ─┐
                ├─► loaders.py ─► Pandera schemas ─► target.py ─► splits.py ─► baseline
data/synthetic ─┘   (source        (7 contracts,     (90 DPD /     (out-of-      (logistic,
                     switch)        fail-closed)      12mo window)  time)         balanced)
```

## What is built

| Phase | Scope | Status |
|---|---|---|
| 1 | Data layer, Pandera contracts, target, OOT splits, logistic baseline | **done** |
| 2 | Polars feature pipeline, WOE/IV, LightGBM + Optuna, challengers, isotonic calibration, MLflow | not started |
| 3 | SHAP service, ECOA reason codes, counterfactuals, Fairlearn + mitigation tradeoff | not started |
| 4 | FastAPI, ONNX export, Redis, Postgres audit log, load test | not started |
| 5 | Next.js frontend, cutoff simulator | not started |
| 6 | Prefect flows, drift monitoring, promotion gate, memo generator, copilot | not started |
| 7 | Model card, validation report, credit policy, ADRs, demo | not started |

## Decision records

- [ADR 0001](docs/adr/0001-out-of-time-splitting.md) — out-of-time splitting, never random
- [ADR 0002](docs/adr/0002-two-datasets-with-separate-roles.md) — why Home Credit and Lending Club serve different roles

## Stack decisions so far

| Choice | Why |
|---|---|
| **Polars**, lazy | 9.6M rows across 7 relational tables; lazy scans keep the bureau_balance join affordable and nothing is read until collect |
| **No pandas** | sklearn consumes polars frames natively since 1.4, so the round-trip and the dependency are both unnecessary |
| **Pandera** at every boundary | Contracts, not documentation. A future-dated bureau line or an implausible age stops ingestion instead of surfacing later as unexplained AUC loss |
| **Logistic baseline first** | Regulators still trust scorecards, and Phase 2's GBDT lift has to be measured against an honest floor |
| **`class_weight`, never SMOTE** | Interpolating between real defaulters invents applicants who never applied and wrecks calibration |
| **Median impute + missingness indicator** | `EXT_SOURCE_1` is missing for ~56% of applicants and missingness is informative — thin-file applicants lack external scores |

## Repo layout

```
src/ingestion/    loaders, Pandera schemas, target rule, OOT splits, synthetic generator
src/models/       baseline scorecard + training entrypoint
src/evaluation/   AUC/Gini/KS/PR-AUC/Brier, decile rank-ordering
src/cli.py        creditlens data | validate | baseline | sources
tests/            46 tests: target rule, split leakage, metrics, loaders, end-to-end
docs/             target definition (binding), ADRs
notebooks/        01_eda.ipynb — exploratory only, never the source of truth
```

`features/ explainability/ fairness/ llm/ api/ monitoring/ flows/ frontend/ infra/`
are scaffolded and empty — they belong to later phases.
