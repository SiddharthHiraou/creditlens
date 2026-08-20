# CreditLens

A credit decisioning and model risk platform for consumer lending. A lender
submits an application; CreditLens returns a probability of default, a mapped
credit score, an approve/decline/refer decision, ECOA-compliant reason codes, and
an LLM-drafted underwriter memo — with the monitoring, fairness testing and
governance artifacts a bank model validation team would ask for.

> **Build status: Phases 1-2 of 7 complete.** Data layer, target definition,
> out-of-time splitting, a 221-feature Polars pipeline, WOE/IV, four model
> tracks with Optuna and monotonic constraints, isotonic calibration, PSI/CSI
> and an MLflow registry are done, tested and reproducible. The SHAP layer,
> API, frontend and MLOps flows are not built yet. Everything claimed below is
> reproducible today with three commands.

---

## Headline numbers

Out-of-time test fold, 16,276 applications, 16.81% bad rate.

| | Baseline (Phase 1) | Champion (Phase 2) |
|---|---|---|
| Model | Logistic, application fields only | CatBoost, calibrated |
| AUC | 0.7624 | **0.7900** |
| Gini | 0.5249 | **0.5801** |
| KS | 0.3818 | **0.4369** |
| PR-AUC | 0.4360 | 0.4448 |
| Brier | 0.2066 | **0.1150** |
| Score PSI (train vs OOT) | — | 0.0175 (stable) |

**Rank ordering: monotonic across all 10 deciles**, 53.9% → 1.4% bad rate,
3.20x lift in decile 1. A model that fails this is not shippable regardless of
AUC: a cutoff drawn near an inversion would admit worse loans than the band
below it.

The check is noise-aware. A bad rate per bin is a binomial proportion, so at
small n a two-loan swing between adjacent deciles flips strict monotonicity on
sampling noise. `is_rank_ordered` only counts inversions exceeding two standard
errors of the difference; `is_strictly_rank_ordered` reports the raw version
alongside it, and both are written to the metrics JSON.

### All four tracks, compared honestly

| Track | Valid AUC | OOT AUC | OOT Gini |
|---|---|---|---|
| Logistic scorecard (WOE + PDO) | 0.7779 | 0.7869 | 0.5738 |
| LightGBM (Optuna, 100 trials) | 0.7809 | 0.7898 | 0.5807 |
| XGBoost | 0.7823 | 0.7908 | 0.5817 |
| **CatBoost — champion** | **0.7857** | 0.7913 | 0.5826 |
| Stacked ensemble (ceiling reference) | 0.7854 | 0.7921 | 0.5841 |

Three things in that table are worth stating plainly rather than burying:

**The scorecard is within 0.0044 OOT AUC of the tuned GBDT champion.** On this
data the WOE logistic gets ~99.4% of the discrimination for a fraction of the
explainability cost. That is the real conversation a credit shop has, and the
honest answer here is that the GBDT's margin is thin. It wins, but not by
enough to make the scorecard track ceremonial.

**The stack beats the champion by 0.0008.** The ceiling is essentially reached;
further tuning of the individual models is not where remaining lift lives. The
stack is excluded from champion selection anyway — it is fitted on validation
predictions, so its validation score is optimistic by construction, and serving
it requires all three bases.

**Calibrated AUC (0.7900) is slightly below raw CatBoost (0.7913).** Isotonic
regression is a step function, so it introduces ties, and ties cost a little
AUC. That trade is correct: the ranking loses 0.0013 and the probability
becomes usable.

### Calibration is the difference between a ranking and a probability

| | Raw | Calibrated |
|---|---|---|
| Brier | 0.19590 | **0.11501** |
| Expected calibration error | 0.26451 | **0.01279** |
| Mean predicted PD | 0.4326 | **0.1731** |
| Actual bad rate | 0.1681 | 0.1681 |

`scale_pos_weight` pushed mean predicted PD to 43% against a true bad rate of
17%. The ranking was fine; the number was meaningless. Expected-loss
arithmetic (`EL = PD × LGD × EAD`) on the raw score would have been wrong by a
factor of 2.5 while every discrimination metric looked healthy. Isotonic
regression, fitted on the held-out calibration fold, closes it.

---

## Quickstart

```bash
make setup && make data && make train
```

`make train` runs the full Phase 2 pipeline: 221 features, selection, four
model tracks, 100 Optuna trials, calibration, PSI, and MLflow registration.
Roughly 6 minutes. For the Phase 1 baseline alone, `make baseline`.

```bash
make test
```

104 tests, 72% coverage on `src/`.

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
| 2 | Polars feature pipeline, WOE/IV, LightGBM + Optuna, challengers, isotonic calibration, MLflow | **done** |
| 3 | SHAP service, ECOA reason codes, counterfactuals, Fairlearn + mitigation tradeoff | not started |
| 4 | FastAPI, ONNX export, Redis, Postgres audit log, load test | not started |
| 5 | Next.js frontend, cutoff simulator | not started |
| 6 | Prefect flows, drift monitoring, promotion gate, memo generator, copilot | not started |
| 7 | Model card, validation report, credit policy, ADRs, demo | not started |

## Decision records

- [ADR 0001](docs/adr/0001-out-of-time-splitting.md) — out-of-time splitting, never random
- [ADR 0002](docs/adr/0002-two-datasets-with-separate-roles.md) — why Home Credit and Lending Club serve different roles
- [ADR 0003](docs/adr/0003-monotonic-constraints.md) — monotonic constraints, and why LightGBM's `basic` method over `advanced`
- [ADR 0004](docs/adr/0004-two-factor-synthetic-generator.md) — why the generator needed two latent factors

## Stack decisions so far

| Choice | Why |
|---|---|
| **Polars**, lazy | 9.6M rows across 7 relational tables; lazy scans keep the bureau_balance join affordable and nothing is read until collect |
| **No pandas** | sklearn consumes polars frames natively since 1.4, so the round-trip and the dependency are both unnecessary |
| **Pandera** at every boundary | Contracts, not documentation. A future-dated bureau line or an implausible age stops ingestion instead of surfacing later as unexplained AUC loss |
| **Logistic baseline first** | Regulators still trust scorecards, and Phase 2's GBDT lift has to be measured against an honest floor |
| **`class_weight`, never SMOTE** | Interpolating between real defaulters invents applicants who never applied and wrecks calibration |
| **Median impute + missingness indicator** | `EXT_SOURCE_1` is missing for ~56% of applicants and missingness is informative — thin-file applicants lack external scores |
| **Optuna TPE + median pruning** | Bayesian search, not a grid. 100 trials with 30 pruned early at intermediate boosting rounds |
| **Monotonic constraints on 47 features** | A model whose PD falls as debt burden rises is indefensible in validation. LightGBM's `basic` method, because `advanced` admitted a real violation in testing — see ADR 0003 |
| **Isotonic, not Platt** | The distortion introduced by class reweighting is not sigmoidal, so a parametric correction cannot fit it |
| **Selection fitted on train only** | Even computing a correlation matrix over the full frame leaks out-of-time information into the choice of features |
| **Versioned feature spec** | `data/feature_spec.yaml` pins the feature list, order and dtypes with a fingerprint. Column *order* matters to a numpy-backed model, so the hash covers it |

## Repo layout

```
src/ingestion/    loaders, Pandera schemas, target rule, OOT splits, synthetic generator
src/features/     221-feature pipeline: relational aggregations, ratios, trends,
                  WOE/IV, selection, monotonic directions, versioned spec
src/models/       baseline, WOE scorecard, GBDT champion + challengers, Optuna,
                  isotonic calibration, stacked ensemble, training entrypoint
src/evaluation/   AUC/Gini/KS/PR-AUC/Brier, decile rank-ordering, PSI/CSI
src/cli.py        creditlens data | validate | features | baseline | train | sources
tests/            104 tests across target, splits, metrics, loaders, WOE, PSI,
                  features, calibration, scorecard, selection, two end-to-end suites
docs/             target definition (binding), 4 ADRs
notebooks/        01_eda.ipynb — exploratory only, never the source of truth
artifacts/        metrics JSON, champion model, MLflow store (gitignored)
```

`explainability/ fairness/ llm/ api/ monitoring/ flows/ frontend/ infra/`
are scaffolded and empty — they belong to later phases.

## The feature pipeline

221 features built from 9.6M rows across 7 relational tables, in 0.8s, lazily.

| Family | Count | Examples |
|---|---|---|
| Bureau aggregations | 38 | recency-weighted debt, active-only overdue, debt-to-credit, product mix |
| Monthly bureau status | 18 | trailing 3/6/12m delinquency counts and shares, status slope |
| Prior applications | 24 | refusal rate, granted-vs-requested ratio, last decision refused |
| Installment behaviour | 34 | late share by window, payment shortfall, days-late slope |
| Revolving utilisation | 32 | utilisation level, trajectory slope, months over limit |
| POS servicing | 15 | DPD depth and share, completion rate |
| Application ratios | 11 | DTI, LTV, residual income per household member, implied term |
| Stability | 8 | employment tenure, share of adult life employed, age bands |
| External scores | 11 | mean/min/max/spread, missingness pattern, interactions |
| Cross-source | 5 | new credit vs existing bureau debt, total debt to income |
| Absent-history flags | 5 | thin-file indicators per source |

Selection cuts 221 → 72 on training data only: IV floor at 0.02 (68 dropped),
correlation pruning above 0.95 (7 dropped), and null importance against a
shuffled target (70 dropped). Every drop carries a recorded reason in the spec.

**Two rules the join layer enforces.** Left joins always, so thin-file
applicants survive with nulls instead of being silently deleted by an inner
join. And history *counts* fill to zero while *ratios and slopes* stay null —
the average utilisation of zero credit cards is undefined, and filling it with
0 would place thin-file applicants at the safest end of a scale they are not on.

### A caveat on the synthetic IV values

Ten features land in the IV > 0.5 "suspicious" band, all of them bureau-balance
delinquency measures. On real data that band means *investigate for leakage*.
Here it is an artifact of the generator: the label and the delinquency history
share a latent factor by construction. Do not read those IVs as a result.
