# CreditLens

A credit decisioning and model risk platform for consumer lending. A lender
submits an application; CreditLens returns a probability of default, a mapped
credit score, an approve/decline/refer decision, ECOA-compliant reason codes, and
an LLM-drafted underwriter memo — with the monitoring, fairness testing and
governance artifacts a bank model validation team would ask for.

> **Build status: Phases 1-4 of 7 complete.** Data layer, target definition,
> out-of-time splitting, a 221-feature Polars pipeline, WOE/IV, four model
> tracks with Optuna and monotonic constraints, smoothed isotonic calibration,
> PSI/CSI, an MLflow registry, a SHAP service, ECOA reason codes,
> counterfactual levers, a Fairlearn audit and a FastAPI scoring service with
> ONNX serving, a Redis feature cache and a Postgres audit log are done, tested
> and reproducible. The frontend and MLOps flows are not built yet. Everything
> claimed below is reproducible today.

---

## Headline numbers

Out-of-time test fold, 16,276 applications, 16.81% bad rate.

| | Baseline (Phase 1) | Champion (Phase 2) |
|---|---|---|
| Model | Logistic, application fields only | CatBoost, calibrated |
| AUC | 0.7624 | **0.7913** |
| Gini | 0.5249 | **0.5827** |
| KS | 0.3818 | **0.4349** |
| Brier | 0.2066 | **0.1148** |
| Score PSI (train vs OOT) | — | 0.0174 (stable) |

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

**Calibration now costs no AUC at all** (0.7913, matching the raw model). Plain
isotonic is a step function: it collapsed 16,091 distinct raw scores into 120
values and cost 0.0013 AUC in pure ties. Phase 3 replaced it with a smoothed
variant that keeps the level and restores the ranking — see
[ADR 0006](docs/adr/0006-smoothed-isotonic-calibration.md).

### Calibration is the difference between a ranking and a probability

| | Raw | Calibrated |
|---|---|---|
| Brier | 0.19590 | **0.11484** |
| Expected calibration error | 0.26451 | **0.01282** |
| Mean predicted PD | 0.4326 | **0.1753** |
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

210 tests, 77% coverage on `src/`.

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
| 3 | SHAP service, ECOA reason codes, counterfactuals, Fairlearn + mitigation tradeoff | **done** |
| 4 | FastAPI, ONNX export, Redis, Postgres audit log, load test | **done** |
| 5 | Next.js frontend, cutoff simulator | not started |
| 6 | Prefect flows, drift monitoring, promotion gate, memo generator, copilot | not started |
| 7 | Model card, validation report, credit policy, ADRs, demo | not started |

## Explainability and adverse action

`make audit` produces the full report; `make explain` prints one applicant.

**The Phase 3 deliverable: 100% of declines produce exactly four distinct,
ranked, human-readable reasons.** A real one, straight from the CLI:

```
applicant 100003  PD 0.2979  score 512  -> decline  (approve at 539)
  1. [Debt-to-income] Proposed monthly payment is high relative to stated income
  2. [Repayment record] Record of late or partial payments on prior obligations
  3. [Delinquency on credit file] Credit file shows past-due amounts or accounts reported delinquent
  4. [Prior application history] Record of recent declined or withdrawn credit applications
```

Three properties make that compliant rather than decorative:

**Distinct, not restated.** The top four raw SHAP contributions are usually four
measurements of one thing. Contributions are summed into 11 ECOA families and
ranked, so four reasons means four different reasons. Verified across every
decline: mean 4.00 distinct families, minimum 4.

**Blind to protected attributes.** The model's single strongest feature is
`EXT_mean_x_age`, which embeds age. It is disclosed as a credit-bureau-score
reason, never as an age reason, and a suppression list keeps age, sex, family
status and dependents out of any disclosure. See
[ADR 0005](docs/adr/0005-reason-codes-and-protected-attributes.md).

**No silent gaps.** All 221 features are either mapped to a family (212) or
explicitly suppressed (9). A test fails the build if that ever stops being true —
an unmapped feature would be silently dropped from a legally required disclosure.

SHAP wiring is checked by additivity: `expected_value + sum(shap)` must equal the
raw margin, and it does to **6.7e-15**. That check earned its place twice —
it caught a stale cached base value (SHAP mutates `expected_value` after the
first call) and a dispatch bug where `LGBMClassifier.predict` silently swallowed
a CatBoost keyword and returned class labels as "margin".

### Counterfactuals: what would have to change

Proposals are expressed as **levers** — one real-world action propagating to
every model feature it would actually move. Feature selection keeps *ratios*
rather than raw amounts, so a naive per-feature sweep would propose an incoherent
applicant who borrows less but repays the same.

| Lever | Moves |
|---|---|
| Request a smaller loan | every amount-derived ratio together, plus residual income |
| Pay down revolving balances | all utilisation and balance measures |
| Bring past-due balances current | every overdue measure |
| Repay existing debt | bureau debt levels and debt-to-income |

Past conduct, external scores, employment tenure and every protected attribute
are **excluded from the search space entirely** — you cannot tell someone to have
had fewer delinquencies last year.

**The honest result: most declines cannot be reversed by any feasible action.**
Of 120 declines, 0 flip on a single action and 1 flips on up to three. Median
achievable gain is 11.7 score points against gaps that are usually larger.

That is not a broken module — it is what the data says, and the segmentation
confirms the levers target the right things:

| Decline driven by | n | Median score gain | Max |
|---|---|---|---|
| Actionable causes (affordability, utilisation, debt) | 19 | **+22.3** | +46.3 |
| Credit history | 101 | +10.3 | +42.3 |

Applicants declined for reasons they can act on get twice the benefit. For
everyone else the module says so plainly rather than manufacturing advice.

## Fairness

Full analysis in [docs/fairness_findings.md](docs/fairness_findings.md). Every
metric is computed twice — once in this repo, once through Fairlearn — and the
two agree to **0.00e+00**.

| Attribute | Disparate impact | Four-fifths rule | Equal-opportunity gap |
|---|---|---|---|
| Gender | 0.9782 | passes | 0.0170 |
| Age band | **0.3843** | **fails** | 0.4921 |

**The model fails the four-fifths rule on age**, approving the 18-24 band at 38%
of the rate of the 65+ band. Stated plainly because a fairness section that only
reports passes is worthless.

Context that belongs alongside it, not instead of it: observed bad rate falls
monotonically from 26.4% (18-24) to 7.1% (65+), so the model is measuring a real
difference rather than inventing one — which explains the gap without justifying
it, since disparate impact is assessed on effect. The model is well calibrated
*within* every band (largest gap +2.3 points), so it is not systematically wrong
about any group. And the disparity runs toward younger applicants, so ECOA's
specific protection for applicants 62 and over is not engaged.

### What mitigation costs

| Approval rate (single group-blind cutoff) | Disparate impact | Bad rate among approved |
|---|---|---|
| 40% | 0.2273 | 4.55% |
| 60% | 0.3843 | 6.73% |
| 90% | 0.7650 | 12.64% |

Most of the disparity is a property of **where the cutoff sits**, not of the
ranking — and no legally deployable cutoff reaches 0.80.

Group-specific thresholds do reach parity (DI 0.995) but are **unlawful to
deploy**: a different cutoff by protected class is disparate treatment even when
intended to reduce disparate impact. They are run to establish the frontier
only — and they get there by approving 96% of applicants, more than doubling the
bad rate among approved from 6.7% to 14.9%. A disparity number read without that
column would be badly misleading.

**No claim is made that bias was removed.** It was measured, the tradeoff was
quantified, and the decision is documented — which is what a model risk team
actually does.

## The API

```bash
make train && make warm-cache && make serve   # http://localhost:8000/docs
```

or the whole stack — API, Postgres, Redis, MLflow — in containers:

```bash
make up
```

```bash
curl -s -X POST localhost:8000/v1/score \
  -H "X-API-Key: demo-key-underwriter" -H "Content-Type: application/json" \
  -d @artifacts/one_payload.json
```

```
pd 0.2979 | score 511.9 | decision decline | latency_ms 9.4
reasons: Debt-to-income, Repayment record, Delinquency on credit file, Prior application history
```

| Endpoint | Purpose |
|---|---|
| `POST /v1/score` | Single application → PD, score, decision, reason codes, SHAP |
| `POST /v1/score/batch` | Up to 1000 applications, async with a job id |
| `GET /v1/score/batch/{id}` | Batch status and results |
| `GET /v1/decisions/{id}` | Full audit record: inputs, features, model version, overrides |
| `POST /v1/decisions/{id}/override` | Underwriter override with mandatory justification |
| `GET /v1/model/metadata` | Active version, feature list, performance snapshot |
| `GET /v1/monitoring/drift` | PSI of served decisions against the training baseline |
| `POST /v1/simulate/cutoff` | Approval rate, bad rate, expected loss, profit at a cutoff |
| `GET /v1/simulate/cutoff/curve` | Profit across the cutoff range, and the maximum |
| `GET /health` `GET /ready` | Liveness and readiness |

API key auth, per-key rate limiting, structured JSON logs with an `X-Request-ID`
on every response, and OpenAPI docs at `/docs`.

### Latency

ONNX Runtime at batch size 1, against native CatBoost `predict`:

| | mean | p50 | p99 |
|---|---|---|---|
| Native CatBoost | 0.194ms | 0.176ms | 0.472ms |
| **ONNX Runtime** | 0.103ms | 0.095ms | **0.169ms** |

Agreement is 3.1e-06 — float32 rounding.

End-to-end, measured with **Locust against a live uvicorn server** (4 workers,
Postgres, Redis), not `TestClient`:

| Concurrent users | p50 | p95 | p99 | req/s |
|---|---|---|---|---|
| 16 | 13ms | 29ms | **46ms** | ~265 |
| 24 | 27ms | 79ms | **110ms** | ~286 |
| 32 | 33ms | 130ms | 170ms | ~330 |

**p99 stays under the 150ms target to ~24 concurrent users on one 4-worker
machine.** Past that the box saturates — and the honest caveat is that Locust
runs on the same 8-core laptop, so above ~24 users the load generator is
competing with the service for cores and the number stops being about the API.

The dominant cost is SHAP, not the model: **6.63ms of a 7.20ms request**, against
0.08ms for the prediction itself. Since an approval needs no adverse action
reasons, the decision is computed first and the explanation runs only when it is
needed:

| `explain` | p50 wall | p99 wall |
|---|---|---|
| `auto` (default) | 3.51ms | 10.94ms |
| `always` | 9.27ms | 12.06ms |
| `never` | 2.53ms | 3.29ms |

### Three things worth knowing before trusting these numbers

**SQLite does not survive concurrency, and it showed.** With four workers
writing to one SQLite file, adding workers bought almost nothing — p99 74ms
against 85ms on a single worker, because they serialize on the writer lock.
Swapping in Postgres took the same test to 46ms. SQLite is the zero-setup dev
fallback; Postgres is the deployment path.

**The rate limiter is in-process**, so with N workers the effective ceiling is N
times the configured limit. Fine for a demo, wrong for production, where the
counter belongs in Redis.

**Throttled requests are counted as failures in the load test, deliberately.** A
429 returns in about a millisecond, so folding them into the success bucket makes
a saturated service look fast — an early run reported a 3ms p50 that was almost
entirely the rate limiter rejecting traffic.

### What the caller does and does not send

The request carries application-level fields only. The 161 relational history
features are looked up server-side by `sk_id_curr` from Redis — the client does
not have the applicant's bureau record and must not be able to assert one. A
cache miss is not an error: it means a thin-file applicant, scored with nulls
exactly as in training, and reported as `history_found: false`.

## Decision records

- [ADR 0001](docs/adr/0001-out-of-time-splitting.md) — out-of-time splitting, never random
- [ADR 0002](docs/adr/0002-two-datasets-with-separate-roles.md) — why Home Credit and Lending Club serve different roles
- [ADR 0003](docs/adr/0003-monotonic-constraints.md) — monotonic constraints, and why LightGBM's `basic` method over `advanced`
- [ADR 0004](docs/adr/0004-two-factor-synthetic-generator.md) — why the generator needed two latent factors
- [ADR 0005](docs/adr/0005-reason-codes-and-protected-attributes.md) — grouped, deduplicated reason codes blind to protected attributes
- [ADR 0006](docs/adr/0006-smoothed-isotonic-calibration.md) — why plain isotonic is unusable as a decisioning score
- [ADR 0007](docs/adr/0007-serving-architecture.md) — ONNX serving, server-side feature cache, append-only audit log

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
src/explainability/  TreeSHAP service, ECOA reason-code mapper + versioned YAML,
                  counterfactual levers
src/fairness/     group metrics, four-fifths rule, Fairlearn cross-check,
                  mitigation tradeoff curves
src/api/          FastAPI app: schemas, auth, rate limiting, structured logs,
                  ONNX scorer, Redis feature cache, Postgres audit log, routers
src/cli.py        creditlens data | validate | features | baseline | train |
                  audit | explain | warm-cache | serve | sources
tests/            210 tests across target, splits, metrics, loaders, WOE, PSI,
                  features, calibration, scorecard, selection, decision, reason
                  codes, SHAP, counterfactuals, fairness, mitigation, ensemble,
                  tuning, API contracts, and four end-to-end suites
docs/             target definition (binding), fairness findings, 6 ADRs
notebooks/        01_eda.ipynb — exploratory only, never the source of truth
artifacts/        metrics JSON, champion model, MLflow store (gitignored)
```

```
infra/            Dockerfile, docker-compose (API + Postgres + Redis + MLflow)
loadtest/         Locust scenarios
```

`llm/ monitoring/ flows/ frontend/` are scaffolded and empty — they belong to
later phases.

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
