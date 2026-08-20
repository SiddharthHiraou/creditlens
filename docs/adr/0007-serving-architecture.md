# ADR 0007: Serving architecture — ONNX, a server-side feature cache, and an append-only audit log

- **Status:** accepted
- **Date:** 2026-08-20
- **Phase:** 4

## Context

The scoring path has a 150ms p99 budget and three hard constraints that pull
against it: 221 features must be assembled, an exact SHAP attribution must be
produced for adverse action, and every decision must be reconstructible months
later.

## Decisions

### Predictions run through ONNX Runtime

Measured at batch size 1, which is the serving case:

| | mean | p50 | p99 |
|---|---|---|---|
| Native CatBoost `predict` | 0.194ms | 0.176ms | 0.472ms |
| ONNX Runtime | 0.103ms | 0.095ms | **0.169ms** |

Agreement with the native model is 3.1e-06 — float32 rounding, nothing more.
Sessions are single-threaded: concurrency comes from the worker pool, and
intra-op threads there just add contention and tail latency.

The calibrator stays in Python. `SmoothedIsotonic` is a handful of numpy
operations over 29 blocks, cheaper to apply directly than to express as a graph,
and the wrapper keeps them together so an uncalibrated PD cannot be served by
accident.

### History features are looked up server-side, not supplied by the caller

161 of the 221 features come from bureau and repayment history. Two reasons they
are cached rather than sent:

- **Latency.** Computing them means scanning 9.6M child rows.
- **Trust.** The client does not have the applicant's bureau record and must not
  be able to assert one. Accepting 161 features over the wire would let a caller
  declare their own credit history.

A cache **miss is not an error**. It means a thin-file applicant, which is a real
segment; those features go to the model as nulls exactly as in training, and the
response reports `history_found: false`.

Redis in deployment; an in-process dict when no `redis_url` is set, so the API
runs with nothing else installed. The fallback names itself in `/ready` — a
silent downgrade would be worse than no cache.

### SHAP is computed only when it is needed

SHAP is **~92% of a request** (6.63ms against 0.08ms for the prediction). The
decision is computed first, and the explanation runs only when the applicant was
not approved, because an approval needs no adverse action reasons:

| `explain` | p50 wall | p99 wall |
|---|---|---|
| `auto` (default) | 3.51ms | 10.94ms |
| `always` | 9.27ms | 12.06ms |
| `never` | 2.53ms | 3.29ms |

On a 60%-approval book, `auto` skips the expensive pass on most requests.

### Overrides append; they never mutate

An underwriter override is a new row, not an edit. The model's decision is the
evidence a validation team needs — what the model said, what the human said, and
why. Overwriting it destroys exactly what makes overrides reviewable.
Justification is mandatory and length-checked; an override with no reason is
unreviewable.

### The audit row carries the whole feature vector

A reason code cannot be re-derived from a probability. Each row stores the
inputs as scored, the reason codes as issued, the model version and the feature
spec fingerprint, so a decision reconstructs against the model that actually
made it even after the champion is retrained.

## Consequences

**The serving image still needs CatBoost, and the original rationale was wrong.**
The intent was an image with no training stack at all. TreeSHAP needs the tree
structure and an ONNX graph does not expose it, so the explainer needs the
booster even though scoring does not. LightGBM, XGBoost, Optuna, MLflow, Pandera
and Fairlearn are all absent, which is still a real reduction, but the honest
statement is "no training stack **except** the champion's own library".
`CREDITLENS_ENABLE_SHAP=false` gives a decision-only tier that needs none of it.

**The rate limiter is in-process.** With N workers the effective ceiling is N
times the configured limit. Correct for a demo, wrong for production, where the
counter belongs in Redis. Stated here rather than left to be discovered.

**SQLite does not survive concurrency.** Four workers writing to one SQLite file
serialize on its writer lock, and adding workers bought almost nothing (p99 74ms
vs 85ms at 16 users). With Postgres the same test gives p99 46ms. SQLite is the
zero-setup dev fallback; Postgres is the deployment path.
