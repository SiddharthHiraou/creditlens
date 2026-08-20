# ADR 0009: Promotion is gated automatically, and the gate has no override

- **Status:** accepted
- **Date:** 2026-08-20
- **Phase:** 6

## Context

Retraining is easy. Deciding whether the result is safe to serve is the hard
part, and it is the part that fails under deadline pressure — a 0.8% AUC
regression looks like noise at 5pm on a Friday.

Four Prefect flows carry the lifecycle: `ingest_and_validate` and
`compute_drift` daily, `retrain_candidate` monthly or on a drift alert, and
`validate_and_promote` as the gate.

## Decision

**Five checks, all of which must pass.** The thresholds come from
`docs/credit_policy.md` §8.5 and changing one is a Credit Committee decision,
not a tuning exercise:

| Check | Threshold | Why |
|---|---|---|
| Discrimination | AUC within 1% of the incumbent | Marginally worse is tolerable; materially worse is not |
| Calibration | ECE not worsening by more than 0.005 | Expected loss and pricing both depend on it |
| Stability | Score PSI below 0.25 | A candidate that disagrees with its own training population is not ready |
| Rank ordering | Monotonic across deciles | Unshippable otherwise, regardless of AUC |
| Fairness | Disparate impact not falling by more than 0.05 | Stops an existing problem getting worse |

**There is no promote-with-a-warning path.** A failing gate writes an issue and
leaves the incumbent serving. The asymmetry is deliberate: the cost of running
the incumbent for another month is bounded and known, and the cost of promoting
a quietly worse model is neither.

**Drift alerts and schedules enter through the same door.** `compute_drift`
writes `drift_alert.json` above PSI 0.25; `drift_triggered_retrain` reads that
file and consumes it. A drift-triggered retrain and a scheduled one take an
identical path, so there is one code path to trust rather than two.

**The gate compares two artifacts, never a model with itself.** The training run
overwrites `champion_model.joblib` in place, so `retrain_candidate` copies the
result aside into `artifacts/candidates/` first. Without that the gate compares
the new model to the new model and passes every check trivially.

## Consequences

**Verified in both directions.** A candidate identical to the incumbent passes
all five. A genuinely crippled candidate — six decision stumps on the same
feature spec, the kind of artifact a bad config or a truncated run produces —
was refused on discrimination (0.7913 → 0.7339) and fairness (0.3843 → 0.3144),
with an issue written and the incumbent left serving. A gate only ever tested
against a good candidate is a gate nobody knows works.

**The fairness check gates degradation, not compliance.** The incumbent already
fails the four-fifths rule on age. The gate stops that getting worse; it is not
a substitute for the fair-lending review that the failure itself requires, and
the issue text says so explicitly so nobody reads a passing gate as a clean bill.

**Issues are files, not GitHub API calls.** These flows run unattended, and a
flow that needs network credentials to report a failure has a second way to fail.
CI can post the file; the record exists either way.

**Registry promotion is best-effort.** An unreachable MLflow must not turn a
passing gate into an exception, but the failure is logged rather than swallowed.
