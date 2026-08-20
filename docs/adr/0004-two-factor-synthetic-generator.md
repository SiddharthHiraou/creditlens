# ADR 0004: The synthetic generator needs two latent factors, not one

- **Status:** accepted
- **Date:** 2026-08-20
- **Phase:** 2 (revises a Phase 1 decision)

## Context

Phase 1's generator drove the application attributes, the external bureau
scores, the bureau history and the repayment conduct from a **single** latent
risk factor per applicant.

When Phase 2's feature pipeline was measured against it, the result was flat:

| Feature set | OOT AUC |
|---|---|
| Application fields only | 0.7702 |
| Relational history only | 0.7625 |
| Both | 0.7800 |

The relational aggregations carried signal on their own but added almost
nothing on top of the application fields. That is the signature of redundancy,
not of weak features: both sets were noisy measurements of the same hidden
variable, so the second one had nothing left to say.

## Decision

Give each applicant **two** correlated latent factors:

- `credit_risk` — drives application attributes and `EXT_SOURCE_*`
- `behavioural_risk` — drives repayment conduct: monthly bureau delinquency,
  late and short installments, revolving utilisation creep

Loading between them is 0.35. Both contribute to the default outcome.

## Consequences

**The measurement now behaves like real data:**

| Feature set | OOT AUC (one factor) | OOT AUC (two factors) |
|---|---|---|
| Application only | 0.7702 | 0.7579 |
| Relational only | 0.7625 | 0.7503 |
| Both | 0.7800 | **0.7884** |

Incremental lift from relational history went from +0.010 to +0.030, which is
the range real Home Credit models show.

**This invalidated Phase 1's reported numbers**, which were regenerated. The
baseline moved from 0.7746 to 0.7624 OOT AUC. That is a property of the data
generator changing, not of the baseline model changing.

**Why this is a fix and not tuning to taste.** Real lenders build repayment
aggregations precisely because conduct carries information a stale external
bureau score does not. A generator in which history is a pure restatement of
the application is not a simplification of reality — it is a
misrepresentation of it, and it made the entire Phase 2 deliverable
unmeasurable. The two-factor structure is the more faithful model.

**This applies only to synthetic data.** Once the real Kaggle CSVs land in
`data/raw/`, the generator is bypassed entirely and none of this is in play.
