# ADR 0002: Home Credit and Lending Club serve different roles

- **Status:** accepted
- **Date:** 2026-08-20
- **Phase:** 1

## Context

Home Credit Default Risk has the relational depth this project needs — six child
tables of bureau, installment, credit-card and POS history — and that aggregation
work is the actual skill being demonstrated. But **it contains no application
date**. It is a single snapshot with a prebuilt binary `TARGET`.

ADR 0001 requires out-of-time splitting. Home Credit cannot satisfy it.

## Decision

Split the roles rather than compromise either.

| Dataset | Role |
|---|---|
| Home Credit | Relational feature engineering; the 150–300 engineered features |
| Lending Club | Out-of-time evaluation, vintage analysis, PSI, reject inference (it has a real `issue_d`) |
| Synthetic generator | Exercises the full pipeline before either download lands; emits a real origination calendar |

The Home Credit loader writes a **null** `origination_date` sentinel. It never
imputes, estimates, or derives a date from row order.

## Consequences

**Accepted cost.** Features engineered on Home Credit are not directly transferable
to the Lending Club schema; the feature module needs a source-aware layer in Phase 2.

**Rejected alternative: derive a pseudo-date from `SK_ID_CURR` order.** IDs are
only loosely time-ordered and the relationship is undocumented. A fabricated date
would produce a split that *looks* out-of-time and silently is not — strictly worse
than no split, because it would be believed.

**Rejected alternative: use Home Credit alone with a random split.** Fails ADR 0001
and forfeits the reject-inference and vintage work entirely.

**Benefit.** "Home Credit has no application date, so out-of-time evaluation runs
on Lending Club" is a strong answer to a question most portfolio projects get wrong.
