# ADR 0001: Out-of-time splitting, never random

- **Status:** accepted
- **Date:** 2026-08-20
- **Phase:** 1

## Context

The modelling population is a book of loans with origination dates spanning
several years. The default convention in tutorials — `train_test_split(shuffle=True)` —
is available and simpler.

## Decision

All splits partition on origination date. `train < calibration < valid < test`,
with no date appearing in two folds. `assert_no_temporal_leakage` runs on every
training invocation and raises rather than warns.

The calibration slice is carved from the **tail** of train, not sampled at random,
so isotonic calibration is fitted on the vintages nearest to validation.

## Consequences

**Accepted costs.** Fold sizes are dictated by origination volume rather than
chosen, so the calibration fold is smaller than an even split would give. Reported
AUC is lower than a random split would produce — this is the point, not a defect.

**Rejected alternative: random split.** It leaks in two ways. Repeat applicants
appear in multiple folds, so the model memorises borrowers. Macro conditions and
the lender's own policy changes are shared across folds, so validation flatters
performance on next quarter's through-the-door population.

**Consequence for Home Credit.** It has no application date, so it cannot be split
this way at all. See ADR 0002.
