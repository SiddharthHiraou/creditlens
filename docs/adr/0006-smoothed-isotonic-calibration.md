# ADR 0006: Isotonic calibration is smoothed before it is used as a score

- **Status:** accepted
- **Date:** 2026-08-20
- **Phase:** 3 (revises a Phase 2 decision)

## Context

Phase 2 calibrated the champion with plain isotonic regression, which fixed the
probability level convincingly: Brier 0.196 → 0.115, ECE 0.265 → 0.013, mean
predicted PD 0.433 → 0.173 against an actual bad rate of 0.168.

Phase 3 then tried to build decisions and counterfactuals on that output and
hit a wall. Isotonic regression is a **step function**. On this champion it
collapsed **16,091 distinct raw scores into 120 calibrated values** across 29
knots, with a single value shared by **1,800 applicants** — 11% of the book
assigned an identical PD.

Three things break as a result:

- **Cutoffs become unusable.** Nudging the approval threshold jumps whole blocks
  of applicants at once, so a target approval rate cannot be hit and a "refer"
  band cannot be placed precisely.
- **Counterfactuals stop working.** A feasible change moves the raw score but
  not the calibrated one, so the search has no surface to climb. Every proposal
  came back as "no change".
- **Ranking is lost.** Ties are pure discrimination loss: OOT AUC fell from
  0.7913 to 0.7900.

## Decision

Keep the isotonic fit and add `SmoothedIsotonic`, which spreads each flat block
across the gap to its neighbours, positioning each point by where its **raw**
score sits inside the block. Bands reach halfway to the adjacent blocks, so they
are ordered and non-overlapping and monotonicity is preserved; the block centre
is unchanged, so the calibrated level is retained.

Separately, the counterfactual search runs against the **uncalibrated**
(continuous) output, with the target expressed as the raw threshold whose
calibrated score reaches the cutoff. Isotonic is monotone, so
`calibrated_score >= target` is exactly `raw_pd <= raw_threshold`.

## Consequences

Measured on the out-of-time fold:

| | Plain isotonic | Smoothed |
|---|---|---|
| Distinct calibrated PDs | 120 | **15,663** |
| OOT AUC | 0.7900 | **0.7913** |
| Brier | 0.11501 | **0.11484** |
| ECE | 0.01279 | 0.01282 |
| Mean PD (actual 0.1681) | 0.1731 | 0.1753 |

The full raw ranking is recovered — zero tie loss — Brier improves slightly, and
ECE is unchanged to four decimal places. There is no tradeoff here to argue
about; the smoothing is a strict improvement.

**What this does not claim.** The 29 knots are what a 4,724-row calibration fold
supports. Smoothing recovers resolution that isotonic had to pool away; it does
not add information about the *level*, and it cannot make the calibration finer
than the data underneath it. A larger calibration fold would.

**Phase 2's reported champion metrics were regenerated** under the smoothed
calibrator.
