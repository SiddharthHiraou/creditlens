# Target Definition

Status: **binding**. Every metric in this repo is computed against this definition.
Changing it invalidates all reported numbers and requires a new model version.

## The definition

A loan is **BAD** if it reached **90 or more days past due within 12 months of
origination**.

| Label | Condition | Used in training | Used in evaluation |
|---|---|---|---|
| `BAD` (1) | max DPD ≥ 90 inside the window | yes | yes |
| `GOOD` (0) | max DPD < 30 through the full window | yes | yes |
| `INDETERMINATE` (-1) | max DPD in [30, 89], never 90+ | **no** | yes, scored |
| `CENSORED` (-2) | window has not closed as of the snapshot date | no | no |

Implemented in [`src/ingestion/target.py`](../src/ingestion/target.py); the rule
is pinned by the tests in `tests/unit/test_target.py`.

## Why each choice

**Why 90 DPD.** It is the Basel reference point for default and the threshold at
which most consumer lenders charge off or move an account to recovery. 30 DPD is
too noisy — a large share of 30-day delinquencies self-cure — and 180 DPD leaves
too few bads to model.

**Why a 12-month performance window.** Long enough for the default curve to
mature past its steepest section, short enough that a model retrained annually
still sees complete outcomes. A longer window would push the most recent, most
decision-relevant vintages out of the training set entirely.

**Why indeterminates are excluded from training.** Accounts that touched 30–89
DPD but never reached 90 are genuinely ambiguous: labelling them GOOD teaches the
model that delinquency is acceptable, labelling them BAD teaches it that a single
missed payment is terminal. Dropping them from the fit sharpens the good/bad
contrast. They are still **scored at evaluation**, because in production they are
part of the through-the-door population and the model has to do something sensible
with them.

**Why censored rows are dropped entirely.** This is the rule that does the most
quiet work. A loan originated three months before the snapshot has not had time
to reach 90 DPD, so calling it GOOD manufactures a clean record out of nothing.
Because recent originations concentrate in the out-of-time test period, keeping
them would deflate the observed bad rate exactly where the model is being judged.
The test suite pins this: a loan already at 200 DPD is still censored if its
window closes after the snapshot date.

## Class balance

The target is imbalanced by construction — roughly 8–15% bad depending on
vintage. Handled with `class_weight="balanced"` (Phase 1) and `scale_pos_weight`
plus threshold tuning (Phase 2).

**SMOTE is not used anywhere in this project, and that is deliberate.** Synthetic
minority oversampling interpolates between real defaulted borrowers to invent
applicants who never applied. On credit data it degrades probability calibration,
which breaks the expected-loss arithmetic (`EL = PD × LGD × EAD`) that the whole
decisioning layer depends on. Reweighting changes the loss function without
fabricating rows.

**Accuracy is not reported anywhere.** At a 10% bad rate, "approve everyone"
scores 90%. The reported metrics are AUC/Gini, KS, PR-AUC, Brier, and the decile
rank-ordering table.

## Splitting

Splits are **out-of-time**, never random ([`src/ingestion/splits.py`](../src/ingestion/splits.py)):

```
train  ──────────►  calibration  ──►  valid  ──►  test (out-of-time)
       earliest vintages                            latest vintages
```

Two reasons a random split is wrong here:

1. **Applicant leakage.** Repeat borrowers land in multiple folds, so the model
   memorises people rather than learning risk.
2. **Macro leakage.** Rates, unemployment and the lender's own policy changes are
   shared across folds, so validation flatters how the model will behave on next
   quarter's population.

The calibration slice is carved from the **tail** of train rather than sampled at
random, so Phase 2's isotonic regression is fitted on the vintages nearest to
validation. `assert_no_temporal_leakage` fails the run if any split's date range
overlaps a later one.

## A caveat that must not be buried

**Home Credit contains no application date.** It is a single snapshot with a
prebuilt binary `TARGET` and no origination timestamp, so a genuine out-of-time
split on it is impossible. This project handles that explicitly rather than
inventing a date:

- **Home Credit** is used for **relational feature engineering** — aggregating
  bureau, installment and credit-card history into applicant-level features.
  Its loader writes a **null** `origination_date` sentinel, never a guess.
- **Lending Club** carries a real `issue_d` and is the backbone for **out-of-time
  evaluation, vintage analysis and PSI**.
- **The synthetic generator** emits a real origination calendar, so the
  out-of-time machinery is exercisable before either download lands.

Anyone claiming a clean out-of-time split on raw Home Credit either invented a
date or leaked one. This is a likely interview question and the answer is above.

## Lending Club status mapping

Lending Club reports a status string rather than a DPD timeline, so the vocabulary
is mapped onto the definition:

| Status | Label |
|---|---|
| `Charged Off`, `Default` | BAD |
| `Fully Paid`, `Current` | GOOD (subject to the window rule) |
| `Late (16-30 days)`, `Late (31-120 days)`, `In Grace Period` | INDETERMINATE |
| anything unrecognised | CENSORED |

Unrecognised statuses are censored rather than coerced to GOOD: a new status
string appearing in a future extract should shrink the population, not silently
add fake good loans.
