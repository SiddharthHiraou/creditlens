# ADR 0003: Monotonic constraints, and why `basic` over `advanced`

- **Status:** accepted
- **Date:** 2026-08-20
- **Phase:** 2

## Context

A GBDT will happily learn "applicants with more prior delinquency default less"
from a quirk of the training sample. That is indefensible in front of a model
validation team regardless of what it does to AUC, and it is the kind of finding
that gets a model sent back.

LightGBM, XGBoost and CatBoost all support per-feature monotonic constraints.
LightGBM additionally offers two enforcement methods, `basic` and `advanced`,
documenting `advanced` as the less conservative and generally more accurate one.

## Decision

Constrain the 47 features whose direction is unambiguous from domain knowledge —
debt burden, delinquency counts, utilisation, external scores, tenure — and use
LightGBM's **`basic`** enforcement method.

Constraints are **fixed, not searched**. Optuna tunes around them; it may not
trade them away for AUC.

## Consequences

**`advanced` does not actually guarantee the constraint.** Testing on this
feature set found a real violation: sweeping `BURO_overdue_to_debt` across its
range with all other features held at their median, predicted PD *fell* by
0.0008 as the feature rose, despite a `+1` constraint. Under `basic` the same
sweep was exactly monotone across every constrained feature.

`basic` also scored marginally better here (OOT AUC 0.7908 vs 0.7890), so the
usual accuracy argument for `advanced` did not apply. Even had it cost accuracy,
a constraint that is not guaranteed is worthless for the regulatory purpose it
exists to serve.

This is pinned by `tests/integration/test_phase2.py::test_monotonic_constraints_hold_in_the_fitted_model`,
which sweeps every constrained feature and fails on any violation.

**Deliberately unconstrained**, with reasons, so the omissions read as decisions:

| Feature | Why unconstrained |
|---|---|
| `STAB_age_years` | Not monotone (young and old both riskier), and ECOA-protected |
| `AMT_INCOME_TOTAL` | Higher income is not monotonically safer once the requested amount scales with it |
| `BURO_n_lines` | Thin file and over-extended are both risky |
| `CC_util_std` | Volatility cuts both ways |

Guessing a direction is worse than leaving one free.
