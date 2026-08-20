# ADR 0005: Reason codes are grouped, deduplicated, and blind to protected attributes

- **Status:** accepted
- **Date:** 2026-08-20
- **Phase:** 3

## Context

Under ECOA and Regulation B a declined applicant must receive the **specific
principal reasons** for the denial. The model produces 72 SHAP contributions per
application; the applicant must receive four sentences.

Three problems stand between one and the other:

1. **Restatement.** The top four raw contributions are frequently four
   measurements of the same thing — `RATIO_annuity_to_income`,
   `RATIO_credit_to_income`, `RATIO_goods_to_income`, `XSRC_total_debt_to_income`
   are one reason wearing four hats. Sending all four deprives the applicant of
   three real reasons.
2. **Protected attributes.** The model's single strongest feature is
   `EXT_mean_x_age`, which embeds age. Disclosing "age" as a basis for denial is
   the violation the whole reason-code layer exists to prevent.
3. **Silent gaps.** A feature with no mapped phrase is dropped from the
   disclosure without anyone noticing.

## Decision

A versioned YAML (`src/explainability/reason_codes.yaml`) maps every feature to
one of 11 **families**, each with an ECOA-style phrase and an `actionable` flag.

- Only **positive** SHAP contributions become reasons. A feature that helped the
  applicant is not a reason for declining them.
- Contributions are **summed within family**, and families are ranked. Four
  reasons means four distinct families.
- Features embedding a protected attribute map to a family describing what the
  applicant can act on. `EXT_mean_x_age` is disclosed as a credit-bureau-score
  reason, never as an age reason.
- A `suppress` list (age, sex, family status, dependents) never reaches a
  disclosure at all, whatever the model does with it.
- Coverage is **asserted, not logged**:
  `test_every_built_feature_is_mapped_or_explicitly_suppressed` fails the build
  if any of the 221 features lacks a family. 212 are mapped, 9 suppressed.

## Consequences

**Measured on the champion:** 100% of declines produce exactly four distinct
families, ranked by contribution. That is the Phase 3 deliverable.

**Accepted cost.** Family-level grouping loses feature-level precision — an
applicant is told "record of late or partial payments", not "your installment
lateness rate over the trailing 12 months is in the 91st percentile". That is
the correct trade: Reg B asks for principal reasons a person can understand and
act on, not a model dump.

**Version is logged with every disclosure**, so a memo produced under version 1
remains reconstructible after the phrases change.

**SHAP attributions are not causal**, and the phrase wording never implies they
are. "What drove the score" and "what would change the decision" are different
questions; the second is answered by the counterfactual module (ADR 0006).
