"""Monotonic constraints for the GBDT track.

A credit model must not be able to learn "more delinquency is safer" from a
quirk of the training sample. Where the direction of a relationship is known
from domain knowledge, it is imposed rather than hoped for.

This is a regulatory expectation as much as a modelling one: a model whose
score improves when an applicant's debt burden rises is indefensible in
front of a validation team, no matter what it does to AUC.

Direction convention (LightGBM/XGBoost):
    +1  feature increases  ->  PD increases   (risk-increasing)
    -1  feature increases  ->  PD decreases   (protective)
     0  unconstrained

Only features whose direction is genuinely unambiguous are constrained.
Guessing a direction is worse than leaving it free.
"""

from __future__ import annotations

# Matched by prefix, longest match wins, so a specific rule can override a
# family rule.
DIRECTION_RULES: dict[str, int] = {
    # --- affordability: more burden is worse -------------------------------
    "RATIO_annuity_to_income": +1,
    "RATIO_credit_to_income": +1,
    "RATIO_goods_to_income": +1,
    "RATIO_annuity_to_credit": +1,
    "RATIO_residual_income": -1,
    "RATIO_residual_income_per_member": -1,
    "RATIO_income_per_family_member": -1,
    "RATIO_downpayment": -1,
    "XSRC_total_debt_to_income": +1,
    "XSRC_bureau_debt_to_income": +1,
    # --- external scores: higher is safer, by construction ------------------
    "EXT_SOURCE_1": -1,
    "EXT_SOURCE_2": -1,
    "EXT_SOURCE_3": -1,
    "EXT_mean": -1,
    "EXT_min": -1,
    "EXT_max": -1,
    "EXT_n_present": -1,
    "EXT_all_missing": +1,
    # --- stability: longer tenure is safer ---------------------------------
    "STAB_employed_years": -1,
    "STAB_employed_share_of_adult_life": -1,
    "STAB_employed_under_1y": +1,
    "STAB_employed_over_5y": -1,
    # --- bureau delinquency: unambiguously risk-increasing -----------------
    "BURO_overdue_total": +1,
    "BURO_overdue_max": +1,
    "BURO_n_overdue_lines": +1,
    "BURO_overdue_line_share": +1,
    "BURO_days_overdue_max": +1,
    "BURO_days_overdue_mean": +1,
    "BURO_overdue_to_debt": +1,
    "BURO_active_overdue_total": +1,
    "BURO_debt_to_credit": +1,
    "BURO_active_debt_to_credit": +1,
    "BURO_prolong_total": +1,
    # --- monthly bureau status ---------------------------------------------
    "BB_n_late_total": +1,
    "BB_late_share_total": +1,
    "BB_max_status": +1,
    "BB_mean_status": +1,
    "BB_n_late_3m": +1,
    "BB_n_late_6m": +1,
    "BB_n_late_12m": +1,
    "BB_late_share_3m": +1,
    "BB_late_share_6m": +1,
    "BB_late_share_12m": +1,
    "BB_max_status_3m": +1,
    "BB_max_status_6m": +1,
    "BB_max_status_12m": +1,
    # --- repayment behaviour -----------------------------------------------
    "INST_n_late_total": +1,
    "INST_late_share_total": +1,
    "INST_max_days_late": +1,
    "INST_mean_days_late": +1,
    "INST_n_late_over_30d": +1,
    "INST_n_late_over_90d": +1,
    "INST_n_short_total": +1,
    "INST_short_share_total": +1,
    "INST_shortfall_total": +1,
    "INST_shortfall_max": +1,
    "INST_paid_to_due": -1,
    "INST_late_share_6m": +1,
    "INST_late_share_12m": +1,
    "INST_late_share_24m": +1,
    # --- revolving utilisation ---------------------------------------------
    "CC_util_mean": +1,
    "CC_util_max": +1,
    "CC_util_latest": +1,
    "CC_share_months_over_90pct": +1,
    "CC_n_months_over_limit": +1,
    "CC_dpd_max": +1,
    "CC_n_dpd_months": +1,
    "CC_dpd_share": +1,
    "CC_util_slope": +1,
    # --- POS servicing ------------------------------------------------------
    "POS_dpd_max": +1,
    "POS_dpd_mean": +1,
    "POS_n_dpd_months": +1,
    "POS_dpd_share": +1,
    "POS_n_dpd_over_30": +1,
    # --- prior decisions by this lender ------------------------------------
    "PREV_n_refused": +1,
    "PREV_refusal_rate": +1,
    "PREV_last_was_refused": +1,
}

# Deliberately unconstrained, with reasons, so the omissions are decisions
# rather than oversights:
#   STAB_age_years        non-monotone (young and very old both riskier) and
#                         age is an ECOA-protected attribute
#   AMT_INCOME_TOTAL      higher income is not monotonically safer once the
#                         requested amount scales with it
#   BURO_n_lines          thin file and over-extended are both risky
#   CC_util_std           volatility cuts both ways
#   *_slope (except CC)   sign depends on the underlying level


def constraints_for(features: list[str]) -> list[int]:
    """Constraint vector aligned to ``features``, in order."""
    return [DIRECTION_RULES.get(f, 0) for f in features]


def constrained_count(features: list[str]) -> tuple[int, int]:
    """(number constrained, total) — reported in the model card."""
    v = constraints_for(features)
    return sum(1 for c in v if c != 0), len(v)
