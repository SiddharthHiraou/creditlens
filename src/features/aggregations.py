"""Relational history -> applicant-level features.

Each function takes the raw child table (and, where needed, its parent) and
returns one row per ``SK_ID_CURR``. Every output column is prefixed with its
source table so provenance survives into the SHAP plots and the reason codes:
a feature named ``INST_late_share_12m`` is traceable to installments_payments
without opening the code.

Aggregations stay lazy end to end. bureau_balance alone is 5.5M rows and gets
joined to bureau before grouping; collecting it eagerly would cost more memory
than the rest of the pipeline combined.
"""

from __future__ import annotations

import polars as pl

# Recency half-life for exponentially weighted bureau aggregates, in days.
# A 2-year-old credit line should not count the same as one opened last month.
RECENCY_HALFLIFE_DAYS = 365.0


def _slope_expr(x: pl.Expr, y: pl.Expr) -> pl.Expr:
    """OLS slope of ``y`` on ``x`` within a group.

    slope = cov(x, y) / var(x), expanded so it computes in a single pass over
    the group. Takes expressions rather than column names because most of the
    trend features regress over a quantity (utilisation, days late) that is
    itself computed and has no column on the frame.

    Returns null where x has no variance -- a single observation carries no
    trend, and a fabricated zero would read as "stable" to the model.
    """
    n = pl.len()
    sx, sy = x.sum(), y.sum()
    denom = n * (x**2).sum() - sx**2
    return pl.when(denom.abs() < 1e-9).then(None).otherwise((n * (x * y).sum() - sx * sy) / denom)


def _safe_ratio(num: pl.Expr, den: pl.Expr) -> pl.Expr:
    """Ratio guarded against divide-by-zero, yielding null rather than inf."""
    return pl.when(den.abs() < 1e-9).then(None).otherwise(num / den)


# --------------------------------------------------------------------------
# bureau
# --------------------------------------------------------------------------
def bureau_features(bureau: pl.LazyFrame) -> pl.LazyFrame:
    """Prior credit lines reported by the credit bureau."""
    is_active = pl.col("CREDIT_ACTIVE") == "Active"
    # Exponential recency weight: DAYS_CREDIT is negative, so this decays with age.
    weight = (pl.col("DAYS_CREDIT") / RECENCY_HALFLIFE_DAYS).exp()

    agg = bureau.group_by("SK_ID_CURR").agg(
        pl.len().alias("BURO_n_lines"),
        is_active.sum().alias("BURO_n_active"),
        (~is_active).sum().alias("BURO_n_closed"),
        is_active.mean().alias("BURO_active_share"),
        pl.col("DAYS_CREDIT").max().alias("BURO_days_since_last_line"),
        pl.col("DAYS_CREDIT").min().alias("BURO_days_since_first_line"),
        pl.col("DAYS_CREDIT").mean().alias("BURO_days_credit_mean"),
        pl.col("DAYS_CREDIT").std().alias("BURO_days_credit_std"),
        pl.col("AMT_CREDIT_SUM").sum().alias("BURO_amt_sum_total"),
        pl.col("AMT_CREDIT_SUM").mean().alias("BURO_amt_sum_mean"),
        pl.col("AMT_CREDIT_SUM").max().alias("BURO_amt_sum_max"),
        pl.col("AMT_CREDIT_SUM_DEBT").sum().alias("BURO_debt_total"),
        pl.col("AMT_CREDIT_SUM_DEBT").mean().alias("BURO_debt_mean"),
        pl.col("AMT_CREDIT_SUM_DEBT").max().alias("BURO_debt_max"),
        pl.col("AMT_CREDIT_SUM_OVERDUE").sum().alias("BURO_overdue_total"),
        pl.col("AMT_CREDIT_SUM_OVERDUE").max().alias("BURO_overdue_max"),
        (pl.col("AMT_CREDIT_SUM_OVERDUE") > 0).sum().alias("BURO_n_overdue_lines"),
        (pl.col("AMT_CREDIT_SUM_OVERDUE") > 0).mean().alias("BURO_overdue_line_share"),
        pl.col("CREDIT_DAY_OVERDUE").max().alias("BURO_days_overdue_max"),
        pl.col("CREDIT_DAY_OVERDUE").mean().alias("BURO_days_overdue_mean"),
        pl.col("CNT_CREDIT_PROLONG").sum().alias("BURO_prolong_total"),
        # Recency-weighted: recent behaviour dominates.
        (weight * pl.col("AMT_CREDIT_SUM_DEBT")).sum().alias("BURO_debt_recency_wtd"),
        (weight * pl.col("AMT_CREDIT_SUM_OVERDUE")).sum().alias("BURO_overdue_recency_wtd"),
        weight.sum().alias("BURO_recency_weight_total"),
        # Active-only slices: a closed overdue line is history, an active one is a problem.
        pl.col("AMT_CREDIT_SUM_DEBT").filter(is_active).sum().alias("BURO_active_debt_total"),
        pl.col("AMT_CREDIT_SUM").filter(is_active).sum().alias("BURO_active_credit_total"),
        pl.col("AMT_CREDIT_SUM_OVERDUE").filter(is_active).sum().alias("BURO_active_overdue_total"),
        # Product mix.
        *[
            (pl.col("CREDIT_TYPE") == t).sum().alias(f"BURO_n_type_{t.lower().replace(' ', '_')}")
            for t in ("Consumer credit", "Credit card", "Car loan", "Mortgage", "Microloan")
        ],
        pl.col("CREDIT_TYPE").n_unique().alias("BURO_n_credit_types"),
    )

    return agg.with_columns(
        BURO_debt_to_credit=_safe_ratio(pl.col("BURO_debt_total"), pl.col("BURO_amt_sum_total")),
        BURO_active_debt_to_credit=_safe_ratio(
            pl.col("BURO_active_debt_total"), pl.col("BURO_active_credit_total")
        ),
        BURO_overdue_to_debt=_safe_ratio(pl.col("BURO_overdue_total"), pl.col("BURO_debt_total")),
        BURO_debt_recency_wtd_norm=_safe_ratio(
            pl.col("BURO_debt_recency_wtd"), pl.col("BURO_recency_weight_total")
        ),
        BURO_lines_per_year=_safe_ratio(
            pl.col("BURO_n_lines") * 365.25, -pl.col("BURO_days_since_first_line")
        ),
    )


# --------------------------------------------------------------------------
# bureau_balance
# --------------------------------------------------------------------------
# Home Credit encodes monthly bureau status as: C = closed, X = unknown,
# 0 = no DPD, 1-5 = increasing DPD buckets (5 is write-off).
_DPD_STATUSES = ["1", "2", "3", "4", "5"]


def bureau_balance_features(bureau_balance: pl.LazyFrame, bureau: pl.LazyFrame) -> pl.LazyFrame:
    """Monthly bureau status history, rolled up to the applicant.

    These trailing-window delinquency counts are consistently among the
    highest-lift features on the real dataset: *when* someone was late matters
    far more than whether they ever were.
    """
    joined = bureau_balance.join(
        bureau.select("SK_ID_BUREAU", "SK_ID_CURR"), on="SK_ID_BUREAU", how="inner"
    )

    is_late = pl.col("STATUS").is_in(_DPD_STATUSES)
    status_num = (
        pl.when(pl.col("STATUS").is_in(_DPD_STATUSES))
        .then(pl.col("STATUS").cast(pl.Int32, strict=False))
        .otherwise(0)
    )

    windows = {"3m": -3, "6m": -6, "12m": -12}
    window_exprs = []
    for label, months in windows.items():
        in_window = pl.col("MONTHS_BALANCE") >= months
        window_exprs += [
            (is_late & in_window).sum().alias(f"BB_n_late_{label}"),
            _safe_ratio(
                (is_late & in_window).sum().cast(pl.Float64), in_window.sum().cast(pl.Float64)
            ).alias(f"BB_late_share_{label}"),
            status_num.filter(in_window).max().alias(f"BB_max_status_{label}"),
        ]

    return joined.group_by("SK_ID_CURR").agg(
        pl.len().alias("BB_n_months"),
        pl.col("SK_ID_BUREAU").n_unique().alias("BB_n_lines"),
        is_late.sum().alias("BB_n_late_total"),
        is_late.mean().alias("BB_late_share_total"),
        status_num.max().alias("BB_max_status"),
        status_num.mean().alias("BB_mean_status"),
        (pl.col("STATUS") == "C").mean().alias("BB_closed_share"),
        (pl.col("STATUS") == "X").mean().alias("BB_unknown_share"),
        # Is delinquency getting worse or better? MONTHS_BALANCE runs
        # negative-to-zero toward the present, so a positive slope means
        # status is climbing as the present approaches: deteriorating.
        _status_slope(),
        *window_exprs,
    )


def _status_slope() -> pl.Expr:
    """Slope of numeric bureau status against month index.

    Written out rather than reusing :func:`_slope` because the status cast has
    to happen inside the expression: there is no numeric status column on the
    frame at aggregation time.
    """
    x = pl.col("MONTHS_BALANCE").cast(pl.Float64)
    y = (
        pl.when(pl.col("STATUS").is_in(_DPD_STATUSES))
        .then(pl.col("STATUS").cast(pl.Int32, strict=False))
        .otherwise(0)
        .cast(pl.Float64)
    )
    n = pl.len()
    sx, sy = x.sum(), y.sum()
    denom = n * (x**2).sum() - sx**2
    return (
        pl.when(denom.abs() < 1e-9)
        .then(None)
        .otherwise((n * (x * y).sum() - sx * sy) / denom)
        .alias("BB_status_slope")
    )


# --------------------------------------------------------------------------
# previous_application
# --------------------------------------------------------------------------
def previous_application_features(prev: pl.LazyFrame) -> pl.LazyFrame:
    """The applicant's history with *this* lender.

    Prior refusals are among the strongest single signals available: the
    lender's own past underwriting decisions encode information the bureau
    does not carry.
    """
    approved = pl.col("NAME_CONTRACT_STATUS") == "Approved"
    refused = pl.col("NAME_CONTRACT_STATUS") == "Refused"

    agg = prev.group_by("SK_ID_CURR").agg(
        pl.len().alias("PREV_n_applications"),
        approved.sum().alias("PREV_n_approved"),
        refused.sum().alias("PREV_n_refused"),
        refused.mean().alias("PREV_refusal_rate"),
        (pl.col("NAME_CONTRACT_STATUS") == "Canceled").sum().alias("PREV_n_canceled"),
        (pl.col("NAME_CONTRACT_STATUS") == "Unused offer").sum().alias("PREV_n_unused"),
        pl.col("DAYS_DECISION").max().alias("PREV_days_since_last"),
        pl.col("DAYS_DECISION").min().alias("PREV_days_since_first"),
        pl.col("DAYS_DECISION").mean().alias("PREV_days_decision_mean"),
        pl.col("AMT_APPLICATION").mean().alias("PREV_amt_application_mean"),
        pl.col("AMT_APPLICATION").max().alias("PREV_amt_application_max"),
        pl.col("AMT_APPLICATION").sum().alias("PREV_amt_application_total"),
        pl.col("AMT_CREDIT").mean().alias("PREV_amt_credit_mean"),
        pl.col("AMT_CREDIT").sum().alias("PREV_amt_credit_total"),
        pl.col("AMT_ANNUITY").mean().alias("PREV_amt_annuity_mean"),
        pl.col("AMT_ANNUITY").max().alias("PREV_amt_annuity_max"),
        pl.col("CNT_PAYMENT").mean().alias("PREV_cnt_payment_mean"),
        pl.col("CNT_PAYMENT").max().alias("PREV_cnt_payment_max"),
        # Most recent decision, which carries more weight than the average.
        pl.col("NAME_CONTRACT_STATUS")
        .sort_by("DAYS_DECISION", descending=True)
        .first()
        .alias("PREV_last_status"),
        pl.col("AMT_APPLICATION").filter(refused).mean().alias("PREV_refused_amt_mean"),
        pl.col("AMT_APPLICATION").filter(approved).mean().alias("PREV_approved_amt_mean"),
        pl.col("NAME_CONTRACT_TYPE").n_unique().alias("PREV_n_contract_types"),
    )

    return agg.with_columns(
        # Did the lender grant less than was asked for? A credit-tightening signal.
        PREV_credit_to_application=_safe_ratio(
            pl.col("PREV_amt_credit_total"), pl.col("PREV_amt_application_total")
        ),
        PREV_applications_per_year=_safe_ratio(
            pl.col("PREV_n_applications") * 365.25, -pl.col("PREV_days_since_first")
        ),
        PREV_last_was_refused=(pl.col("PREV_last_status") == "Refused").cast(pl.Int8),
    ).drop("PREV_last_status")


# --------------------------------------------------------------------------
# installments_payments
# --------------------------------------------------------------------------
def installments_features(inst: pl.LazyFrame) -> pl.LazyFrame:
    """Actual repayment behaviour: did they pay, on time, in full?

    ``DAYS_ENTRY_PAYMENT`` and ``DAYS_INSTALMENT`` are both negative offsets, so
    payment minus due is positive when the payment landed late.
    """
    days_late = pl.col("DAYS_ENTRY_PAYMENT") - pl.col("DAYS_INSTALMENT")
    is_late = days_late > 0
    shortfall = pl.col("AMT_INSTALMENT") - pl.col("AMT_PAYMENT")
    is_short = shortfall > 0.01

    windows = {"6m": -180, "12m": -365, "24m": -730}
    window_exprs = []
    for label, days in windows.items():
        in_window = pl.col("DAYS_INSTALMENT") >= days
        window_exprs += [
            (is_late & in_window).sum().alias(f"INST_n_late_{label}"),
            _safe_ratio(
                (is_late & in_window).sum().cast(pl.Float64), in_window.sum().cast(pl.Float64)
            ).alias(f"INST_late_share_{label}"),
            days_late.filter(in_window).max().alias(f"INST_max_days_late_{label}"),
            (is_short & in_window).sum().alias(f"INST_n_short_{label}"),
        ]

    agg = inst.group_by("SK_ID_CURR").agg(
        pl.len().alias("INST_n_payments"),
        pl.col("SK_ID_PREV").n_unique().alias("INST_n_loans"),
        is_late.sum().alias("INST_n_late_total"),
        is_late.mean().alias("INST_late_share_total"),
        days_late.max().alias("INST_max_days_late"),
        days_late.mean().alias("INST_mean_days_late"),
        days_late.std().alias("INST_std_days_late"),
        days_late.filter(is_late).mean().alias("INST_mean_days_late_when_late"),
        (days_late > 30).sum().alias("INST_n_late_over_30d"),
        (days_late > 90).sum().alias("INST_n_late_over_90d"),
        is_short.sum().alias("INST_n_short_total"),
        is_short.mean().alias("INST_short_share_total"),
        shortfall.sum().alias("INST_shortfall_total"),
        shortfall.max().alias("INST_shortfall_max"),
        pl.col("AMT_INSTALMENT").sum().alias("INST_amt_due_total"),
        pl.col("AMT_PAYMENT").sum().alias("INST_amt_paid_total"),
        pl.col("AMT_INSTALMENT").mean().alias("INST_amt_due_mean"),
        pl.col("DAYS_INSTALMENT").max().alias("INST_days_since_last_due"),
        pl.col("DAYS_INSTALMENT").min().alias("INST_days_since_first_due"),
        # Is lateness worsening? DAYS_INSTALMENT rises toward the present, so a
        # positive slope means later payments are getting later.
        _slope_expr(pl.col("DAYS_INSTALMENT").cast(pl.Float64), days_late.cast(pl.Float64)).alias(
            "INST_days_late_slope"
        ),
        *window_exprs,
    )

    return agg.with_columns(
        INST_paid_to_due=_safe_ratio(pl.col("INST_amt_paid_total"), pl.col("INST_amt_due_total")),
        INST_payments_per_loan=_safe_ratio(
            pl.col("INST_n_payments").cast(pl.Float64), pl.col("INST_n_loans").cast(pl.Float64)
        ),
    )


# --------------------------------------------------------------------------
# credit_card_balance
# --------------------------------------------------------------------------
def credit_card_features(cc: pl.LazyFrame) -> pl.LazyFrame:
    """Revolving behaviour. Utilisation *trajectory* matters more than level:
    a borrower climbing from 20% to 80% is a different risk from one sitting
    flat at 80%."""
    util = _safe_ratio(pl.col("AMT_BALANCE"), pl.col("AMT_CREDIT_LIMIT_ACTUAL"))
    has_dpd = pl.col("SK_DPD") > 0

    windows = {"3m": -3, "6m": -6, "12m": -12}
    window_exprs = []
    for label, months in windows.items():
        in_window = pl.col("MONTHS_BALANCE") >= months
        window_exprs += [
            util.filter(in_window).mean().alias(f"CC_util_mean_{label}"),
            util.filter(in_window).max().alias(f"CC_util_max_{label}"),
            (has_dpd & in_window).sum().alias(f"CC_n_dpd_{label}"),
        ]

    agg = cc.group_by("SK_ID_CURR").agg(
        pl.len().alias("CC_n_months"),
        pl.col("SK_ID_PREV").n_unique().alias("CC_n_cards"),
        util.mean().alias("CC_util_mean"),
        util.max().alias("CC_util_max"),
        util.std().alias("CC_util_std"),
        util.last().alias("CC_util_latest"),
        (util > 0.9).mean().alias("CC_share_months_over_90pct"),
        (util > 1.0).sum().alias("CC_n_months_over_limit"),
        pl.col("AMT_BALANCE").mean().alias("CC_balance_mean"),
        pl.col("AMT_BALANCE").max().alias("CC_balance_max"),
        pl.col("AMT_CREDIT_LIMIT_ACTUAL").mean().alias("CC_limit_mean"),
        pl.col("AMT_CREDIT_LIMIT_ACTUAL").max().alias("CC_limit_max"),
        pl.col("AMT_DRAWINGS_CURRENT").mean().alias("CC_drawings_mean"),
        pl.col("AMT_DRAWINGS_CURRENT").sum().alias("CC_drawings_total"),
        pl.col("AMT_PAYMENT_CURRENT").mean().alias("CC_payment_mean"),
        pl.col("AMT_PAYMENT_CURRENT").sum().alias("CC_payment_total"),
        pl.col("SK_DPD").max().alias("CC_dpd_max"),
        has_dpd.sum().alias("CC_n_dpd_months"),
        has_dpd.mean().alias("CC_dpd_share"),
        _slope_expr(pl.col("MONTHS_BALANCE").cast(pl.Float64), util.cast(pl.Float64)).alias(
            "CC_util_slope"
        ),
        _slope_expr(
            pl.col("MONTHS_BALANCE").cast(pl.Float64), pl.col("AMT_BALANCE").cast(pl.Float64)
        ).alias("CC_balance_slope"),
        *window_exprs,
    )

    return agg.with_columns(
        CC_payment_to_drawings=_safe_ratio(pl.col("CC_payment_total"), pl.col("CC_drawings_total")),
        # Recent utilisation minus long-run: positive means a recent run-up.
        CC_util_recent_vs_overall=pl.col("CC_util_mean_3m") - pl.col("CC_util_mean"),
    )


# --------------------------------------------------------------------------
# POS_CASH_balance
# --------------------------------------------------------------------------
def pos_cash_features(pos: pl.LazyFrame) -> pl.LazyFrame:
    """Point-of-sale and cash loan servicing history."""
    has_dpd = pl.col("SK_DPD") > 0

    return pos.group_by("SK_ID_CURR").agg(
        pl.len().alias("POS_n_months"),
        pl.col("SK_ID_PREV").n_unique().alias("POS_n_loans"),
        pl.col("SK_DPD").max().alias("POS_dpd_max"),
        pl.col("SK_DPD").mean().alias("POS_dpd_mean"),
        has_dpd.sum().alias("POS_n_dpd_months"),
        has_dpd.mean().alias("POS_dpd_share"),
        (pl.col("SK_DPD") > 30).sum().alias("POS_n_dpd_over_30"),
        pl.col("CNT_INSTALMENT").mean().alias("POS_cnt_instalment_mean"),
        pl.col("CNT_INSTALMENT").max().alias("POS_cnt_instalment_max"),
        pl.col("CNT_INSTALMENT_FUTURE").mean().alias("POS_cnt_future_mean"),
        pl.col("CNT_INSTALMENT_FUTURE").last().alias("POS_cnt_future_latest"),
        (pl.col("NAME_CONTRACT_STATUS") == "Completed").mean().alias("POS_completed_share"),
        (pl.col("NAME_CONTRACT_STATUS") == "Active").sum().alias("POS_n_active_months"),
        pl.col("MONTHS_BALANCE").max().alias("POS_days_since_last"),
        _slope_expr(
            pl.col("MONTHS_BALANCE").cast(pl.Float64), pl.col("SK_DPD").cast(pl.Float64)
        ).alias("POS_dpd_slope"),
    )
