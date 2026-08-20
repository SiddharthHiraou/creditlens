"""Default-target construction with an enforced performance window.

The target is *not* "did this loan ever go bad". It is:

    BAD  = reached 90+ days past due within 12 months of origination
    GOOD = survived the full 12-month window without reaching 90 DPD
    INDETERMINATE = ever reached 30-89 DPD but never 90+, inside the window

Indeterminates are dropped from training and scored at evaluation only. Loans
whose 12-month window has not fully elapsed as of the data snapshot are
right-censored and excluded entirely -- keeping them would label a loan "good"
purely because it has not had time to default yet, which silently deflates the
observed bad rate on the most recent (and most decision-relevant) vintages.
"""

from __future__ import annotations

import datetime as dt
from enum import IntEnum

import polars as pl

from src.config import TargetConfig


class Label(IntEnum):
    GOOD = 0
    BAD = 1
    INDETERMINATE = -1
    CENSORED = -2


# Lending Club reports a terminal or current status string rather than a DPD
# timeline, so the status vocabulary is mapped onto the DPD definition above.
LC_BAD_STATUSES = frozenset(
    {"Charged Off", "Default", "Does not meet the credit policy. Status:Charged Off"}
)
LC_INDETERMINATE_STATUSES = frozenset(
    {"Late (31-120 days)", "Late (16-30 days)", "In Grace Period"}
)
LC_GOOD_STATUSES = frozenset(
    {"Fully Paid", "Current", "Does not meet the credit policy. Status:Fully Paid"}
)


def window_close_date(origination: dt.date, months: int) -> dt.date:
    """Date on which the performance window closes for a loan."""
    year = origination.year + (origination.month - 1 + months) // 12
    month = (origination.month - 1 + months) % 12 + 1
    day = min(origination.day, _days_in_month(year, month))
    return dt.date(year, month, day)


def _days_in_month(year: int, month: int) -> int:
    if month == 12:
        return 31
    return (dt.date(year, month + 1, 1) - dt.timedelta(days=1)).day


def assign_labels_from_dpd(
    lf: pl.LazyFrame,
    cfg: TargetConfig,
    *,
    origination_col: str = "origination_date",
    max_dpd_col: str = "max_dpd_in_window",
) -> pl.LazyFrame:
    """Label a frame that carries an observed max-DPD inside the window.

    This is the path used for the synthetic data and for any source where a
    real delinquency timeline is available.
    """
    window_close = pl.col(origination_col).dt.offset_by(f"{cfg.performance_window_months}mo")

    return lf.with_columns(
        window_close_date=window_close,
        label=pl.when(window_close > pl.lit(cfg.snapshot_date))
        .then(pl.lit(int(Label.CENSORED)))
        .when(pl.col(max_dpd_col) >= cfg.dpd_bad_threshold)
        .then(pl.lit(int(Label.BAD)))
        .when(pl.col(max_dpd_col) >= cfg.dpd_indeterminate_floor)
        .then(pl.lit(int(Label.INDETERMINATE)))
        .otherwise(pl.lit(int(Label.GOOD)))
        .cast(pl.Int8),
    )


def assign_labels_from_status(
    lf: pl.LazyFrame,
    cfg: TargetConfig,
    *,
    origination_col: str = "issue_d",
    status_col: str = "loan_status",
) -> pl.LazyFrame:
    """Label Lending Club rows from their status string.

    ``Current`` is only trustworthy as GOOD once the performance window has
    closed; before that the row is censored by the window rule below anyway.
    Unrecognised statuses are censored rather than silently coerced to good.
    """
    window_close = pl.col(origination_col).dt.offset_by(f"{cfg.performance_window_months}mo")

    return lf.with_columns(
        window_close_date=window_close,
        label=pl.when(window_close > pl.lit(cfg.snapshot_date))
        .then(pl.lit(int(Label.CENSORED)))
        .when(pl.col(status_col).is_in(list(LC_BAD_STATUSES)))
        .then(pl.lit(int(Label.BAD)))
        .when(pl.col(status_col).is_in(list(LC_INDETERMINATE_STATUSES)))
        .then(pl.lit(int(Label.INDETERMINATE)))
        .when(pl.col(status_col).is_in(list(LC_GOOD_STATUSES)))
        .then(pl.lit(int(Label.GOOD)))
        .otherwise(pl.lit(int(Label.CENSORED)))
        .cast(pl.Int8),
    )


def modelling_population(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Rows eligible for *training*: good and bad only."""
    return lf.filter(pl.col("label").is_in([int(Label.GOOD), int(Label.BAD)]))


def evaluation_population(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Rows eligible for *scoring at evaluation*: adds back indeterminates."""
    return lf.filter(pl.col("label") != int(Label.CENSORED))


def label_summary(lf: pl.LazyFrame) -> pl.DataFrame:
    """Counts and rates per label, for the README and the data dictionary."""
    names = {
        int(Label.GOOD): "good",
        int(Label.BAD): "bad",
        int(Label.INDETERMINATE): "indeterminate",
        int(Label.CENSORED): "censored",
    }
    out = (
        lf.group_by("label")
        .agg(pl.len().alias("n"))
        .with_columns(share=pl.col("n") / pl.col("n").sum())
        .sort("label")
        .collect()
    )
    return out.with_columns(
        label_name=pl.col("label").replace_strict(names, default="unknown", return_dtype=pl.Utf8)
    ).select("label", "label_name", "n", "share")


def observed_bad_rate(lf: pl.LazyFrame) -> float:
    """Bad rate over the modelling population (excludes indeterminate/censored)."""
    df = modelling_population(lf).select(pl.col("label").mean()).collect()
    return float(df.item()) if df.height else float("nan")
