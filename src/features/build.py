"""Assemble the applicant-level feature matrix.

One row per ``SK_ID_CURR``: application fields, derived application features,
and every relational aggregation left-joined on.

Two rules this module enforces:

**Left joins, always.** An applicant with no bureau file must survive with
nulls, not vanish. Thin-file applicants are a real and risk-relevant segment;
an inner join would silently drop them and inflate every metric.

**No target-derived features.** Nothing here may touch the label. Target
encoding, out-of-fold means and similar constructions leak; if one is ever
needed it belongs behind a fold-aware transformer, not in this module.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from src.features.aggregations import (
    bureau_balance_features,
    bureau_features,
    credit_card_features,
    installments_features,
    pos_cash_features,
    previous_application_features,
)
from src.features.application import application_features
from src.ingestion.loaders import Source, load

# Columns that exist for bookkeeping and must never reach a model.
NON_FEATURE_COLUMNS = frozenset(
    {
        "SK_ID_CURR",
        "label",
        "origination_date",
        "window_close_date",
        "vintage",
        "max_dpd_in_window",
        "TARGET",
    }
)

# Counts of history that are genuinely zero rather than unknown when an
# applicant has no rows in a child table.
_ZERO_FILL_PREFIXES = ("BURO_n_", "BB_n_", "PREV_n_", "INST_n_", "CC_n_", "POS_n_")


@dataclass(frozen=True)
class FeatureMatrix:
    frame: pl.LazyFrame
    feature_names: list[str]
    categorical_names: list[str]

    @property
    def n_features(self) -> int:
        return len(self.feature_names)


def build(
    application: pl.LazyFrame | None = None,
    source: Source = Source.AUTO,
    *,
    validate: bool = False,
) -> FeatureMatrix:
    """Build the full feature matrix.

    ``application`` may be passed in already labelled and split; when omitted
    it is loaded fresh. Everything stays lazy until the caller collects.
    """
    app = application if application is not None else load("application", source, validate=validate)

    bureau = load("bureau", source, validate=validate)
    parts = {
        "bureau": bureau_features(bureau),
        "bureau_balance": bureau_balance_features(
            load("bureau_balance", source, validate=validate), bureau
        ),
        "previous": previous_application_features(
            load("previous_application", source, validate=validate)
        ),
        "installments": installments_features(
            load("installments_payments", source, validate=validate)
        ),
        "credit_card": credit_card_features(load("credit_card_balance", source, validate=validate)),
        "pos": pos_cash_features(load("POS_CASH_balance", source, validate=validate)),
    }

    out = application_features(app)
    for part in parts.values():
        out = out.join(part, on="SK_ID_CURR", how="left")

    out = _fill_absent_history(out)
    out = _cross_source_features(out)

    schema = out.collect_schema()
    categorical = [c for c, dt in schema.items() if dt == pl.Utf8 and c not in NON_FEATURE_COLUMNS]
    features = [c for c in schema.names() if c not in NON_FEATURE_COLUMNS]

    return FeatureMatrix(frame=out, feature_names=features, categorical_names=categorical)


def _fill_absent_history(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Distinguish "no history" from "unknown".

    After a left join, an applicant with no bureau file has null in every
    bureau column. For *counts* that null genuinely means zero and filling it
    helps the model. For *means, ratios and slopes* it does not -- the average
    utilisation of zero credit cards is undefined, and filling it with 0 would
    place thin-file applicants at the safest end of a scale they are not on.
    Those stay null and let the GBDT learn its own split for missing.
    """
    schema = lf.collect_schema()
    zero_fill = [
        c for c in schema.names() if c.startswith(_ZERO_FILL_PREFIXES) and schema[c].is_numeric()
    ]
    has_history = [
        ("BURO", "BURO_n_lines"),
        ("PREV", "PREV_n_applications"),
        ("INST", "INST_n_payments"),
        ("CC", "CC_n_months"),
        ("POS", "POS_n_months"),
    ]
    flags = [
        pl.col(col).is_null().cast(pl.Int8).alias(f"FLAG_no_{prefix.lower()}_history")
        for prefix, col in has_history
        if col in schema.names()
    ]
    return lf.with_columns(flags).with_columns([pl.col(c).fill_null(0) for c in zero_fill])


def _cross_source_features(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Features that only exist by relating one source to another.

    The application's requested credit measured against the applicant's
    existing obligations is more informative than either in isolation.
    """
    from src.features.aggregations import _safe_ratio

    return lf.with_columns(
        XSRC_new_credit_to_bureau_debt=_safe_ratio(pl.col("AMT_CREDIT"), pl.col("BURO_debt_total")),
        XSRC_total_debt_to_income=_safe_ratio(
            pl.col("BURO_debt_total") + pl.col("AMT_CREDIT"), pl.col("AMT_INCOME_TOTAL")
        ),
        XSRC_bureau_debt_to_income=_safe_ratio(
            pl.col("BURO_debt_total"), pl.col("AMT_INCOME_TOTAL")
        ),
        XSRC_annuity_to_prev_annuity=_safe_ratio(
            pl.col("AMT_ANNUITY"), pl.col("PREV_amt_annuity_mean")
        ),
        XSRC_credit_to_prev_credit=_safe_ratio(
            pl.col("AMT_CREDIT"), pl.col("PREV_amt_credit_mean")
        ),
    )
