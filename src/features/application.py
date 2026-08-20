"""Application-level features: ratios, stability proxies, and interactions.

These are computed from the applicant table alone. The relational history lives
in ``aggregations.py``; the two are joined in ``build.py``.
"""

from __future__ import annotations

import polars as pl

from src.features.aggregations import _safe_ratio

EXT_SOURCES = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]


def affordability_ratios(lf: pl.LazyFrame) -> pl.LazyFrame:
    """The ratios a credit analyst computes before looking at anything else."""
    income = pl.col("AMT_INCOME_TOTAL")
    credit = pl.col("AMT_CREDIT")
    annuity = pl.col("AMT_ANNUITY")
    goods = pl.col("AMT_GOODS_PRICE")

    return lf.with_columns(
        RATIO_annuity_to_income=_safe_ratio(annuity, income),
        RATIO_credit_to_income=_safe_ratio(credit, income),
        RATIO_goods_to_income=_safe_ratio(goods, income),
        RATIO_credit_to_goods=_safe_ratio(credit, goods),
        RATIO_annuity_to_credit=_safe_ratio(annuity, credit),
        # Implied term in months: how long to repay at this annuity.
        RATIO_implied_term=_safe_ratio(credit, annuity),
        # Income left after servicing this loan, per household member.
        RATIO_income_per_family_member=_safe_ratio(
            income, pl.col("CNT_FAM_MEMBERS").cast(pl.Float64)
        ),
        RATIO_residual_income=income - (annuity * 12),
        RATIO_residual_income_per_member=_safe_ratio(
            income - (annuity * 12), pl.col("CNT_FAM_MEMBERS").cast(pl.Float64)
        ),
        RATIO_children_to_family=_safe_ratio(
            pl.col("CNT_CHILDREN").cast(pl.Float64), pl.col("CNT_FAM_MEMBERS").cast(pl.Float64)
        ),
        # Down payment implied by the gap between goods price and credit taken.
        RATIO_downpayment=_safe_ratio(goods - credit, goods),
    )


def stability_features(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Tenure and life-stage proxies.

    Home Credit encodes DAYS_* as negative offsets from the application date.
    Converting to positive years makes the direction of every downstream
    monotonic constraint obvious instead of inverted.
    """
    age_years = -pl.col("DAYS_BIRTH") / 365.25
    employed_years = -pl.col("DAYS_EMPLOYED") / 365.25

    return lf.with_columns(
        STAB_age_years=age_years,
        STAB_employed_years=employed_years,
        STAB_id_published_years=-pl.col("DAYS_ID_PUBLISH") / 365.25,
        # Share of adult life spent in the current job.
        STAB_employed_share_of_adult_life=_safe_ratio(employed_years, age_years - 18),
        STAB_age_at_employment_start=age_years - employed_years,
        STAB_employed_under_1y=(employed_years < 1).cast(pl.Int8),
        STAB_employed_over_5y=(employed_years > 5).cast(pl.Int8),
        STAB_age_band=pl.when(age_years < 25)
        .then(pl.lit("18-24"))
        .when(age_years < 35)
        .then(pl.lit("25-34"))
        .when(age_years < 45)
        .then(pl.lit("35-44"))
        .when(age_years < 55)
        .then(pl.lit("45-54"))
        .when(age_years < 65)
        .then(pl.lit("55-64"))
        .otherwise(pl.lit("65+")),
    )


def external_score_features(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Aggregate the three external bureau scores.

    Missingness is preserved as an explicit count, not imputed away: applicants
    with no external score are thin-file, and thin-file is itself predictive.
    """
    present = [pl.col(c).is_not_null().cast(pl.Int8) for c in EXT_SOURCES]

    n_present = sum(present[1:], present[0])
    return lf.with_columns(
        [pl.col(c).is_null().cast(pl.Int8).alias(f"EXT_{c[-1]}_missing") for c in EXT_SOURCES]
        + [
            pl.mean_horizontal(EXT_SOURCES).alias("EXT_mean"),
            pl.min_horizontal(EXT_SOURCES).alias("EXT_min"),
            pl.max_horizontal(EXT_SOURCES).alias("EXT_max"),
            n_present.alias("EXT_n_present"),
            (n_present == 0).cast(pl.Int8).alias("EXT_all_missing"),
        ]
    ).with_columns(
        # Spread between the strongest and weakest external opinion. A wide
        # spread means the bureaus disagree about this applicant.
        EXT_spread=pl.col("EXT_max") - pl.col("EXT_min"),
        # Interactions with the two strongest raw drivers.
        EXT_mean_x_annuity_ratio=pl.col("EXT_mean") * pl.col("RATIO_annuity_to_income"),
        EXT_mean_x_age=pl.col("EXT_mean") * pl.col("STAB_age_years"),
    )


def application_features(lf: pl.LazyFrame) -> pl.LazyFrame:
    """All application-level derived features, in dependency order."""
    return external_score_features(stability_features(affordability_ratios(lf)))
