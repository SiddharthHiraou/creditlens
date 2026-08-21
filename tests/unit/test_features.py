"""Feature engineering: aggregations, spec contract, monotonic constraints."""

from __future__ import annotations

import polars as pl
import pytest

from src.features.aggregations import (
    _safe_ratio,
    _slope_expr,
    bureau_features,
    credit_card_features,
    installments_features,
)
from src.features.build import NON_FEATURE_COLUMNS, build
from src.features.monotonic import DIRECTION_RULES, constrained_count, constraints_for
from src.features.spec import FeatureSpec


def test_safe_ratio_returns_null_instead_of_infinity():
    df = pl.DataFrame({"a": [1.0, 1.0], "b": [2.0, 0.0]})
    out = df.select(_safe_ratio(pl.col("a"), pl.col("b")).alias("r"))["r"].to_list()
    assert out[0] == 0.5
    assert out[1] is None


def test_slope_is_null_when_x_has_no_variance():
    """A single observation carries no trend; a zero would read as 'stable'."""
    df = pl.DataFrame({"g": [1, 1], "x": [5.0, 5.0], "y": [1.0, 2.0]})
    out = df.group_by("g").agg(_slope_expr(pl.col("x"), pl.col("y")).alias("s"))
    assert out["s"][0] is None


def test_slope_recovers_a_known_gradient():
    df = pl.DataFrame({"g": [1] * 5, "x": [1.0, 2, 3, 4, 5], "y": [3.0, 5, 7, 9, 11]})
    out = df.group_by("g").agg(_slope_expr(pl.col("x"), pl.col("y")).alias("s"))
    assert out["s"][0] == pytest.approx(2.0)


def test_bureau_aggregation_produces_one_row_per_applicant():
    bureau = pl.LazyFrame(
        {
            "SK_ID_CURR": [1, 1, 2],
            "SK_ID_BUREAU": [10, 11, 12],
            "CREDIT_ACTIVE": ["Active", "Closed", "Active"],
            "CREDIT_TYPE": ["Credit card"] * 3,
            "DAYS_CREDIT": [-100, -800, -50],
            "DAYS_CREDIT_ENDDATE": [100, -200, 300],
            "AMT_CREDIT_SUM": [1000.0, 2000.0, 500.0],
            "AMT_CREDIT_SUM_DEBT": [400.0, 0.0, 250.0],
            "AMT_CREDIT_SUM_OVERDUE": [0.0, 0.0, 50.0],
            "CREDIT_DAY_OVERDUE": [0, 0, 12],
            "CNT_CREDIT_PROLONG": [0, 1, 0],
        }
    )
    out = bureau_features(bureau).collect().sort("SK_ID_CURR")
    assert out.height == 2
    assert out.filter(pl.col("SK_ID_CURR") == 1)["BURO_n_lines"].item() == 2
    assert out.filter(pl.col("SK_ID_CURR") == 1)["BURO_n_active"].item() == 1
    # debt 400 / credit 3000
    assert out.filter(pl.col("SK_ID_CURR") == 1)["BURO_debt_to_credit"].item() == pytest.approx(
        400 / 3000
    )


def test_installments_treats_a_later_payment_date_as_late():
    """Both columns are negative offsets, so payment minus due is positive when late."""
    inst = pl.LazyFrame(
        {
            "SK_ID_PREV": [1, 1],
            "SK_ID_CURR": [7, 7],
            "NUM_INSTALMENT_NUMBER": [1, 2],
            "DAYS_INSTALMENT": [-100, -70],
            "DAYS_ENTRY_PAYMENT": [-90, -75],  # first is 10 days late, second 5 days early
            "AMT_INSTALMENT": [100.0, 100.0],
            "AMT_PAYMENT": [100.0, 60.0],
        }
    )
    out = installments_features(inst).collect()
    assert out["INST_n_late_total"].item() == 1
    assert out["INST_max_days_late"].item() == 10
    assert out["INST_n_short_total"].item() == 1
    assert out["INST_paid_to_due"].item() == pytest.approx(160 / 200)


def test_credit_card_utilisation_uses_the_limit_not_the_balance():
    cc = pl.LazyFrame(
        {
            "SK_ID_PREV": [1, 1],
            "SK_ID_CURR": [3, 3],
            "MONTHS_BALANCE": [-2, -1],
            "AMT_BALANCE": [500.0, 900.0],
            "AMT_CREDIT_LIMIT_ACTUAL": [1000.0, 1000.0],
            "AMT_DRAWINGS_CURRENT": [0.0, 0.0],
            "AMT_PAYMENT_CURRENT": [0.0, 0.0],
            "SK_DPD": [0, 5],
        }
    )
    out = credit_card_features(cc).collect()
    assert out["CC_util_max"].item() == pytest.approx(0.9)
    assert out["CC_util_mean"].item() == pytest.approx(0.7)
    # Utilisation rising toward the present is a positive slope.
    assert out["CC_util_slope"].item() > 0


@pytest.fixture(scope="module")
def matrix():
    fm = build()
    return fm, fm.frame.head(2000).collect()


def test_feature_matrix_is_one_row_per_applicant(matrix):
    fm, df = matrix
    assert df["SK_ID_CURR"].n_unique() == df.height


def test_bookkeeping_columns_never_leak_into_the_feature_list(matrix):
    fm, _ = matrix
    assert not (set(fm.feature_names) & NON_FEATURE_COLUMNS)
    assert "label" not in fm.feature_names
    assert "max_dpd_in_window" not in fm.feature_names


def test_thin_file_applicants_survive_the_join(matrix):
    """A left join is the difference between modelling thin files and
    silently deleting them.

    Asserted against the source table's own row count rather than a literal,
    so the test does not depend on how large a dataset was generated.
    """
    fm, _ = matrix
    assert "FLAG_no_buro_history" in fm.feature_names

    from src.ingestion.loaders import load

    applicants = load("application", validate=False).select(pl.len()).collect().item()
    assert build().frame.select(pl.len()).collect().item() == applicants


def test_history_counts_fill_to_zero_but_ratios_stay_null():
    """Zero cards means zero months, but not 'average utilisation of 0'."""
    df = build().frame.collect()
    no_cards = df.filter(pl.col("CC_n_months") == 0)
    if no_cards.height:
        assert no_cards["CC_util_mean"].null_count() == no_cards.height


def test_monotonic_directions_are_signed_correctly():
    assert DIRECTION_RULES["RATIO_annuity_to_income"] == +1  # more burden, more risk
    assert DIRECTION_RULES["EXT_mean"] == -1  # higher score, less risk
    assert DIRECTION_RULES["BB_n_late_total"] == +1
    assert DIRECTION_RULES["STAB_employed_years"] == -1


def test_age_is_left_unconstrained_because_it_is_not_monotone():
    """Also an ECOA-protected attribute; a forced direction would be indefensible."""
    assert "STAB_age_years" not in DIRECTION_RULES


def test_constraint_vector_aligns_with_feature_order():
    feats = ["EXT_mean", "unknown_feature", "BB_n_late_total"]
    assert constraints_for(feats) == [-1, 0, +1]
    assert constrained_count(feats) == (2, 3)


def test_spec_fingerprint_changes_when_feature_order_changes():
    """Order matters to a numpy-backed model, so the hash must cover it."""
    df = pl.DataFrame({"a": [1.0], "b": [2.0]})
    s1 = FeatureSpec.build(df, ["a", "b"], [])
    s2 = FeatureSpec.build(df, ["b", "a"], [])
    assert s1.fingerprint != s2.fingerprint


def test_spec_round_trips_through_yaml(tmp_path):
    df = pl.DataFrame({"EXT_mean": [0.5], "BB_n_late_total": [2]})
    spec = FeatureSpec.build(df, ["EXT_mean", "BB_n_late_total"], [])
    path = spec.save(tmp_path / "spec.yaml")
    loaded = FeatureSpec.load(path)
    assert loaded.features == spec.features
    assert loaded.fingerprint == spec.fingerprint
    assert loaded.monotonic == {"EXT_mean": -1, "BB_n_late_total": 1}


def test_spec_rejects_a_frame_missing_its_features():
    df = pl.DataFrame({"a": [1.0], "b": [2.0]})
    spec = FeatureSpec.build(df, ["a", "b"], [])
    with pytest.raises(ValueError, match="missing"):
        spec.matrix(pl.DataFrame({"a": [1.0]}))


def test_spec_matrix_returns_columns_in_spec_order():
    df = pl.DataFrame({"b": [2.0], "a": [1.0]})
    spec = FeatureSpec.build(df, ["a", "b"], [])
    assert spec.matrix(df).columns == ["a", "b"]
