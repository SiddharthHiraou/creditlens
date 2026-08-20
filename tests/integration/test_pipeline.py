"""End-to-end Phase 1: generate -> validate -> label -> split -> fit -> score.

Runs on a small synthetic draw so it stays inside a normal CI budget.
"""

from __future__ import annotations

import polars as pl
import pytest

from src.config import SYNTHETIC_SPLIT, SYNTHETIC_TARGET
from src.evaluation.metrics import decile_table, discrimination, is_rank_ordered
from src.ingestion.schemas import SCHEMAS
from src.ingestion.splits import assert_no_temporal_leakage, split_by_time
from src.ingestion.synthetic import SyntheticConfig, generate
from src.ingestion.target import assign_labels_from_dpd, modelling_population
from src.models.baseline import fit, prepare


@pytest.fixture(scope="module")
def tables() -> dict[str, pl.DataFrame]:
    return generate(SyntheticConfig(n_applicants=8_000))


def test_generator_emits_every_expected_table(tables):
    assert set(tables) == set(SCHEMAS)


def test_all_tables_satisfy_their_pandera_contract(tables):
    for name, df in tables.items():
        SCHEMAS[name].validate(df, lazy=True)


def test_referential_integrity_across_relational_tables(tables):
    app_ids = set(tables["application"]["SK_ID_CURR"].to_list())
    for child in ("bureau", "previous_application", "installments_payments"):
        assert set(tables[child]["SK_ID_CURR"].to_list()) <= app_ids

    bureau_ids = set(tables["bureau"]["SK_ID_BUREAU"].to_list())
    assert set(tables["bureau_balance"]["SK_ID_BUREAU"].to_list()) <= bureau_ids

    prev_ids = set(tables["previous_application"]["SK_ID_PREV"].to_list())
    for child in ("installments_payments", "credit_card_balance", "POS_CASH_balance"):
        assert set(tables[child]["SK_ID_PREV"].to_list()) <= prev_ids


def test_bad_rate_is_in_a_plausible_consumer_lending_range(tables):
    lf = assign_labels_from_dpd(tables["application"].lazy(), SYNTHETIC_TARGET)
    rate = modelling_population(lf).select(pl.col("label").mean()).collect().item()
    assert 0.03 < rate < 0.30, f"bad rate {rate:.2%} is not a realistic consumer book"


def test_baseline_beats_random_out_of_time_and_ranks_monotonically(tables):
    lf = prepare(
        modelling_population(assign_labels_from_dpd(tables["application"].lazy(), SYNTHETIC_TARGET))
    )
    splits = split_by_time(lf, SYNTHETIC_SPLIT)
    assert_no_temporal_leakage(splits)

    train = splits.train.collect()
    test = splits.test.collect()
    assert train.height > 0 and test.height > 0

    model = fit(train)
    pd_hat = model.predict_pd(test)
    rep = discrimination(test["label"].to_numpy(), pd_hat)

    # A floor, not a target. Phase 2 has to beat whatever this actually scores.
    assert rep.auc > 0.65, f"baseline AUC {rep.auc:.4f} is below the usable floor"
    assert rep.ks > 0.25, f"baseline KS {rep.ks:.4f} does not separate"

    # Deciles, checked noise-aware: this fixture holds ~2.6k test rows at a
    # ~16% bad rate, so a decile carries only ~40 bads and a two-loan swing
    # between adjacent bins would flip a strict monotonicity test on noise.
    # `is_rank_ordered` only counts inversions larger than 2 standard errors.
    assert is_rank_ordered(decile_table(test["label"].to_numpy(), pd_hat))


def test_predicted_pd_is_a_valid_probability(tables):
    lf = prepare(
        modelling_population(assign_labels_from_dpd(tables["application"].lazy(), SYNTHETIC_TARGET))
    )
    splits = split_by_time(lf, SYNTHETIC_SPLIT)
    model = fit(splits.train.collect())
    p = model.predict_pd(splits.test.collect())
    assert p.min() >= 0.0 and p.max() <= 1.0


def test_schema_rejects_a_future_dated_bureau_line(tables):
    corrupted = tables["bureau"].with_columns(
        DAYS_CREDIT=pl.when(pl.int_range(pl.len()) == 0).then(5).otherwise(pl.col("DAYS_CREDIT"))
    )
    with pytest.raises(Exception, match="DAYS_CREDIT"):
        SCHEMAS["bureau"].validate(corrupted, lazy=True)


def test_schema_rejects_an_implausible_age(tables):
    corrupted = tables["application"].with_columns(
        DAYS_BIRTH=pl.when(pl.int_range(pl.len()) == 0).then(-2000).otherwise(pl.col("DAYS_BIRTH"))
    )
    with pytest.raises(Exception, match="DAYS_BIRTH|plausible_applicant_age"):
        SCHEMAS["application"].validate(corrupted, lazy=True)
