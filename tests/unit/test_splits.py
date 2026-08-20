"""Out-of-time splitting must never let a later vintage into an earlier fold."""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from src.config import SplitConfig
from src.ingestion.splits import (
    Splits,
    assert_no_temporal_leakage,
    split_by_time,
    vintage_column,
)

CFG = SplitConfig(
    train_end=dt.date(2021, 12, 31),
    valid_end=dt.date(2022, 6, 30),
    test_end=dt.date(2023, 6, 30),
)


def _frame(n: int = 4000) -> pl.LazyFrame:
    start = dt.date(2020, 1, 1)
    return pl.LazyFrame(
        {"origination_date": [start + dt.timedelta(days=i % 1200) for i in range(n)]}
    )


def test_splits_are_disjoint_and_ordered_in_time():
    splits = split_by_time(_frame(), CFG)
    assert_no_temporal_leakage(splits)  # raises on overlap

    bounds = {}
    for name in ("train", "calibration", "valid", "test"):
        df = getattr(splits, name).collect()
        bounds[name] = (df["origination_date"].min(), df["origination_date"].max())

    assert bounds["train"][1] < bounds["calibration"][0]
    assert bounds["calibration"][1] <= CFG.train_end < bounds["valid"][0]
    assert bounds["valid"][1] <= CFG.valid_end < bounds["test"][0]
    assert bounds["test"][1] <= CFG.test_end


def test_every_row_in_range_lands_in_exactly_one_split():
    lf = _frame()
    total_in_range = (
        lf.filter(pl.col("origination_date") <= pl.lit(CFG.test_end))
        .select(pl.len())
        .collect()
        .item()
    )
    assert sum(split_by_time(lf, CFG).counts().values()) == total_in_range


def test_calibration_is_carved_from_the_tail_of_train_not_sampled():
    splits = split_by_time(_frame(), CFG)
    train_max = splits.train.collect()["origination_date"].max()
    cal_min = splits.calibration.collect()["origination_date"].min()
    assert cal_min > train_max


def test_leakage_detector_actually_fires():
    overlap = pl.LazyFrame({"origination_date": [dt.date(2021, 1, 1), dt.date(2023, 1, 1)]})
    early = pl.LazyFrame({"origination_date": [dt.date(2022, 1, 1)]})
    bad = Splits(train=overlap, calibration=early, valid=early, test=early)
    with pytest.raises(AssertionError, match="Temporal leakage"):
        assert_no_temporal_leakage(bad)


def test_split_config_rejects_out_of_order_boundaries():
    with pytest.raises(ValueError, match="strictly increasing"):
        SplitConfig(
            train_end=dt.date(2022, 1, 1),
            valid_end=dt.date(2021, 1, 1),
            test_end=dt.date(2023, 1, 1),
        )


def test_empty_development_period_raises_clearly():
    late = pl.LazyFrame({"origination_date": [dt.date(2023, 1, 1)]})
    with pytest.raises(ValueError, match="No rows on or before"):
        split_by_time(late, CFG)


def test_vintage_column_formats_year_quarter():
    lf = pl.LazyFrame({"origination_date": [dt.date(2022, 2, 3), dt.date(2022, 11, 30)]})
    assert vintage_column(lf).collect()["vintage"].to_list() == ["2022-Q1", "2022-Q4"]
