"""Out-of-time splitting.

Random splitting is prohibited in this project. Two reasons, both of which get
raised in credit risk interviews:

1. Repeat applicants appear in multiple folds, so the model memorises borrowers
   rather than learning risk.
2. Macro conditions (rates, unemployment, the lender's own policy changes) are
   shared across folds, so validation performance is optimistic about how the
   model behaves on next quarter's through-the-door population.

The calibration slice is carved from the *tail* of train rather than sampled at
random, so isotonic regression is fitted on the vintages closest to validation.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import polars as pl

from src.config import SplitConfig


@dataclass(frozen=True)
class Splits:
    train: pl.LazyFrame
    calibration: pl.LazyFrame
    valid: pl.LazyFrame
    test: pl.LazyFrame

    def counts(self) -> dict[str, int]:
        return {
            name: int(getattr(self, name).select(pl.len()).collect().item())
            for name in ("train", "calibration", "valid", "test")
        }


def split_by_time(
    lf: pl.LazyFrame,
    cfg: SplitConfig,
    *,
    date_col: str = "origination_date",
) -> Splits:
    """Partition on origination date into train / calibration / valid / test."""
    dev = lf.filter(pl.col(date_col) <= pl.lit(cfg.train_end))

    # Emptiness must be checked before the quantile: polars panics rather
    # than returning null when asked for a quantile of an empty Date series.
    if dev.select(pl.len()).collect().item() == 0:
        raise ValueError(f"No rows on or before train_end={cfg.train_end}; check {date_col} range.")

    # Quantile cut on the date itself keeps whole days together and is stable
    # regardless of how volume is distributed across the development period.
    cut = dev.select(pl.col(date_col).quantile(1.0 - cfg.calibration_fraction, "nearest")).collect()
    if cut.item() is None:
        raise ValueError(f"All {date_col} values are null on or before {cfg.train_end}.")
    cal_start: dt.date = cut.item()

    return Splits(
        train=dev.filter(pl.col(date_col) < pl.lit(cal_start)),
        calibration=dev.filter(pl.col(date_col) >= pl.lit(cal_start)),
        valid=lf.filter(
            (pl.col(date_col) > pl.lit(cfg.train_end)) & (pl.col(date_col) <= pl.lit(cfg.valid_end))
        ),
        test=lf.filter(
            (pl.col(date_col) > pl.lit(cfg.valid_end)) & (pl.col(date_col) <= pl.lit(cfg.test_end))
        ),
    )


def assert_no_temporal_leakage(splits: Splits, *, date_col: str = "origination_date") -> None:
    """Fail loudly if any split's date range overlaps a later split's."""
    ordered = ["train", "calibration", "valid", "test"]
    prev_max: dt.date | None = None
    prev_name = ""
    for name in ordered:
        frame = getattr(splits, name)
        bounds = frame.select(
            pl.col(date_col).min().alias("lo"), pl.col(date_col).max().alias("hi")
        ).collect()
        lo, hi = bounds.item(0, "lo"), bounds.item(0, "hi")
        if lo is None:
            continue
        if prev_max is not None and lo < prev_max:
            raise AssertionError(
                f"Temporal leakage: {name} starts {lo} but {prev_name} runs to {prev_max}"
            )
        prev_max, prev_name = hi, name


def vintage_column(lf: pl.LazyFrame, *, date_col: str = "origination_date") -> pl.LazyFrame:
    """Attach a YYYY-Qn origination cohort, used for vintage analysis."""
    return lf.with_columns(
        vintage=pl.col(date_col).dt.year().cast(pl.Utf8)
        + pl.lit("-Q")
        + pl.col(date_col).dt.quarter().cast(pl.Utf8)
    )
