"""The target rule is the single most consequential piece of Phase 1.

If the performance window or the indeterminate band is wrong, every metric in
the project is wrong in a way no downstream test would catch.
"""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from src.config import TargetConfig
from src.ingestion.target import (
    Label,
    assign_labels_from_dpd,
    assign_labels_from_status,
    evaluation_population,
    modelling_population,
    observed_bad_rate,
    window_close_date,
)

CFG = TargetConfig(snapshot_date=dt.date(2024, 6, 30))


def _labels(dpd_rows: list[tuple[dt.date, int]]) -> list[int]:
    lf = pl.LazyFrame(
        {
            "origination_date": [d for d, _ in dpd_rows],
            "max_dpd_in_window": [v for _, v in dpd_rows],
        }
    )
    return assign_labels_from_dpd(lf, CFG).collect()["label"].to_list()


def test_90_plus_dpd_is_bad():
    assert _labels([(dt.date(2022, 1, 1), 90), (dt.date(2022, 1, 1), 271)]) == [
        Label.BAD,
        Label.BAD,
    ]


def test_30_to_89_dpd_is_indeterminate_not_good():
    assert _labels([(dt.date(2022, 1, 1), 30), (dt.date(2022, 1, 1), 89)]) == [
        Label.INDETERMINATE,
        Label.INDETERMINATE,
    ]


def test_under_30_dpd_is_good():
    assert _labels([(dt.date(2022, 1, 1), 0), (dt.date(2022, 1, 1), 29)]) == [
        Label.GOOD,
        Label.GOOD,
    ]


def test_open_performance_window_is_censored_even_when_already_bad():
    """The rule that stops recent vintages looking artificially clean."""
    # Window closes 2024-07-01, one day after the snapshot.
    assert _labels([(dt.date(2023, 7, 1), 200)]) == [Label.CENSORED]
    # One day earlier and the window has closed.
    assert _labels([(dt.date(2023, 6, 29), 200)]) == [Label.BAD]


def test_modelling_population_drops_indeterminate_and_censored():
    lf = pl.LazyFrame(
        {
            "origination_date": [dt.date(2022, 1, 1)] * 4,
            "max_dpd_in_window": [0, 45, 120, 5],
        }
    )
    labelled = assign_labels_from_dpd(lf, CFG)
    assert modelling_population(labelled).collect().height == 3
    assert evaluation_population(labelled).collect().height == 4


def test_observed_bad_rate_excludes_indeterminates():
    lf = pl.LazyFrame(
        {
            "origination_date": [dt.date(2022, 1, 1)] * 4,
            "max_dpd_in_window": [0, 45, 120, 0],
        }
    )
    # 1 bad of 3 modellable rows, not 1 of 4.
    assert observed_bad_rate(assign_labels_from_dpd(lf, CFG)) == pytest.approx(1 / 3)


@pytest.mark.parametrize(
    ("start", "months", "expected"),
    [
        (dt.date(2022, 1, 15), 12, dt.date(2023, 1, 15)),
        (dt.date(2022, 1, 31), 1, dt.date(2022, 2, 28)),  # clamps to month end
        (dt.date(2023, 12, 31), 12, dt.date(2024, 12, 31)),
        (dt.date(2020, 2, 29), 12, dt.date(2021, 2, 28)),  # leap-day origination
    ],
)
def test_window_close_date_handles_month_ends(start, months, expected):
    assert window_close_date(start, months) == expected


def test_status_mapping_treats_unknown_status_as_censored():
    lf = pl.LazyFrame(
        {
            "issue_d": [dt.date(2016, 1, 1)] * 4,
            "loan_status": ["Charged Off", "Fully Paid", "Late (31-120 days)", "Martian"],
        }
    )
    labels = assign_labels_from_status(lf, CFG).collect()["label"].to_list()
    assert labels == [Label.BAD, Label.GOOD, Label.INDETERMINATE, Label.CENSORED]
