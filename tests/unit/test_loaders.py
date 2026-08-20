"""Source resolution: the switch that lets real Kaggle CSVs replace synthetic
parquet with no code change. If this silently picks the wrong source, every
downstream number is computed on the wrong data.
"""

from __future__ import annotations

import polars as pl
import pytest

from src.ingestion import loaders
from src.ingestion.loaders import (
    HOME_CREDIT_FILES,
    MissingTableError,
    Source,
    describe_sources,
    resolve,
    scan,
)


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    raw, syn = tmp_path / "raw", tmp_path / "synthetic"
    raw.mkdir()
    syn.mkdir()
    monkeypatch.setattr(loaders, "RAW", raw)
    monkeypatch.setattr(loaders, "SYNTHETIC", syn)
    return raw, syn


def test_raw_csv_wins_over_synthetic_parquet_when_both_exist(isolated):
    raw, syn = isolated
    (raw / HOME_CREDIT_FILES["bureau"]).write_text("SK_ID_CURR\n1\n")
    pl.DataFrame({"SK_ID_CURR": [1]}).write_parquet(syn / "bureau.parquet")

    path, source = resolve("bureau")
    assert source is Source.RAW
    assert path.name == HOME_CREDIT_FILES["bureau"]


def test_falls_back_to_synthetic_when_no_raw_file(isolated):
    _, syn = isolated
    pl.DataFrame({"SK_ID_CURR": [1]}).write_parquet(syn / "bureau.parquet")
    assert resolve("bureau")[1] is Source.SYNTHETIC


def test_explicit_source_does_not_silently_fall_back(isolated):
    _, syn = isolated
    pl.DataFrame({"SK_ID_CURR": [1]}).write_parquet(syn / "bureau.parquet")
    # Synthetic exists, but RAW was demanded: fail loudly rather than substitute.
    with pytest.raises(MissingTableError, match="not found"):
        resolve("bureau", Source.RAW)


def test_missing_everywhere_names_the_remedy(isolated):
    with pytest.raises(MissingTableError, match="make data"):
        resolve("bureau")


def test_unknown_table_name_is_rejected():
    with pytest.raises(KeyError, match="Unknown table"):
        resolve("not_a_table")


def test_real_application_csv_is_normalised_onto_the_internal_contract(isolated):
    """Home Credit has no application date and no DPD depth; both gaps must be
    filled explicitly so nothing downstream mistakes them for observations."""
    raw, _ = isolated
    (raw / HOME_CREDIT_FILES["application"]).write_text(
        "SK_ID_CURR,TARGET,AMT_CREDIT\n1,1,1000\n2,0,2000\n"
    )
    df = scan("application").collect()

    assert df["origination_date"].dtype == pl.Date
    assert df["origination_date"].null_count() == df.height  # sentinel, not a guess
    assert df["max_dpd_in_window"].to_list() == [90, 0]


def test_synthetic_scan_is_not_rewritten(isolated):
    _, syn = isolated
    pl.DataFrame({"SK_ID_CURR": [1], "origination_date": [None]}).write_parquet(
        syn / "application.parquet"
    )
    assert scan("application").collect().columns == ["SK_ID_CURR", "origination_date"]


def test_describe_sources_reports_every_table(isolated):
    _, syn = isolated
    pl.DataFrame({"SK_ID_CURR": [1]}).write_parquet(syn / "bureau.parquet")
    df = describe_sources()
    assert df.height == len(HOME_CREDIT_FILES)
    assert df.filter(pl.col("table") == "bureau")["found"].item() is True
    assert df.filter(pl.col("table") == "application")["found"].item() is False
