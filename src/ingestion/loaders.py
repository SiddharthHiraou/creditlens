"""Lazy Polars loaders with a single switch between real and synthetic data.

Resolution order for each table:

1. ``data/raw/<real csv name>``  -- the Kaggle download, if present
2. ``data/synthetic/<name>.parquet`` -- the generator's output

Everything downstream calls :func:`scan` and never touches a path, so dropping
the real CSVs into ``data/raw/`` switches the whole pipeline over with no code
change. Scans are lazy: nothing is read until a ``.collect()``, which is what
keeps the 5.5M-row bureau_balance join affordable.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

import polars as pl

from src.config import RAW, SYNTHETIC
from src.ingestion.schemas import SCHEMAS


class Source(StrEnum):
    AUTO = "auto"
    SYNTHETIC = "synthetic"
    RAW = "raw"


# Home Credit ships the applicant table split into train/test; only the
# labelled half is usable here.
HOME_CREDIT_FILES = {
    "application": "application_train.csv",
    "bureau": "bureau.csv",
    "bureau_balance": "bureau_balance.csv",
    "previous_application": "previous_application.csv",
    "installments_payments": "installments_payments.csv",
    "credit_card_balance": "credit_card_balance.csv",
    "POS_CASH_balance": "POS_CASH_balance.csv",
}


class MissingTableError(FileNotFoundError):
    pass


def resolve(name: str, source: Source = Source.AUTO) -> tuple[Path, Source]:
    """Return the path that will be read and which source it came from."""
    if name not in HOME_CREDIT_FILES:
        raise KeyError(f"Unknown table {name!r}; expected one of {sorted(HOME_CREDIT_FILES)}")

    raw_path = RAW / HOME_CREDIT_FILES[name]
    syn_path = SYNTHETIC / f"{name}.parquet"

    if source is Source.RAW:
        if not raw_path.exists():
            raise MissingTableError(f"{raw_path} not found. See README 'Getting the data'.")
        return raw_path, Source.RAW
    if source is Source.SYNTHETIC:
        if not syn_path.exists():
            raise MissingTableError(f"{syn_path} not found. Run `make data`.")
        return syn_path, Source.SYNTHETIC

    if raw_path.exists():
        return raw_path, Source.RAW
    if syn_path.exists():
        return syn_path, Source.SYNTHETIC
    raise MissingTableError(
        f"No data for {name!r}. Run `make data` for synthetic, or place "
        f"{HOME_CREDIT_FILES[name]} in {RAW}."
    )


def scan(name: str, source: Source = Source.AUTO) -> pl.LazyFrame:
    """Lazily scan a table, normalising the real CSVs onto the internal shape."""
    path, resolved = resolve(name, source)
    if resolved is Source.SYNTHETIC:
        return pl.scan_parquet(path)

    lf = pl.scan_csv(path, infer_schema_length=20_000, null_values=["", "XNA", "NA"])
    if name == "application":
        lf = _normalise_home_credit_application(lf)
    return lf


def _normalise_home_credit_application(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Bridge the real application table onto this project's contract.

    Home Credit has no application date and no DPD depth -- only a prebuilt
    binary ``TARGET``. Both gaps are filled explicitly here rather than
    silently, so nothing downstream mistakes a placeholder for an observation.
    """
    cols = lf.collect_schema().names()
    out = lf
    if "origination_date" not in cols:
        # Sentinel, not an estimate. Out-of-time work runs on Lending Club;
        # see docs/target_definition.md for why.
        out = out.with_columns(origination_date=pl.lit(None, dtype=pl.Date))
    if "max_dpd_in_window" not in cols and "TARGET" in cols:
        # Map the vendor label onto the DPD scale the target rule expects.
        out = out.with_columns(
            max_dpd_in_window=pl.when(pl.col("TARGET") == 1).then(90).otherwise(0).cast(pl.Int32)
        )
    return out


def load(name: str, source: Source = Source.AUTO, *, validate: bool = True) -> pl.LazyFrame:
    """Scan a table and, by default, enforce its Pandera contract."""
    lf = scan(name, source)
    if validate:
        SCHEMAS[name].validate(lf.collect(), lazy=True)
    return lf


def load_all(source: Source = Source.AUTO, *, validate: bool = True) -> dict[str, pl.LazyFrame]:
    return {name: load(name, source, validate=validate) for name in HOME_CREDIT_FILES}


def describe_sources() -> pl.DataFrame:
    """What the pipeline will actually read, for the run log."""
    rows = []
    for name in HOME_CREDIT_FILES:
        try:
            path, src = resolve(name)
            rows.append({"table": name, "source": str(src), "path": path.name, "found": True})
        except MissingTableError:
            rows.append({"table": name, "source": "-", "path": "-", "found": False})
    return pl.DataFrame(rows)
