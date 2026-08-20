"""Central paths and run configuration.

Everything that training and serving both need to agree on lives here or in
``data/feature_spec.yaml``. Nothing else should hardcode a path.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
SYNTHETIC = DATA / "synthetic"
DOCS = ROOT / "docs"
ARTIFACTS = ROOT / "artifacts"

for _d in (RAW, PROCESSED, SYNTHETIC, ARTIFACTS):
    _d.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 20260820


class SplitConfig(BaseModel):
    """Out-of-time split boundaries, expressed as origination-date cutoffs.

    Train on the earliest vintages, validate on the middle, test on the latest.
    A random split would leak future macro conditions and repeat applicants
    across folds; every credit risk interviewer asks about this.
    """

    train_end: dt.date
    valid_end: dt.date
    test_end: dt.date
    calibration_fraction: float = Field(
        0.20,
        gt=0,
        lt=1,
        description="Slice held out of the tail of train, used only for isotonic calibration.",
    )

    def model_post_init(self, _ctx) -> None:
        if not (self.train_end < self.valid_end < self.test_end):
            raise ValueError(
                f"Split boundaries must be strictly increasing, got "
                f"{self.train_end} / {self.valid_end} / {self.test_end}"
            )


class TargetConfig(BaseModel):
    """Performance-window parameters for the default target.

    See ``docs/target_definition.md`` for the written definition.
    """

    performance_window_months: int = 12
    dpd_bad_threshold: int = 90
    dpd_indeterminate_floor: int = 30
    snapshot_date: dt.date


# Lending Club spans 2007-2018; these boundaries put roughly 60/15/25 of the
# post-2013 volume into train/valid/test while keeping whole vintages intact.
LENDING_CLUB_SPLIT = SplitConfig(
    train_end=dt.date(2015, 12, 31),
    valid_end=dt.date(2016, 6, 30),
    test_end=dt.date(2017, 12, 31),
)

LENDING_CLUB_TARGET = TargetConfig(snapshot_date=dt.date(2018, 12, 31))

# The synthetic generator emits application dates on this calendar so the
# out-of-time machinery is exercisable before the Kaggle data lands.
SYNTHETIC_SPLIT = SplitConfig(
    train_end=dt.date(2021, 12, 31),
    valid_end=dt.date(2022, 6, 30),
    test_end=dt.date(2023, 6, 30),
)

SYNTHETIC_TARGET = TargetConfig(snapshot_date=dt.date(2024, 6, 30))
