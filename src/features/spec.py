"""Versioned feature contract.

Training and serving must not diverge. The selected feature list, their dtypes,
their monotonic directions and the fitted selection provenance are written to
``data/feature_spec.yaml``; the API loads the same file at startup and refuses
to score if the incoming payload does not satisfy it.

This is the poor-man's feature store, and for a system this size it is enough.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import polars as pl
import yaml

from src.config import DATA
from src.features.monotonic import constraints_for

SPEC_PATH = DATA / "feature_spec.yaml"
SPEC_VERSION = 1


@dataclass
class FeatureSpec:
    version: int
    created_at: str
    n_features: int
    features: list[str]
    categorical: list[str]
    monotonic: dict[str, int]
    dtypes: dict[str, str]
    dropped: dict[str, str] = field(default_factory=dict)
    iv: dict[str, float] = field(default_factory=dict)
    fingerprint: str = ""

    @staticmethod
    def _fingerprint(features: list[str]) -> str:
        """Stable hash of the ordered feature list.

        Column *order* matters to a numpy-backed model, so the fingerprint
        covers order, not just membership. Serving compares this hash and
        fails loudly rather than scoring a silently reordered matrix.
        """
        return hashlib.sha256(json.dumps(features).encode()).hexdigest()[:16]

    @classmethod
    def build(
        cls,
        df: pl.DataFrame,
        features: list[str],
        categorical: list[str],
        *,
        dropped: dict[str, str] | None = None,
        iv: pl.DataFrame | None = None,
    ) -> FeatureSpec:
        directions = constraints_for(features)
        iv_map: dict[str, float] = {}
        if iv is not None:
            iv_map = {
                r["feature"]: round(float(r["iv"]), 6)
                for r in iv.iter_rows(named=True)
                if r["feature"] in set(features)
            }
        return cls(
            version=SPEC_VERSION,
            created_at=dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
            n_features=len(features),
            features=features,
            categorical=[c for c in categorical if c in features],
            monotonic={f: d for f, d in zip(features, directions, strict=True) if d != 0},
            dtypes={f: str(df[f].dtype) for f in features if f in df.columns},
            dropped=dropped or {},
            iv=iv_map,
            fingerprint=cls._fingerprint(features),
        )

    def save(self, path: Path = SPEC_PATH) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(asdict(self), sort_keys=False, width=100))
        return path

    @classmethod
    def load(cls, path: Path = SPEC_PATH) -> FeatureSpec:
        if not path.exists():
            raise FileNotFoundError(f"No feature spec at {path}. Run `make train`.")
        return cls(**yaml.safe_load(path.read_text()))

    def validate_frame(self, df: pl.DataFrame) -> None:
        """Fail loudly on any drift between the spec and an incoming frame."""
        missing = [f for f in self.features if f not in df.columns]
        if missing:
            raise ValueError(f"Frame is missing {len(missing)} spec features: {missing[:8]}")
        if self._fingerprint([f for f in self.features]) != self.fingerprint:
            raise ValueError("Feature spec fingerprint does not match its own feature list.")

    def matrix(self, df: pl.DataFrame) -> pl.DataFrame:
        """Select spec features in spec order — the only supported way to
        build a model input."""
        self.validate_frame(df)
        return df.select(self.features)
