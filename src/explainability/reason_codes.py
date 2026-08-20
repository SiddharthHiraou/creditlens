"""Turn SHAP contributions into ECOA-compliant adverse action reasons.

The requirement being satisfied: under ECOA and Regulation B a declined
applicant must receive the *specific principal reasons* for the denial, not a
generic statement. This module produces them from the model's own attributions
so the disclosure reflects what actually drove the decision.

The mapping lives in ``reason_codes.yaml`` and is versioned, because the API
logs which version produced each disclosure.

Three behaviours worth stating, all of them compliance-driven:

* **Only positive contributions count.** A reason for denial must be something
  that pushed the applicant *toward* default. Features that helped are not
  reasons for a decline.
* **Deduplicated at family level.** Four flavours of high debt-to-income is one
  reason. Returning it four times would deprive the applicant of three real
  ones, which defeats the purpose of the disclosure.
* **Protected attributes never surface.** Suppressed features are dropped
  entirely, and features embedding a protected attribute (the age-bearing
  ``EXT_mean_x_age`` interaction) are mapped to a family that describes what
  the applicant can act on.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import polars as pl
import yaml

MAPPING_PATH = Path(__file__).with_name("reason_codes.yaml")

# Regulation B expects the principal reasons; four is the industry norm and
# what the major bureaus' adverse action templates carry.
DEFAULT_TOP_N = 4


@dataclass(frozen=True)
class ReasonCode:
    rank: int
    family: str
    label: str
    phrase: str
    actionable: bool
    contribution: float
    driving_features: tuple[str, ...]

    def as_dict(self) -> dict:
        return {
            "rank": self.rank,
            "family": self.family,
            "label": self.label,
            "phrase": self.phrase,
            "actionable": self.actionable,
            "contribution": round(self.contribution, 6),
            "driving_features": list(self.driving_features),
        }


@dataclass(frozen=True)
class ReasonCodeMapper:
    version: int
    families: dict[str, dict]
    feature_family: dict[str, str]
    suppressed: frozenset[str]

    @classmethod
    def load(cls, path: Path = MAPPING_PATH) -> ReasonCodeMapper:
        raw = yaml.safe_load(path.read_text())
        return cls(
            version=int(raw["version"]),
            families=raw["families"],
            feature_family=raw["features"],
            suppressed=frozenset(raw.get("suppress", [])),
        )

    def unmapped(self, features: list[str]) -> list[str]:
        """Model features with no family and no suppression.

        A new feature reaching an applicant with no phrase is a compliance
        failure, so this is asserted in the test suite rather than logged.
        """
        return [f for f in features if f not in self.feature_family and f not in self.suppressed]

    def explain(
        self,
        feature_names: list[str],
        shap_values: np.ndarray,
        *,
        top_n: int = DEFAULT_TOP_N,
    ) -> list[ReasonCode]:
        """Rank reasons for a single application, most influential first."""
        v = np.asarray(shap_values, dtype=float).ravel()
        if v.size != len(feature_names):
            raise ValueError(
                f"Got {v.size} SHAP values for {len(feature_names)} features; "
                "the matrix and the model are out of alignment."
            )

        totals: dict[str, float] = {}
        drivers: dict[str, list[tuple[str, float]]] = {}
        for name, contribution in zip(feature_names, v, strict=True):
            if contribution <= 0:
                continue  # only reasons that pushed toward default
            if name in self.suppressed:
                continue
            family = self.feature_family.get(name)
            if family is None:
                continue
            totals[family] = totals.get(family, 0.0) + float(contribution)
            drivers.setdefault(family, []).append((name, float(contribution)))

        ranked = sorted(totals.items(), key=lambda kv: -kv[1])[:top_n]
        out: list[ReasonCode] = []
        for i, (family, total) in enumerate(ranked, start=1):
            meta = self.families[family]
            top_features = tuple(f for f, _ in sorted(drivers[family], key=lambda kv: -kv[1])[:3])
            out.append(
                ReasonCode(
                    rank=i,
                    family=family,
                    label=meta["label"],
                    phrase=meta["phrase"].strip(),
                    actionable=bool(meta.get("actionable", False)),
                    contribution=total,
                    driving_features=top_features,
                )
            )
        return out

    def explain_frame(
        self, feature_names: list[str], shap_values: np.ndarray, *, top_n: int = DEFAULT_TOP_N
    ) -> pl.DataFrame:
        """Reason codes for a batch, one row per (application, rank)."""
        rows = []
        for i, v in enumerate(np.asarray(shap_values, dtype=float)):
            for rc in self.explain(feature_names, v, top_n=top_n):
                rows.append({"row": i, **rc.as_dict()})
        return pl.DataFrame(
            rows,
            schema={
                "row": pl.Int64,
                "rank": pl.Int64,
                "family": pl.Utf8,
                "label": pl.Utf8,
                "phrase": pl.Utf8,
                "actionable": pl.Boolean,
                "contribution": pl.Float64,
                "driving_features": pl.List(pl.Utf8),
            },
        )


@lru_cache(maxsize=1)
def default_mapper() -> ReasonCodeMapper:
    return ReasonCodeMapper.load()
