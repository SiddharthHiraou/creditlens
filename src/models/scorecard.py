"""Logistic scorecard on WOE-binned features, with standard PDO scaling.

This track is non-negotiable even though the GBDT will beat it. It is what
credit shops actually run, what regulators are comfortable validating, and the
reference the challenger models are argued against. It also produces a printable
points table, which no GBDT can.

Scaling follows the industry convention:

    score = offset + factor * ln(odds_good_to_bad)
    factor = PDO / ln(2)
    offset = base_score - factor * ln(base_odds)

With base 600 at 50:1 odds and PDO 20, every 20 points doubles the odds of a
good outcome. Points per attribute are then:

    points_bin = -(woe_bin * beta_feature + intercept / n_features) * factor
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression

from src.config import RANDOM_SEED
from src.features.woe import WoeBinning, fit_woe_set, transform_woe

BASE_SCORE = 600.0
BASE_ODDS = 50.0
PDO = 20.0


@dataclass
class Scorecard:
    binnings: dict[str, WoeBinning]
    model: LogisticRegression
    feature_names: list[str]
    factor: float = field(init=False)
    offset: float = field(init=False)

    def __post_init__(self) -> None:
        self.factor = PDO / np.log(2)
        self.offset = BASE_SCORE - self.factor * np.log(BASE_ODDS)

    def _woe_matrix(self, df: pl.DataFrame) -> np.ndarray:
        woe = transform_woe(df, self.binnings)
        return woe.select([f"woe_{f}" for f in self.feature_names]).to_numpy().astype(float)

    def predict_pd(self, df: pl.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self._woe_matrix(df))[:, 1]

    def score(self, df: pl.DataFrame) -> np.ndarray:
        """Map PD onto the 300-850 credit score scale.

        Higher score = lower risk, matching every consumer scoring convention.
        """
        pd_hat = np.clip(self.predict_pd(df), 1e-6, 1 - 1e-6)
        odds_good = (1 - pd_hat) / pd_hat
        return np.clip(self.offset + self.factor * np.log(odds_good), 300, 850)

    def points_table(self) -> pl.DataFrame:
        """The printable scorecard: points awarded per feature bin."""
        rows = []
        n = max(len(self.feature_names), 1)
        intercept_share = float(self.model.intercept_[0]) / n
        for i, feature in enumerate(self.feature_names):
            beta = float(self.model.coef_[0][i])
            binning = self.binnings[feature]
            for bin_label, woe in sorted(binning.mapping.items()):
                points = -(woe * beta + intercept_share) * self.factor
                rows.append(
                    {
                        "feature": feature,
                        "bin": bin_label,
                        "woe": round(woe, 5),
                        "beta": round(beta, 5),
                        "points": round(points, 2),
                    }
                )
        return pl.DataFrame(rows).sort(["feature", "bin"])

    def feature_contributions(self) -> pl.DataFrame:
        """Points range per feature — how much each can move a decision."""
        return (
            self.points_table()
            .group_by("feature")
            .agg(
                min_points=pl.col("points").min(),
                max_points=pl.col("points").max(),
                n_bins=pl.len(),
            )
            .with_columns(points_range=pl.col("max_points") - pl.col("min_points"))
            .sort("points_range", descending=True)
        )


def fit_scorecard(
    train: pl.DataFrame,
    features: list[str],
    *,
    label_col: str = "label",
    n_bins: int = 10,
    c: float = 1.0,
) -> Scorecard:
    """Fit WOE binnings on train, then a regularised logistic on the WOE matrix."""
    binnings = fit_woe_set(train, features, label_col=label_col, n_bins=n_bins)
    usable = [f for f in features if f in binnings]

    woe = transform_woe(train, binnings).select([f"woe_{f}" for f in usable])
    x = woe.to_numpy().astype(float)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    y = train[label_col].to_numpy().astype(int)

    model = LogisticRegression(
        C=c,
        l1_ratio=0.0,
        solver="lbfgs",
        max_iter=3000,
        class_weight="balanced",
        random_state=RANDOM_SEED,
    )
    model.fit(x, y)
    return Scorecard(binnings=binnings, model=model, feature_names=usable)
