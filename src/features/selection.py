"""Feature selection: IV screen, correlation pruning, null importance.

Selection runs on **training data only**. Screening on the full frame — even
just to compute a correlation matrix — leaks out-of-time information into the
choice of features, which is a subtle enough form of leakage that it usually
survives review.

Every survivor carries a recorded reason, written into the feature spec.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

from src.config import RANDOM_SEED
from src.features.woe import information_values


@dataclass
class SelectionReport:
    kept: list[str]
    dropped: dict[str, str] = field(default_factory=dict)
    iv_table: pl.DataFrame | None = None
    importance_table: pl.DataFrame | None = None

    def summary(self) -> pl.DataFrame:
        reasons = (
            pl.DataFrame({"feature": list(self.dropped), "reason": list(self.dropped.values())})
            if self.dropped
            else pl.DataFrame(schema={"feature": pl.Utf8, "reason": pl.Utf8})
        )
        return (
            reasons.group_by("reason")
            .agg(pl.len().alias("n_dropped"))
            .sort("n_dropped", descending=True)
        )


def drop_degenerate(
    df: pl.DataFrame, features: list[str], *, max_null_share: float = 0.95
) -> tuple[list[str], dict[str, str]]:
    """Remove constants and near-empty columns before anything expensive runs."""
    kept, dropped = [], {}
    for f in features:
        s = df[f]
        null_share = s.null_count() / max(df.height, 1)
        if null_share > max_null_share:
            dropped[f] = f"null_share>{max_null_share}"
        elif s.n_unique() <= 1:
            dropped[f] = "constant"
        else:
            kept.append(f)
    return kept, dropped


def drop_low_iv(
    df: pl.DataFrame, features: list[str], *, min_iv: float = 0.02, label_col: str = "label"
) -> tuple[list[str], dict[str, str], pl.DataFrame]:
    """Drop features below the conventional IV floor of 0.02."""
    iv = information_values(df, features, label_col=label_col)
    weak = iv.filter(pl.col("iv") < min_iv)["feature"].to_list()
    kept = [f for f in features if f not in set(weak)]
    return kept, {f: f"iv<{min_iv}" for f in weak}, iv


def prune_correlated(
    df: pl.DataFrame,
    features: list[str],
    ranking: dict[str, float],
    *,
    threshold: float = 0.95,
) -> tuple[list[str], dict[str, str]]:
    """Drop the weaker of any pair correlated above ``threshold``.

    "Weaker" is decided by ``ranking`` (IV), so the survivor of each collinear
    pair is the one that carries more univariate signal. Many aggregations here
    are near-duplicates by construction -- ``BURO_debt_total`` and
    ``BURO_debt_mean`` move together -- and keeping both adds variance to the
    scorecard's coefficients without adding information.
    """
    numeric = [f for f in features if df[f].dtype.is_numeric()]
    if len(numeric) < 2:
        return features, {}

    mat = df.select(numeric).fill_null(strategy="mean").to_numpy().astype(float)
    mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
    # Guard zero-variance columns, which would make corrcoef emit nan.
    std = mat.std(axis=0)
    ok = std > 1e-12
    corr = np.zeros((len(numeric), len(numeric)))
    if ok.sum() >= 2:
        sub = np.corrcoef(mat[:, ok], rowvar=False)
        idx = np.where(ok)[0]
        corr[np.ix_(idx, idx)] = np.nan_to_num(sub, nan=0.0)

    order = sorted(range(len(numeric)), key=lambda i: -ranking.get(numeric[i], 0.0))
    dropped: dict[str, str] = {}
    survivors: list[int] = []
    for i in order:
        clash = next(
            (numeric[j] for j in survivors if abs(corr[i, j]) > threshold),
            None,
        )
        if clash is not None:
            dropped[numeric[i]] = f"corr>{threshold} with {clash}"
        else:
            survivors.append(i)

    keep = {numeric[i] for i in survivors}
    return [f for f in features if f not in dropped or f in keep], dropped


def null_importance(
    df: pl.DataFrame,
    features: list[str],
    *,
    label_col: str = "label",
    n_runs: int = 5,
    seed: int = RANDOM_SEED,
) -> pl.DataFrame:
    """Compare real gain against gain earned on a shuffled target.

    A feature that scores as highly against a randomised label as against the
    real one is fitting noise -- typically a high-cardinality or mostly-null
    column. This catches what a plain importance ranking cannot: importance is
    relative, so a useless feature still gets a nonzero score.
    """
    import lightgbm as lgb

    x = df.select(features).to_numpy().astype(np.float32)
    x = np.nan_to_num(x, nan=np.nan)  # LightGBM handles nan natively
    y = df[label_col].to_numpy().astype(int)

    params = dict(
        objective="binary",
        n_estimators=120,
        learning_rate=0.1,
        num_leaves=31,
        verbose=-1,
        n_jobs=-1,
        seed=seed,
    )

    real = lgb.LGBMClassifier(**params).fit(x, y).booster_.feature_importance("gain")

    rng = np.random.default_rng(seed)
    null_runs = []
    for _ in range(n_runs):
        shuffled = rng.permutation(y)
        null_runs.append(
            lgb.LGBMClassifier(**params).fit(x, shuffled).booster_.feature_importance("gain")
        )
    null_mean = np.mean(null_runs, axis=0)

    return pl.DataFrame(
        {
            "feature": features,
            "gain_real": real.astype(float),
            "gain_null_mean": null_mean.astype(float),
            # Log ratio: how many times the noise floor does this feature clear?
            "score": np.log1p(real) - np.log1p(null_mean),
        }
    ).sort("score", descending=True)


def select(
    train: pl.DataFrame,
    features: list[str],
    *,
    label_col: str = "label",
    min_iv: float = 0.02,
    corr_threshold: float = 0.95,
    use_null_importance: bool = True,
    min_null_importance: float = 0.0,
) -> SelectionReport:
    """Full selection pipeline. Fit on train only."""
    dropped: dict[str, str] = {}

    kept, d = drop_degenerate(train, features)
    dropped.update(d)

    kept, d, iv = drop_low_iv(train, kept, min_iv=min_iv, label_col=label_col)
    dropped.update(d)

    ranking = dict(zip(iv["feature"].to_list(), iv["iv"].to_list(), strict=True))
    kept, d = prune_correlated(train, kept, ranking, threshold=corr_threshold)
    dropped.update(d)

    imp = None
    if use_null_importance:
        numeric = [f for f in kept if train[f].dtype.is_numeric()]
        if numeric:
            imp = null_importance(train, numeric, label_col=label_col)
            noise = imp.filter(pl.col("score") <= min_null_importance)["feature"].to_list()
            for f in noise:
                dropped[f] = "no lift over shuffled-target baseline"
            kept = [f for f in kept if f not in set(noise)]

    return SelectionReport(kept=kept, dropped=dropped, iv_table=iv, importance_table=imp)
