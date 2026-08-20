"""Discrimination and calibration metrics for a binary default model.

Accuracy is deliberately absent. With an 8-10% bad rate, predicting "everyone
is good" scores 90%+ and is worthless, so it is not reported anywhere in this
project.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


@dataclass(frozen=True)
class DiscriminationReport:
    n: int
    n_bad: int
    bad_rate: float
    auc: float
    gini: float
    ks: float
    ks_threshold: float
    pr_auc: float
    brier: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)

    def render(self) -> str:
        return (
            f"n={self.n:,}  bad={self.n_bad:,} ({self.bad_rate:.2%})\n"
            f"AUC   {self.auc:.4f}\n"
            f"Gini  {self.gini:.4f}\n"
            f"KS    {self.ks:.4f} @ p={self.ks_threshold:.4f}\n"
            f"PR-AUC {self.pr_auc:.4f}   (baseline {self.bad_rate:.4f})\n"
            f"Brier {self.brier:.5f}"
        )


def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    """Kolmogorov-Smirnov separation and the score at which it is achieved.

    KS is the maximum vertical gap between the cumulative distributions of bads
    and goods. Credit teams quote it alongside Gini; below ~0.25 a scorecard is
    generally considered not to separate.
    """
    order = np.argsort(y_score, kind="mergesort")
    y, s = y_true[order], y_score[order]
    n_bad, n_good = y.sum(), (1 - y).sum()
    if n_bad == 0 or n_good == 0:
        return float("nan"), float("nan")
    cum_bad = np.cumsum(y) / n_bad
    cum_good = np.cumsum(1 - y) / n_good
    gaps = np.abs(cum_bad - cum_good)
    i = int(np.argmax(gaps))
    return float(gaps[i]), float(s[i])


def discrimination(y_true: np.ndarray, y_score: np.ndarray) -> DiscriminationReport:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    auc = float(roc_auc_score(y_true, y_score))
    ks, ks_at = ks_statistic(y_true, y_score)
    return DiscriminationReport(
        n=int(y_true.size),
        n_bad=int(y_true.sum()),
        bad_rate=float(y_true.mean()),
        auc=auc,
        gini=2 * auc - 1,
        ks=ks,
        ks_threshold=ks_at,
        pr_auc=float(average_precision_score(y_true, y_score)),
        brier=float(brier_score_loss(y_true, y_score)),
    )


def decile_table(y_true: np.ndarray, y_score: np.ndarray, *, n_bins: int = 10) -> pl.DataFrame:
    """Rank-ordering table. Bad rate must fall monotonically as score improves.

    Bin 1 is the riskiest decile by predicted PD. A model whose observed bad
    rate does not decrease across bins is not shippable regardless of AUC.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    ranks = np.argsort(np.argsort(-y_score, kind="mergesort"), kind="mergesort")
    bins = np.minimum((ranks * n_bins) // max(len(y_score), 1), n_bins - 1) + 1

    df = pl.DataFrame({"decile": bins.astype(np.int32), "y": y_true, "pd": y_score})
    out = (
        df.group_by("decile")
        .agg(
            n=pl.len(),
            n_bad=pl.col("y").sum(),
            bad_rate=pl.col("y").mean(),
            mean_pd=pl.col("pd").mean(),
            min_pd=pl.col("pd").min(),
            max_pd=pl.col("pd").max(),
        )
        .sort("decile")
    )
    return out.with_columns(
        cum_bad_capture=pl.col("n_bad").cum_sum() / pl.col("n_bad").sum(),
        lift=pl.col("bad_rate") / (pl.col("n_bad").sum() / pl.col("n").sum()),
    )


def is_strictly_rank_ordered(decile_df: pl.DataFrame) -> bool:
    """True when observed bad rate never rises between adjacent bins.

    Strict and unforgiving: a single loan moving between bins can flip it. Use
    :func:`is_rank_ordered` for the shippability decision and this one only
    when the population is large enough that sampling noise is negligible.
    """
    rates = decile_df.sort("decile")["bad_rate"].to_list()
    return all(a >= b for a, b in zip(rates, rates[1:], strict=False))


def rank_order_violations(decile_df: pl.DataFrame, *, z: float = 2.0) -> pl.DataFrame:
    """Adjacent-bin inversions that exceed sampling noise.

    Each bin's bad rate is a binomial proportion with standard error
    ``sqrt(p(1-p)/n)``. An inversion between adjacent bins is only evidence of
    broken rank ordering if it is larger than ``z`` standard errors of the
    difference; below that it is indistinguishable from noise. At ``z=2`` this
    is roughly a 95% test.

    Returns one row per inversion, empty when the model rank-orders.
    """
    d = decile_df.sort("decile")
    rates = d["bad_rate"].to_list()
    ns = d["n"].to_list()
    bins = d["decile"].to_list()

    rows = []
    for i in range(len(rates) - 1):
        lo, hi = rates[i], rates[i + 1]
        if hi <= lo:
            continue  # correctly ordered
        se = float(np.sqrt(lo * (1 - lo) / max(ns[i], 1) + hi * (1 - hi) / max(ns[i + 1], 1)))
        excess = (hi - lo) - z * se
        if excess > 0:
            rows.append(
                {
                    "from_decile": bins[i],
                    "to_decile": bins[i + 1],
                    "bad_rate_from": lo,
                    "bad_rate_to": hi,
                    "gap": hi - lo,
                    "z_threshold": z * se,
                }
            )
    schema = {
        "from_decile": pl.Int32,
        "to_decile": pl.Int32,
        "bad_rate_from": pl.Float64,
        "bad_rate_to": pl.Float64,
        "gap": pl.Float64,
        "z_threshold": pl.Float64,
    }
    return pl.DataFrame(rows, schema=schema)


def is_rank_ordered(decile_df: pl.DataFrame, *, z: float = 2.0) -> bool:
    """Shippability check: no inversion larger than sampling noise.

    A model that fails this does not rank-order and is not shippable regardless
    of its AUC, because a cutoff drawn anywhere near the inversion would admit
    worse loans than the band below it.
    """
    return rank_order_violations(decile_df, z=z).height == 0
