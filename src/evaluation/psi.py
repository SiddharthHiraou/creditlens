"""Population Stability Index and Characteristic Stability Index.

PSI is the metric credit risk teams actually page on. It answers "has the
population moved since the model was built", which is a different and often
earlier question than "has performance degraded" -- performance requires
outcomes, and outcomes arrive 12 months late.

    PSI = sum over bins of (actual_share - expected_share) * ln(actual/expected)

Industry thresholds, used throughout this project:

| PSI | reading | action |
|---|---|---|
| < 0.10 | stable | none |
| 0.10 - 0.25 | moderate shift | investigate |
| > 0.25 | significant shift | alarm; consider retraining |

CSI is the same computation applied per input feature rather than to the score,
which is how you find *which* input moved once PSI on the score fires.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

PSI_INVESTIGATE = 0.10
PSI_ALARM = 0.25

# Floor on a bin's share, so an empty bin yields a large-but-finite
# contribution instead of infinity.
_EPS = 1e-6


@dataclass(frozen=True)
class PsiResult:
    psi: float
    table: pl.DataFrame

    @property
    def verdict(self) -> str:
        if self.psi < PSI_INVESTIGATE:
            return "stable"
        if self.psi < PSI_ALARM:
            return "moderate shift"
        return "significant shift"

    @property
    def is_alarm(self) -> bool:
        return self.psi >= PSI_ALARM


def _bin_edges(expected: np.ndarray, n_bins: int) -> np.ndarray:
    """Quantile edges from the *expected* (baseline) distribution.

    Both populations must be binned on the same edges, and those edges belong
    to the baseline. Re-binning each period on its own quantiles would make PSI
    structurally near-zero and useless.
    """
    finite = expected[np.isfinite(expected)]
    if finite.size == 0:
        return np.array([-np.inf, np.inf])
    qs = np.linspace(0, 1, n_bins + 1)[1:-1]
    inner = np.unique(np.quantile(finite, qs))
    return np.concatenate([[-np.inf], inner, [np.inf]])


def psi(expected: np.ndarray, actual: np.ndarray, *, n_bins: int = 10) -> PsiResult:
    """PSI between a baseline and a current distribution."""
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    edges = _bin_edges(expected, n_bins)

    e_counts, _ = np.histogram(expected[np.isfinite(expected)], bins=edges)
    a_counts, _ = np.histogram(actual[np.isfinite(actual)], bins=edges)

    e_share = np.maximum(e_counts / max(e_counts.sum(), 1), _EPS)
    a_share = np.maximum(a_counts / max(a_counts.sum(), 1), _EPS)
    contrib = (a_share - e_share) * np.log(a_share / e_share)

    table = pl.DataFrame(
        {
            "bin": np.arange(len(contrib), dtype=np.int32),
            "lower": edges[:-1],
            "upper": edges[1:],
            "expected_share": e_share,
            "actual_share": a_share,
            "contribution": contrib,
        }
    )
    return PsiResult(psi=float(contrib.sum()), table=table)


def csi(
    expected: pl.DataFrame,
    actual: pl.DataFrame,
    features: list[str],
    *,
    n_bins: int = 10,
) -> pl.DataFrame:
    """Per-feature stability, worst first.

    Categorical columns are compared on their own category shares rather than
    quantile bins, since quantiles are meaningless for them.
    """
    rows = []
    for f in features:
        if f not in expected.columns or f not in actual.columns:
            continue
        if expected[f].dtype.is_numeric():
            value = psi(
                expected[f].to_numpy().astype(float),
                actual[f].to_numpy().astype(float),
                n_bins=n_bins,
            ).psi
        else:
            value = _categorical_psi(expected[f], actual[f])
        rows.append({"feature": f, "csi": value})

    return (
        pl.DataFrame(rows, schema={"feature": pl.Utf8, "csi": pl.Float64})
        .with_columns(
            verdict=pl.when(pl.col("csi") < PSI_INVESTIGATE)
            .then(pl.lit("stable"))
            .when(pl.col("csi") < PSI_ALARM)
            .then(pl.lit("moderate shift"))
            .otherwise(pl.lit("significant shift"))
        )
        .sort("csi", descending=True)
    )


def _categorical_psi(expected: pl.Series, actual: pl.Series) -> float:
    e = expected.value_counts(normalize=True)
    a = actual.value_counts(normalize=True)
    key = expected.name
    joined = e.join(a, on=key, how="full", suffix="_actual", coalesce=True).fill_null(_EPS)
    e_share = np.maximum(joined["proportion"].to_numpy(), _EPS)
    a_share = np.maximum(joined["proportion_actual"].to_numpy(), _EPS)
    return float(np.sum((a_share - e_share) * np.log(a_share / e_share)))
