"""Weight of Evidence encoding and Information Value.

WOE is the encoding the logistic scorecard track runs on, and IV is the
univariate screen credit teams use to decide what is worth keeping at all.

    WOE_bin = ln( (bads_in_bin / total_bads) / (goods_in_bin / total_goods) )
    IV      = sum over bins of (bad_share - good_share) * WOE

Conventional IV reading:

| IV | interpretation |
|---|---|
| < 0.02 | unpredictive, drop |
| 0.02 - 0.1 | weak |
| 0.1 - 0.3 | medium |
| 0.3 - 0.5 | strong |
| > 0.5 | suspiciously strong -- check for leakage before celebrating |

Two properties make WOE worth the trouble on the scorecard track: it turns a
non-linear relationship into a monotone one the logistic model can use, and it
handles nulls as a first-class bin rather than requiring imputation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

# Laplace-style smoothing. Without it, a bin containing no bads produces
# log(0) = -inf, which propagates a silent inf through the whole scorecard.
SMOOTHING = 0.5

NULL_BIN = "__NULL__"


@dataclass
class WoeBinning:
    """A fitted binning for one feature. Applying it is a pure lookup."""

    feature: str
    is_numeric: bool
    breaks: list[float] = field(default_factory=list)
    mapping: dict[str, float] = field(default_factory=dict)
    iv: float = 0.0
    table: pl.DataFrame | None = None

    def bin_labels(self, s: pl.Series) -> pl.Series:
        if not self.is_numeric:
            return s.cast(pl.Utf8).fill_null(NULL_BIN)
        idx = np.searchsorted(np.asarray(self.breaks), s.to_numpy().astype(float), side="right")
        labels = pl.Series([f"b{i}" for i in idx], dtype=pl.Utf8)
        return pl.select(
            pl.when(s.is_null().to_frame("m")["m"]).then(pl.lit(NULL_BIN)).otherwise(labels)
        ).to_series()

    def transform(self, s: pl.Series) -> pl.Series:
        """Map raw values onto their bin's WOE.

        A category unseen at fit time maps to 0.0 -- neutral evidence. That is
        the conservative choice: it neither penalises nor rewards an applicant
        for a value the model has never observed.
        """
        return self.bin_labels(s).replace_strict(self.mapping, default=0.0, return_dtype=pl.Float64)


def _quantile_breaks(values: np.ndarray, n_bins: int) -> list[float]:
    """Interior cut points at equal-frequency quantiles.

    Deduplicated, because a feature where 60% of rows share one value produces
    repeated quantiles and would otherwise yield empty bins.
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return []
    qs = np.linspace(0, 1, n_bins + 1)[1:-1]
    return sorted(set(np.quantile(finite, qs).tolist()))


def fit_woe(
    df: pl.DataFrame,
    feature: str,
    *,
    label_col: str = "label",
    n_bins: int = 10,
    min_bin_share: float = 0.02,
) -> WoeBinning:
    """Fit a WOE binning for one feature against the binary label."""
    s = df[feature]
    y = df[label_col].to_numpy().astype(int)
    is_numeric = s.dtype.is_numeric()

    if is_numeric:
        breaks = _quantile_breaks(s.to_numpy().astype(float), n_bins)
        binning = WoeBinning(feature=feature, is_numeric=True, breaks=breaks)
    else:
        binning = WoeBinning(feature=feature, is_numeric=False)

    labels = binning.bin_labels(s)
    work = pl.DataFrame({"bin": labels, "y": y})

    total_bad = max(int(y.sum()), 1)
    total_good = max(int((1 - y).sum()), 1)

    tab = (
        work.group_by("bin")
        .agg(n=pl.len(), n_bad=pl.col("y").sum())
        .with_columns(n_good=pl.col("n") - pl.col("n_bad"))
        .sort("bin")
    )

    # Rare categories carry unstable WOE; fold them into a shared bucket.
    if not is_numeric:
        rare = tab.filter(pl.col("n") / len(y) < min_bin_share)["bin"].to_list()
        if rare:
            work = work.with_columns(
                bin=pl.when(pl.col("bin").is_in(rare))
                .then(pl.lit("__RARE__"))
                .otherwise(pl.col("bin"))
            )
            tab = (
                work.group_by("bin")
                .agg(n=pl.len(), n_bad=pl.col("y").sum())
                .with_columns(n_good=pl.col("n") - pl.col("n_bad"))
                .sort("bin")
            )
            binning.mapping.update({r: 0.0 for r in rare})

    tab = tab.with_columns(
        bad_share=(pl.col("n_bad") + SMOOTHING) / (total_bad + SMOOTHING * tab.height),
        good_share=(pl.col("n_good") + SMOOTHING) / (total_good + SMOOTHING * tab.height),
    ).with_columns(
        woe=(pl.col("bad_share") / pl.col("good_share")).log(),
        bad_rate=pl.col("n_bad") / pl.col("n"),
    )
    tab = tab.with_columns(
        iv_contribution=(pl.col("bad_share") - pl.col("good_share")) * pl.col("woe")
    )

    mapping = dict(zip(tab["bin"].to_list(), tab["woe"].to_list(), strict=True))
    if not is_numeric:
        rare_woe = mapping.get("__RARE__", 0.0)
        for r in list(binning.mapping):
            binning.mapping[r] = rare_woe
    binning.mapping.update(mapping)
    binning.iv = float(tab["iv_contribution"].sum())
    binning.table = tab
    return binning


def information_values(
    df: pl.DataFrame, features: list[str], *, label_col: str = "label", n_bins: int = 10
) -> pl.DataFrame:
    """IV for every feature, strongest first, with the conventional band."""
    rows = []
    for f in features:
        try:
            b = fit_woe(df, f, label_col=label_col, n_bins=n_bins)
        except Exception:  # noqa: BLE001 - a feature that cannot be binned has no IV
            rows.append({"feature": f, "iv": 0.0, "n_bins": 0})
            continue
        rows.append(
            {"feature": f, "iv": b.iv, "n_bins": b.table.height if b.table is not None else 0}
        )

    return (
        pl.DataFrame(rows, schema={"feature": pl.Utf8, "iv": pl.Float64, "n_bins": pl.Int64})
        .with_columns(
            strength=pl.when(pl.col("iv") < 0.02)
            .then(pl.lit("unpredictive"))
            .when(pl.col("iv") < 0.10)
            .then(pl.lit("weak"))
            .when(pl.col("iv") < 0.30)
            .then(pl.lit("medium"))
            .when(pl.col("iv") < 0.50)
            .then(pl.lit("strong"))
            .otherwise(pl.lit("suspicious"))
        )
        .sort("iv", descending=True)
    )


def fit_woe_set(
    df: pl.DataFrame, features: list[str], *, label_col: str = "label", n_bins: int = 10
) -> dict[str, WoeBinning]:
    out: dict[str, WoeBinning] = {}
    for f in features:
        try:
            out[f] = fit_woe(df, f, label_col=label_col, n_bins=n_bins)
        except Exception:  # noqa: BLE001
            continue
    return out


def transform_woe(df: pl.DataFrame, binnings: dict[str, WoeBinning]) -> pl.DataFrame:
    """Apply fitted binnings, producing one ``woe_*`` column per feature."""
    return pl.DataFrame(
        {f"woe_{name}": b.transform(df[name]) for name, b in binnings.items() if name in df.columns}
    )
