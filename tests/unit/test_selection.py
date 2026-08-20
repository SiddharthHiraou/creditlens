"""Feature selection: degenerate columns, IV floor, correlation, null importance."""

from __future__ import annotations

import numpy as np
import polars as pl

from src.features.selection import drop_degenerate, prune_correlated, select


def _frame(n: int = 6_000) -> pl.DataFrame:
    rng = np.random.default_rng(0)
    signal = rng.normal(size=n)
    y = rng.binomial(1, 1 / (1 + np.exp(-(-2.0 + 1.6 * signal))))
    return pl.DataFrame(
        {
            "signal": signal,
            "signal_copy": signal * 2.0 + 1e-9,  # perfectly collinear
            "noise": rng.normal(size=n),
            "constant": np.ones(n),
            "mostly_null": [None] * (n - 20) + list(rng.normal(size=20)),
            "label": y,
        }
    )


def test_constant_and_mostly_null_columns_are_dropped():
    df = _frame()
    kept, dropped = drop_degenerate(df, ["signal", "constant", "mostly_null"])
    assert kept == ["signal"]
    assert dropped["constant"] == "constant"
    assert "null_share" in dropped["mostly_null"]


def test_correlated_pair_keeps_the_higher_ranked_member():
    df = _frame()
    ranking = {"signal": 0.9, "signal_copy": 0.1}
    kept, dropped = prune_correlated(df, ["signal", "signal_copy"], ranking, threshold=0.95)
    assert "signal" in kept
    assert "signal_copy" in dropped
    assert "corr>" in dropped["signal_copy"]


def test_full_pipeline_keeps_signal_and_discards_noise():
    df = _frame()
    features = ["signal", "signal_copy", "noise", "constant", "mostly_null"]
    report = select(df, features, use_null_importance=False)
    assert "signal" in report.kept
    assert "noise" not in report.kept
    assert "constant" not in report.kept


def test_every_dropped_feature_carries_a_recorded_reason():
    df = _frame()
    features = ["signal", "signal_copy", "noise", "constant", "mostly_null"]
    report = select(df, features, use_null_importance=False)
    dropped = set(features) - set(report.kept)
    assert dropped == set(report.dropped)
    assert all(report.dropped.values())


def test_null_importance_ranks_real_signal_above_noise():
    from src.features.selection import null_importance

    df = _frame()
    out = null_importance(df, ["signal", "noise"], n_runs=2)
    assert out["feature"][0] == "signal"
    assert (
        out.filter(pl.col("feature") == "signal")["score"].item()
        > out.filter(pl.col("feature") == "noise")["score"].item()
    )
