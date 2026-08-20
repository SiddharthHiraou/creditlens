"""WOE encoding and Information Value."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from src.features.woe import NULL_BIN, fit_woe, information_values, transform_woe


@pytest.fixture
def signal_frame() -> pl.DataFrame:
    rng = np.random.default_rng(0)
    n = 20_000
    x = rng.normal(size=n)
    y = rng.binomial(1, 1 / (1 + np.exp(-(-2.0 + 1.6 * x))))
    return pl.DataFrame({"x": x, "noise": rng.normal(size=n), "label": y})


def test_iv_separates_signal_from_noise(signal_frame):
    iv = information_values(signal_frame, ["x", "noise"])
    signal = iv.filter(pl.col("feature") == "x")["iv"].item()
    noise = iv.filter(pl.col("feature") == "noise")["iv"].item()
    assert signal > 0.3, f"a strong predictor should clear IV 0.3, got {signal:.4f}"
    assert noise < 0.02, f"pure noise should fall under the IV floor, got {noise:.4f}"


def test_woe_is_monotone_in_bad_rate(signal_frame):
    """Higher WOE must mean higher bad rate; the sign convention matters
    because the scorecard's points are the negated product of WOE and beta."""
    b = fit_woe(signal_frame, "x")
    tab = b.table.sort("woe")
    rates = tab["bad_rate"].to_list()
    assert rates[0] < rates[-1]
    assert np.corrcoef(tab["woe"].to_numpy(), tab["bad_rate"].to_numpy())[0, 1] > 0.9


def test_a_bin_with_no_bads_does_not_produce_infinite_woe():
    """Without smoothing this is log(0) and it silently poisons the scorecard."""
    df = pl.DataFrame({"x": [0.0] * 100 + [1.0] * 100, "label": [0] * 100 + [1] * 100})
    b = fit_woe(df, "x", n_bins=2)
    assert all(np.isfinite(v) for v in b.mapping.values())
    assert np.isfinite(b.iv)


def test_nulls_get_their_own_bin_rather_than_being_dropped():
    df = pl.DataFrame({"x": [1.0, 2.0, None, None, 3.0, None], "label": [0, 0, 1, 1, 0, 1]})
    b = fit_woe(df, "x", n_bins=2)
    assert NULL_BIN in b.mapping
    # Nulls are all bad here, so their WOE must be positive (risk-increasing).
    assert b.mapping[NULL_BIN] > 0


def test_unseen_category_maps_to_neutral_evidence():
    """A category absent at fit time must neither reward nor penalise."""
    train = pl.DataFrame({"c": ["a", "a", "b", "b"], "label": [0, 1, 0, 1]})
    b = fit_woe(train, "c")
    out = b.transform(pl.Series("c", ["a", "zzz_unseen"]))
    assert out[1] == 0.0


def test_transform_is_deterministic_and_shape_preserving(signal_frame):
    b = {"x": fit_woe(signal_frame, "x")}
    a = transform_woe(signal_frame, b)
    c = transform_woe(signal_frame, b)
    assert a.height == signal_frame.height
    assert a.columns == ["woe_x"]
    assert a.equals(c)


def test_iv_bands_are_labelled(signal_frame):
    iv = information_values(signal_frame, ["x", "noise"])
    assert iv.filter(pl.col("feature") == "noise")["strength"].item() == "unpredictive"
