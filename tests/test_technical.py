"""Tests for technical indicators against known values."""

import numpy as np
import pandas as pd

from concinvest.features import technical


def test_sma_known_values():
    s = pd.Series([1, 2, 3, 4, 5], dtype=float)
    out = technical.sma(s, 3)
    assert np.isnan(out.iloc[1])
    assert out.iloc[2] == 2.0  # mean(1,2,3)
    assert out.iloc[4] == 4.0  # mean(3,4,5)


def test_ema_first_value_equals_seed():
    s = pd.Series([10, 20, 30], dtype=float)
    out = technical.ema(s, span=2)
    assert out.iloc[0] == 10.0  # adjust=False seeds with first value


def test_rsi_all_gains_is_100():
    s = pd.Series(np.arange(1, 40, dtype=float))  # strictly increasing
    out = technical.rsi(s, 14)
    assert out.dropna().iloc[-1] == 100.0


def test_macd_columns_and_zero_for_flat_series():
    s = pd.Series(np.full(60, 50.0))
    out = technical.macd(s)
    assert list(out.columns) == ["macd", "macd_signal"]
    assert abs(out["macd"].iloc[-1]) < 1e-9  # flat -> no divergence


def test_add_technical_features_columns(synth_market):
    df = synth_market["SIE.DE"]
    for col in ("sma_50", "ema_12", "rsi_14", "macd", "price_sma50_ratio", "volume_sma20_ratio"):
        assert col in df.columns
    # ratio around 1 once warmed up
    assert df["price_sma50_ratio"].dropna().between(0.5, 2.0).all()
