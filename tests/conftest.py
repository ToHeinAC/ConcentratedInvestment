"""Shared fixtures: deterministic synthetic market data (no network)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from concinvest.features import technical


def _synth_close(n: int, seed: int, drift: float = 0.0005, vol: float = 0.015) -> pd.Series:
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, vol, n)
    price = 100.0 * np.cumprod(1.0 + rets)
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.Series(price, index=idx)


@pytest.fixture
def synth_raw() -> dict[str, pd.DataFrame]:
    """Raw OHLCV-like frames for two stocks + the cross-asset tickers."""
    tickers = {
        "SIE.DE": 1, "TSLA": 2,
        "GC=F": 3, "CL=F": 4, "HG=F": 5, "^VIX": 6,
        "DX-Y.NYB": 7, "^TNX": 8, "^GSPC": 9, "^IXIC": 10,
    }
    out: dict[str, pd.DataFrame] = {}
    for t, seed in tickers.items():
        close = _synth_close(400, seed)
        df = pd.DataFrame(
            {
                "open": close.values,
                "high": close.values * 1.01,
                "low": close.values * 0.99,
                "close": close.values,
                "adj_close": close.values,
                "volume": 1_000_000.0,
            },
            index=close.index.date,
        )
        df.index.name = "date"
        out[t] = df
    return out


@pytest.fixture
def synth_market(synth_raw) -> dict[str, pd.DataFrame]:
    """Stocks only, with technical features attached."""
    return {
        t: technical.add_technical_features(synth_raw[t])
        for t in ("SIE.DE", "TSLA")
    }
