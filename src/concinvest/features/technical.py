"""Technical indicators computed from daily OHLCV (pandas rolling / ewm).

Each helper takes/returns plain pandas objects so they are trivially unit-testable
without any I/O. ``add_technical_features`` assembles the Table-1 feature columns.
"""

from __future__ import annotations

import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    # Zero average loss -> rs = +inf -> RSI = 100 (handled naturally by the formula).
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    return pd.DataFrame({"macd": macd_line, "macd_signal": signal_line})


def bollinger(series: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    mid = sma(series, window)
    std = series.rolling(window=window, min_periods=window).std()
    return pd.DataFrame(
        {"bollinger_upper": mid + num_std * std, "bollinger_lower": mid - num_std * std}
    )


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``df`` augmented with the Table-1 technical feature columns.

    ``df`` must contain at least a ``close`` column and a ``volume`` column,
    indexed by date in ascending order.
    """
    out = df.copy()
    close = out["close"]

    for w in (5, 10, 20, 50, 100, 200):
        out[f"sma_{w}"] = sma(close, w)
    for s in (12, 26, 50):
        out[f"ema_{s}"] = ema(close, s)

    out["rsi_14"] = rsi(close, 14)
    out = out.join(macd(close))
    out = out.join(bollinger(close))

    out["price_sma50_ratio"] = close / out["sma_50"]
    out["price_sma200_ratio"] = close / out["sma_200"]
    out["sma50_sma200_ratio"] = out["sma_50"] / out["sma_200"]
    if "volume" in out.columns:
        out["volume_sma20_ratio"] = out["volume"] / sma(out["volume"], 20)
    return out
