"""Yahoo Finance data fetching via yfinance.

OHLCV is downloaded in batch; analyst / news / options metadata is fetched per
ticker with a small delay to respect rate limits. All functions degrade
gracefully (return empty / None) when the network or a field is unavailable, so
the pipeline never hard-fails on a single missing ticker.
"""

from __future__ import annotations

import datetime as _dt
import time

import pandas as pd
import yfinance as yf

from .. import config

# Polite delay between per-ticker metadata requests (seconds).
_META_DELAY = 0.5

_OHLCV_COLS = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adj_close",
    "Volume": "volume",
}


def download_ohlcv(
    tickers: list[str],
    start: _dt.date | str = config.START_DATE,
    end: _dt.date | str | None = None,
    retries: int = 3,
) -> dict[str, pd.DataFrame]:
    """Download daily OHLCV for ``tickers``.

    Returns a mapping ticker -> tidy DataFrame indexed by date with columns
    open/high/low/close/adj_close/volume. Tickers with no data are omitted.
    """
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            raw = yf.download(
                tickers,
                start=str(start),
                end=str(end) if end else None,
                interval="1d",
                auto_adjust=False,
                group_by="ticker",
                threads=True,
                progress=False,
            )
            return _split_ohlcv(raw, tickers)
        except Exception as exc:  # noqa: BLE001 - retry on any download error
            last_err = exc
            time.sleep(1.0 * (attempt + 1))
    raise RuntimeError(f"yfinance download failed after {retries} retries: {last_err}")


def _split_ohlcv(raw: pd.DataFrame, tickers: list[str]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    if raw is None or raw.empty:
        return out
    multi = isinstance(raw.columns, pd.MultiIndex)
    for ticker in tickers:
        try:
            sub = raw[ticker] if multi else raw
        except KeyError:
            continue
        sub = sub.rename(columns=_OHLCV_COLS)
        keep = [c for c in _OHLCV_COLS.values() if c in sub.columns]
        sub = sub[keep].dropna(how="all")
        if sub.empty:
            continue
        sub.index = pd.to_datetime(sub.index).date
        sub.index.name = "date"
        out[ticker] = sub
    return out


def fetch_recommendation_mean(ticker: str) -> float | None:
    """Analyst recommendation mean (1=strong buy ... 5=strong sell)."""
    try:
        info = yf.Ticker(ticker).info
        val = info.get("recommendationMean")
        time.sleep(_META_DELAY)
        return float(val) if val is not None else None
    except Exception:  # noqa: BLE001
        return None


def fetch_news_headlines(ticker: str, count: int = 10) -> list[str]:
    """Recent news headlines for ``ticker`` (best effort, may be empty)."""
    try:
        news = yf.Ticker(ticker).get_news(count=count)
        time.sleep(_META_DELAY)
    except Exception:  # noqa: BLE001
        return []
    headlines: list[str] = []
    for item in news or []:
        # yfinance news schema varies; try common shapes.
        title = item.get("title") or item.get("content", {}).get("title")
        if title:
            headlines.append(str(title))
    return headlines


def fetch_put_call_ratio(ticker: str) -> float | None:
    """Put/call open-interest ratio from the nearest expiry option chain."""
    try:
        tk = yf.Ticker(ticker)
        expiries = tk.options
        if not expiries:
            return None
        chain = tk.option_chain(expiries[0])
        time.sleep(_META_DELAY)
        put_oi = float(chain.puts["openInterest"].fillna(0).sum())
        call_oi = float(chain.calls["openInterest"].fillna(0).sum())
        if call_oi <= 0:
            return None
        return put_oi / call_oi
    except Exception:  # noqa: BLE001
        return None
