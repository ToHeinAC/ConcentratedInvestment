"""Yahoo Finance data fetching via yfinance.

OHLCV is downloaded in batch; analyst / news / options metadata is fetched per
ticker with a small delay to respect rate limits. All functions degrade
gracefully (return empty / None) when the network or a field is unavailable, so
the pipeline never hard-fails on a single missing ticker.
"""

from __future__ import annotations

import datetime as _dt
import sys
import time

import pandas as pd
import requests
import yfinance as yf

from .. import config

# Browser-like UA so the German news sites return the normal markup.
_UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) concinvest/0.0"
_SCRAPE_TIMEOUT = 6.0

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

    A threaded batch download of the full universe is regularly rate-limited by
    Yahoo, which drops some tickers from the result *without raising*. Any ticker
    missing after the batch is therefore retried **individually** (single-ticker
    requests survive rate limiting far better), spaced by ``_META_DELAY``; a ticker
    still unavailable after that is skipped (degrade, don't hard-fail) and reported
    on stderr so the cron log shows which symbols came back empty.
    """
    result = _download_batch(tickers, start, end, retries)
    for ticker in [t for t in tickers if t not in result]:
        time.sleep(_META_DELAY)
        try:
            result.update(_download_batch([ticker], start, end, retries))
        except Exception:  # noqa: BLE001 - a single flaky ticker must not fail the run
            pass
    still_missing = [t for t in tickers if t not in result]
    if still_missing:
        print(
            f"[concinvest] no OHLCV returned for {len(still_missing)} ticker(s): "
            f"{', '.join(still_missing)}",
            file=sys.stderr,
        )
    return result


def _download_batch(
    tickers: list[str],
    start: _dt.date | str,
    end: _dt.date | str | None,
    retries: int,
) -> dict[str, pd.DataFrame]:
    """One batched ``yf.download`` with retry-on-exception; partial results (some
    tickers dropped by rate limiting) are returned as-is for the caller to backfill."""
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


def fetch_eps_revisions(ticker: str) -> tuple[float | None, float | None]:
    """(up, down) analyst EPS revisions over the last 7 days for the current quarter."""
    try:
        rev = yf.Ticker(ticker).eps_revisions
        time.sleep(_META_DELAY)
        if rev is None or rev.empty:
            return None, None
        row = rev.loc["0q"] if "0q" in rev.index else rev.iloc[0]
        up = row.get("upLast7days")
        down = row.get("downLast7days")
        return (
            float(up) if up is not None and not pd.isna(up) else None,
            float(down) if down is not None and not pd.isna(down) else None,
        )
    except Exception:  # noqa: BLE001
        return None, None


def fetch_analyst_target_mean(ticker: str) -> float | None:
    """Mean analyst price target, or None."""
    try:
        info = yf.Ticker(ticker).info
        val = info.get("targetMeanPrice")
        time.sleep(_META_DELAY)
        return float(val) if val is not None else None
    except Exception:  # noqa: BLE001
        return None


def fetch_iv_skew(ticker: str) -> float | None:
    """Implied-vol skew = OTM-put IV minus ATM-call IV (nearest expiry)."""
    try:
        tk = yf.Ticker(ticker)
        expiries = tk.options
        if not expiries:
            return None
        spot = float(tk.fast_info["last_price"])
        chain = tk.option_chain(expiries[0])
        time.sleep(_META_DELAY)
        atm_call_iv = _iv_at(chain.calls, spot)
        otm_put_iv = _iv_at(chain.puts, 0.9 * spot)
        if atm_call_iv is None or otm_put_iv is None:
            return None
        return otm_put_iv - atm_call_iv
    except Exception:  # noqa: BLE001
        return None


def _iv_at(opts: pd.DataFrame, strike: float) -> float | None:
    """Implied vol of the contract whose strike is nearest ``strike``."""
    if opts is None or opts.empty:
        return None
    row = opts.iloc[(opts["strike"] - strike).abs().argmin()]
    iv = row.get("impliedVolatility")
    return float(iv) if iv is not None and not pd.isna(iv) else None


def fetch_german_headlines(query: str, max_items: int = 10) -> list[str]:
    """Scrape recent German-language headlines for ``query`` (best effort)."""
    url = "https://www.finanznachrichten.de/suche/uebersicht.htm"
    try:
        resp = requests.get(
            url, params={"suche": query},
            headers={"User-Agent": _UA}, timeout=_SCRAPE_TIMEOUT,
        )
        time.sleep(_META_DELAY)
        if resp.status_code != 200:
            return []
        return _parse_finanznachrichten(resp.text, max_items)
    except Exception:  # noqa: BLE001
        return []


def _parse_finanznachrichten(html: str, max_items: int = 10) -> list[str]:
    """Extract article headlines from a finanznachrichten.de search page."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    seen: dict[str, None] = {}
    for a in soup.select("a.news-headline, a[href*='/nachrichten-']"):
        title = a.get_text(strip=True)
        if len(title) > 15:
            seen.setdefault(title, None)
        if len(seen) >= max_items:
            break
    return list(seen)
