"""Daily ETL (Phase 5 cron building block): offline, no network.

Verifies that ``pipeline.daily_etl`` persists every table and that repeated runs on
distinct dates *accumulate* dated sentiment snapshots — the history the live analyst
signals need before they can become model features.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from concinvest import pipeline
from concinvest.data import store


def _fake_sentiment_row(ticker: str, as_of=None) -> pd.DataFrame:
    return pd.DataFrame([{
        "date": as_of, "ticker": ticker,
        "recommendation_mean": 2.0, "news_sentiment_score": 0.5,
        "put_call_ratio": 1.0, "eps_revision_up_7d": 1, "eps_revision_down_7d": 0,
        "analyst_target_mean": 100.0, "iv_skew": 0.0,
    }])


def test_daily_etl_accumulates_sentiment_history(tmp_path, synth_raw, monkeypatch):
    db = tmp_path / "etl.sqlite"
    monkeypatch.setattr(pipeline.fetch, "download_ohlcv", lambda *a, **k: synth_raw)
    monkeypatch.setattr(pipeline.analyst, "build_sentiment_row", _fake_sentiment_row)

    s1 = pipeline.daily_etl(as_of=dt.date(2025, 6, 1), db_path=db)
    s2 = pipeline.daily_etl(as_of=dt.date(2025, 6, 2), db_path=db)

    # synth_raw carries 2 of the 5 portfolio stocks -> 2 sentiment rows per run.
    assert s1["sentiment_rows"] == s2["sentiment_rows"] == 2
    assert s1["tickers"] > 0 and s1["cross_rows"] > 0

    conn = store.connect(db)
    sent = store.read_table(conn, "sentiment_analyst")
    conn.close()
    # Two distinct snapshot dates accumulated per ticker (history, not overwrite).
    dates = set(pd.to_datetime(sent["date"]))
    assert dates >= {pd.Timestamp("2025-06-01"), pd.Timestamp("2025-06-02")}
    assert (sent.groupby("ticker")["date"].nunique() == 2).all()


def test_daily_etl_without_sentiment_skips_snapshot(tmp_path, synth_raw, monkeypatch):
    db = tmp_path / "etl2.sqlite"
    monkeypatch.setattr(pipeline.fetch, "download_ohlcv", lambda *a, **k: synth_raw)

    summary = pipeline.daily_etl(with_sentiment=False, db_path=db)
    assert summary["sentiment_rows"] == 0


def _capture_download(synth_raw, calls):
    """Stub download_ohlcv that records the ``start`` it was asked for, and only
    returns rows on/after that start (so the merge with stored history is exercised)."""
    def _dl(universe, start=None, end=None, **k):
        calls.append(pd.to_datetime(start).date())
        cutoff = pd.to_datetime(start).date()
        return {t: synth_raw[t][synth_raw[t].index >= cutoff]
                for t in universe if t in synth_raw}
    return _dl


def test_fetch_and_store_incremental_tail(tmp_path, synth_raw, monkeypatch):
    """Second run fetches only the tail (~last stored date − overlap), not from start."""
    db = tmp_path / "inc.sqlite"
    calls: list = []
    monkeypatch.setattr(pipeline.fetch, "download_ohlcv",
                        _capture_download(synth_raw, calls))
    universe = list(synth_raw)

    pipeline.fetch_and_store(universe, start=dt.date(2020, 1, 1), db_path=db)
    market2, _cross, raw2 = pipeline.fetch_and_store(
        universe, start=dt.date(2020, 1, 1), db_path=db)

    # First call pulls full history from start; second only the recent tail.
    assert calls[0] == dt.date(2020, 1, 1)
    last_stored = max(synth_raw["TSLA"].index)
    expected = last_stored - dt.timedelta(days=pipeline._REFETCH_OVERLAP_DAYS)
    assert calls[1] == expected
    # Despite the short second fetch, the merged history keeps full depth: long SMAs
    # are populated and the full date range survives.
    assert raw2["TSLA"].index[0] == min(synth_raw["TSLA"].index)
    assert market2["TSLA"]["sma_200"].notna().any()


def test_fetch_and_store_full_refetch(tmp_path, synth_raw, monkeypatch):
    """full=True forces a fetch from start even when history is already stored."""
    db = tmp_path / "full.sqlite"
    calls: list = []
    monkeypatch.setattr(pipeline.fetch, "download_ohlcv",
                        _capture_download(synth_raw, calls))
    universe = list(synth_raw)

    pipeline.fetch_and_store(universe, start=dt.date(2020, 1, 1), db_path=db)
    pipeline.fetch_and_store(universe, start=dt.date(2020, 1, 1), db_path=db, full=True)
    assert calls[1] == dt.date(2020, 1, 1)


def test_fetch_and_store_partial_db_full_fetch(tmp_path, synth_raw, monkeypatch):
    """A db missing some universe tickers falls back to a full fetch (self-healing)."""
    db = tmp_path / "partial.sqlite"
    calls: list = []
    monkeypatch.setattr(pipeline.fetch, "download_ohlcv",
                        _capture_download(synth_raw, calls))

    pipeline.fetch_and_store(["TSLA"], start=dt.date(2020, 1, 1), db_path=db)
    # Now ask for the full universe; SIE.DE etc. are absent -> fetch from start again.
    pipeline.fetch_and_store(list(synth_raw), start=dt.date(2020, 1, 1), db_path=db)
    assert calls[1] == dt.date(2020, 1, 1)
