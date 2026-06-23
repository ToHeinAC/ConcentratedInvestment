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
