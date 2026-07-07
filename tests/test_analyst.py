"""Offline tests for analyst.build_sentiment: the sentiment row + scored news records
that feed the UI headline expander. All network fetchers are monkeypatched."""

from __future__ import annotations

import datetime as dt

from concinvest.features import analyst, sentiment


def _stub_fetch(monkeypatch):
    """Neutralise every network call analyst.build_sentiment makes except the news items."""
    for name, val in [("fetch_recommendation_mean", 2.0),
                      ("fetch_put_call_ratio", 1.0),
                      ("fetch_analyst_target_mean", 100.0),
                      ("fetch_iv_skew", 0.0)]:
        monkeypatch.setattr(analyst.fetch, name, lambda _t, v=val: v)
    monkeypatch.setattr(analyst.fetch, "fetch_eps_revisions", lambda _t: (1, 0))


def test_build_sentiment_row_and_records(monkeypatch):
    _stub_fetch(monkeypatch)
    as_of = dt.date(2026, 7, 7)  # cutoff (days=7) -> 2026-06-30
    utc = dt.timezone.utc
    yf_items = [
        {"title": "Record profit and raised guidance", "link": "https://ex.com/new",
         "published": dt.datetime(2026, 7, 5, tzinfo=utc)},        # recent -> kept
        {"title": "Fraud scandal and huge losses reported", "link": "https://ex.com/old",
         "published": dt.datetime(2026, 6, 1, tzinfo=utc)},        # >7d old -> dropped
    ]
    de_items = [{"title": "Siemens hebt Prognose an", "link": "https://fn.de/x",
                 "published": None}]                               # undated -> kept
    monkeypatch.setattr(analyst.fetch, "fetch_news_items", lambda _t: yf_items)
    monkeypatch.setattr(analyst.fetch, "fetch_german_news_items", lambda _q: de_items)

    row, records = analyst.build_sentiment("SIE.DE", as_of=as_of)

    # Aggregate score = mean over ALL fetched articles (pre-filter), unchanged behaviour.
    all_titles = [i["title"] for i in yf_items + de_items]
    expected = sum(sentiment.score_texts(all_titles)) / len(all_titles)
    assert abs(row["news_sentiment_score"].iloc[0] - expected) < 1e-9
    assert row["ticker"].iloc[0] == "SIE.DE"

    # Display records: recent yfinance + undated German, most-positive first; old dropped.
    titles = [r["title"] for r in records]
    assert "Record profit and raised guidance" in titles
    assert "Siemens hebt Prognose an" in titles
    assert "Fraud scandal and huge losses reported" not in titles
    assert [r["score"] for r in records] == sorted((r["score"] for r in records),
                                                   reverse=True)
    assert {r["source"] for r in records} == {"yfinance", "finanznachrichten"}
    assert all("link" in r and "score" in r for r in records)


def test_build_sentiment_row_wrapper_matches(monkeypatch):
    _stub_fetch(monkeypatch)
    monkeypatch.setattr(analyst.fetch, "fetch_news_items", lambda _t: [])
    monkeypatch.setattr(analyst.fetch, "fetch_german_news_items", lambda _q: [])
    row = analyst.build_sentiment_row("SIE.DE", as_of=dt.date(2026, 7, 7))
    assert row["news_sentiment_score"].iloc[0] == 0.0  # no articles -> neutral
    assert list(row["ticker"]) == ["SIE.DE"]
