"""Offline tests for the pure (non-network) helpers in data.fetch."""

import pandas as pd

from concinvest.data import fetch


def test_iv_at_picks_nearest_strike():
    opts = pd.DataFrame(
        {"strike": [90.0, 100.0, 110.0], "impliedVolatility": [0.30, 0.20, 0.25]}
    )
    # Nearest strike to 101 is 100 -> IV 0.20.
    assert fetch._iv_at(opts, 101.0) == 0.20
    # Nearest to 89 is 90 -> IV 0.30.
    assert fetch._iv_at(opts, 89.0) == 0.30


def test_iv_at_handles_empty():
    assert fetch._iv_at(pd.DataFrame(), 100.0) is None
    assert fetch._iv_at(None, 100.0) is None


class _FakeTicker:
    def __init__(self, news):
        self._news = news

    def get_news(self, count=10):
        return self._news


def test_fetch_news_items_handles_both_schemas(monkeypatch):
    news = [
        {"title": "Flat schema headline", "link": "https://ex.com/a",
         "providerPublishTime": 1_700_000_000},
        {"content": {"title": "Nested schema headline",
                     "canonicalUrl": {"url": "https://ex.com/b"},
                     "pubDate": "2026-07-01T09:00:00Z"}},
        {"content": {}},  # no title -> skipped
    ]
    monkeypatch.setattr(fetch.yf, "Ticker", lambda t: _FakeTicker(news))
    monkeypatch.setattr(fetch.time, "sleep", lambda *_: None)

    items = fetch.fetch_news_items("SIE.DE")
    assert [i["title"] for i in items] == ["Flat schema headline", "Nested schema headline"]
    assert items[0]["link"] == "https://ex.com/a"
    assert items[1]["link"] == "https://ex.com/b"
    assert items[0]["published"].year == 2023  # epoch 1.7e9 -> 2023
    assert items[1]["published"].date().isoformat() == "2026-07-01"
    # The title-only wrapper still returns plain strings.
    assert fetch.fetch_news_headlines("SIE.DE") == [i["title"] for i in items]


def test_parse_finanznachrichten_items_resolves_links():
    html = """
    <html><body>
      <a class="news-headline" href="/nachrichten-1">Siemens hebt Jahresprognose deutlich an</a>
    </body></html>
    """
    items = fetch._parse_finanznachrichten_items(html, max_items=10)
    assert items[0]["link"] == "https://www.finanznachrichten.de/nachrichten-1"
    assert items[0]["published"] is None
    # The title-only wrapper still returns plain strings (back-compat).
    assert fetch._parse_finanznachrichten(html, max_items=10) == [items[0]["title"]]


def test_parse_finanznachrichten_extracts_headlines():
    html = """
    <html><body>
      <a class="news-headline" href="/nachrichten-1">Siemens hebt Jahresprognose deutlich an</a>
      <a href="/nachrichten-2">Munich Re meldet starkes Quartalsergebnis heute</a>
      <a href="/other">kurz</a>
      <a class="news-headline" href="/nachrichten-1">Siemens hebt Jahresprognose deutlich an</a>
    </body></html>
    """
    out = fetch._parse_finanznachrichten(html, max_items=10)
    assert "Siemens hebt Jahresprognose deutlich an" in out
    assert "Munich Re meldet starkes Quartalsergebnis heute" in out
    # Short link text (<=15 chars) is dropped; duplicates de-duplicated.
    assert "kurz" not in out
    assert len(out) == 2


def test_parse_finanznachrichten_respects_max_items():
    links = "".join(
        f'<a href="/nachrichten-{i}">Eine lange Schlagzeile Nummer {i}</a>' for i in range(20)
    )
    out = fetch._parse_finanznachrichten(f"<html><body>{links}</body></html>", max_items=5)
    assert len(out) == 5


def test_download_ohlcv_retries_dropped_tickers_individually(monkeypatch):
    """A batch that drops a ticker (rate-limited) triggers an individual retry that
    backfills it, so a partial batch result is not silently accepted."""
    calls: list = []

    def fake_batch(tickers, start, end, retries):
        calls.append(list(tickers))
        if len(tickers) > 1:  # batch drops TSLA, returns only SIE.DE
            return {"SIE.DE": pd.DataFrame({"close": [1.0]})}
        return {tickers[0]: pd.DataFrame({"close": [2.0]})}  # individual retry succeeds

    monkeypatch.setattr(fetch, "_download_batch", fake_batch)
    monkeypatch.setattr(fetch.time, "sleep", lambda *_: None)

    out = fetch.download_ohlcv(["SIE.DE", "TSLA"])
    assert set(out) == {"SIE.DE", "TSLA"}
    assert calls[0] == ["SIE.DE", "TSLA"]  # batch first
    assert ["TSLA"] in calls               # then the dropped ticker individually


def test_download_ohlcv_degrades_on_persistently_missing_ticker(monkeypatch):
    """A ticker that keeps failing its individual retry is skipped, not fatal."""
    def fake_batch(tickers, start, end, retries):
        if len(tickers) > 1:
            return {"SIE.DE": pd.DataFrame({"close": [1.0]})}
        raise RuntimeError("rate limited")

    monkeypatch.setattr(fetch, "_download_batch", fake_batch)
    monkeypatch.setattr(fetch.time, "sleep", lambda *_: None)

    out = fetch.download_ohlcv(["SIE.DE", "TSLA"])
    assert set(out) == {"SIE.DE"}  # TSLA dropped, run still returns what it could get
