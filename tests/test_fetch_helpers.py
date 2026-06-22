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
