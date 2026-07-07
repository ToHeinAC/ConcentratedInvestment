"""Analyst & options sentiment row assembly (Table 2).

Assembles the live ``sentiment_analyst`` row: analyst recommendation mean and price
target, a news-sentiment score (yfinance + German-source headlines, VADER/FinBERT),
the options put/call ratio, IV skew, and the 7-day EPS revision up/down counts.

These signals have no usable history (yfinance exposes only recent news/options), so
they are fetched live and stored/displayed — they are *not* part of the model's
``FEATURE_COLS`` (a tree gains nothing from columns that are constant over training).
"""

from __future__ import annotations

import datetime as _dt

import pandas as pd

from ..data import fetch, tickers
from . import sentiment


def _scored_headlines(ticker: str) -> list[dict]:
    """Fetch yfinance + German news items and attach a per-article sentiment score.

    Each record is ``{title, link, published, source, score}`` on the [-3, 3] scale.
    """
    items = [dict(i, source="yfinance") for i in fetch.fetch_news_items(ticker)]
    query = tickers.GERMAN_QUERY.get(ticker)
    if query:
        items += [dict(i, source="finanznachrichten")
                  for i in fetch.fetch_german_news_items(query)]
    scores = sentiment.score_texts([i["title"] for i in items])
    for item, s in zip(items, scores):
        item["score"] = s
    return items


def _recent_records(records: list[dict], as_of: _dt.date, days: int) -> list[dict]:
    """yfinance items within ``days`` + all undated (German) items, most-positive first."""
    cutoff = as_of - _dt.timedelta(days=days)
    recent = [r for r in records
              if r["published"] is None or r["published"].date() >= cutoff]
    return sorted(recent, key=lambda r: r["score"], reverse=True)


def build_sentiment(
    ticker: str, as_of: _dt.date | None = None, days: int = 7
) -> tuple[pd.DataFrame, list[dict]]:
    """Assemble the sentiment_analyst row for ``ticker`` **and** the scored per-article
    records behind its news score (for the UI headline expander).

    News items are fetched and scored **once**: the row's ``news_sentiment_score`` is the
    mean over all fetched articles, while the returned records are the display subset —
    yfinance items within the last ``days`` plus all German items (which carry no date) —
    sorted most-positive first.
    """
    as_of = as_of or _dt.date.today()
    records = _scored_headlines(ticker)
    score = (sum(r["score"] for r in records) / len(records)) if records else 0.0
    up_7d, down_7d = fetch.fetch_eps_revisions(ticker)
    row = {
        "date": as_of,
        "ticker": ticker,
        "recommendation_mean": fetch.fetch_recommendation_mean(ticker),
        "news_sentiment_score": score,
        "put_call_ratio": fetch.fetch_put_call_ratio(ticker),
        "eps_revision_up_7d": up_7d,
        "eps_revision_down_7d": down_7d,
        "analyst_target_mean": fetch.fetch_analyst_target_mean(ticker),
        "iv_skew": fetch.fetch_iv_skew(ticker),
    }
    return pd.DataFrame([row]), _recent_records(records, as_of, days)


def build_sentiment_row(ticker: str, as_of: _dt.date | None = None) -> pd.DataFrame:
    """One-row sentiment_analyst frame for ``ticker`` (see :func:`build_sentiment`)."""
    return build_sentiment(ticker, as_of=as_of)[0]
