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


def _news_score(ticker: str) -> float:
    """Score combined yfinance + German-source headlines on the [-3, 3] scale."""
    headlines = fetch.fetch_news_headlines(ticker)
    query = tickers.GERMAN_QUERY.get(ticker)
    if query:
        headlines = headlines + fetch.fetch_german_headlines(query)
    return sentiment.score_headlines(headlines)


def build_sentiment_row(ticker: str, as_of: _dt.date | None = None) -> pd.DataFrame:
    """Fetch and assemble a one-row sentiment_analyst frame for ``ticker``."""
    as_of = as_of or _dt.date.today()
    up_7d, down_7d = fetch.fetch_eps_revisions(ticker)
    row = {
        "date": as_of,
        "ticker": ticker,
        "recommendation_mean": fetch.fetch_recommendation_mean(ticker),
        "news_sentiment_score": _news_score(ticker),
        "put_call_ratio": fetch.fetch_put_call_ratio(ticker),
        "eps_revision_up_7d": up_7d,
        "eps_revision_down_7d": down_7d,
        "analyst_target_mean": fetch.fetch_analyst_target_mean(ticker),
        "iv_skew": fetch.fetch_iv_skew(ticker),
    }
    return pd.DataFrame([row])
