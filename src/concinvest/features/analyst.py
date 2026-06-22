"""Analyst & options sentiment row assembly (Table 2).

Phase 1 keeps the numeric essentials: analyst recommendation mean, a VADER news
sentiment score, and the options put/call ratio. Richer revision-momentum signals
arrive in Phase 2.
"""

from __future__ import annotations

import datetime as _dt

import pandas as pd

from ..data import fetch
from . import sentiment


def build_sentiment_row(ticker: str, as_of: _dt.date | None = None) -> pd.DataFrame:
    """Fetch and assemble a one-row sentiment_analyst frame for ``ticker``."""
    as_of = as_of or _dt.date.today()
    headlines = fetch.fetch_news_headlines(ticker)
    row = {
        "date": as_of,
        "ticker": ticker,
        "recommendation_mean": fetch.fetch_recommendation_mean(ticker),
        "news_sentiment_score": sentiment.score_headlines(headlines),
        "put_call_ratio": fetch.fetch_put_call_ratio(ticker),
    }
    return pd.DataFrame([row])
