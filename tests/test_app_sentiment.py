"""Pure-logic tests for the Live tab's analyst/sentiment visual (`streamlit_app`)."""

import pandas as pd

from concinvest.app import streamlit_app as app


def _market(price: float) -> dict:
    return {"SIE.DE": {"close": pd.Series([price - 1.0, price])}}


def test_sentiment_rows_derives_rating_news_and_upside():
    sent = pd.DataFrame([{"ticker": "SIE.DE", "recommendation_mean": 1.3,
                          "news_sentiment_score": 0.9,
                          "analyst_target_mean": 124.0}])
    (row,) = app._sentiment_rows(sent, _market(100.0))
    assert row["rating"] == "Strong Buy"
    assert row["news"] == "Positive"
    assert row["upside"] == 124.0 / 100.0 - 1.0  # +24%


def test_sentiment_rows_missing_target_yields_none_upside():
    sent = pd.DataFrame([{"ticker": "SIE.DE", "recommendation_mean": None,
                          "news_sentiment_score": 0.0,
                          "analyst_target_mean": None}])
    (row,) = app._sentiment_rows(sent, _market(100.0))
    assert row["upside"] is None
    assert row["rating"] == "—"
    assert row["news"] == "Neutral"


def test_rating_colors_and_news_icons_cover_all_labels():
    for _, label in app._RATINGS:
        assert label in app._RATING_COLORS
    assert set(app._NEWS_ICONS) == {"Positive", "Neutral", "Negative", "—"}
