"""Tests for the live analyst/sentiment forecast overlay (live-only tilt)."""

import pandas as pd

from concinvest.ml.forecast import Forecast
from concinvest.ml.overlay import apply_overlay, risk_gate, sentiment_tilt


def _row(**kw) -> pd.Series:
    base = {"recommendation_mean": None, "eps_revision_up_7d": None,
            "eps_revision_down_7d": None, "analyst_target_mean": None,
            "put_call_ratio": None, "iv_skew": None}
    base.update(kw)
    return pd.Series(base)


def test_sentiment_tilt_bullish_vs_bearish():
    bull = sentiment_tilt(
        _row(recommendation_mean=1.0, eps_revision_up_7d=5, eps_revision_down_7d=0,
             analyst_target_mean=120.0), price=100.0)
    bear = sentiment_tilt(
        _row(recommendation_mean=5.0, eps_revision_up_7d=0, eps_revision_down_7d=5,
             analyst_target_mean=80.0), price=100.0)
    assert bull > 0.5 and bear < -0.5


def test_sentiment_tilt_neutral_when_no_signals():
    assert sentiment_tilt(_row(), price=100.0) == 0.0


def test_risk_gate_drops_on_crash_fear():
    assert risk_gate(_row()) == 1.0  # no signals -> calm
    assert risk_gate(_row(put_call_ratio=2.0)) == 0.0  # heavy put buying
    assert risk_gate(_row(iv_skew=0.10)) == 0.0  # steep put skew


def test_apply_overlay_bullish_raises_buy_confidence():
    fc = [Forecast("AAA", "buy", 10_000.0, "stock", 0.6)]
    sent = pd.DataFrame([{"ticker": "AAA", "recommendation_mean": 1.0,
                          "eps_revision_up_7d": 5, "eps_revision_down_7d": 0,
                          "analyst_target_mean": 120.0, "put_call_ratio": 0.9,
                          "iv_skew": 0.0}])
    out = apply_overlay(fc, sent, {"AAA": 100.0})
    assert out[0].confidence > 0.6 and out[0].amount_eur > 10_000.0


def test_apply_overlay_crash_fear_caps_leverage():
    fc = [Forecast("AAA", "buy", 10_000.0, "3x", 0.7)]
    sent = pd.DataFrame([{"ticker": "AAA", "recommendation_mean": 3.0,
                          "eps_revision_up_7d": 0, "eps_revision_down_7d": 0,
                          "analyst_target_mean": None, "put_call_ratio": 2.0,
                          "iv_skew": 0.10}])
    out = apply_overlay(fc, sent, {"AAA": 100.0})
    assert out[0].leverage == "stock"  # gate ~0 -> leverage capped down


def test_apply_overlay_noop_without_sentiment():
    fc = [Forecast("AAA", "buy", 10_000.0, "2x", 0.6)]
    assert apply_overlay(fc, pd.DataFrame(), {"AAA": 100.0}) == fc
