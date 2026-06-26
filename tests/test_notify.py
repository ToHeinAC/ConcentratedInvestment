"""Tests for the email-on-trigger alert builder (pure, offline)."""

from concinvest.ml.forecast import Forecast
from concinvest.notify import build_alert
from concinvest.portfolio.rules import Trade


def test_no_trigger_returns_none():
    assert build_alert([], [], portfolio_name="book", strategy="default") is None


def test_forecast_trigger_renders_alert():
    fcs = [Forecast(ticker="TSLA", action="buy", amount_eur=9000.0,
                    leverage="3x", confidence=0.71)]
    alert = build_alert(fcs, [], portfolio_name="book", strategy="default")
    assert alert is not None
    assert "1 trigger(s)" in alert.subject and "book" in alert.subject
    assert "BUY TSLA 3x" in alert.body and "71%" in alert.body


def test_strategy_action_trigger_renders_alert():
    actions = [Trade(ticker="FCX", action="sell", amount_eur=1500.0, tier=3)]
    alert = build_alert([], actions, portfolio_name="book", strategy="aggressive")
    assert alert is not None
    assert "1 trigger(s)" in alert.subject and "aggressive" in alert.subject
    assert "SELL FCX 3x" in alert.body and "mandatory risk-rule sells" in alert.body
