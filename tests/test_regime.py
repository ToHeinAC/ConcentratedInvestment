"""Rising-market regime classifier (explainable Fear&Greed-style votes)."""

import numpy as np
import pandas as pd

from concinvest.features.regime import detect_regime

_NAMES = ["S&P vs 50d MA", "S&P vs 125d MA", "Breadth", "VIX vs 50d MA",
          "Gold vs 50d MA", "Oil vs 50d MA"]


def _series(start: float, drift: float, n: int = 200, seed: int = 0) -> pd.Series:
    idx = pd.bdate_range("2020-01-01", periods=n)
    steps = np.full(n, drift)
    return pd.Series(start * np.cumprod(1.0 + steps), index=idx)


def _five_stocks(drift: float) -> dict[str, pd.Series]:
    return {f"S{i}": _series(100.0, drift, seed=i) for i in range(5)}


def test_clearly_rising() -> None:
    """All six components bullish (gold & oil below their MA) -> Rising, 1.0."""
    reg = detect_regime(_series(4000.0, 0.002), _series(30.0, -0.003),
                        _five_stocks(0.002), gold=_series(2000.0, -0.003),
                        oil=_series(80.0, -0.003))
    assert reg.label == "Rising"
    assert reg.score == 1.0
    assert all(s.bullish for s in reg.signals)


def test_clearly_falling() -> None:
    """All six bearish (gold & oil above their MA) -> Falling, no bullish votes."""
    reg = detect_regime(_series(4000.0, -0.002), _series(15.0, 0.004),
                        _five_stocks(-0.002), gold=_series(2000.0, 0.003),
                        oil=_series(80.0, 0.003))
    assert reg.label == "Falling"
    assert reg.score == 0.0
    assert not any(s.bullish for s in reg.signals)


def test_half_is_neutral() -> None:
    """Three of six bullish (S&P x2 + gold, 0.5) -> Neutral."""
    reg = detect_regime(_series(4000.0, 0.002), _series(15.0, 0.004),
                        _five_stocks(-0.002), gold=_series(2000.0, -0.003),
                        oil=_series(80.0, 0.004))
    assert reg.label == "Neutral"
    assert reg.score == 0.5


def test_signals_order_and_explainability() -> None:
    """Six named votes in order; breadth label is descriptive, not a bare fraction."""
    reg = detect_regime(_series(4000.0, 0.002), _series(30.0, -0.003),
                        _five_stocks(0.002), gold=_series(2000.0, -0.003),
                        oil=_series(80.0, -0.003))
    assert [s.name for s in reg.signals] == _NAMES
    assert all(s.detail and s.label != "n/a" for s in reg.signals)
    assert reg.signals[2].label == "5 > 125d MA"  # breadth: all five above MA


def test_commodities_optional() -> None:
    """Omitting gold & oil yields the four core votes (graceful degrade)."""
    reg = detect_regime(_series(4000.0, 0.002), _series(30.0, -0.003),
                        _five_stocks(0.002))
    assert [s.name for s in reg.signals] == _NAMES[:4]


def test_short_history_does_not_crash() -> None:
    """Series shorter than the MA windows vote non-bullish, no error."""
    short = {f"S{i}": _series(100.0, 0.002, n=10, seed=i) for i in range(5)}
    reg = detect_regime(_series(4000.0, 0.002, n=10), _series(20.0, 0.0, n=10),
                        short, gold=_series(2000.0, 0.0, n=10),
                        oil=_series(80.0, 0.0, n=10))
    assert reg.label == "Falling"
    assert all("insufficient" in s.detail for s in reg.signals)
