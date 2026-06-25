"""Rising-market regime classifier.

A small, explainable signal built from three Fear&Greed-style components
(Story.md): S&P 500 momentum (price vs 125-day MA), market breadth (how many of
the portfolio stocks sit above their 125-day MA), and volatility (VIX vs its
50-day MA). Each casts one bullish/bearish vote; the majority sets the regime
label. Pure pandas — no network, no model.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .technical import sma


@dataclass(frozen=True)
class RegimeSignal:
    """One component vote plus a human-readable explanation."""

    name: str
    bullish: bool
    detail: str  # full explanation (chart hover / text)
    label: str  # short metric for the chart bar, e.g. "+4.7%", "2/5", "n/a"


@dataclass(frozen=True)
class Regime:
    """Aggregate regime read: label, bullish-vote fraction, and the components."""

    label: str  # "Rising" | "Neutral" | "Falling"
    score: float  # fraction of bullish votes (0..1)
    signals: tuple[RegimeSignal, ...]


def _last(series: pd.Series) -> float | None:
    series = series.dropna()
    return float(series.iloc[-1]) if not series.empty else None


def _momentum(sp500: pd.Series, window: int, name: str) -> RegimeSignal:
    price, ma = _last(sp500), _last(sma(sp500, window))
    if price is None or ma is None:
        return RegimeSignal(name, False, "S&P 500 momentum: insufficient history", "n/a")
    dev = price / ma - 1
    return RegimeSignal(name, price > ma,
                        f"S&P 500 {price:,.0f} vs {window}d MA {ma:,.0f} ({dev:+.1%})",
                        f"{dev:+.1%}")


def _below_ma(series: pd.Series, window: int, name: str, asset: str,
              hi_word: str, lo_word: str) -> RegimeSignal:
    """A 'price above its MA = bearish' vote (gold/oil): bullish when below."""
    level, ma = _last(series), _last(sma(series, window))
    if level is None or ma is None:
        return RegimeSignal(name, False, f"{asset}: insufficient history", "n/a")
    bullish = level < ma  # below MA = bullish (rising gold/oil is risk-off)
    dev = level / ma - 1
    return RegimeSignal(name, bullish,
                        f"{asset} {level:,.1f} vs {window}d MA {ma:,.1f} "
                        f"({lo_word if bullish else hi_word})",
                        f"{dev:+.1%}")


def _breadth(stock_closes: list[pd.Series], window: int) -> RegimeSignal:
    above = total = 0
    for s in stock_closes:
        price, ma = _last(s), _last(sma(s, window))
        if price is None or ma is None:
            continue
        total += 1
        above += int(price > ma)
    if not total:
        return RegimeSignal("Breadth", False, "breadth: insufficient history", "n/a")
    return RegimeSignal("Breadth", above / total > 0.5,
                        f"{above}/{total} stocks above their {window}d MA",
                        f"{above} > {window}d MA")


def _volatility(vix: pd.Series, window: int) -> RegimeSignal:
    level, ma = _last(vix), _last(sma(vix, window))
    if level is None or ma is None:
        return RegimeSignal("VIX vs 50d MA", False, "VIX: insufficient history", "n/a")
    bullish = level < ma  # calm = below MA
    dev = level / ma - 1
    return RegimeSignal("VIX vs 50d MA", bullish,
                        f"VIX {level:.1f} vs {window}d MA {ma:.1f} "
                        f"({'calm' if bullish else 'elevated'})",
                        f"{dev:+.1%}")


def detect_regime(
    sp500: pd.Series,
    vix: pd.Series,
    stock_closes: dict[str, pd.Series] | list[pd.Series],
    gold: pd.Series | None = None,
    oil: pd.Series | None = None,
    *,
    sp_fast_window: int = 50,
    mom_window: int = 125,
    breadth_window: int = 125,
    vix_window: int = 50,
    commodity_window: int = 50,
) -> Regime:
    """Classify the current market regime from the latest available date.

    ``stock_closes`` is the portfolio stocks' close-price Series (dict or list);
    ``gold``/``oil`` are commodity closes — each appended as a 'price above its
    50d MA = risk-off/bearish' vote when provided. Votes: bullish fraction > 0.6 =
    Rising, < 0.4 = Falling, else Neutral.
    """
    stocks = (list(stock_closes.values()) if isinstance(stock_closes, dict)
              else list(stock_closes))
    signals = [
        _momentum(sp500, sp_fast_window, "S&P vs 50d MA"),
        _momentum(sp500, mom_window, "S&P vs 125d MA"),
        _breadth(stocks, breadth_window),
        _volatility(vix, vix_window),
    ]
    if gold is not None:
        signals.append(_below_ma(gold, commodity_window, "Gold vs 50d MA", "Gold",
                                 "safe-haven bid", "risk-on"))
    if oil is not None:
        signals.append(_below_ma(oil, commodity_window, "Oil vs 50d MA", "Oil",
                                 "costly", "cheap"))
    votes = sum(s.bullish for s in signals)
    frac = votes / len(signals)
    label = "Rising" if frac > 0.6 else "Falling" if frac < 0.4 else "Neutral"
    return Regime(label, frac, tuple(signals))
