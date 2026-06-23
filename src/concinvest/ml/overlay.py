"""Live analyst/sentiment overlay on the forecast (live-only, not backtested).

The `sentiment_analyst` signals (analyst consensus, EPS revisions, price target, news
sentiment, put/call ratio, IV skew) have **no usable history** — yfinance exposes only
the current snapshot — so they cannot be replayed in the backtest/walk-forward. This
module uses them as a deterministic, interpretable tilt on the *live* 5-field forecast
only:

- `sentiment_tilt` raises/lowers conviction from analyst consensus + EPS-revision
  momentum + price-vs-target upside.
- `risk_gate` cuts leverage when options markets price crash fear (high put/call or a
  steep put IV skew).

Once the Phase 5 cron accumulates daily history, these signals can be validated and
promoted into the model proper.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from .forecast import Forecast

_TILT_GAIN = 0.5  # max ±50% conviction shift from sentiment
_TIER_ORDER = ["stock", "2x", "3x"]


def _clip(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return float(max(lo, min(hi, x)))


def sentiment_tilt(row: pd.Series, price: float | None = None) -> float:
    """Bullish/bearish tilt in [-1, 1] from the live analyst signals (NaN parts skipped)."""
    parts: list[float] = []
    rec = row.get("recommendation_mean")
    if rec is not None and not pd.isna(rec):
        parts.append(_clip((3.0 - float(rec)) / 2.0))  # 1=strong buy -> +1, 5=sell -> -1
    up, down = row.get("eps_revision_up_7d"), row.get("eps_revision_down_7d")
    if not (pd.isna(up) if up is not None else True) or not (pd.isna(down) if down is not None else True):
        net = float(up or 0.0) - float(down or 0.0)
        parts.append(_clip(net / 5.0))  # 5 net revisions -> full
    target = row.get("analyst_target_mean")
    if price and target is not None and not pd.isna(target) and price > 0:
        parts.append(_clip((float(target) / price - 1.0) / 0.20))  # 20% upside -> full
    return float(np.mean(parts)) if parts else 0.0


def risk_gate(row: pd.Series) -> float:
    """Leverage gate in [0, 1]: 1 = calm, →0 as options price crash fear (NaN skipped)."""
    gates: list[float] = []
    pcr = row.get("put_call_ratio")
    if pcr is not None and not pd.isna(pcr):
        gates.append(_clip(1.0 - max(0.0, float(pcr) - 1.0), 0.0, 1.0))  # pcr 2.0 -> 0
    skew = row.get("iv_skew")
    if skew is not None and not pd.isna(skew):
        gates.append(_clip(1.0 - max(0.0, float(skew)) / 0.10, 0.0, 1.0))  # skew 0.10 -> 0
    return min(gates) if gates else 1.0


def _capped_leverage(leverage: str, gate: float) -> str:
    """Cap the leverage tier as the risk gate falls (gate<0.34 -> stock, <0.67 -> 2x)."""
    max_tier = 0 if gate < 0.34 else (1 if gate < 0.67 else 2)
    if _TIER_ORDER.index(leverage) <= max_tier:
        return leverage
    return _TIER_ORDER[max_tier]


def apply_overlay(
    forecasts: list[Forecast], sentiment_df: pd.DataFrame, prices: dict[str, float]
) -> list[Forecast]:
    """Tilt each forecast's confidence/amount by `sentiment_tilt` and cap its leverage by
    `risk_gate`. No-op when there is no live sentiment. Live-only — not used in backtests."""
    if sentiment_df is None or sentiment_df.empty:
        return forecasts
    rows = sentiment_df.set_index("ticker")
    out: list[Forecast] = []
    for fc in forecasts:
        if fc.ticker not in rows.index:
            out.append(fc)
            continue
        row = rows.loc[fc.ticker]
        tilt = sentiment_tilt(row, prices.get(fc.ticker))
        gate = risk_gate(row)
        # A sell's conviction rises when sentiment is bearish (negative tilt).
        signed = tilt if fc.action == "buy" else -tilt
        factor = 1.0 + _TILT_GAIN * signed
        out.append(replace(
            fc,
            confidence=round(float(min(1.0, max(0.0, fc.confidence * factor))), 4),
            amount_eur=round(fc.amount_eur * max(0.0, factor), 2),
            leverage=_capped_leverage(fc.leverage, gate) if fc.action == "buy" else fc.leverage,
        ))
    return out
