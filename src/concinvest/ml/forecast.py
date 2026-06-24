"""Forecast emission: the five required fields per Story.md.

For each stock we enumerate candidate actions (buy/sell at each leverage tier),
score them with the trained model, and keep the highest-confidence action that
clears a threshold. Base case is *hold* — most days produce no trade.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np
import pandas as pd

from .. import config
from .dataset import FEATURE_COLS
from .model import TrainedModel

_LEVERAGE_LABEL = {1: "stock", 2: "2x", 3: "3x"}


@dataclass
class Forecast:
    ticker: str
    action: str  # "buy" | "sell"
    amount_eur: float
    leverage: str  # "stock" | "2x" | "3x"
    confidence: float

    def as_dict(self) -> dict:
        return asdict(self)


def _candidates(snapshot: pd.Series, leverages: tuple[int, ...]) -> pd.DataFrame:
    """All buy/sell x leverage candidate feature rows for one snapshot."""
    base = {c: float(snapshot.get(c, 0.0)) for c in FEATURE_COLS}
    rows = []
    meta = []
    for is_sell in (0, 1):
        for lev in leverages:
            row = dict(base)
            row["is_sell"] = is_sell
            row["leverage"] = lev
            rows.append(row)
            meta.append((is_sell, lev))
    df = pd.DataFrame(rows, columns=FEATURE_COLS).fillna(0.0)
    df["_is_sell"] = [m[0] for m in meta]
    df["_lev"] = [m[1] for m in meta]
    return df


def forecast(
    model: TrainedModel,
    snapshots: dict[str, pd.Series],
    portfolio_value: float = config.INITIAL_CAPITAL_EUR,
    threshold: float = 0.55,
    leverages: tuple[int, ...] = config.LEVERAGE_TIERS,
) -> list[Forecast]:
    """Emit the best above-threshold action per ticker (else no trade). ``leverages``
    restricts the candidate tiers — the aggressive strategy passes ``(3,)`` (3x only)."""
    out: list[Forecast] = []
    for ticker, snap in snapshots.items():
        cand = _candidates(snap, leverages)
        conf = model.predict_confidence(cand)
        best = int(np.argmax(conf))
        if conf[best] < threshold:
            continue  # hold
        is_sell = int(cand.iloc[best]["_is_sell"])
        lev = int(cand.iloc[best]["_lev"])
        out.append(
            Forecast(
                ticker=ticker,
                action="sell" if is_sell else "buy",
                amount_eur=round(0.10 * portfolio_value, 2),
                leverage=_LEVERAGE_LABEL[lev],
                confidence=round(float(conf[best]), 4),
            )
        )
    return out


def apply_book_limits(
    forecasts: list[Forecast],
    cash: float,
    held: dict[tuple[str, str], float],
) -> list[Forecast]:
    """Cap each action by the live book (Story.md: buy only with cash on hand, sell only
    from open positions). A buy is clamped to the *remaining* cash (decremented as buys
    are funded); a sell is clamped to the value held in that name's leverage tier.
    Actions that can't be funded — or that fall below ``MIN_TRADE_EUR`` after capping
    (Story.md: no order < €500) — are dropped. Applied after the sentiment overlay."""
    remaining = cash
    out: list[Forecast] = []
    for fc in forecasts:
        if fc.action == "buy":
            amount = min(fc.amount_eur, remaining)
        else:
            amount = min(fc.amount_eur, held.get((fc.ticker, fc.leverage), 0.0))
        if amount < config.MIN_TRADE_EUR:  # Story.md: drop orders < €500
            continue
        if fc.action == "buy":
            remaining -= amount
        out.append(replace(fc, amount_eur=round(amount, 2)))
    return out


def forecasts_to_frame(forecasts: list[Forecast]) -> pd.DataFrame:
    cols = ["ticker", "action", "amount_eur", "leverage", "confidence"]
    if not forecasts:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame([f.as_dict() for f in forecasts])[cols]
