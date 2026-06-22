"""Synthetic buy/sell training-data generation (Story.md ML dataset).

A *feature panel* is a per-(date, ticker) matrix of point-in-time features. From it
we sample stochastic trade datapoints: half buys, half sells, each a market snapshot
with an action encoding (``is_sell``, ``leverage``). The label is whether the action
was profitable over a forward horizon — strictly forward-looking to avoid leakage.

Phase 1 generates a modest dataset; the full 100k generator (Story.md) is Phase 3.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config

# Feature columns the model consumes. Per-stock technicals + cross-asset context +
# sentiment placeholders (neutral defaults over history; real values at forecast
# time) + the action encoding.
TECH_FEATURES = [
    "rsi_14",
    "macd",
    "price_sma50_ratio",
    "price_sma200_ratio",
    "sma50_sma200_ratio",
    "volume_sma20_ratio",
]
CROSS_FEATURES = [
    "vix_level",
    "vix_sma20_ratio",
    "gold_oil_ratio",
    "copper_gold_ratio",
    "yield_10y",
]
SENTIMENT_FEATURES = ["news_sentiment_score", "put_call_ratio"]
ACTION_FEATURES = ["is_sell", "leverage"]

FEATURE_COLS = TECH_FEATURES + CROSS_FEATURES + SENTIMENT_FEATURES + ACTION_FEATURES


def build_feature_panel(
    market: dict[str, pd.DataFrame],
    cross: pd.DataFrame,
) -> pd.DataFrame:
    """Join per-stock technical features with date-aligned cross-asset features.

    ``market`` maps ticker -> DataFrame (indexed by date) that already contains the
    technical feature columns (see ``features.technical.add_technical_features``).
    Returns a long DataFrame indexed by a (date, ticker) MultiIndex.
    """
    cross = cross.copy()
    cross.index = pd.to_datetime(cross.index)
    frames = []
    for ticker, df in market.items():
        d = df.copy()
        d.index = pd.to_datetime(d.index)
        d = d.join(cross, how="left")
        # Sentiment placeholders: neutral over history (no historical news feed).
        d["news_sentiment_score"] = 0.0
        d["put_call_ratio"] = np.nan
        d["ticker"] = ticker
        d = d.set_index("ticker", append=True)
        frames.append(d)
    panel = pd.concat(frames)
    panel.index.names = ["date", "ticker"]
    return panel


def generate_dataset(
    panel: pd.DataFrame,
    prices: dict[str, pd.Series],
    n: int = 4_000,
    horizon: int = 20,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """Sample ``n`` synthetic datapoints (half buys, half sells).

    ``prices`` maps ticker -> close Series for forward-return labelling. Returns
    ``(X, y)`` where X has ``FEATURE_COLS`` and y is the binary "profitable action".
    """
    rng = np.random.default_rng(seed)
    tickers = [t for t in prices if t in panel.index.get_level_values("ticker").unique()]
    rows: list[dict] = []
    labels: list[int] = []

    n_sell_target = n // 2
    attempts = 0
    max_attempts = n * 50
    n_sells = 0
    while len(rows) < n and attempts < max_attempts:
        attempts += 1
        ticker = tickers[rng.integers(len(tickers))]
        price = prices[ticker]
        if len(price) <= horizon + 1:
            continue
        i = int(rng.integers(0, len(price) - horizon - 1))
        date = price.index[i]
        try:
            feat = panel.loc[(pd.Timestamp(date), ticker)]
        except KeyError:
            continue
        if feat[TECH_FEATURES + CROSS_FEATURES].isna().any():
            continue

        is_sell = 1 if n_sells < n_sell_target else 0
        # Balance toward sells until target met, then buys.
        if is_sell == 0 and (len(rows) - n_sells) >= (n - n_sell_target):
            is_sell = 1
        leverage = int(rng.choice(config.LEVERAGE_TIERS))

        fwd_return = leverage * (float(price.iloc[i + horizon]) / float(price.iloc[i]) - 1.0)
        # Buy is "good" if price rose; sell (exit) is "good" if price fell.
        label = int(fwd_return > 0) if is_sell == 0 else int(fwd_return < 0)

        record = {c: float(feat.get(c, np.nan)) for c in TECH_FEATURES + CROSS_FEATURES}
        record["news_sentiment_score"] = 0.0
        record["put_call_ratio"] = 0.0  # neutral default over history
        record["is_sell"] = is_sell
        record["leverage"] = leverage
        rows.append(record)
        labels.append(label)
        n_sells += is_sell

    X = pd.DataFrame(rows, columns=FEATURE_COLS).fillna(0.0)
    y = pd.Series(labels, name="profitable")
    return X, y
