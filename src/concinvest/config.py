"""Central configuration: paths, dates, and shared constants.

Keep this free of heavy imports so every module can import it cheaply.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

# --- Paths ---------------------------------------------------------------
# Repo root = two levels up from this file (src/concinvest/config.py).
ROOT_DIR: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = ROOT_DIR / "data"
DB_PATH: Path = DATA_DIR / "concinvest.sqlite"

# --- Time window (Story.md: start at 2020-01-01) -------------------------
START_DATE: _dt.date = _dt.date(2020, 1, 1)

# Train/validate split: first 4 years train, last 1 year validate.
VALIDATION_YEARS: int = 1

# --- Portfolio constants (Story.md base case) ----------------------------
INITIAL_CAPITAL_EUR: float = 100_000.0
BASE_STOCK_ALLOCATION: float = 0.90  # 90% stocks
BASE_CASH_ALLOCATION: float = 0.10  # 10% cash

# Base-case per-name allocation as absolute fractions of the portfolio
# (Story.md: 12% underlying + 3% 2x + 3% 3x = 18% per name; 5 names = 90%).
BASE_PER_NAME_SPLIT: dict[str, float] = {"stock": 0.12, "2x": 0.03, "3x": 0.03}

# Risk rules (fractions of total portfolio value).
PER_NAME_CAP: float = 0.33  # trim when a single name exceeds this
TRIM_FRACTION: float = 0.03  # trim 3% of portfolio value
MAX_DAILY_SELL: float = 0.10  # each sell < 10% of portfolio/day
MAX_DRAWDOWN: float = 0.20  # de-risk to cash beyond this drawdown
CRISIS_REVERT_DAYS: int = 60  # ~2 months to return to base case

# Crisis / black-swan: a basket drop exceeding CRISIS_DROP over CRISIS_LOOKBACK
# trading days deploys the cash reserve to ~100% invested (buy-the-dip), then
# reverts to the base case within CRISIS_REVERT_DAYS.
CRISIS_DROP: float = 0.15
CRISIS_LOOKBACK: int = 10

# Tax (Story.md: German flat Abgeltungsteuer with loss offsetting).
CAPITAL_GAINS_TAX_RATE: float = 0.25

# Leverage tiers available (long only).
LEVERAGE_TIERS: tuple[int, ...] = (1, 2, 3)

# Rebalance dead-band: only act when invested fraction deviates from the model's
# target exposure by more than this, to keep the base case "mostly no trades".
REBALANCE_BAND: float = 0.10

# Risk lever (Phase 3 tightening) — leverage-aware de-risk: during a confirmed
# >MAX_DRAWDOWN drawdown, the per-name daily sell (still capped at MAX_DAILY_SELL,
# Story.md) is drawn from the 3x tier first, then 2x, then stock — cutting the most
# damaging leverage first. (A vol-aware leverage throttle was evaluated and dropped:
# the walk-forward showed it cut the strategy's leverage edge and fought the crisis
# dip-buy, lowering mean outperformance from +5.7% to +3.5%.)

# --- Sentiment -----------------------------------------------------------
# News-sentiment backend: "vader" (light, default) or "finbert" (transformers,
# opt-in via the ``sentiment`` extra). FinBERT is loaded lazily on first use.
SENTIMENT_MODEL: str = "vader"

# --- Benchmark -----------------------------------------------------------
BENCHMARK_TICKER: str = "^IXIC"  # NASDAQ Composite

# --- Streamlit -----------------------------------------------------------
STREAMLIT_PORT: int = 8505


def ensure_dirs() -> None:
    """Create runtime directories that are gitignored."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
