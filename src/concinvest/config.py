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
# (Story.md: 9% underlying + 4.5% 2x + 4.5% 3x = 18% per name; 5 names = 90%).
# Underlying == leveraged (9% == 4.5+4.5) sits exactly on the underlying-dominance
# boundary, so any up-move tips leverage ahead and triggers the dominance trim.
BASE_PER_NAME_SPLIT: dict[str, float] = {"stock": 0.09, "2x": 0.045, "3x": 0.045}

# Risk rules (fractions of total portfolio value).
PER_NAME_CAP: float = 0.33  # trim when a single name exceeds this
MIN_NAME_WEIGHT: float = 0.06  # floor: never de-risk a name below this (held as underlying)
MAX_CASH: float = 0.70  # cash must stay below this (implied by 5 names x MIN_NAME_WEIGHT)
TRIM_FRACTION: float = 0.03  # trim 3% of portfolio value
MAX_DAILY_SELL: float = 0.10  # each sell < 10% of portfolio/day
MIN_TRADE_EUR: float = 500.0  # never buy or sell an order smaller than this
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

# --- Aggressive strategy (selectable; the base case above stays the default) ---
# An all-3x book with minimal rules (no 33%/dominance/drawdown guardrails): start
# 90% in 3x (per-name 18%) + 10% cash; cut a 3x lot fully at -60%; once a lot is
# +60% skim an ML amount (>=30%) split 50/50 into cash and a permanent underlying
# buy-and-hold lot; ML buy events deploy a fixed 10%-of-portfolio 3x chunk; a crisis
# deploys the accumulated cash (buy-the-dip).
AGG_BASE_SPLIT: dict[str, float] = {"3x": 0.18}  # per-name 18% all-3x; 5 names = 90%
AGG_STOP_LOSS: float = 0.40  # exit a 3x lot when value <= 40% of cost basis (-60%)
AGG_TAKE_PROFIT: float = 1.60  # skim eligible when value >= 160% of the tp-reference (+60%)
AGG_MIN_TP_FRACTION: float = 0.30  # take profit on at least 30% of the position
AGG_TP_TO_UNDERLYING: float = 0.50  # half of net profit-taking proceeds buy the underlying
AGG_ENTRY_THRESHOLD: float = 0.55  # ML buy-confidence required to fire an entry event
AGG_ENTRY_CHUNK: float = 0.10  # deploy 10% of portfolio value per entry event

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
