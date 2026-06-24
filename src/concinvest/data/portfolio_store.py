"""Persistence for the Live tab's user portfolios (the live portfolio store).

Each named portfolio is a CSV **file** under ``data/portfolios/`` that the user selects
between. One row per **position** — ``ticker, tier, invested_eur, buy_date`` — so every
tier (1x / 2x / 3x) of every stock carries **its own buy date**, evaluated separately;
a single ``tier == 0`` / ``ticker == 'CASH'`` row carries the cash balance. Pure local
file I/O (no network), mirroring the optional-path testability of ``data.store``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .. import config

POSITION_COLS = ["ticker", "tier", "invested_eur", "buy_date"]
_CASH = "CASH"


def portfolio_dir(base: Path | None = None) -> Path:
    """Directory holding the portfolio CSVs (created on demand)."""
    d = (base or config.DATA_DIR) / "portfolios"
    d.mkdir(parents=True, exist_ok=True)
    return d


def list_portfolios(base: Path | None = None) -> list[str]:
    """Names (file stems) of the saved portfolios, alphabetically."""
    return sorted(p.stem for p in portfolio_dir(base).glob("*.csv"))


def save_portfolio(
    name: str, positions: pd.DataFrame, cash: float, base: Path | None = None
) -> Path:
    """Write ``positions`` (``POSITION_COLS``) plus a CASH row to ``<name>.csv``."""
    rows = positions[POSITION_COLS].copy()
    cash_row = pd.DataFrame([{"ticker": _CASH, "tier": 0,
                              "invested_eur": float(cash), "buy_date": pd.NaT}])
    out = pd.concat([rows, cash_row], ignore_index=True)
    path = portfolio_dir(base) / f"{name}.csv"
    out.to_csv(path, index=False)
    return path


def load_portfolio(name: str, base: Path | None = None) -> tuple[pd.DataFrame, float]:
    """Read ``<name>.csv`` → ``(positions without the cash row, cash)``."""
    df = pd.read_csv(portfolio_dir(base) / f"{name}.csv", parse_dates=["buy_date"])
    is_cash = df["ticker"] == _CASH
    cash = float(df.loc[is_cash, "invested_eur"].sum()) if is_cash.any() else 0.0
    positions = df.loc[~is_cash, POSITION_COLS].reset_index(drop=True)
    return positions, cash
