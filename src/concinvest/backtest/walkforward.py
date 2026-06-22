"""Walk-forward (multi-window) validation.

A single 1-year holdout can flatter or punish the strategy depending on the year.
Walk-forward splits history into several consecutive windows; for each window the
model is trained only on prior data (with a horizon embargo to keep labels from
bleeding across the boundary) and the forecast backtest is run over the window.
The aggregate win rate / mean outperformance vs NASDAQ is a far more honest read
than any one window.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..ml import dataset, model
from .engine import run_forecast_backtest


@dataclass
class WalkForwardResult:
    windows: pd.DataFrame  # start, end, portfolio, benchmark, outperformance, beats

    @property
    def win_rate(self) -> float:
        return float(self.windows["beats"].mean()) if not self.windows.empty else float("nan")

    @property
    def mean_outperformance(self) -> float:
        col = self.windows["outperformance"]
        return float(col.mean()) if not col.empty else float("nan")


def _windows(dates: pd.DatetimeIndex, n_windows: int, window: int) -> list[tuple[int, pd.Timestamp, pd.Timestamp]]:
    """Consecutive non-overlapping windows of ``window`` trading days, walking back
    from the most recent date. Returns ``(start_index, start_date, end_date)`` tuples."""
    uniq = pd.DatetimeIndex(sorted(pd.DatetimeIndex(dates).unique()))
    out = []
    for i in range(n_windows):
        end_i = len(uniq) - i * window
        start_i = end_i - window
        if start_i < 0:
            break
        out.append((start_i, uniq[start_i], uniq[end_i - 1]))
    return list(reversed(out))


def walk_forward_validate(
    market: dict[str, pd.DataFrame],
    benchmark_close: pd.Series,
    panel: pd.DataFrame,
    prices: dict[str, pd.Series],
    n_windows: int = 4,
    window: int = 252,
    n_dataset: int = 10_000,
    horizon: int = 20,
    tune: bool = True,
    seed: int = 42,
) -> WalkForwardResult:
    """Train-then-test across ``n_windows`` consecutive ``window``-day windows."""
    X, y = dataset.generate_dataset(panel, prices, n=n_dataset, horizon=horizon, seed=seed)
    uniq = pd.DatetimeIndex(sorted(pd.DatetimeIndex(panel.index.get_level_values("date")).unique()))
    rows = []
    for start_i, w_start, w_end in _windows(uniq, n_windows, window):
        if start_i - horizon <= 0:
            continue  # not enough history before the window for an embargoed train set
        cutoff = uniq[start_i - horizon]  # embargo: labels must end before the window
        X_tr, y_tr = X[X.index < cutoff], y[y.index < cutoff]
        if len(X_tr) < 50 or y_tr.nunique() < 2:
            continue
        trained = model.tune_and_train(X_tr, y_tr) if tune else model.train(X_tr, y_tr)
        bt = run_forecast_backtest(
            market, benchmark_close, trained, panel,
            start=w_start.strftime("%Y-%m-%d"), end=w_end.strftime("%Y-%m-%d"),
        )
        rows.append({
            "start": w_start.date(), "end": w_end.date(),
            "portfolio": bt.portfolio_return, "benchmark": bt.benchmark_return,
            "outperformance": bt.outperformance, "beats": bt.beats_benchmark,
        })
    return WalkForwardResult(windows=pd.DataFrame(rows))
