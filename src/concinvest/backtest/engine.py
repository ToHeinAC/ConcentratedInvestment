"""Phase 1 backtest: model-timed concentrated portfolio vs. NASDAQ.

A deliberately simple, point-in-time strategy to validate the end-to-end loop:
hold an equal-weight basket of the 5 stocks, but scale daily equity *exposure* by
the model's average buy-confidence (the rest sits in cash). Exposure is lagged one
day to avoid look-ahead. Returns cumulative value vs. a NASDAQ buy-and-hold.

Full allocation/risk/leverage/tax logic (Story.md) lands in Phase 4.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .. import config
from ..ml.dataset import FEATURE_COLS
from ..ml.model import TrainedModel
from ..portfolio import rules
from ..portfolio import state as pstate

_TIER_LABEL = {1: "stock", 2: "2x", 3: "3x"}


@dataclass
class BacktestResult:
    curve: pd.DataFrame  # index=date, columns: portfolio, benchmark
    portfolio_return: float
    benchmark_return: float
    trades: list[rules.Trade] = field(default_factory=list)  # forecast backtest only
    final_state: pstate.PortfolioState | None = None  # end-of-window book (forecast bt)
    tier_curve: pd.DataFrame | None = None  # daily per-(ticker, tier) value (forecast bt)
    cash_curve: pd.Series | None = None  # daily cash balance (forecast bt)

    @property
    def outperformance(self) -> float:
        return self.portfolio_return - self.benchmark_return

    @property
    def beats_benchmark(self) -> bool:
        return self.portfolio_return > self.benchmark_return


def _daily_exposure(model: TrainedModel, panel: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.Series:
    """Mean P(buy profitable, leverage=1) across stocks per date, lagged 1 day."""
    rows = panel.loc[panel.index.get_level_values("date").isin(dates)].copy()
    if rows.empty:
        return pd.Series(1.0, index=dates)
    feats = rows.reindex(columns=FEATURE_COLS).copy()
    feats["is_sell"] = 0
    feats["leverage"] = 1
    feats = feats.fillna(0.0)
    conf = model.predict_confidence(feats)
    rows = rows.assign(_conf=conf)
    expo = rows.groupby(level="date")["_conf"].mean()
    expo = expo.reindex(dates).ffill().fillna(1.0)
    return expo.shift(1).bfill().clip(0.0, 1.0)


def _name_confidence(
    model: TrainedModel,
    panel: pd.DataFrame,
    dates: pd.DatetimeIndex,
    is_sell: int = 0,
    leverage: int = 1,
) -> pd.DataFrame:
    """Per-stock action confidence as a date×ticker frame, lagged 1 day.

    Unlike ``_daily_exposure`` (which averages across stocks into one series), this
    keeps each name's own confidence so the forecast backtest can trade names
    independently (Story.md: the forecast is per ticker). ``is_sell``/``leverage`` pick
    the action scored — the default book uses buy/1x; the aggressive book also asks for
    buy/3x and sell/3x."""
    rows = panel.loc[panel.index.get_level_values("date").isin(dates)].copy()
    if rows.empty:
        return pd.DataFrame(index=dates)
    feats = rows.reindex(columns=FEATURE_COLS).copy()
    feats["is_sell"] = is_sell
    feats["leverage"] = leverage
    feats = feats.fillna(0.0)
    conf = pd.Series(model.predict_confidence(feats), index=rows.index)
    wide = conf.unstack("ticker").reindex(dates).ffill().fillna(1.0)
    return wide.shift(1).bfill().clip(0.0, 1.0)


def run_backtest(
    market: dict[str, pd.DataFrame],
    benchmark_close: pd.Series,
    model: TrainedModel,
    panel: pd.DataFrame,
    start: str | None = None,
) -> BacktestResult:
    """Run the Phase 1 backtest over the window starting at ``start``."""
    # Equal-weight daily returns across available stocks.
    closes = pd.DataFrame({t: df["close"] for t, df in market.items()})
    closes.index = pd.to_datetime(closes.index)
    closes = closes.sort_index()
    if start:
        closes = closes.loc[start:]
    rets = closes.pct_change().mean(axis=1)  # equal-weight basket return
    dates = pd.DatetimeIndex(rets.index)

    exposure = _daily_exposure(model, panel, dates)
    strat_ret = (rets * exposure).fillna(0.0)
    portfolio = config.INITIAL_CAPITAL_EUR * (1.0 + strat_ret).cumprod()

    bench = benchmark_close.copy()
    bench.index = pd.to_datetime(bench.index)
    bench = bench.reindex(dates).ffill()
    benchmark = config.INITIAL_CAPITAL_EUR * (bench / bench.iloc[0])

    curve = pd.DataFrame({"portfolio": portfolio, "benchmark": benchmark}).dropna()
    p_ret = float(curve["portfolio"].iloc[-1] / curve["portfolio"].iloc[0] - 1.0)
    b_ret = float(curve["benchmark"].iloc[-1] / curve["benchmark"].iloc[0] - 1.0)
    return BacktestResult(curve=curve, portfolio_return=p_ret, benchmark_return=b_ret)


def _dividend_yields(
    market: dict[str, pd.DataFrame], dates: pd.DatetimeIndex
) -> pd.DataFrame:
    """Per-day dividend yield per stock = total-return minus price return.

    With ``auto_adjust=False``, ``adj_close`` is the dividend/split-adjusted total
    return while ``close`` is price-only, so their daily return difference recovers
    the dividend yield (Story.md: dividends accrue to the underlying only). Stocks
    without an ``adj_close`` column contribute zero.
    """
    cols = {}
    for ticker, df in market.items():
        if "adj_close" not in df.columns:
            continue
        idx = pd.to_datetime(df.index)
        adj = pd.Series(df["adj_close"].values, index=idx).pct_change()
        price = pd.Series(df["close"].values, index=idx).pct_change()
        cols[ticker] = (adj - price).clip(lower=0.0)
    if not cols:
        return pd.DataFrame(index=dates)
    return pd.DataFrame(cols).reindex(dates).fillna(0.0)


def _benchmark_curve(benchmark_close: pd.Series, dates: pd.DatetimeIndex) -> pd.Series:
    """NASDAQ buy-and-hold rebased to the initial capital over ``dates``."""
    bench = benchmark_close.copy()
    bench.index = pd.to_datetime(bench.index)
    # ffill interior gaps; bfill a leading NaN when the window opens on a date the
    # benchmark didn't trade (e.g. a US holiday while EU/JP stocks traded).
    bench = bench.reindex(dates).ffill().bfill()
    return config.INITIAL_CAPITAL_EUR * (bench / bench.iloc[0])


def run_rules_backtest(
    market: dict[str, pd.DataFrame],
    benchmark_close: pd.Series,
    start: str | None = None,
    capital: float = config.INITIAL_CAPITAL_EUR,
) -> BacktestResult:
    """Replay the Story.md base-case leveraged book under the daily risk guardrails.

    Starts from the 90/10 base case (per-name 9%/4.5%/4.5% stock/2x/3x), marks every
    lot to market each day, then applies the sell-side guardrails (drawdown de-risk,
    per-name trim, 10%/day cap) with German tax on realized gains. No re-entry yet —
    forecast-driven buys/sells are the next Phase 4 increment.
    """
    closes = pd.DataFrame({t: df["close"] for t, df in market.items()})
    closes.index = pd.to_datetime(closes.index)
    closes = closes.sort_index()
    if start:
        closes = closes.loc[start:]
    rets = closes.pct_change().fillna(0.0)
    dates = pd.DatetimeIndex(rets.index)

    divs = _dividend_yields(market, dates)
    state = pstate.build_base_case(capital, stocks=list(market))
    values: list[float] = []
    for date, row in rets.iterrows():
        state.mark(row.to_dict())
        state.pay_dividends(divs.loc[date].to_dict() if date in divs.index else {})
        rules.apply_guardrails(state)
        values.append(state.total_value())

    portfolio = pd.Series(values, index=dates)
    benchmark = _benchmark_curve(benchmark_close, dates)
    curve = pd.DataFrame({"portfolio": portfolio, "benchmark": benchmark}).dropna()
    p_ret = float(curve["portfolio"].iloc[-1] / curve["portfolio"].iloc[0] - 1.0)
    b_ret = float(curve["benchmark"].iloc[-1] / curve["benchmark"].iloc[0] - 1.0)
    return BacktestResult(curve=curve, portfolio_return=p_ret, benchmark_return=b_ret)


# Classifier-neutral confidence: at/above this the model is not bearish, so the
# book stays fully at the base-case allocation; below it we de-risk toward cash.
_NEUTRAL_CONF: float = 0.5


def _target_exposure(confidence: float) -> float:
    """Base-case-faithful target equity fraction from mean buy-confidence.

    Holds the 90% base case while the model is neutral-to-bullish
    (``confidence >= 0.5``); only a bearish read (< 0.5) scales exposure down
    proportionally toward cash. The drawdown guardrail de-risks crashes separately.
    """
    return config.BASE_STOCK_ALLOCATION * min(1.0, confidence / _NEUTRAL_CONF)


# Per-name base weight (9%+4.5%+4.5%) and a dead-band scaled from the book-level
# REBALANCE_BAND by the per-name share of the base allocation, so a single name's
# rebalance fires at the same relative sensitivity as the old aggregate dial.
_PER_NAME_BASE: float = sum(config.BASE_PER_NAME_SPLIT.values())
_PER_NAME_BAND: float = config.REBALANCE_BAND * _PER_NAME_BASE / config.BASE_STOCK_ALLOCATION


def _target_name_fraction(confidence: float) -> float:
    """Base-case-faithful target portfolio fraction for one name from its own
    buy-confidence (holds the per-name base while neutral-to-bullish, de-risks below),
    floored at ``MIN_NAME_WEIGHT`` so each stock stays ≥ 6% of the book (Story.md)."""
    target = _PER_NAME_BASE * min(1.0, confidence / _NEUTRAL_CONF)
    return max(target, config.MIN_NAME_WEIGHT)


def _rebalance_names_to_target(
    state: pstate.PortfolioState, targets: dict[str, float], stocks: list[str]
) -> list[rules.Trade]:
    """Nudge **each name** toward its own target portfolio fraction, within the daily
    move cap and a per-name dead-band. Names are handled independently (Story.md), so a
    bearish read on one stock trims only that stock. Returns the trades performed."""
    total = state.total_value()
    if total <= 0:
        return []
    trades: list[rules.Trade] = []
    for ticker in stocks:
        dev = targets.get(ticker, _PER_NAME_BASE) - state.name_value(ticker) / total
        if abs(dev) < _PER_NAME_BAND:
            continue
        move = min(abs(dev), config.MAX_DAILY_SELL) * total  # cap daily turnover
        if dev > 0:
            trades += _deploy_name(state, ticker, move)
        # Routine confidence-rebalance sells pro-rata across tiers (keeps the leverage
        # edge in up-markets); tier-graded shedding is reserved for the two de-risking
        # events Story.md names — the crash drawdown and the post-upstreak 33% trim.
        else:
            trades += _sell_proportional(state, ticker, move)
    return trades


def _is_crisis(basket_ret: pd.Series, i: int) -> bool:
    """True if the basket fell more than ``CRISIS_DROP`` over the trailing lookback."""
    if i + 1 < config.CRISIS_LOOKBACK:
        return False
    window = basket_ret.iloc[i + 1 - config.CRISIS_LOOKBACK : i + 1]
    return float((1.0 + window).prod() - 1.0) <= -config.CRISIS_DROP


def _deploy_name(
    state: pstate.PortfolioState, ticker: str, amount: float
) -> list[rules.Trade]:
    """Deploy ``amount`` of cash into one name by the base-case tier split (9/4.5/4.5).
    Returns one buy ``Trade`` **per funded tier** (each with its actual € invested).
    Orders below ``MIN_TRADE_EUR`` are skipped (Story.md: no trade < €500)."""
    if amount < config.MIN_TRADE_EUR:
        return []
    split = config.BASE_PER_NAME_SPLIT
    weight_sum = sum(split.values())
    trades: list[rules.Trade] = []
    for tier_name, weight in split.items():
        tier = rules._TIER_OF[tier_name]
        invested = state.buy(ticker, tier, amount * weight / weight_sum)
        if invested > 0:
            trades.append(rules.Trade(ticker, "buy", invested, tier=tier))
    return trades


def _deploy(state: pstate.PortfolioState, amount: float, stocks: list[str]) -> list[rules.Trade]:
    """Deploy ``amount`` of cash equally across stocks by the base-case tier split
    (crisis buy-the-dip). One buy ``Trade`` per funded (stock, tier)."""
    per_stock = amount / len(stocks)
    trades: list[rules.Trade] = []
    for ticker in stocks:
        trades += _deploy_name(state, ticker, per_stock)
    return trades


def _sell_proportional(
    state: pstate.PortfolioState, ticker: str, amount: float
) -> list[rules.Trade]:
    """Sell ``amount`` EUR of ``ticker`` pro-rata across its tiers (selling logic is
    unchanged — `state.sell_name`), returning one sell ``Trade`` per tier with the
    actual € shed from that tier (so the trade log shows real per-tier amounts, not one
    aggregate 'all (pro-rata)' row)."""
    name_val = state.name_value(ticker)
    gross = min(amount, name_val)
    if gross < config.MIN_TRADE_EUR:  # Story.md: no order smaller than €500
        return []
    frac = gross / name_val
    by_tier: dict[int, float] = {}
    for lot in state.lots:
        if lot.ticker == ticker:
            by_tier[lot.tier] = by_tier.get(lot.tier, 0.0) + lot.value * frac
    state.sell_name(ticker, amount)
    return [rules.Trade(ticker, "sell", v, tier=t) for t, v in sorted(by_tier.items())]


def run_forecast_backtest(
    market: dict[str, pd.DataFrame],
    benchmark_close: pd.Series,
    model: TrainedModel,
    panel: pd.DataFrame,
    start: str | None = None,
    end: str | None = None,
    capital: float = config.INITIAL_CAPITAL_EUR,
) -> BacktestResult:
    """Rules + forecast backtest: the base-case leveraged book where **each name's**
    target portfolio fraction tracks that name's own buy-confidence (lagged, scaled by
    its per-name base weight), with cash re-entry, daily guardrails, and German tax.
    Names are rebalanced independently (Story.md: the forecast is per ticker), so a
    bearish read on one stock trims only that stock. ``start``/``end`` bound the window
    (inclusive)."""
    closes = pd.DataFrame({t: df["close"] for t, df in market.items()})
    closes.index = pd.to_datetime(closes.index)
    closes = closes.sort_index()
    if start is not None or end is not None:
        closes = closes.loc[start:end]
    rets = closes.pct_change().fillna(0.0)
    dates = pd.DatetimeIndex(rets.index)

    name_conf = _name_confidence(model, panel, dates)  # per-stock buy-confidence, lagged
    basket_ret = rets.mean(axis=1)  # equal-weight basket return, for crisis detection
    divs = _dividend_yields(market, dates)
    stocks = list(market)
    state = pstate.build_base_case(capital, stocks=stocks)
    crisis_day: int | None = None
    values: list[float] = []
    cash_vals: list[float] = []  # end-of-day cash for the Strategy tab's Cash view
    tier_rows: list[dict] = []  # end-of-day value per (ticker, tier) for the Strategy tab
    trades: list[rules.Trade] = []
    for i, (date, row) in enumerate(rets.iterrows()):
        state.mark(row.to_dict())
        state.pay_dividends(divs.loc[date].to_dict() if date in divs.index else {})
        # Structural caps apply even in crisis: per-name 33% trim + underlying ≥ 2x+3x.
        day = rules.trim_overweight(state) + rules.enforce_underlying_dominance(state)
        in_crisis = crisis_day is not None and (i - crisis_day) < config.CRISIS_REVERT_DAYS
        if not in_crisis:
            crisis_day = None
            day += rules.drawdown_derisk(state)  # riskiest-tier-first de-risk
            if _is_crisis(basket_ret, i):
                crisis_day = i
                day += _deploy(state, state.cash, stocks)  # buy the dip
            else:
                conf_row = name_conf.loc[date] if date in name_conf.index else None
                targets = {
                    t: _target_name_fraction(
                        float(conf_row[t])
                        if conf_row is not None and t in conf_row else 1.0
                    )
                    for t in stocks
                }
                day += _rebalance_names_to_target(state, targets, stocks)
        # During crisis: stay fully invested (no de-risk, no rebalance toward cash).
        for t in day:
            t.date = date
        trades.extend(day)
        values.append(state.total_value())
        cash_vals.append(state.cash)
        tier_rows.append(_tier_snapshot(state))

    return _assemble_result(values, cash_vals, tier_rows, dates, benchmark_close,
                            trades, state)


def _tier_snapshot(state: pstate.PortfolioState) -> dict[tuple[str, str], float]:
    """End-of-day value per ``(ticker, tier_label)`` for the Strategy tab's tier_curve."""
    snap: dict[tuple[str, str], float] = {}
    for lot in state.lots:
        key = (lot.ticker, _TIER_LABEL[lot.tier])
        snap[key] = snap.get(key, 0.0) + lot.value
    return snap


def _assemble_result(
    values: list[float],
    cash_vals: list[float],
    tier_rows: list[dict],
    dates: pd.DatetimeIndex,
    benchmark_close: pd.Series,
    trades: list[rules.Trade],
    state: pstate.PortfolioState,
) -> BacktestResult:
    """Build the ``BacktestResult`` (curve + tier_curve + cash_curve + returns) shared by
    the forecast and aggressive backtests."""
    portfolio = pd.Series(values, index=dates)
    cash_curve = pd.Series(cash_vals, index=dates)
    tier_curve = pd.DataFrame(tier_rows, index=dates).fillna(0.0)
    if not tier_curve.empty:
        tier_curve.columns = pd.MultiIndex.from_tuples(
            tier_curve.columns, names=["ticker", "tier"]
        )
    benchmark = _benchmark_curve(benchmark_close, dates)
    curve = pd.DataFrame({"portfolio": portfolio, "benchmark": benchmark}).dropna()
    p_ret = float(curve["portfolio"].iloc[-1] / curve["portfolio"].iloc[0] - 1.0)
    b_ret = float(curve["benchmark"].iloc[-1] / curve["benchmark"].iloc[0] - 1.0)
    return BacktestResult(curve=curve, portfolio_return=p_ret, benchmark_return=b_ret,
                          trades=trades, final_state=state, tier_curve=tier_curve,
                          cash_curve=cash_curve)


# --- Aggressive strategy (all-3x book, selectable in the UI) ------------------
# Minimal-rule alternative to the guardrailed default book: only the four rules below
# (no 33%/dominance/drawdown guardrails, no daily-sell cap). See config.AGG_* and
# IMPLEMENTATION.md "Strategies".


def _agg_stop_loss(state: pstate.PortfolioState) -> list[rules.Trade]:
    """Auto-sell every 3x lot down to <= AGG_STOP_LOSS of its cost basis (-60%); the full
    proceeds go to cash (redeployable in a crash). Tier-1 underlying lots are untouched."""
    trades: list[rules.Trade] = []
    for lot in list(state.lots):
        if lot.tier != 3 or lot.value > config.AGG_STOP_LOSS * lot.cost_basis:
            continue
        gross = lot.value
        if gross < config.MIN_TRADE_EUR:
            continue
        state.sell_lot(lot, gross)
        trades.append(rules.Trade(lot.ticker, "sell", gross, tier=3))
    return trades


def _agg_take_profit(
    state: pstate.PortfolioState, sell_row: pd.Series | None
) -> list[rules.Trade]:
    """Skim every 3x lot that is >= AGG_TAKE_PROFIT of its take-profit reference (+60%):
    sell ``max(MIN_TP, sell-confidence)`` of it, re-base the lot's reference to its
    remainder (needs another +60% to re-trigger), and route AGG_TP_TO_UNDERLYING of the
    net proceeds into a permanent tier-1 underlying lot (the rest stays cash)."""
    trades: list[rules.Trade] = []
    for lot in list(state.lots):
        if lot.tier != 3 or lot.value < config.AGG_TAKE_PROFIT * lot.tp_basis:
            continue
        conf = float(sell_row[lot.ticker]) if (
            sell_row is not None and lot.ticker in sell_row) else config.AGG_MIN_TP_FRACTION
        frac = min(1.0, max(config.AGG_MIN_TP_FRACTION, conf))
        gross = lot.value * frac
        if gross < config.MIN_TRADE_EUR:
            continue
        ticker = lot.ticker
        net = state.sell_lot(lot, gross)
        if lot.value > 1e-9:  # re-base the surviving remainder's profit reference
            lot.tp_basis = lot.value
        trades.append(rules.Trade(ticker, "sell", gross, tier=3))
        invested = state.buy(ticker, 1, config.AGG_TP_TO_UNDERLYING * net)
        if invested > 0:
            trades.append(rules.Trade(ticker, "buy", invested, tier=1))
    return trades


def _agg_entries(
    state: pstate.PortfolioState, buy_row: pd.Series | None, total: float
) -> list[rules.Trade]:
    """ML buy events (3x only): each name whose buy-confidence clears AGG_ENTRY_THRESHOLD
    deploys a fixed AGG_ENTRY_CHUNK of portfolio value into a new 3x lot, funded by cash.
    Names already at the per-name cap are skipped (don't add to an over-concentrated name)."""
    if buy_row is None:
        return []
    cap = config.PER_NAME_CAP * total
    trades: list[rules.Trade] = []
    for ticker in buy_row.index:
        if float(buy_row[ticker]) < config.AGG_ENTRY_THRESHOLD:
            continue
        if state.name_value(ticker) >= cap:  # already concentrated -> no new entry
            continue
        amount = min(config.AGG_ENTRY_CHUNK * total, state.cash)
        if amount < config.MIN_TRADE_EUR:
            continue
        invested = state.buy(ticker, 3, amount)
        if invested > 0:
            trades.append(rules.Trade(ticker, "buy", invested, tier=3))
    return trades


def _agg_cap_overweight(state: pstate.PortfolioState) -> list[rules.Trade]:
    """Cap single-stock concentration: trim any name whose total (underlying + 3x) exceeds
    ``PER_NAME_CAP`` of the book back to the cap, shedding the **3x tier first** (proceeds
    to cash). The aggressive book has no other concentration controls, so this limits the
    damage from a single-stock blow-up. Reuses the default book's ``sell_riskiest_first``."""
    total = state.total_value()
    if total <= 0:
        return []
    cap = config.PER_NAME_CAP * total
    trades: list[rules.Trade] = []
    for ticker in sorted({lot.ticker for lot in state.lots}):
        excess = state.name_value(ticker) - cap
        if excess >= config.MIN_TRADE_EUR:
            trades += rules.sell_riskiest_first(state, ticker, excess)
    return trades


def _agg_deploy_crisis(
    state: pstate.PortfolioState, stocks: list[str]
) -> list[rules.Trade]:
    """Crisis buy-the-dip: split the accumulated cash equally across names into 3x lots."""
    cash = state.cash
    if cash < config.MIN_TRADE_EUR:
        return []
    per_stock = cash / len(stocks)
    trades: list[rules.Trade] = []
    for ticker in stocks:
        invested = state.buy(ticker, 3, per_stock)
        if invested > 0:
            trades.append(rules.Trade(ticker, "buy", invested, tier=3))
    return trades


def run_aggressive_backtest(
    market: dict[str, pd.DataFrame],
    benchmark_close: pd.Series,
    model: TrainedModel,
    panel: pd.DataFrame,
    start: str | None = None,
    end: str | None = None,
    capital: float = config.INITIAL_CAPITAL_EUR,
) -> BacktestResult:
    """Aggressive all-3x backtest (the selectable alternative to ``run_forecast_backtest``).

    Starts 90% in 3x (per-name 18%) + 10% cash, then each day: mark, pay dividends on the
    growing underlying base, run the -60% stop-loss and +60%/ML take-profit, either deploy
    all cash on a crisis (buy-the-dip) or fire ML 3x entries, and finally cap single-stock
    concentration at ``PER_NAME_CAP`` (33%). No drawdown/dominance guardrails beyond that.
    """
    closes = pd.DataFrame({t: df["close"] for t, df in market.items()})
    closes.index = pd.to_datetime(closes.index)
    closes = closes.sort_index()
    if start is not None or end is not None:
        closes = closes.loc[start:end]
    rets = closes.pct_change().fillna(0.0)
    dates = pd.DatetimeIndex(rets.index)

    buy_conf = _name_confidence(model, panel, dates, is_sell=0, leverage=3)
    sell_conf = _name_confidence(model, panel, dates, is_sell=1, leverage=3)
    basket_ret = rets.mean(axis=1)  # for crisis detection
    divs = _dividend_yields(market, dates)
    stocks = list(market)
    state = pstate.build_base_case(capital, stocks=stocks, split=config.AGG_BASE_SPLIT)
    values: list[float] = []
    cash_vals: list[float] = []
    tier_rows: list[dict] = []
    trades: list[rules.Trade] = []
    for i, (date, row) in enumerate(rets.iterrows()):
        state.mark(row.to_dict())
        state.pay_dividends(divs.loc[date].to_dict() if date in divs.index else {})
        day = _agg_stop_loss(state)
        day += _agg_take_profit(state, sell_conf.loc[date] if date in sell_conf.index else None)
        if _is_crisis(basket_ret, i):
            day += _agg_deploy_crisis(state, stocks)  # buy the dip with the cash hoard
        else:
            buy_row = buy_conf.loc[date] if date in buy_conf.index else None
            day += _agg_entries(state, buy_row, state.total_value())
        day += _agg_cap_overweight(state)  # enforce the per-name concentration ceiling
        for t in day:
            t.date = date
        trades.extend(day)
        values.append(state.total_value())
        cash_vals.append(state.cash)
        tier_rows.append(_tier_snapshot(state))

    return _assemble_result(values, cash_vals, tier_rows, dates, benchmark_close,
                            trades, state)
