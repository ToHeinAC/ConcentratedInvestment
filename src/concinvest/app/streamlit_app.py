"""Streamlit entrypoint.

Run on port 8505 (project convention)::

    streamlit run src/concinvest/app/streamlit_app.py --server.port 8505

Three tabs: **Current market** (cross-asset correlation + live analyst/sentiment),
**Forecast & Backtest** (5-field forecast, portfolio-vs-NASDAQ curve, feature
importance), and **Strategy** (per-asset buy/sell events with tier on the price curve,
NASDAQ below — interactive Plotly).
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from concinvest import config
from concinvest.app import exit_button
from concinvest.data import tickers
from concinvest.ml.forecast import forecasts_to_frame


_TIER_LABEL = {1: "stock", 2: "2x", 3: "3x"}


def _state_to_frame(state) -> pd.DataFrame:
    """Actual end-of-backtest book -> positions DataFrame (per-tier value + weight).

    Reflects the evolved portfolio (unequal names, time-varying cash), not the static
    base-case template. ``state`` is the forecast backtest's final ``PortfolioState``.
    """
    cols = ["ticker", "name", "type", "weight", "value_eur"]
    if state is None:
        return pd.DataFrame(columns=cols)
    total = state.total_value()
    rows = []
    for ticker in tickers.STOCKS:
        for tier in (1, 2, 3):
            value = sum(l.value for l in state.lots
                        if l.ticker == ticker and l.tier == tier)
            if value <= 0:
                continue
            rows.append({"ticker": ticker, "name": tickers.NAMES.get(ticker, ticker),
                         "type": _TIER_LABEL[tier], "weight": value / total,
                         "value_eur": value})
    rows.append({"ticker": "CASH", "name": "Cash", "type": "cash",
                 "weight": state.cash / total, "value_eur": state.cash})
    return pd.DataFrame(rows, columns=cols)


def _positions_pie(positions: pd.DataFrame):
    """Plotly donut of relative sizing per name (tiers summed) + cash."""
    import plotly.graph_objects as go

    by_name = positions.groupby("ticker", sort=False)["value_eur"].sum()
    fig = go.Figure(go.Pie(labels=list(by_name.index), values=list(by_name.values),
                           hole=0.35, sort=False, textinfo="label+percent"))
    fig.update_layout(height=340, margin={"l": 0, "r": 0, "t": 10, "b": 0},
                      showlegend=False)
    return fig


def _tier_label(tier) -> str:
    """Human-readable position tier; tier-less (proportional) sells are pro-rata."""
    return _TIER_LABEL.get(tier, "all (pro-rata)")


def _trades_to_frame(trades) -> pd.DataFrame:
    """Backtest trade log -> tidy DataFrame for the Strategy tab."""
    cols = ["date", "ticker", "action", "amount_eur", "tier", "position"]
    if not trades:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(
        [{"date": pd.to_datetime(t.date), "ticker": t.ticker, "action": t.action,
          "amount_eur": t.amount_eur, "tier": t.tier, "position": _tier_label(t.tier)}
         for t in trades]
    )


@st.cache_data(show_spinner="Fetching data, training model, running backtest…", ttl=3600)
def _load(n_dataset: int, with_sentiment: bool, strategy: str):
    # Import here so the module loads fast even if heavy deps lag.
    from concinvest.pipeline import run_phase1

    res = run_phase1(n_dataset=n_dataset, with_sentiment=with_sentiment, strategy=strategy)
    return {
        "strategy": strategy,
        "forecasts": forecasts_to_frame(res.forecasts),
        "curve": res.backtest.curve,
        "portfolio_return": res.backtest.portfolio_return,
        "benchmark_return": res.backtest.benchmark_return,
        "beats": res.backtest.beats_benchmark,
        "importance": res.model.feature_importance,
        "mean_cv": res.model.mean_cv,
        "correlation": res.correlation,
        "sentiment": res.sentiment,
        "market": res.market,
        "nasdaq": res.nasdaq,
        "trades": _trades_to_frame(res.backtest.trades),
        # getattr: tolerate a stale hot-reloaded engine without these fields.
        "positions": _state_to_frame(getattr(res.backtest, "final_state", None)),
        "tier_curve": getattr(res.backtest, "tier_curve", None),
        "cash_curve": getattr(res.backtest, "cash_curve", None),
    }


_DEFAULT_POINTS = """\
**Default (balanced)** — the Story.md base case
- Start **90% / 10% cash**; per name **9% stock + 4.5% 2x + 4.5% 3x** (18%)
- ML buy-confidence sets each name's target weight (per-stock, lagged)
- Guardrails: **33% per-name trim**, **underlying ≥ 2x+3x**, **20% drawdown → de-risk**
- Crisis: go **~100% invested** (buy-the-dip), revert to base within ~2 months
- 25% German tax with loss offset; dividends on the underlying
- *Lower variance — the all-rounder.*"""

_AGGRESSIVE_POINTS = """\
**Aggressive (3x)** — high-leverage, minimal rules
- Start **90% in 3x** (per name **18%**) + 10% cash; **3x only**
- **Stop-loss:** cut a position fully at **−60%** (underlying −20% vs entry) → cash
- **Take-profit:** at **+60%**, skim an ML amount (**≥30%**), split **50/50** into
  cash and a permanent **underlying buy-and-hold** base (pays dividends)
- **Entry:** ML buy event deploys a fixed **10% of portfolio** into 3x
- **Crash:** deploy the piled-up cash into 3x (buy-the-dip)
- **33% per-name cap** so one stock can't dominate
- *Higher variance — bigger upside, harder hits.*"""


def _render_strategy_help(st) -> None:
    """Popover (top-right of the page) showing both strategies' main points."""
    with st.popover("ℹ️ Compare strategies", width="stretch"):
        st.markdown(_DEFAULT_POINTS)
        st.divider()
        st.markdown(_AGGRESSIVE_POINTS)
        st.divider()
        st.caption("Pick the strategy in the sidebar (Default unless you switch). Both "
                   "share the same model, data, tax and dividend handling.")


def main() -> None:
    st.set_page_config(page_title="ConcentratedInvestment", page_icon="📈", layout="wide")
    title_col, help_col = st.columns([4, 1])
    title_col.title("ConcentratedInvestment")
    title_col.caption("V1.0")
    with help_col:
        _render_strategy_help(st)

    with st.sidebar:
        st.header("Controls")
        n_dataset = st.select_slider(
            "Training datapoints", options=[1000, 2000, 4000, 8000], value=4000
        )
        strategy_label = st.radio(
            "Strategy", ["Default (balanced)", "Aggressive (3x)"], index=0,
            help="Default: 90/10 guardrailed book. Aggressive: all-3x book with "
                 "−60% stop-loss / +60% take-profit and a growing underlying base.",
        )
        strategy = "aggressive" if strategy_label.startswith("Aggressive") else "default"
        with_sentiment = st.checkbox("Live sentiment (slower)", value=False)
        run = st.button("Run / refresh", type="primary")
        st.divider()
        exit_button.render(st)

    if not run and "loaded" not in st.session_state:
        st.info("Set options and press **Run / refresh** to fetch data and generate a forecast.")
        return

    if run:
        st.session_state["loaded"] = True
        _load.clear()

    data = _load(n_dataset, with_sentiment, strategy)

    tab_current, tab_forecast, tab_strategy = st.tabs(
        ["Current market", "Forecast & Backtest", "Strategy"]
    )
    with tab_current:
        _render_current(data)
    with tab_forecast:
        _render_forecast(data)
    with tab_strategy:
        _render_strategy(data)

    st.caption(f"Benchmark: {config.BENCHMARK_TICKER} · start {config.START_DATE}")


def _render_current(data: dict) -> None:
    """First tab: current portfolio, market state — correlation, live signals."""
    positions = data.get("positions")
    if positions is None or positions.empty:
        st.info("Press **Run / refresh** to compute the current portfolio.")
        return
    total_value = float(positions["value_eur"].sum())
    cash_w = float(positions.loc[positions["ticker"] == "CASH", "weight"].sum())

    st.subheader("Current portfolio")
    c1, c2 = st.columns(2)
    c1.metric("Portfolio value", f"€{total_value:,.0f}")
    c2.metric("Cash", f"{cash_w:.0%}")
    col_tbl, col_pie = st.columns([3, 2])
    with col_tbl:
        st.dataframe(
            positions.style.format({"weight": "{:.1%}", "value_eur": "€{:,.0f}"}),
            width="stretch", hide_index=True,
        )
    with col_pie:
        st.plotly_chart(_positions_pie(positions), width="stretch")
    if data.get("strategy") == "aggressive":
        st.caption(
            "Actual end-of-backtest book — aggressive all-3x strategy. Started ~90% in "
            "3x + 10% cash; the **stock** tier is the underlying buy-and-hold base built "
            "from take-profits, and cash rises as winners are skimmed / losers stopped out."
        )
    else:
        st.caption(
            "Actual end-of-backtest book — evolved from the 90/10 base case over the "
            "validation window. Weights and cash are live (cash rises on de-risk, falls "
            "to ~0% on a crisis buy-the-dip), not the static template."
        )

    _render_correlation(data["correlation"])
    _render_sentiment(data.get("sentiment"), data.get("market", {}))


def _render_correlation(corr: pd.DataFrame) -> None:
    """Cross-asset correlation (recent 60 days) with three selectable views."""
    head, legend = st.columns([4, 1])
    head.subheader("Cross-asset correlation (recent 60 days)")
    with legend.popover("Ticker legend", width="stretch"):
        st.dataframe(
            pd.DataFrame({"ticker": list(corr.columns),
                          "name": [tickers.NAMES.get(t, t) for t in corr.columns]}),
            width="stretch", hide_index=True, height=400,
        )
    view = st.radio("View", ["5 stocks vs NASDAQ", "All assets", "One stock vs all"],
                    horizontal=True, label_visibility="collapsed")
    grad = {"cmap": "RdYlGn", "vmin": -1, "vmax": 1}
    if view == "5 stocks vs NASDAQ":
        keys = [t for t in (*tickers.STOCKS, config.BENCHMARK_TICKER) if t in corr.columns]
        st.dataframe(corr.loc[keys, keys].style.format("{:.2f}")
                     .background_gradient(**grad), width="stretch")
    elif view == "All assets":
        st.dataframe(corr.style.format("{:.2f}").background_gradient(**grad),
                     width="stretch")
    else:
        import plotly.graph_objects as go

        choices = [t for t in tickers.STOCKS if t in corr.columns]
        stock = st.selectbox("Stock", choices,
                             format_func=lambda t: f"{t} — {tickers.NAMES.get(t, t)}")
        ser = corr[stock].drop(labels=[stock]).sort_values()
        fig = go.Figure(go.Scatter(
            x=ser.values, y=list(ser.index), mode="markers",
            marker={"size": 11, "color": ser.values, "colorscale": "RdYlGn",
                    "cmin": -1, "cmax": 1, "showscale": True}))
        fig.update_layout(height=460, xaxis_title=f"correlation with {stock}",
                          xaxis_range=[-1, 1], margin={"l": 0, "r": 0, "t": 10, "b": 0})
        st.plotly_chart(fig, width="stretch")


_RATINGS = [(1.5, "Strong Buy"), (2.5, "Buy"), (3.5, "Hold"), (4.5, "Sell")]


def _rating(mean) -> str:
    if mean is None or pd.isna(mean):
        return "—"
    return next((label for hi, label in _RATINGS if mean < hi), "Strong Sell")


def _news_label(score) -> str:
    if score is None or pd.isna(score):
        return "—"
    return "Positive" if score > 0.5 else "Negative" if score < -0.5 else "Neutral"


def _render_sentiment(sent, market: dict) -> None:
    """Simplified, human-readable analyst/sentiment summary per stock."""
    if sent is None or sent.empty:
        return
    rows = []
    for _, r in sent.iterrows():
        t = r["ticker"]
        price = float(market[t]["close"].iloc[-1]) if t in market else None
        target = r.get("analyst_target_mean")
        upside = (target / price - 1.0) if (target and price) else None
        rows.append({"Stock": tickers.NAMES.get(t, t),
                     "Analyst rating": _rating(r.get("recommendation_mean")),
                     "News": _news_label(r.get("news_sentiment_score")),
                     "Target upside": upside})
    df = pd.DataFrame(rows)
    st.subheader("Analyst & sentiment (live)")
    st.dataframe(df.style.format({"Target upside": "{:+.0%}"}, na_rep="—"),
                 width="stretch", hide_index=True)
    st.caption("Analyst consensus rating · news-headline tone · upside to mean price "
               "target. Live signals (display only; no history to train on yet).")


def _render_forecast(data: dict) -> None:
    """Second tab: forecast, backtest, and model feature importance."""
    st.subheader("Forecast")
    fc = data["forecasts"]
    if fc.empty:
        st.success("Base case: **hold** — no trade triggered by current conditions.")
    else:
        st.dataframe(fc, width="stretch", hide_index=True)

    st.subheader("Backtest vs. NASDAQ (validation window)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Portfolio return", f"{data['portfolio_return']:.1%}")
    c2.metric("NASDAQ return", f"{data['benchmark_return']:.1%}")
    c3.metric(
        "Outperformance",
        f"{data['portfolio_return'] - data['benchmark_return']:+.1%}",
        delta="beats benchmark" if data["beats"] else "below benchmark",
    )
    curve = data["curve"]
    pct = (curve / curve.iloc[0] - 1.0) * 100.0  # cumulative % return, both rebased to 0
    st.line_chart(pct, y_label="cumulative return (%)")
    st.caption(f"Model CV ROC-AUC (mean): {data['mean_cv']:.3f}")

    st.subheader("Feature importance")
    import plotly.graph_objects as go

    imp = (pd.Series(data["importance"])
           .drop(labels=["is_sell"], errors="ignore")  # action flag, not a market signal
           .sort_values(ascending=False))
    show_all = st.toggle("Show all features", value=False)
    shown = imp if show_all else imp.head(10)
    # Horizontal bars: x = importance, y = feature label, highest at the top.
    shown = shown.sort_values(ascending=True)
    fig = go.Figure(go.Bar(x=shown.values, y=list(shown.index), orientation="h",
                           marker_color="#1f77b4"))
    fig.update_layout(height=max(320, 22 * len(shown)),
                      margin={"l": 0, "r": 0, "t": 10, "b": 0},
                      xaxis_title="importance")
    st.plotly_chart(fig, width="stretch")


def _series(frame_or_series) -> pd.Series:
    """Coerce a close column / series to a datetime-indexed float Series."""
    idx = pd.to_datetime(list(frame_or_series.index))
    vals = frame_or_series["close"] if hasattr(frame_or_series, "columns") else frame_or_series
    return pd.Series(list(vals), index=idx)


def _markers_at(price: pd.Series, dates) -> pd.Series:
    """Price level at each trade date (ffill across non-trading days for that asset)."""
    dates = pd.DatetimeIndex(dates)
    if len(dates) == 0:
        return pd.Series(dtype=float)
    # Dedupe the price index before reindexing: a doubled market bar (repeated
    # timestamp) makes the union axis non-unique and breaks reindex. dates may still
    # repeat (several trades share one signal bar) — that's a valid reindex target.
    s = price[~price.index.duplicated(keep="last")].sort_index()
    s = s.reindex(s.index.union(pd.Index(dates).unique())).ffill()
    return s.reindex(dates)


def _signal_bar(index, dates) -> pd.DatetimeIndex:
    """Map each execution date to the **prior trading bar** in ``index`` (the signal day).

    The backtest decides from day T-1's features (confidence is lagged one day) and
    executes on T; markers are drawn on T-1 so they align with the price bar that
    triggered the trade. Display-only — the backtest is unchanged."""
    idx = pd.DatetimeIndex(index)
    if len(idx) == 0:
        return pd.DatetimeIndex(dates)
    pos = (idx.searchsorted(pd.DatetimeIndex(dates)) - 1).clip(0)
    return idx[pos]


_TIER_COLOR = {"stock": "#1f77b4", "2x": "#ff7f0e", "3x": "#d62728"}


def _add_markers(fig, row: int, line: pd.Series, ev, size: int) -> None:
    """Add buy/sell markers on ``line`` (subplot ``row``) for trade rows ``ev`` (columns
    date/action/position/amount_eur), drawn on the signal day (T-1)."""
    import plotly.graph_objects as go

    if ev is None or ev.empty:
        return
    for action, symbol, color in [("buy", "triangle-up", "green"),
                                  ("sell", "triangle-down", "red")]:
        rows = ev[ev["action"] == action]
        if rows.empty:
            continue
        sig = _signal_bar(line.index, rows["date"])
        fig.add_trace(go.Scatter(
            x=sig, y=_markers_at(line, sig).values, mode="markers", showlegend=False,
            marker={"symbol": symbol, "color": color, "size": size,
                    "line": {"width": 1, "color": "white"}},
            hovertext=[f"{action} {p} €{a:,.0f}"
                       for p, a in zip(rows["position"], rows["amount_eur"])],
        ), row=row, col=1)


def _render_cash(data: dict) -> None:
    """Strategy tab's Cash view: cash (€) over NASDAQ on a shared x-axis."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    cash = data.get("cash_curve")
    if cash is None or cash.empty:
        st.info("Cash evolution unavailable — press **Run / refresh**.")
        return
    window = data["curve"].index
    start, end = window[0], window[-1]
    cash = cash.loc[start:end]
    nq = _series(data["nasdaq"]).loc[start:end]

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
        row_heights=[0.55, 0.45],
        subplot_titles=("Cash (€)", f"NASDAQ ({config.BENCHMARK_TICKER})"),
    )
    fig.add_trace(go.Scatter(x=cash.index, y=cash.values, mode="lines",
                             name="cash", line={"color": "#2ca02c"}), row=1, col=1)
    fig.add_trace(go.Scatter(x=nq.index, y=nq.values, mode="lines",
                             name=f"NASDAQ ({config.BENCHMARK_TICKER})",
                             line={"color": "#888"}), row=2, col=1)
    fig.update_xaxes(range=[start, end])
    fig.update_yaxes(title_text="€", row=1, col=1)
    fig.update_layout(height=620, margin={"l": 0, "r": 0, "t": 50, "b": 0},
                      legend={"orientation": "h", "y": 1.1})
    st.plotly_chart(fig, width="stretch")
    st.caption("Cash balance over the validation window — rises as the book de-risks, "
               "falls toward ~0% on a crisis buy-the-dip.")


def _render_strategy(data: dict) -> None:
    """Third tab: price, per-tier balances, and NASDAQ for one asset stacked on a
    **shared x-axis**, with buy/sell signals (drawn on the decision day, T-1)."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    st.subheader("Strategy per asset")
    stock = st.selectbox(
        "Asset", [*tickers.STOCKS, "CASH"],
        format_func=lambda t: "Cash" if t == "CASH" else f"{t} — {tickers.NAMES.get(t, t)}",
    )
    if stock == "CASH":
        _render_cash(data)
        return
    window = data["curve"].index
    start, end = window[0], window[-1]
    mkt = data["market"].get(stock)
    if mkt is None:
        st.warning("No price data for this asset.")
        return
    price = _series(mkt).loc[start:end]
    trades = data["trades"]
    tdf = (trades[(trades["ticker"] == stock) & trades["date"].between(start, end)]
           if not trades.empty else trades)
    if not tdf.empty:  # derive position from tier (robust to a stale cached frame)
        tdf = tdf.assign(position=tdf["tier"].map(_tier_label))

    tier_curve = data.get("tier_curve")
    has_tiers = (tier_curve is not None and not tier_curve.empty
                 and stock in tier_curve.columns.get_level_values("ticker"))

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.045,
        row_heights=[0.4, 0.35, 0.25],
        subplot_titles=(f"{stock} — price & signals (decision day, T-1)",
                        f"{stock} — balance per tier (€)",
                        f"NASDAQ ({config.BENCHMARK_TICKER})"),
    )

    # Row 1: price + one aggregated marker per (date, action) showing the total €.
    fig.add_trace(go.Scatter(x=price.index, y=price.values, mode="lines",
                             name=stock, line={"color": "#1f77b4"}), row=1, col=1)
    if not tdf.empty:
        agg = (tdf.groupby(["date", "action"], as_index=False)["amount_eur"].sum()
               .assign(position="total"))
        _add_markers(fig, 1, price, agg, size=11)

    # Row 2: per-tier balance lines + per-tier markers (actual € for each tier).
    if has_tiers:
        sub = tier_curve[stock].loc[start:end]
        for tier in [t for t in ("stock", "2x", "3x") if t in sub.columns]:
            line = sub[tier]
            fig.add_trace(go.Scatter(x=sub.index, y=line.values, mode="lines",
                                     name=tier, line={"color": _TIER_COLOR.get(tier)}),
                          row=2, col=1)
            if not tdf.empty:
                # This tier's trades, plus any aggregate "all (pro-rata)" rows (the
                # latter only appear with a stale pre-per-tier engine — still mark them
                # on every tier so signals aren't lost before a restart).
                ev = tdf[tdf["position"].isin([tier, "all (pro-rata)"])]
                _add_markers(fig, 2, line, ev, size=9)

    # Row 3: NASDAQ benchmark over the same window.
    nq = _series(data["nasdaq"]).loc[start:end]
    fig.add_trace(go.Scatter(x=nq.index, y=nq.values, mode="lines",
                             name=f"NASDAQ ({config.BENCHMARK_TICKER})",
                             line={"color": "#888"}), row=3, col=1)

    fig.update_xaxes(range=[start, end])  # identical x-axis across all three panels
    fig.update_yaxes(title_text="€", row=2, col=1)
    fig.update_layout(height=760, margin={"l": 0, "r": 0, "t": 50, "b": 0},
                      legend={"orientation": "h", "y": 1.07})
    st.plotly_chart(fig, width="stretch")
    if not has_tiers:
        st.info("Per-tier balances unavailable — press **Run / refresh**.")

    n = 0 if tdf is None or tdf.empty else len(tdf)
    with st.popover(f"Show buys & sells ({n})", width="stretch"):
        if tdf is not None and not tdf.empty:
            st.dataframe(
                tdf[["date", "action", "position", "amount_eur"]]
                .rename(columns={"amount_eur": "amount_eur (gross)"})
                .style.format({"amount_eur (gross)": "€{:,.0f}",
                               "date": lambda d: d.strftime("%Y-%m-%d")}),
                width="stretch", hide_index=True,
            )
            st.caption("**position**: the leverage tier (stock / 2x / 3x) traded; each "
                       "row is the actual € for that tier.")
        else:
            st.write(f"No trades on {stock} over the validation window.")


if __name__ == "__main__":
    main()
