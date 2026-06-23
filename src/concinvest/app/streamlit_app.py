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

from concinvest import __version__, config
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
def _load(n_dataset: int, with_sentiment: bool):
    # Import here so the module loads fast even if heavy deps lag.
    from concinvest.pipeline import run_phase1

    res = run_phase1(n_dataset=n_dataset, with_sentiment=with_sentiment)
    return {
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
    }


def main() -> None:
    st.set_page_config(page_title="ConcentratedInvestment", page_icon="📈", layout="wide")
    st.title("ConcentratedInvestment")
    st.caption(f"v{__version__} · forecast-driven leveraged portfolio, rules & German tax")

    with st.sidebar:
        st.header("Controls")
        n_dataset = st.select_slider(
            "Training datapoints", options=[1000, 2000, 4000, 8000], value=4000
        )
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

    data = _load(n_dataset, with_sentiment)

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
        st.plotly_chart(_positions_pie(positions), use_container_width=True)
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
    with legend.popover("Ticker legend", use_container_width=True):
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
        st.plotly_chart(fig, use_container_width=True)


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
    imp = pd.Series(data["importance"]).sort_values(ascending=False)
    st.bar_chart(imp)


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
    s = price.reindex(price.index.union(dates)).sort_index().ffill()
    return s.reindex(dates)


def _render_strategy(data: dict) -> None:
    """Third tab: per-asset buy/sell events (with tier) on the price curve, NASDAQ below."""
    import plotly.graph_objects as go

    st.subheader("Strategy per asset")
    stock = st.selectbox(
        "Asset", tickers.STOCKS,
        format_func=lambda t: f"{t} — {tickers.NAMES.get(t, t)}",
    )
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

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=price.index, y=price.values, mode="lines",
                             name=stock, line={"color": "#1f77b4"}))
    for action, symbol, color in [("buy", "triangle-up", "green"),
                                  ("sell", "triangle-down", "red")]:
        ev = tdf[tdf["action"] == action] if not tdf.empty else tdf
        if not ev.empty:
            fig.add_trace(go.Scatter(
                x=ev["date"], y=_markers_at(price, ev["date"]).values, mode="markers",
                name=action, marker={"symbol": symbol, "color": color, "size": 11},
                hovertext=[f"{action} {pos} €{a:,.0f}"
                           for pos, a in zip(ev["position"], ev["amount_eur"])],
            ))
    fig.update_layout(height=320, margin={"l": 0, "r": 0, "t": 30, "b": 0},
                      title=f"{stock} — price & buy/sell events (validation window)")
    st.plotly_chart(fig, use_container_width=True)

    _render_tier_balances(data.get("tier_curve"), stock, start, end, tdf)

    nq = _series(data["nasdaq"]).loc[start:end]
    nfig = go.Figure()
    nfig.add_trace(go.Scatter(x=nq.index, y=nq.values, mode="lines",
                              name="NASDAQ", line={"color": "#888"}))
    nfig.update_layout(height=260, margin={"l": 0, "r": 0, "t": 30, "b": 0},
                       title=f"{config.BENCHMARK_TICKER} — same window")
    st.plotly_chart(nfig, use_container_width=True)

    n = 0 if tdf is None or tdf.empty else len(tdf)
    with st.popover(f"Show buys & sells ({n})", use_container_width=True):
        if tdf is not None and not tdf.empty:
            st.dataframe(
                tdf[["date", "action", "position", "amount_eur"]]
                .rename(columns={"amount_eur": "amount_eur (gross)"})
                .style.format({"amount_eur (gross)": "€{:,.0f}",
                               "date": lambda d: d.strftime("%Y-%m-%d")}),
                width="stretch", hide_index=True,
            )
            st.caption("**position**: stock/2x/3x for the riskiest-first de-risk and the "
                       "33% trim; *all (pro-rata)* for the routine rebalance.")
        else:
            st.write(f"No trades on {stock} over the validation window.")


_TIER_COLOR = {"stock": "#1f77b4", "2x": "#ff7f0e", "3x": "#d62728"}


def _tier_markers(fig, tier: str, line: pd.Series, tdf) -> None:
    """Add buy/sell markers (with amounts) onto one tier's balance line. Tier-specific
    trades land on their tier; aggregate buys / pro-rata sells land on every tier."""
    import plotly.graph_objects as go

    if tdf is None or tdf.empty:
        return
    rel = tdf[tdf["position"].isin([tier, "all (pro-rata)"])]
    for action, symbol, color in [("buy", "triangle-up", "green"),
                                  ("sell", "triangle-down", "red")]:
        ev = rel[rel["action"] == action]
        if ev.empty:
            continue
        fig.add_trace(go.Scatter(
            x=ev["date"], y=_markers_at(line, ev["date"]).values, mode="markers",
            marker={"symbol": symbol, "color": color, "size": 9,
                    "line": {"width": 1, "color": "white"}},
            name=f"{tier} {action}", showlegend=False,
            hovertext=[f"{action} {p} €{a:,.0f}"
                       for p, a in zip(ev["position"], ev["amount_eur"])],
        ))


def _render_tier_balances(tier_curve, stock: str, start, end, tdf=None) -> None:
    """Balance (EUR) evolution of each leverage tier of ``stock``, with buy/sell markers."""
    import plotly.graph_objects as go

    if (tier_curve is None or tier_curve.empty
            or stock not in tier_curve.columns.get_level_values("ticker")):
        st.info("Per-tier balances unavailable — press **Run / refresh**.")
        return
    sub = tier_curve[stock].loc[start:end]
    order = [t for t in ("stock", "2x", "3x") if t in sub.columns]
    fig = go.Figure()
    for tier in order:
        line = sub[tier]
        fig.add_trace(go.Scatter(x=sub.index, y=line.values, mode="lines",
                                 name=tier, line={"color": _TIER_COLOR.get(tier)}))
        _tier_markers(fig, tier, line, tdf)
    fig.update_layout(height=320, margin={"l": 0, "r": 0, "t": 30, "b": 0},
                      yaxis_title="balance (€)",
                      title=f"{stock} — balance per tier with buy/sell signals")
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
