# Architecture

Deep reference for the `concinvest` package. For the compact build plan and phase
status see [../IMPLEMENTATION.md](../IMPLEMENTATION.md); for the database tables see
[SCHEMA.md](SCHEMA.md).

## Data flow

```
                    yfinance (network)
                          │
            ┌─────────────┴──────────────┐
            ▼                            ▼
   data.fetch.download_ohlcv     data.fetch.fetch_* (analyst/news/options)
            │                            │
            ▼                            ▼
   features.technical          features.analyst / sentiment / options
   features.cross_asset                 │
            │                            │
            ├──────────► data.store (SQLite) ◄──────────┤
            │            ohlcv_raw, daily_market,
            │            cross_asset, sentiment_analyst
            ▼
   ml.dataset.build_feature_panel        (per-(date,ticker) feature matrix)
            │
            ▼
   ml.dataset.generate_dataset           (synthetic buy/sell datapoints, X/y)
            │
            ▼
   ml.model.train                        (RandomForest + TimeSeriesSplit CV)
            │
        ┌───┴────────────────────────────┐
        ▼                                ▼
   ml.forecast.forecast          backtest.engine.run_forecast_backtest
   (5-field recommendation)      (portfolio value vs NASDAQ)
        │                                │
        └────────────► app.streamlit_app ◄──────────┘
```

`pipeline.run_phase1` wires the whole chain; `pipeline.fetch_and_store` is the
reusable daily-ETL building block (later driven by the Phase 5 cron job).

## Module responsibilities

### `data/`
- **`tickers.py`** — universe constants. `STOCKS` (5 fixed), `INDICES`, `COMMODITIES`,
  `BONDS`, `MACRO`, `CRYPTO`. `CORE_TICKERS` (13) is the Phase 1 slice; `ALL_TICKERS`
  (27, de-duplicated) is the full universe; `NAMES` is a flat lookup.
- **`fetch.py`** — all network access. `download_ohlcv()` batches via
  `yf.download(group_by="ticker", threads=True)` with retries; per-ticker
  `fetch_recommendation_mean()`, `fetch_news_headlines()`, `fetch_put_call_ratio()`,
  `fetch_eps_revisions()`, `fetch_analyst_target_mean()`, `fetch_iv_skew()` carry a
  `_META_DELAY` (0.5s) pause and degrade to `None`/`[]` on error.
  `fetch_german_headlines()` scrapes `finanznachrichten.de` (best-effort, pure parse
  in `_parse_finanznachrichten`); `_iv_at` picks the nearest-strike implied vol.
- **`store.py`** — SQLite. `connect()` creates the schema and runs `_migrate()`
  (additive `ALTER TABLE`s from `_MIGRATIONS` for pre-Phase-2 DBs); `upsert()` does
  generic `INSERT OR REPLACE`; `read_table()` reads back. Date columns normalised to
  ISO text PKs. See [SCHEMA.md](SCHEMA.md).

### `features/`
- **`technical.py`** — pure pandas. `sma`/`ema`/`rsi`/`macd`/`bollinger` helpers and
  `add_technical_features()` which appends the Table-1 columns to an OHLCV frame.
  RSI uses Wilder-style EWM; zero average loss yields RSI = 100 via the natural
  `avg_gain/0 → +inf` path.
- **`cross_asset.py`** — `build_cross_asset_frame()` builds Table-3 ratios
  (gold/oil, copper/gold, VIX level + sma20 ratio, 10y yield, 10y-5y spread, VVIX
  level, GSCI sma20 ratio, dollar index, BTC sma20 ratio) from a dict of close-price
  Series, aligned on the date union; series absent from the dict are skipped.
- **`sentiment.py`** — `score_headlines(model=…)` scores on the `[-3, 3]` scale via
  one of two lazily-loaded backends behind a shared lock: NLTK VADER (default, mean
  compound × 3) or FinBERT (`P(pos) − P(neg)` × 3, opt-in `sentiment` extra). Backend
  defaults to `config.SENTIMENT_MODEL`.
- **`analyst.py`** — `build_sentiment_row()` assembles a one-row Table-2 frame
  (recommendation mean, news sentiment over yfinance + German headlines, put/call,
  EPS revisions, analyst target, IV skew); these are stored/displayed only, not
  model features.
- **`options.py`** — `put_call_ratio()` and `iv_skew()` feature-facing wrappers over
  the fetches.

### `ml/`
- **`dataset.py`** — `FEATURE_COLS` is the model contract (technical + cross-asset +
  **momentum lags** of those at `_lag{3,10,30,100}` + sentiment placeholders + action
  encoding). `build_feature_panel()` joins per-stock technicals with date-aligned
  cross-asset features into a `(date, ticker)` MultiIndex panel, then appends each base
  feature's lagged values (per-ticker `shift`, leading edge → 0; strictly past data, no
  leakage). `generate_dataset()` samples `n` datapoints (half buys / half sells; 100k is
  the Story.md target); each is a market snapshot plus `is_sell`/`leverage`; the label
  is "profitable action" from a forward-return horizon (buy good if price rose, sell
  good if it fell). Rows are **returned sorted by snapshot date** (DatetimeIndex) so
  `TimeSeriesSplit` is honest; features are point-in-time, labels strictly forward — no
  leakage. `train_validate_split()` carves the last `VALIDATION_YEARS` off by calendar
  date (Story.md 4y-train / 1y-validate).
- **`model.py`** — `train()` fits a `RandomForestClassifier` with `TimeSeriesSplit`
  ROC-AUC CV and feature importances, returning a `TrainedModel` whose
  `predict_confidence()` gives `P(profitable)`. `tune()` selects the best `PARAM_GRID`
  entry by mean TSCV AUC; `select_features()` prunes features below `MIN_IMPORTANCE`
  (action encoding always kept); `tune_and_train(prune=True)` tunes, prunes, and
  refits. `FEATURE_COLS` stays the stable superset callers build, while
  `TrainedModel.features` records the columns actually used. The pipeline trains on
  the pre-validation split only, so the validation-window backtest is out-of-sample.
- **`forecast.py`** — `forecast()` enumerates buy/sell × leverage candidates per
  stock, scores them, and keeps the best above `threshold` (else hold). Emits the
  five Story.md fields via the `Forecast` dataclass; `forecasts_to_frame()` tabulates.
  `apply_book_limits()` (applied after the overlay) caps each buy at the remaining cash
  and each sell at the held tier value, dropping unfundable actions (Story.md: buy only
  with cash on hand, sell only from open positions).
- **`overlay.py`** — live analyst/sentiment overlay on the forecast (**live-only**, not
  backtested — these signals have no history). `sentiment_tilt` (recommendation mean +
  EPS-revision momentum + price-vs-target) scales confidence/amount; `risk_gate`
  (put/call + IV skew) caps the leverage tier on crash fear; `apply_overlay` applies
  both to the `run_phase1` forecast.

### `portfolio/` (Phase 4)
- **`state.py`** — `PortfolioState` (cash + leveraged `Lot`s with cost basis,
  `loss_carry`, `high_water`). `mark()` applies daily constant-leverage returns;
  `buy()`/`sell_name()` open lots and realize tax-adjusted proceeds; `sell_tier()`
  sells one tier only (tier-targeted de-risk) — both route through `_sell_lots`;
  `pay_dividends()` credits cash on tier-1 (underlying) lots only, net of flat tax
  (Story.md: leveraged lots earn no dividend); `build_base_case()` constructs the
  Story.md 90/10 book (per-name 9%/4.5%/4.5%).
- **`tax.py`** — `tax_on_sale()`: 25% flat Abgeltungsteuer with a single
  **full-portfolio** realized-loss carry pool (never expiring) that offsets future gains
  before tax, so gains and losses net across the whole book over time (Story.md).
- **`rules.py`** — deterministic sell-side guardrails returning dated `Trade`s
  (`ticker, action, amount_eur, tier, date`): per-name trim (33%→3%),
  `enforce_underlying_dominance` (keep underlying ≥ 2x+3x by selling the leverage excess),
  and drawdown de-risk (>20%→cash, but **never below the 6% `MIN_NAME_WEIGHT` floor** —
  the floor is underlying-only, and across 5 names keeps cash < 70% / `MAX_CASH`) all shed
  the **riskiest tier first** (3x→2x→stock) via the shared `sell_riskiest_first` (built on
  `state.sell_tier`), within a 10%/day sell cap and an order minimum of `MIN_TRADE_EUR`
  (€500 — gated in `sell_riskiest_first`, so trivial orders, incl. the tiny daily
  dominance micro-trims, are skipped); `apply_guardrails()` runs them per day
  (de-risk, dominance, trim). (The routine confidence-rebalance in
  `backtest.engine` sells pro-rata instead — grading it cost ~5pp; IMPLEMENTATION §5c.) (A vol-aware leverage throttle was
  evaluated here and dropped — walk-forward showed it hurt; see IMPLEMENTATION §5c.)
  The crisis buy-the-dip path lives in `backtest.engine` (`_is_crisis`/`_deploy`).

### `backtest/`
- **`engine.py`** — three backtests, all returning a `BacktestResult` (curve +
  portfolio/benchmark returns + `beats_benchmark` + `trades` + `final_state` (the
  forecast backtest's end-of-window `PortfolioState` for the Current-portfolio view) +
  `tier_curve` (daily per-`(ticker, tier)` value for the Strategy tab)): `run_backtest()` (Phase 1
  confidence-scaled equal-weight basket), `run_rules_backtest()` (base-case leveraged
  book under guardrails, sell-side only), and `run_forecast_backtest()` — **the
  pipeline's backtest** — the leveraged book where **each name's** target portfolio
  fraction follows `_target_name_fraction(that name's buy-confidence)`: it holds the
  per-name base weight while the name is neutral-to-bullish (≥ 0.5) and only de-risks
  that name below 0.5, rebalancing names independently (`_rebalance_names_to_target`,
  per-name dead-band) so a bearish read on one stock trims only that stock — Story.md's
  per-ticker forecast. Cash re-entry, daily guardrails, and German tax apply.
  (The book-level `_target_exposure` dial is retained for the Phase-1 `run_backtest`.)
  `_is_crisis` (a basket
  drop > `CRISIS_DROP` over `CRISIS_LOOKBACK` days) overrides this for
  `CRISIS_REVERT_DAYS`: it `_deploy`s the cash reserve to ~100% invested (buy-the-dip)
  and holds — no de-risk/rebalance-to-cash — then reverts to base (Story.md crisis
  rule). The per-name trim still fires in crisis. `_dividend_yields` recovers each
  day's underlying dividend yield (`adj_close` minus `close` return) and
  `state.pay_dividends` credits it to the tier-1 lots; `_benchmark_curve` ffills
  interior gaps and bfills a leading NaN (window opening on a benchmark holiday). Every
  buy/sell (`_deploy`/`_rebalance_names_to_target`/guardrails) is collected **per tier**
  with its actual € (deploys split 9/4.5/4.5; the pro-rata rebalance sell via
  `_sell_proportional`), date-stamped, and returned as `BacktestResult.trades` (the
  Strategy tab's source); each day's per-`(ticker, tier)` value is `BacktestResult.tier_curve`.
- **`walkforward.py`** — `walk_forward_validate()` trains-then-tests across several
  consecutive `window`-day windows (model trained only on prior data, with a horizon
  embargo so labels don't bleed across the boundary), returning a `WalkForwardResult`
  with per-window returns + `win_rate` / `mean_outperformance`. Surfaced by
  `concinvest validate`; the honest read vs any single-year backtest.

### `app/`
- **`streamlit_app.py`** — UI in three tabs: **Current market** (the *actual*
  end-of-backtest book from `BacktestResult.final_state` — real per-tier values and live
  cash level, not the static template; Plotly donut; cross-asset correlation with three
  views — 5 stocks vs NASDAQ / all assets / one stock vs all as a point chart — and a
  ticker→name legend popover; a simplified analyst/sentiment summary — rating, news tone,
  target upside), **Forecast & Backtest** (5-field forecast sized to the live book,
  portfolio-vs-NASDAQ curve as cumulative %, feature importances),
  **Strategy** (per asset, three panels on a shared x-axis with a legend: price with
  aggregated buy/sell markers, a per-tier balance-evolution chart from
  `BacktestResult.tier_curve` with per-tier markers (actual € each), and NASDAQ; the full
  trade table behind a popover button. Markers drawn on the decision day T-1 — interactive
  Plotly). Cached via `st.cache_data`.
- **`exit_button.py`** — safe-exit helper: discovers the running port (default 8505),
  `lsof -ti:PORT | kill -9` filtered to skip any `ssh` process.

### Top level
- **`config.py`** — paths, dates (`START_DATE` 2020-01-01), portfolio/risk/tax
  constants, `BENCHMARK_TICKER` (`^IXIC`), `STREAMLIT_PORT` (8505).
- **`pipeline.py`** — `run_phase1` / `fetch_and_store` orchestration; `daily_etl`
  (the Phase 5 cron building block — `fetch_and_store` + a dated `sentiment_analyst`
  snapshot via `_fetch_sentiment(as_of=…)`, so the live analyst signals accumulate
  history).
- **`cli.py`** — `concinvest {info,update,run,validate}`; `update --sentiment` runs
  `daily_etl` (the daily cron entry, wrapped by `scripts/daily_update.sh`).

## Key design decisions

- **Raw vs. derived split** — `ohlcv_raw` is stored separately from computed feature
  tables so parameters can be re-tuned without re-downloading (Story.md).
- **Sentiment placeholders over history** — yfinance only exposes recent news, so
  historical `news_sentiment_score`/`put_call_ratio` default to neutral in the panel;
  live values are filled only at forecast time. Keeps `FEATURE_COLS` consistent
  between training and inference. The other live analyst signals (EPS revisions,
  target, IV skew) likewise have no history and so are stored/displayed only, never
  trained on (a tree gains nothing from columns constant over training).
- **Leverage** — 2x/3x modelled as daily-rebalanced constant-leverage return
  multipliers (documented assumption; real LETF instruments revisited later).
- **Network isolation** — all network calls live in `data.fetch`; every other module
  is pure and unit-tested offline with the synthetic fixtures in `tests/conftest.py`.
