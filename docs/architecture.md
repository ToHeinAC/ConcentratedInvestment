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
reusable daily-ETL building block (later driven by the Phase 5 cron job). It is
**incremental**: only bars newer than the stored maximum (minus a `_REFETCH_OVERLAP_DAYS`
overlap, since yfinance revises recent bars and posts dividends retroactively to
`adj_close`) are downloaded, then merged with the full stored history read back via
`store.read_ohlcv` so feature windows and training keep full depth. An empty/partial DB
(or `full=True`, e.g. after a split rescales deep history) falls back to a full fetch
from `start`.

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
  generic `INSERT OR REPLACE`; `read_table()` reads back. `latest_date()` returns the
  max stored date per ticker and `read_ohlcv()` reconstructs the `download_ohlcv` shape
  from `ohlcv_raw` — together they drive **incremental fetching** (`fetch_and_store`).
  Date columns normalised to ISO text PKs. See [SCHEMA.md](SCHEMA.md).
- **`portfolio_store.py`** — persistence for the Live tab's user portfolios. Each named
  portfolio is a CSV **file** under `data/portfolios/` (`list_portfolios` / `save_portfolio`
  / `load_portfolio`, all with an optional `base` dir for tests). One row per position
  (`ticker, tier, invested_eur, buy_date` — every tier carries its own buy date) plus a
  `tier 0` / `CASH` row for the cash balance. Pure local file I/O (no network).

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
- **`regime.py`** — `detect_regime()` classifies the current market (Rising / Neutral /
  Falling) from six explainable Fear&Greed-style votes: S&P 500 vs its 50-day MA,
  S&P 500 vs its 125-day MA, breadth (portfolio stocks above their 125-day MA), VIX vs
  its 50-day MA, gold (`GC=F`) vs its 50-day MA, and oil (`CL=F`) vs its 50-day MA. For
  VIX/gold/oil, *below* the MA is bullish (a rising gold/oil = safe-haven/inflation
  risk-off). Bullish-vote fraction > 0.6 = Rising, < 0.4 = Falling, else Neutral; returns
  a `Regime` (label + bullish-vote fraction + per-component `RegimeSignal`s, each with a
  human-readable `detail` and a short chart `label` — breadth's is "N > 125d MA", not a
  bare fraction). `gold`/`oil` are optional (omitted → the four core votes); the dense
  raw commodity closes are used (the cross-asset gold/oil *ratio* has interspersed NaNs
  that break its MA). Pure pandas; insufficient history votes non-bullish. Computed in
  `run_phase1` and surfaced as the **ML: Current market** gauge + vote bars.

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
  entry by mean TSCV AUC; `select_features()` prunes features below a cutoff that
  **scales with the feature count** — `min(MIN_IMPORTANCE, KEEP_UNIFORM_FRAC / n)`, so a
  wide correlated set (the momentum lags) can't dilute every importance below the
  absolute floor and collapse the model to the action encoding (action encoding always
  kept); `tune_and_train(prune=True)` tunes, prunes, and refits. `FEATURE_COLS` stays the stable superset callers build, while
  `TrainedModel.features` records the columns actually used. The pipeline trains on
  the pre-validation split only, so the validation-window backtest is out-of-sample.
- **`forecast.py`** — `forecast()` enumerates buy/sell × leverage candidates per
  stock (the `leverages` arg restricts the tiers — the aggressive strategy passes
  `(3,)`), scores them, and keeps the best above `threshold` (else hold). Emits the
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
  `loss_carry`, `high_water`). Each `Lot` also carries `tp_basis` (a take-profit
  reference used only by the aggressive strategy, re-based after each skim). `mark()`
  applies daily constant-leverage returns; `buy()`/`sell_name()` open lots and realize
  tax-adjusted proceeds; `sell_tier()` sells one tier only (tier-targeted de-risk);
  `sell_lot()` sells a single lot (aggressive stop-loss / take-profit) — all route
  through `_sell_lots`; `pay_dividends()` credits cash on tier-1 (underlying) lots only,
  net of flat tax (Story.md: leveraged lots earn no dividend); `build_base_case()`
  constructs a book from a per-name tier `split` — the Story.md 90/10 default
  (9%/4.5%/4.5%) or the aggressive all-3x `config.AGG_BASE_SPLIT` (18% 3x).
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
- **`engine.py`** — four backtests, all returning a `BacktestResult` (curve +
  portfolio/benchmark returns + `beats_benchmark` + `trades` + `final_state` (the
  forecast backtest's end-of-window `PortfolioState` for the Current-portfolio view) +
  `tier_curve` (daily per-`(ticker, tier)` value for the Strategy tab) + `cash_curve`
  (daily cash balance, for the Strategy tab's Cash view)): `run_backtest()` (Phase 1
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
  The per-day snapshot and result assembly are shared via `_tier_snapshot` /
  `_assemble_result`. **`run_aggressive_backtest()`** is the fourth, selectable book (the
  all-3x "Aggressive" strategy; IMPLEMENTATION.md §4b): the `config.AGG_BASE_SPLIT` 90/10
  all-3x base case, then per day `_agg_stop_loss` (full exit of a 3x lot ≤ −60% of cost),
  `_agg_take_profit` (skim ≥30% of a lot ≥ +60% of its `tp_basis`, re-base the remainder,
  seed half the net proceeds into a permanent tier-1 underlying lot), and either
  `_agg_deploy_crisis` (deploy the cash hoard into 3x on `_is_crisis`) or `_agg_entries`
  (fixed 10%-of-portfolio 3x chunks for names with buy-confidence ≥ `AGG_ENTRY_THRESHOLD`,
  skipping already-capped names), then `_agg_cap_overweight` (cap each name's total at
  `PER_NAME_CAP` = 33%, shedding 3x first → cash, via the shared `rules.sell_riskiest_first`)
  to bound single-stock concentration. No drawdown/dominance guardrails or daily-cap beyond
  that. Per-name buy/sell confidence comes from `_name_confidence(..., is_sell, leverage)`.
- **`walkforward.py`** — `walk_forward_validate()` trains-then-tests across several
  consecutive `window`-day windows (model trained only on prior data, with a horizon
  embargo so labels don't bleed across the boundary), returning a `WalkForwardResult`
  with per-window returns + `win_rate` / `mean_outperformance`. Surfaced by
  `concinvest validate`; the honest read vs any single-year backtest.

### `app/`
- **`streamlit_app.py`** — title bar carries two popovers: **ℹ️ About** (`_render_about`
  — the `_VERSION` string + the 5 portfolio stocks from `tickers.STOCKS`) and **Compare
  strategies**. UI in four tabs. **Live: Sample Portfolio** (first tab):
  a **persisted, selectable** user portfolio. A dropdown lists the saved portfolios from
  `data.portfolio_store` (or "New portfolio"); a 15-row `st.data_editor` grid holds one row
  per position — the € **invested** and a **separate buy date** for each tier (stock / 2x /
  3x) of each stock — plus cash; a **💾 Save / update** button writes it back to the chosen
  CSV (buy dates default to **today** — a book entered "as it is right now" starts at
  current ≈ invested). `pipeline.build_dated_book` derives each lot's **current value** by the
  daily-rebalanced Nx-leverage path (`_lot_value_path`: cumprod of `1 + tier ×
  underlying-daily-return` since the buy date — a real leveraged ETF, matching the
  backtest's `state.mark`; daily factor floored at 0), keeps the
  real **cost basis**, and computes the book's **high-water** (always ≥ current, so the
  drawdown can't go negative), shown as invested / current / drawdown metrics + a **Plotly
  pie** of current value per position (invested € and P&L on hover) **next to** a
  **performance-since-inception vs NASDAQ** line chart (`pipeline.dated_book_value_path`
  → daily combined book €-value; both lines rebased to 0 at the earliest buy date,
  portfolio dark green / NASDAQ dark red — the `_PORTFOLIO_COLOR`/`_NASDAQ_COLOR`
  convention applied to every NASDAQ plot: this chart, the backtest curve, the Strategy
  tab, the Cash view). A **Run live
  analysis** button fetches live news/sentiment and calls `pipeline.recommend_for_portfolio`,
  surfacing the strategy's **actions** (default: drawdown de-risk / dominance / 33% trim;
  aggressive: −60% stop-loss / +60% take-profit / cap — both need the derived cost basis)
  and the **ML + sentiment signals** (the 5-field forecast sized to that book) plus the live
  analyst/sentiment summary. Reuses the trained model from `_load` (no retrain). The other
  three are the ML views (prefixed **ML:**): **ML: Current market** (the *actual*
  end-of-backtest book from `BacktestResult.final_state` — real per-tier values and live
  cash level, not the static template; Plotly donut; a **rising-market regime panel**
  (`_render_regime` over `Phase1Result.regime` — a Plotly **vote gauge** (verdict-coloured
  title over a red→white→green gradient arc, the bullish-vote count marked by a dark
  threshold line) + a diverging **vote bar** per
  component, reason on hover); cross-asset correlation with three
  views — 5 stocks vs NASDAQ / all assets / one stock vs all as a point chart — and a
  ticker→name legend popover; a simplified analyst/sentiment summary — rating, news tone,
  target upside), **ML: Forecast & Backtest** (5-field forecast sized to the live book,
  portfolio-vs-NASDAQ curve as cumulative %, feature importances),
  **ML: Strategy** (per asset, three panels on a shared x-axis with a legend: price with
  aggregated buy/sell markers, a per-tier balance-evolution chart from
  `BacktestResult.tier_curve` with per-tier markers (actual € each), and NASDAQ; the full
  trade table behind a popover button. Markers drawn on the decision day T-1 — interactive
  Plotly). The asset selector also offers **Cash** as a 6th option: cash (€) over NASDAQ
  on a shared x-axis, from `BacktestResult.cash_curve`. Cached via `st.cache_data`.
- **`exit_button.py`** — safe-exit helper: `shutdown()` sends `SIGTERM` to the app's
  **own** PID (`os.kill(os.getpid(), …)`) — no `lsof`/port kill, so anything sharing or
  forwarding the port (SSH tunnel, IDE port-forward) is left untouched; the port frees
  when the app exits.

### Top level
- **`config.py`** — paths, dates (`START_DATE` 2020-01-01), portfolio/risk/tax
  constants, `BENCHMARK_TICKER` (`^IXIC`), `STREAMLIT_PORT` (8505).
- **`pipeline.py`** — `run_phase1` / `fetch_and_store` orchestration (`Phase1Result`
  also carries the feature `panel` for the Live tab). `run_phase1` takes an optional
  `progress(fraction, label)` callback that fires at each main step (fetch → train →
  backtest → forecast → correlation/regime), driving the app's `st.progress` bar
  (no-op for the CLI). `daily_etl`
  (the Phase 5 cron building block — `fetch_and_store` + a dated `sentiment_analyst`
  snapshot via `_fetch_sentiment(as_of=…)`, so the live analyst signals accumulate
  history); `build_dated_book(positions, market, cash)` — pure: turns **per-position dated
  invested amounts** (`ticker, tier, invested_eur, buy_date` — each tier its own date) into
  a `PortfolioState` (current value = the daily-rebalanced Nx-leverage path, cumprod of
  `1 + tier × underlying-daily-return` since the buy date — `_lot_value_path`; real cost
  basis, derived high-water that always includes "now", so drawdown ≥ 0 even for lots
  bought past the last close); `dated_book_value_path(positions, market, cash)` — pure: the
  daily combined book €-value (cash + each lot's leverage path, the same series whose peak
  sets the high-water),
  for the Live tab's performance-vs-NASDAQ chart (`build_dated_book` and it share the
  `_dated_lots_and_paths` / `_combine_paths` helpers);
  `recommend_for_portfolio(state, model, panel, market, strategy, …)` — side-effect-free
  live recommendations for a **user-supplied** book (reuses a trained model): live
  sentiment → forecast sized to the book (`apply_book_limits`) + overlay + the strategy's
  value/cost-aware actions (`_strategy_actions`: `rules.apply_guardrails` for default;
  `_agg_stop_loss` + `_agg_take_profit` + `trim_overweight` for aggressive; on a deep copy).
  Powers the Live tab.
- **`cli.py`** — `concinvest {info,update,run,validate}`; `update --sentiment` runs
  `daily_etl` (the daily cron entry, wrapped by `scripts/daily_update.sh`).

## Key design decisions

- **Raw vs. derived split** — `ohlcv_raw` is stored separately from computed feature
  tables so parameters can be re-tuned without re-downloading (Story.md). It is also the
  cache that incremental fetching reads back (`store.read_ohlcv`): only the recent tail
  is pulled from the network each run, then merged with stored history before features
  recompute. The volume thus becomes the source of truth read into the model, not just
  redeploy persistence — but the empty/partial-DB full-fetch fallback keeps a fresh
  volume self-healing.
- **Sentiment placeholders over history** — yfinance only exposes recent news, so
  historical `news_sentiment_score`/`put_call_ratio` default to neutral in the panel;
  live values are filled only at forecast time. Keeps `FEATURE_COLS` consistent
  between training and inference. The other live analyst signals (EPS revisions,
  target, IV skew) likewise have no history and so are stored/displayed only, never
  trained on (a tree gains nothing from columns constant over training).
- **Leverage** — 2x/3x are daily-rebalanced constant-leverage return multipliers
  everywhere (documented assumption; real LETF instruments revisited later): a tier-k
  lot's value evolves by `(1 + k × underlying_daily_return)` each day, so a +1% underlying
  day moves a 3x lot +3% *that day* and the compounding/volatility-decay is captured. The
  backtests apply this via `state.mark`; the Live tab's user book reuses the **same** model
  as a per-lot path from each lot's buy date (`_lot_value_path`: cumprod of the daily
  factors, floored at 0). (Previously the Live tab used a simpler `invested × (1 + tier ×
  total return)` that under-stated leverage and diverged from the backtest — now unified.)
- **Network isolation** — all network calls live in `data.fetch`; every other module
  is pure and unit-tested offline with the synthetic fixtures in `tests/conftest.py`.
