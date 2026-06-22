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
   ml.forecast.forecast          backtest.engine.run_backtest
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
  sentiment placeholders + action encoding). `build_feature_panel()` joins per-stock
  technicals with date-aligned cross-asset features into a `(date, ticker)` MultiIndex
  panel. `generate_dataset()` samples `n` datapoints (half buys / half sells; 100k is
  the Story.md target); each is a market snapshot plus `is_sell`/`leverage`; the label
  is "profitable action" from a forward-return horizon (buy good if price rose, sell
  good if it fell). Rows are **returned sorted by snapshot date** (DatetimeIndex) so
  `TimeSeriesSplit` is honest; features are point-in-time, labels strictly forward — no
  leakage. `train_validate_split()` carves the last `VALIDATION_YEARS` off by calendar
  date (Story.md 4y-train / 1y-validate).
- **`model.py`** — `train()` fits a `RandomForestClassifier` with `TimeSeriesSplit`
  ROC-AUC CV and feature importances, returning a `TrainedModel` whose
  `predict_confidence()` gives `P(profitable)`. `tune()` selects the best `PARAM_GRID`
  entry by mean TSCV AUC; `tune_and_train()` tunes then fits. The pipeline trains on
  the pre-validation split only, so the validation-window backtest is out-of-sample.
- **`forecast.py`** — `forecast()` enumerates buy/sell × leverage candidates per
  stock, scores them, and keeps the best above `threshold` (else hold). Emits the
  five Story.md fields via the `Forecast` dataclass; `forecasts_to_frame()` tabulates.

### `portfolio/` (Phase 4)
- **`state.py`** — `PortfolioState` (cash + leveraged `Lot`s with cost basis,
  `loss_carry`, `high_water`). `mark()` applies daily constant-leverage returns;
  `buy()`/`sell_name()` open lots and realize tax-adjusted proceeds;
  `build_base_case()` constructs the Story.md 90/10 book (per-name 12%/3%/3%).
- **`tax.py`** — `tax_on_sale()`: 25% flat Abgeltungsteuer with a realized-loss carry
  that offsets future gains before tax.
- **`rules.py`** — deterministic sell-side guardrails returning `Trade`s: per-name
  trim (33%→3%), drawdown de-risk (>20%→cash), 10%/day sell cap; `apply_guardrails()`
  runs them per day. Crisis/dividends/forecast-driven trading are the next increment.

### `backtest/`
- **`engine.py`** — `run_backtest()` holds an equal-weight 5-stock basket but scales
  daily equity exposure by the model's mean buy-confidence (rest in cash), lagged one
  day to stay point-in-time. `run_rules_backtest()` replays the base-case leveraged
  book under the `portfolio` guardrails + tax. Both return a `BacktestResult` (curve +
  portfolio/benchmark returns + `beats_benchmark`).

### `app/`
- **`streamlit_app.py`** — UI: forecast table, portfolio-vs-NASDAQ curve, live
  analyst/sentiment table, feature importances, cross-asset correlation matrix.
  Cached via `st.cache_data`.
- **`exit_button.py`** — safe-exit helper: discovers the running port (default 8505),
  `lsof -ti:PORT | kill -9` filtered to skip any `ssh` process.

### Top level
- **`config.py`** — paths, dates (`START_DATE` 2020-01-01), portfolio/risk/tax
  constants, `BENCHMARK_TICKER` (`^IXIC`), `STREAMLIT_PORT` (8505).
- **`pipeline.py`** — `run_phase1` / `fetch_and_store` orchestration.
- **`cli.py`** — `concinvest {info,update,run}`.

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
