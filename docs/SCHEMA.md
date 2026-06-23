# Database Schema

SQLite database at `data/concinvest.sqlite` (path from `config.DB_PATH`; the
`data/` dir is gitignored). Created/managed by
[`concinvest.data.store`](../src/concinvest/data/store.py). Raw OHLCV is kept
separate from derived features so features can be recomputed without re-downloading
(Story.md). All writes are idempotent `INSERT OR REPLACE` keyed on the primary key;
date columns are stored as ISO `YYYY-MM-DD` text.

## `ohlcv_raw`
Raw daily price/volume, one row per `(date, ticker)`.

| Column | Type | Notes |
|--------|------|-------|
| date | TEXT | PK part |
| ticker | TEXT | PK part |
| open, high, low, close, adj_close | REAL | from `history()` |
| volume | REAL | |

**PK:** `(date, ticker)`.

## `daily_market` (Table 1)
Per-stock OHLCV plus computed technical features. Written for the 5 portfolio stocks.

| Column | Source |
|--------|--------|
| date, ticker | PK |
| open, high, low, close, volume | `ohlcv_raw` |
| sma_5, sma_10, sma_20, sma_50, sma_100, sma_200 | `features.technical` |
| ema_12, ema_26, ema_50 | `features.technical` |
| rsi_14, macd, macd_signal | `features.technical` |
| bollinger_upper, bollinger_lower | `features.technical` |
| price_sma50_ratio, price_sma200_ratio, sma50_sma200_ratio | derived |
| volume_sma20_ratio | derived |

**PK:** `(date, ticker)`.

## `sentiment_analyst` (Table 2)
Analyst + sentiment signals per stock. Fetched live and stored/displayed only — no
usable history, so these are **not** model features.

| Column | Source |
|--------|--------|
| date, ticker | PK |
| recommendation_mean | `info['recommendationMean']` |
| news_sentiment_score | VADER/FinBERT on `get_news()` + German headlines, scaled to [-3, 3] |
| put_call_ratio | derived from `option_chain()` |
| eps_revision_up_7d, eps_revision_down_7d | `eps_revisions` (`0q` row, last 7 days) |
| analyst_target_mean | `info['targetMeanPrice']` |
| iv_skew | OTM-put IV − ATM-call IV, nearest expiry (`option_chain()`) |

**PK:** `(date, ticker)`.

## `cross_asset` (Table 3)
One row per date; market-wide derived ratios.

| Column | Formula / Source |
|--------|------------------|
| date | PK |
| gold_oil_ratio | `GC=F` / `CL=F` |
| copper_gold_ratio | `HG=F` / `GC=F` |
| vix_level | `^VIX` close |
| vix_sma20_ratio | `^VIX` / SMA20(`^VIX`) |
| yield_10y | `^TNX` |
| yield_spread_10y_5y | `^TNX` − `^FVX` |
| vvix_level | `^VVIX` close |
| gsci_sma20_ratio | `^SPGSCI` / SMA20(`^SPGSCI`) |
| dollar_index | `DX-Y.NYB` |
| btc_sma20_ratio | `BTC-USD` / SMA20(`BTC-USD`) |

**PK:** `(date)`.

## Model feature contract

The ML layer does not read these tables directly at train time; it builds an
in-memory `(date, ticker)` panel (`ml.dataset.build_feature_panel`). The model's
column contract is `ml.dataset.FEATURE_COLS`:

- **Technical:** `rsi_14, macd, price_sma50_ratio, price_sma200_ratio,
  sma50_sma200_ratio, volume_sma20_ratio`
- **Cross-asset:** `vix_level, vix_sma20_ratio, vvix_level, gold_oil_ratio,
  copper_gold_ratio, yield_10y, yield_spread_10y_5y`
- **Momentum lags:** each technical + cross-asset feature also carried at `_lag3`,
  `_lag10`, `_lag30`, `_lag100` (its value that many trading days back) so the trees
  see recent trajectory, not just the point-in-time level. Strictly past data — no
  forward leakage; leading edge fills to 0.
- **Sentiment:** `news_sentiment_score, put_call_ratio` (neutral defaults over
  history; live values at forecast time)
- **Action encoding:** `is_sell, leverage`

## Planned (Phase 3+)

A positions/trade-ledger table for the Phase 4 rules + tax engine. Additive columns
are applied to existing databases by `store._migrate` (no rebuild).
