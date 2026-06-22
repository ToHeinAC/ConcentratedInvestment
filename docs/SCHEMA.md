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
Analyst + sentiment signals per stock. Phase 1 stores the numeric essentials;
revision-momentum and richer fields arrive in Phase 2.

| Column | Source |
|--------|--------|
| date, ticker | PK |
| recommendation_mean | `info['recommendationMean']` |
| news_sentiment_score | VADER on `get_news()`, scaled to [-3, 3] |
| put_call_ratio | derived from `option_chain()` |

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
| dollar_index | `DX-Y.NYB` |
| btc_sma20_ratio | `BTC-USD` / SMA20(`BTC-USD`) |

**PK:** `(date)`.

## Model feature contract

The ML layer does not read these tables directly at train time; it builds an
in-memory `(date, ticker)` panel (`ml.dataset.build_feature_panel`). The model's
column contract is `ml.dataset.FEATURE_COLS`:

- **Technical:** `rsi_14, macd, price_sma50_ratio, price_sma200_ratio,
  sma50_sma200_ratio, volume_sma20_ratio`
- **Cross-asset:** `vix_level, vix_sma20_ratio, gold_oil_ratio, copper_gold_ratio,
  yield_10y`
- **Sentiment:** `news_sentiment_score, put_call_ratio` (neutral defaults over
  history; live values at forecast time)
- **Action encoding:** `is_sell, leverage`

## Planned (Phase 2+)

Additional `sentiment_analyst` columns (eps revisions, analyst targets, FinBERG/
German-source scores), full-universe rows in `cross_asset` (extra bonds, VVIX,
GSCI), and a positions/trade-ledger table for the Phase 4 rules + tax engine.
