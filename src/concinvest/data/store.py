"""SQLite storage for raw OHLCV and computed feature tables.

Raw OHLCV is kept separate from derived features (Story.md) so features can be
recomputed without re-downloading. All writes are idempotent upserts keyed on the
table's primary key.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from .. import config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS ohlcv_raw (
    date TEXT NOT NULL,
    ticker TEXT NOT NULL,
    open REAL, high REAL, low REAL, close REAL, adj_close REAL, volume REAL,
    PRIMARY KEY (date, ticker)
);
CREATE TABLE IF NOT EXISTS daily_market (
    date TEXT NOT NULL,
    ticker TEXT NOT NULL,
    open REAL, high REAL, low REAL, close REAL, volume REAL,
    sma_5 REAL, sma_10 REAL, sma_20 REAL, sma_50 REAL, sma_100 REAL, sma_200 REAL,
    ema_12 REAL, ema_26 REAL, ema_50 REAL,
    rsi_14 REAL, macd REAL, macd_signal REAL,
    bollinger_upper REAL, bollinger_lower REAL,
    price_sma50_ratio REAL, price_sma200_ratio REAL, sma50_sma200_ratio REAL,
    volume_sma20_ratio REAL,
    PRIMARY KEY (date, ticker)
);
CREATE TABLE IF NOT EXISTS sentiment_analyst (
    date TEXT NOT NULL,
    ticker TEXT NOT NULL,
    recommendation_mean REAL,
    news_sentiment_score REAL,
    put_call_ratio REAL,
    eps_revision_up_7d REAL, eps_revision_down_7d REAL,
    analyst_target_mean REAL, iv_skew REAL,
    PRIMARY KEY (date, ticker)
);
CREATE TABLE IF NOT EXISTS cross_asset (
    date TEXT NOT NULL,
    gold_oil_ratio REAL, copper_gold_ratio REAL,
    vix_level REAL, vix_sma20_ratio REAL,
    yield_10y REAL, yield_spread_10y_5y REAL,
    vvix_level REAL, gsci_sma20_ratio REAL,
    dollar_index REAL, btc_sma20_ratio REAL,
    PRIMARY KEY (date)
);
"""

# Columns added after the initial schema; applied idempotently to pre-existing DBs
# so additive Phase 2 features don't require a rebuild.
_MIGRATIONS: dict[str, dict[str, str]] = {
    "sentiment_analyst": {
        "eps_revision_up_7d": "REAL", "eps_revision_down_7d": "REAL",
        "analyst_target_mean": "REAL", "iv_skew": "REAL",
    },
    "cross_asset": {
        "yield_spread_10y_5y": "REAL", "vvix_level": "REAL",
        "gsci_sma20_ratio": "REAL",
    },
}


def _migrate(conn: sqlite3.Connection) -> None:
    """Add any missing columns from ``_MIGRATIONS`` (additive, idempotent)."""
    for table, cols in _MIGRATIONS.items():
        existing = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
        for name, sqltype in cols.items():
            if name not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {sqltype}")
    conn.commit()


def connect(db_path: Path | str | None = None) -> sqlite3.Connection:
    """Open a connection, creating the data dir and schema if needed."""
    config.ensure_dirs()
    conn = sqlite3.connect(str(db_path or config.DB_PATH))
    # WAL lets the Streamlit app read while the daily cron writes, without locking
    # (single-writer/many-reader). Idempotent — the mode is persisted on the db file.
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(_SCHEMA)
    _migrate(conn)
    return conn


def upsert(conn: sqlite3.Connection, table: str, df: pd.DataFrame) -> int:
    """INSERT OR REPLACE every row of ``df`` into ``table``.

    The DataFrame index is written as ordinary column(s); callers should
    ``reset_index()`` so that key columns (e.g. ``date``, ``ticker``) are present.
    Returns the number of rows written.
    """
    if df is None or df.empty:
        return 0
    df = df.copy()
    # Normalise date columns to ISO strings for stable text PKs.
    for col in df.columns:
        if "date" in col.lower():
            df[col] = pd.to_datetime(df[col]).dt.strftime("%Y-%m-%d")
    cols = list(df.columns)
    placeholders = ", ".join(["?"] * len(cols))
    collist = ", ".join(cols)
    sql = f"INSERT OR REPLACE INTO {table} ({collist}) VALUES ({placeholders})"
    rows = [tuple(None if pd.isna(v) else v for v in row) for row in df.itertuples(index=False)]
    conn.executemany(sql, rows)
    conn.commit()
    return len(rows)


def latest_date(conn: sqlite3.Connection, table: str = "ohlcv_raw") -> dict[str, str]:
    """Most recent stored date per ticker -> ``{ticker: 'YYYY-MM-DD'}``.

    Drives incremental fetching: only bars newer than the stored maximum need to be
    re-downloaded (see ``pipeline.fetch_and_store``).
    """
    rows = conn.execute(f"SELECT ticker, MAX(date) FROM {table} GROUP BY ticker").fetchall()
    return {t: d for t, d in rows if d is not None}


def read_ohlcv(conn: sqlite3.Connection, tickers: list[str]) -> dict[str, pd.DataFrame]:
    """Reconstruct the ``fetch.download_ohlcv`` shape (ticker -> date-indexed OHLCV
    frame) from ``ohlcv_raw``, so a freshly-fetched tail can be merged with full
    stored history before features are recomputed (SMA-200 etc. need the full depth).
    """
    df = pd.read_sql_query("SELECT * FROM ohlcv_raw", conn)
    if df.empty:
        return {}
    df["date"] = pd.to_datetime(df["date"]).dt.date
    out: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        sub = df[df["ticker"] == ticker].drop(columns="ticker").set_index("date").sort_index()
        if not sub.empty:
            sub.index.name = "date"
            out[ticker] = sub
    return out


def read_table(conn: sqlite3.Connection, table: str, ticker: str | None = None) -> pd.DataFrame:
    """Read a table back as a DataFrame, optionally filtered by ticker."""
    sql = f"SELECT * FROM {table}"
    params: tuple = ()
    if ticker is not None:
        sql += " WHERE ticker = ?"
        params = (ticker,)
    sql += " ORDER BY date"
    df = pd.read_sql_query(sql, conn, params=params)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df
