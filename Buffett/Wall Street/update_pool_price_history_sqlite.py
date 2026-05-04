# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pandas as pd
import yfinance as yf

from expanded_pool_config import DATA_END, DB_PATH, PRICE_TICKERS, ROOT, TRAIN_START


def configure_runtime_dirs() -> None:
    cache_dir = ROOT / ".sqlite_cache"
    cache_dir.mkdir(exist_ok=True)
    os.environ.setdefault("LOCALAPPDATA", str(cache_dir))
    os.environ.setdefault("TEMP", str(cache_dir))
    os.environ.setdefault("TMP", str(cache_dir))


def ensure_schema(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS price_history (
            date TIMESTAMP,
            ticker TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            adj_close REAL,
            volume REAL
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_price_history_date_ticker
        ON price_history(date, ticker)
        """
    )
    connection.commit()


def load_existing_max_dates(connection: sqlite3.Connection) -> dict[str, pd.Timestamp]:
    placeholders = ",".join(["?"] * len(PRICE_TICKERS))
    query = f"""
    SELECT ticker, MAX(date) AS max_date
    FROM price_history
    WHERE ticker IN ({placeholders})
    GROUP BY ticker
    """
    frame = pd.read_sql_query(query, connection, params=PRICE_TICKERS, parse_dates=["max_date"])
    return {
        row["ticker"]: pd.Timestamp(row["max_date"])
        for _, row in frame.iterrows()
        if pd.notna(row["max_date"])
    }


def download_ticker_history(ticker: str, start: str, end: str) -> pd.DataFrame:
    raw = yf.download(
        tickers=ticker,
        start=pd.Timestamp(start),
        end=pd.Timestamp(end) + pd.Timedelta(days=1),
        auto_adjust=False,
        progress=False,
    )
    if raw.empty:
        return pd.DataFrame(columns=["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"])

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    frame = raw.rename_axis("date").reset_index()
    frame["ticker"] = ticker
    frame.columns = [str(column).lower().replace(" ", "_") for column in frame.columns]
    frame = frame[["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]].copy()
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.dropna(subset=["adj_close"]).reset_index(drop=True)


def replace_ticker_range(connection: sqlite3.Connection, dataframe: pd.DataFrame) -> None:
    if dataframe.empty:
        return
    ticker = str(dataframe["ticker"].iloc[0])
    start_date = pd.Timestamp(dataframe["date"].min()).strftime("%Y-%m-%d")
    end_date = pd.Timestamp(dataframe["date"].max()).strftime("%Y-%m-%d")
    connection.execute(
        """
        DELETE FROM price_history
        WHERE ticker = ? AND DATE(date) >= ? AND DATE(date) <= ?
        """,
        (ticker, start_date, end_date),
    )
    dataframe.to_sql("price_history", connection, if_exists="append", index=False)
    connection.commit()


def update_metadata(connection: sqlite3.Connection) -> None:
    table_exists = connection.execute(
        """
        SELECT COUNT(*)
        FROM sqlite_master
        WHERE type = 'table' AND name = 'metadata'
        """
    ).fetchone()[0]
    if not table_exists:
        return
    rows = [
        ("data_end", DATA_END),
        ("price_tickers", ",".join(PRICE_TICKERS)),
    ]
    connection.executemany(
        """
        INSERT INTO metadata(key, value) VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """,
        rows,
    )
    connection.commit()


def main() -> None:
    configure_runtime_dirs()
    end_date = pd.Timestamp(DATA_END)

    with sqlite3.connect(DB_PATH) as connection:
        ensure_schema(connection)
        max_dates = load_existing_max_dates(connection)

        rows: list[dict[str, str | int]] = []
        for ticker in PRICE_TICKERS:
            existing_max = max_dates.get(ticker)
            if existing_max is not None and existing_max.normalize() >= end_date.normalize():
                rows.append({"ticker": ticker, "status": "up_to_date", "rows_added": 0})
                continue

            start_date = TRAIN_START if existing_max is None else (existing_max + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            if pd.Timestamp(start_date) > end_date:
                rows.append({"ticker": ticker, "status": "up_to_date", "rows_added": 0})
                continue

            data = download_ticker_history(ticker, start_date, DATA_END)
            if data.empty:
                rows.append({"ticker": ticker, "status": "no_data", "rows_added": 0})
                continue

            replace_ticker_range(connection, data)
            rows.append({"ticker": ticker, "status": "updated", "rows_added": int(len(data))})

        update_metadata(connection)

        coverage = pd.read_sql_query(
            """
            SELECT ticker, MIN(date) AS min_date, MAX(date) AS max_date, COUNT(*) AS rows_count
            FROM price_history
            WHERE ticker IN ({})
            GROUP BY ticker
            ORDER BY ticker
            """.format(",".join(["?"] * len(PRICE_TICKERS))),
            connection,
            params=PRICE_TICKERS,
            parse_dates=["min_date", "max_date"],
        )

    report = pd.DataFrame(rows)
    print("SQLite price_history update completed")
    if not report.empty:
        print(report.to_string(index=False))
        print()
    print("Coverage after update:")
    print(coverage.to_string(index=False))


if __name__ == "__main__":
    main()
