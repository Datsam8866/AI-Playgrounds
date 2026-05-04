# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "stock_forecast.sqlite"
PREDICTIONS_CSV = ROOT / "multi_asset_xgboost_regression_predictions.csv"
BACKTEST_CSV = ROOT / "final_backtest_report.csv"
PORTFOLIO_PREDICTIONS_CSV = ROOT / "portfolio_universe_xgboost_regression_predictions.csv"
PORTFOLIO_SUMMARY_CSV = ROOT / "portfolio_universe_xgboost_regression_summary.csv"
MODEL_TICKERS = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMZN", "META", "TSLA"]
PORTFOLIO_TICKERS = [
    "VOO",
    "GOOGL",
    "NVDA",
    "AVGO",
    "TSLA",
    "TSM",
    "SSO",
    "SOXX",
    "QQQ",
    "PLTR",
    "MSTR",
    "VRT",
    "AMD",
    "DELL",
    "MRVL",
    "QCOM",
]
PRICE_TICKERS = sorted(set(MODEL_TICKERS + PORTFOLIO_TICKERS))
MACRO_TICKERS = ["^VIX", "^TNX"]
TRAIN_START = "2000-01-01"
DATA_END = "2026-04-10"
IC_WINDOW = 90
IC_THRESHOLD = 0.08
HORIZON_DAYS = 20
K_HIGH = 3
K_LOW = 8


def configure_runtime_dirs() -> None:
    cache_dir = ROOT / ".sqlite_cache"
    cache_dir.mkdir(exist_ok=True)
    os.environ.setdefault("LOCALAPPDATA", str(cache_dir))
    os.environ.setdefault("TEMP", str(cache_dir))
    os.environ.setdefault("TMP", str(cache_dir))


def download_price_history(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    raw = yf.download(
        tickers=tickers,
        start=pd.Timestamp(start),
        end=pd.Timestamp(end) + pd.Timedelta(days=1),
        auto_adjust=False,
        progress=False,
        group_by="ticker",
    )

    rows = []
    for ticker in tickers:
        ticker_frame = raw[ticker].copy().rename_axis("date").reset_index()
        ticker_frame["ticker"] = ticker
        ticker_frame.columns = [column.lower().replace(" ", "_") for column in ticker_frame.columns]
        rows.append(ticker_frame)

    price_history = pd.concat(rows, ignore_index=True).sort_values(["date", "ticker"]).reset_index(drop=True)
    return price_history[["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]]


def download_macro_history(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    raw = yf.download(
        tickers=tickers,
        start=pd.Timestamp(start),
        end=pd.Timestamp(end) + pd.Timedelta(days=1),
        auto_adjust=False,
        progress=False,
        group_by="ticker",
    )

    rows = []
    for ticker in tickers:
        ticker_frame = raw[ticker].copy().rename_axis("date").reset_index()
        ticker_frame["ticker"] = ticker
        ticker_frame.columns = [column.lower().replace(" ", "_") for column in ticker_frame.columns]
        rows.append(ticker_frame)

    macro_history = pd.concat(rows, ignore_index=True).sort_values(["date", "ticker"]).reset_index(drop=True)
    return macro_history[["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]]


def load_predictions() -> pd.DataFrame:
    predictions = pd.read_csv(PREDICTIONS_CSV, parse_dates=["Date"])
    predictions = predictions.rename(
        columns={
            "Date": "date",
            "predicted_return": "predicted_return",
            "forward_20d_return": "forward_20d_return",
            "next_day_return": "next_day_return",
        }
    )
    return predictions.sort_values(["date", "ticker"]).reset_index(drop=True)


def load_optional_csv(csv_path: Path) -> pd.DataFrame | None:
    if not csv_path.exists():
        return None
    dataframe = pd.read_csv(csv_path)
    for candidate in ["Date", "date"]:
        if candidate in dataframe.columns:
            dataframe[candidate] = pd.to_datetime(dataframe[candidate])
            if candidate != "date":
                dataframe = dataframe.rename(columns={candidate: "date"})
            break
    return dataframe


def load_backtest_monthly() -> pd.DataFrame:
    monthly = pd.read_csv(BACKTEST_CSV)
    monthly = monthly.rename(columns={"Unnamed: 0": "date"}).assign(
        date=lambda frame: pd.to_datetime(frame["date"])
    )
    return monthly.sort_values("date").reset_index(drop=True)


def compute_daily_ic(predictions: pd.DataFrame) -> pd.Series:
    rows = []

    for date_value, group in predictions.groupby("date"):
        valid = group.dropna(subset=["predicted_return", "forward_20d_return"])
        if len(valid) < 3:
            continue

        ic, _ = spearmanr(valid["predicted_return"], valid["forward_20d_return"])
        rows.append({"date": date_value, "ic": ic})

    daily_ic = pd.DataFrame(rows)
    if daily_ic.empty:
        return pd.Series(dtype=float, name="daily_ic")

    return daily_ic.set_index("date")["ic"].sort_index()


def build_daily_signals(predictions: pd.DataFrame) -> pd.DataFrame:
    daily_ic = compute_daily_ic(predictions)
    lagged_ic = daily_ic.shift(HORIZON_DAYS)
    rolling_ic = lagged_ic.rolling(window=IC_WINDOW, min_periods=IC_WINDOW // 2).mean()

    frames = []

    for date_value, group in predictions.groupby("date"):
        current_ic = rolling_ic.loc[date_value] if date_value in rolling_ic.index else np.nan
        k_value = K_HIGH if (pd.notna(current_ic) and current_ic > IC_THRESHOLD) else K_LOW

        ranked = group.sort_values("predicted_return", ascending=False).reset_index(drop=True).copy()
        rank = np.arange(1, len(ranked) + 1)
        selected = rank <= k_value
        ranked = ranked.assign(
            rank=rank,
            selected=selected,
            weight=np.where(selected, 1.0 / k_value, 0.0),
            rolling_ic=current_ic,
            k_value=k_value,
            signal_mode="high_confidence" if k_value == K_HIGH else "low_confidence",
        )
        frames.append(ranked)

    signals = pd.concat(frames, ignore_index=True)
    return signals[
        [
            "date",
            "ticker",
            "split",
            "predicted_return",
            "forward_20d_return",
            "next_day_return",
            "rolling_ic",
            "k_value",
            "rank",
            "selected",
            "weight",
            "signal_mode",
        ]
    ].sort_values(["date", "rank"]).reset_index(drop=True)


def ensure_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        DROP TABLE IF EXISTS price_history;
        DROP TABLE IF EXISTS macro_history;
        DROP TABLE IF EXISTS model_predictions_regression;
        DROP TABLE IF EXISTS daily_signals_ic_adaptive;
        DROP TABLE IF EXISTS backtest_monthly;
        DROP TABLE IF EXISTS portfolio_model_predictions_regression;
        DROP TABLE IF EXISTS portfolio_model_summary;
        DROP TABLE IF EXISTS metadata;

        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        """
    )
    connection.commit()


def write_table(connection: sqlite3.Connection, dataframe: pd.DataFrame, table_name: str) -> None:
    dataframe.to_sql(table_name, connection, if_exists="replace", index=False)


def create_indexes(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE INDEX IF NOT EXISTS idx_price_history_date_ticker
        ON price_history(date, ticker);

        CREATE INDEX IF NOT EXISTS idx_macro_history_date_ticker
        ON macro_history(date, ticker);

        CREATE INDEX IF NOT EXISTS idx_model_predictions_date_ticker
        ON model_predictions_regression(date, ticker);

        CREATE INDEX IF NOT EXISTS idx_daily_signals_date_ticker
        ON daily_signals_ic_adaptive(date, ticker);

        CREATE INDEX IF NOT EXISTS idx_backtest_monthly_date
        ON backtest_monthly(date);

        CREATE INDEX IF NOT EXISTS idx_portfolio_predictions_date_ticker
        ON portfolio_model_predictions_regression(date, ticker);
        """
    )
    connection.commit()


def populate_metadata(connection: sqlite3.Connection) -> None:
    metadata_rows = [
        ("train_start", TRAIN_START),
        ("data_end", DATA_END),
        ("model_name", "xgboost_regression_ic_adaptive"),
        ("signal_rule", f"rolling_ic_{IC_WINDOW}d > {IC_THRESHOLD} => k={K_HIGH}, else k={K_LOW}"),
        ("predictions_csv", PREDICTIONS_CSV.name),
        ("backtest_csv", BACKTEST_CSV.name),
        ("price_tickers", ",".join(PRICE_TICKERS)),
        ("portfolio_predictions_csv", PORTFOLIO_PREDICTIONS_CSV.name if PORTFOLIO_PREDICTIONS_CSV.exists() else ""),
    ]
    connection.executemany("INSERT INTO metadata(key, value) VALUES (?, ?)", metadata_rows)
    connection.commit()


def main() -> None:
    configure_runtime_dirs()

    predictions = load_predictions()
    portfolio_predictions = load_optional_csv(PORTFOLIO_PREDICTIONS_CSV)
    portfolio_summary = load_optional_csv(PORTFOLIO_SUMMARY_CSV)
    backtest_monthly = load_backtest_monthly()
    daily_signals = build_daily_signals(predictions)
    price_history = download_price_history(PRICE_TICKERS, TRAIN_START, DATA_END)
    macro_history = download_macro_history(MACRO_TICKERS, TRAIN_START, DATA_END)

    with sqlite3.connect(DB_PATH) as connection:
        ensure_schema(connection)
        write_table(connection, price_history, "price_history")
        write_table(connection, macro_history, "macro_history")
        write_table(connection, predictions, "model_predictions_regression")
        write_table(connection, daily_signals, "daily_signals_ic_adaptive")
        write_table(connection, backtest_monthly, "backtest_monthly")
        if portfolio_predictions is not None:
            write_table(connection, portfolio_predictions, "portfolio_model_predictions_regression")
        if portfolio_summary is not None:
            write_table(connection, portfolio_summary, "portfolio_model_summary")
        populate_metadata(connection)
        create_indexes(connection)

        counts = {
            table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in [
                "price_history",
                "macro_history",
                "model_predictions_regression",
                "daily_signals_ic_adaptive",
                "backtest_monthly",
            ]
        }
        if portfolio_predictions is not None:
            counts["portfolio_model_predictions_regression"] = connection.execute(
                "SELECT COUNT(*) FROM portfolio_model_predictions_regression"
            ).fetchone()[0]
        if portfolio_summary is not None:
            counts["portfolio_model_summary"] = connection.execute(
                "SELECT COUNT(*) FROM portfolio_model_summary"
            ).fetchone()[0]

    print(f"SQLite database created: {DB_PATH}")
    for table_name, row_count in counts.items():
        print(f"{table_name}: {row_count:,} rows")


if __name__ == "__main__":
    main()
