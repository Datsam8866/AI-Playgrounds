# -*- coding: utf-8 -*-
from __future__ import annotations

import sqlite3
import warnings

import pandas as pd

from daily_signal import BASE_NUMERIC_FEATURES, build_and_predict_today
from expanded_pool_config import DB_PATH, NON_LEVERAGED_EXPANDED_POOL_TICKERS, REFERENCE_TICKERS, ROOT
from multi_asset_logistic_baseline import (
    TEST_START,
    TRAIN_START,
    VALIDATION_END,
    VALIDATION_START,
    build_feature_frame,
    build_monthly_periods,
    compute_rsi,
)
from multi_asset_xgboost_regime_baseline import build_regime_features
from multi_asset_xgboost_regression import compute_daily_ic, icir, run_walk_forward_regression

warnings.filterwarnings("ignore", category=FutureWarning, module=r"yfinance(\..*)?$")
warnings.filterwarnings("ignore", category=FutureWarning, message=r".*ChainedAssignmentError.*")


PORTFOLIO_TICKERS = NON_LEVERAGED_EXPANDED_POOL_TICKERS
MACRO_TICKERS = ["^VIX", "^TNX"]
ALL_TICKERS = list(dict.fromkeys(PORTFOLIO_TICKERS + REFERENCE_TICKERS))
PREDICTIONS_PATH = ROOT / "expanded_pool_xgboost_regression_no_leverage_predictions_v2.csv"
SUMMARY_PATH = ROOT / "expanded_pool_xgboost_regression_no_leverage_summary_v2.csv"


def load_market_data_from_sqlite(
    tickers: list[str],
    start: str,
    end: str,
    table_name: str = "price_history",
) -> dict[str, pd.DataFrame]:
    placeholders = ",".join(["?"] * len(tickers))
    query = f"""
    SELECT date, ticker, open, high, low, close, adj_close, volume
    FROM {table_name}
    WHERE ticker IN ({placeholders})
      AND date >= ?
      AND date < ?
    ORDER BY date ASC, ticker ASC
    """
    end_exclusive = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    params = list(tickers) + [pd.Timestamp(start).strftime("%Y-%m-%d"), end_exclusive]

    with sqlite3.connect(DB_PATH) as connection:
        frame = pd.read_sql_query(query, connection, params=params, parse_dates=["date"])

    columns = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume", "ticker"]
    if frame.empty:
        return {ticker: pd.DataFrame(columns=columns) for ticker in tickers}

    frame = frame.rename(
        columns={
            "date": "Date",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "adj_close": "Adj Close",
            "volume": "Volume",
        }
    )
    frame["Date"] = pd.to_datetime(frame["Date"])

    data_by_ticker: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        data_by_ticker[ticker] = frame[frame["ticker"] == ticker].copy().reset_index(drop=True)
    return data_by_ticker


def load_macro_data_from_sqlite(start: str, end: str) -> pd.DataFrame:
    macro_raw = load_market_data_from_sqlite(MACRO_TICKERS, start, end, table_name="macro_history")

    vix = macro_raw["^VIX"].sort_values("Date").copy()
    tnx = macro_raw["^TNX"].sort_values("Date").copy()
    vix_price = vix["Close"].astype(float)
    tnx_price = tnx["Close"].astype(float)

    macro = pd.DataFrame(
        {
            "Date": vix["Date"],
            "vix_close": vix_price,
            "vix_ret_5": vix_price.pct_change(5),
            "vix_vs_sma20": (vix_price / vix_price.rolling(20).mean()) - 1.0,
            "vix_rsi_14": compute_rsi(vix_price, 14),
        }
    )
    tnx_features = pd.DataFrame(
        {
            "Date": tnx["Date"],
            "tnx_close": tnx_price,
            "tnx_change_5": tnx_price.diff(5),
            "tnx_vs_sma20": (tnx_price / tnx_price.rolling(20).mean()) - 1.0,
        }
    )
    return macro.merge(tnx_features, on="Date", how="outer").sort_values("Date").reset_index(drop=True)

def build_feature_columns(features: pd.DataFrame) -> list[str]:
    ticker_dummies = sorted([column for column in features.columns if column.startswith("ticker_")])
    return BASE_NUMERIC_FEATURES + ticker_dummies


def get_latest_complete_end_date() -> str:
    with sqlite3.connect(DB_PATH) as connection:
        price_placeholders = ",".join(["?"] * len(ALL_TICKERS))
        price_query = f"""
        SELECT MAX(date) AS max_date
        FROM price_history
        WHERE ticker IN ({price_placeholders})
        """
        price_max = pd.read_sql_query(price_query, connection, params=ALL_TICKERS, parse_dates=["max_date"])

        macro_placeholders = ",".join(["?"] * len(MACRO_TICKERS))
        macro_query = f"""
        SELECT MAX(date) AS max_date
        FROM macro_history
        WHERE ticker IN ({macro_placeholders})
        """
        macro_max = pd.read_sql_query(macro_query, connection, params=MACRO_TICKERS, parse_dates=["max_date"])

    if price_max["max_date"].isna().all():
        raise ValueError("No price history found in sqlite.")
    if macro_max["max_date"].isna().all():
        raise ValueError("No macro history found in sqlite.")

    latest_complete = min(price_max["max_date"].iloc[0], macro_max["max_date"].iloc[0])
    return pd.Timestamp(latest_complete).date().isoformat()


def build_latest_inference(
    data_all: dict[str, pd.DataFrame],
    feature_columns: list[str],
    latest_end: str,
) -> pd.DataFrame:
    portfolio_data = {ticker: data_all[ticker] for ticker in PORTFOLIO_TICKERS}
    base_features = build_feature_frame(portfolio_data, keep_all_rows=True)
    macro_data = load_macro_data_from_sqlite(TRAIN_START, latest_end)
    features_all = build_regime_features(base_features, data_all, macro_data, keep_all_rows=True)

    predictions_latest = build_and_predict_today(features_all, feature_columns, pd.Timestamp(latest_end))
    predictions_latest = predictions_latest.assign(
        split="inference",
        forward_20d_return=pd.NA,
        next_day_return=pd.NA,
    )
    return predictions_latest[
        ["Date", "ticker", "split", "predicted_return", "forward_20d_return", "next_day_return"]
    ].copy()


def main() -> None:
    latest_end = get_latest_complete_end_date()

    print("Running expanded-pool XGBoost regression without leveraged ETFs (v2)...")
    print(f"Expanded non-leverage pool ({len(PORTFOLIO_TICKERS)}): {', '.join(PORTFOLIO_TICKERS)}")
    print(f"Latest complete sqlite date: {latest_end}")

    data_all = load_market_data_from_sqlite(ALL_TICKERS, TRAIN_START, latest_end)
    portfolio_data = {ticker: data_all[ticker] for ticker in PORTFOLIO_TICKERS}
    base_features = build_feature_frame(portfolio_data)
    macro_data = load_macro_data_from_sqlite(TRAIN_START, latest_end)
    features = build_regime_features(base_features, data_all, macro_data)
    feature_columns = build_feature_columns(features)

    periods = (
        build_monthly_periods(VALIDATION_START, VALIDATION_END, "validation")
        + build_monthly_periods(TEST_START, latest_end, "test")
    )

    predictions = run_walk_forward_regression(features, feature_columns, periods)
    predictions = predictions[
        ["Date", "ticker", "split", "predicted_return", "forward_20d_return", "next_day_return"]
    ].copy()

    inference_predictions = build_latest_inference(data_all, feature_columns, latest_end)
    predictions = pd.concat([predictions, inference_predictions], ignore_index=True)
    predictions = predictions.sort_values(["Date", "split", "ticker"]).reset_index(drop=True)
    predictions.to_csv(PREDICTIONS_PATH, index=False)

    val_preds = predictions[predictions["split"] == "validation"].copy()
    test_preds = predictions[predictions["split"] == "test"].copy()
    val_ic = compute_daily_ic(val_preds)
    test_ic = compute_daily_ic(test_preds)

    summary = pd.DataFrame(
        [
            {
                "universe_size": len(PORTFOLIO_TICKERS),
                "historical_prediction_rows": int((predictions["split"] != "inference").sum()),
                "inference_rows": int((predictions["split"] == "inference").sum()),
                "predictions_rows": len(predictions),
                "validation_ic_mean": float(val_ic.mean()) if not val_ic.empty else float("nan"),
                "validation_icir": icir(val_ic) if not val_ic.empty else float("nan"),
                "test_ic_mean": float(test_ic.mean()) if not test_ic.empty else float("nan"),
                "test_icir": icir(test_ic) if not test_ic.empty else float("nan"),
                "latest_historical_snapshot_date": predictions.loc[
                    predictions["split"] != "inference", "Date"
                ].max().date().isoformat(),
                "latest_inference_date": inference_predictions["Date"].max().date().isoformat(),
                "predictions_csv": PREDICTIONS_PATH.name,
            }
        ]
    )
    summary.to_csv(SUMMARY_PATH, index=False)

    coverage = (
        predictions.groupby(["split", "ticker"])["Date"]
        .agg(["min", "max", "count"])
        .reset_index()
        .sort_values(["split", "ticker"])
        .reset_index(drop=True)
    )

    latest_ranked = inference_predictions.sort_values("predicted_return", ascending=False).reset_index(drop=True)
    latest_ranked["rank"] = range(1, len(latest_ranked) + 1)

    print()
    print("Prediction coverage by split and ticker:")
    print(coverage.to_string(index=False))
    print()
    print("Latest inference snapshot:")
    print(latest_ranked.head(10).to_string(index=False))
    print()
    print("Summary:")
    print(summary.to_string(index=False))
    print()
    print(f"Saved CSV: {PREDICTIONS_PATH.name}")


if __name__ == "__main__":
    main()
