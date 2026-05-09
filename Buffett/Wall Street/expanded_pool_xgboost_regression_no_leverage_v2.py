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
from sector_etf_config import SECTOR_ETFS, TICKER_TO_SECTOR_ETF

warnings.filterwarnings("ignore", category=FutureWarning, module=r"yfinance(\..*)?$")
warnings.filterwarnings("ignore", category=FutureWarning, message=r".*ChainedAssignmentError.*")


PORTFOLIO_TICKERS = NON_LEVERAGED_EXPANDED_POOL_TICKERS
MACRO_TICKERS = ["^VIX", "^TNX"]
ALL_TICKERS = list(dict.fromkeys(PORTFOLIO_TICKERS + REFERENCE_TICKERS + SECTOR_ETFS))
PREDICTIONS_PATH = ROOT / "expanded_pool_xgboost_regression_no_leverage_predictions_v2.csv"
SUMMARY_PATH = ROOT / "expanded_pool_xgboost_regression_no_leverage_summary_v2.csv"

SECTOR_FEATURE_NAMES: list[str] = [
    "sector_ret_20",
    "sector_ret_60",
    "sector_vs_sma200",
    "sector_vol_20",
    "rel_ret20_vs_sector",
    "rel_ret60_vs_sector",
    "sector_rank_pct",
]


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

def build_sector_features(
    features: pd.DataFrame,
    data_all: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Add sector ETF momentum features to the feature frame.

    For each ticker, the corresponding sector ETF (from TICKER_TO_SECTOR_ETF)
    contributes:
      sector_ret_20, sector_ret_60, sector_vs_sma200, sector_vol_20
      rel_ret20_vs_sector, rel_ret60_vs_sector
      sector_rank_pct  (cross-sector rotation signal per date)

    Tickers not in the mapping fall back to SPY.
    """
    import numpy as np

    # ── Build per-ETF time series keyed by (Date) ──────────────────────────
    # Store as dict: etf -> {col: Series indexed by Date}
    sector_ts: dict[str, pd.DataFrame] = {}
    for etf in SECTOR_ETFS + ["SPY"]:
        if etf not in data_all or data_all[etf].empty:
            continue
        df = data_all[etf].sort_values("Date").reset_index(drop=True)
        price = df["Adj Close"].astype(float)
        sector_ts[etf] = pd.DataFrame(
            {
                "ret_20": price.pct_change(20).values,
                "ret_60": price.pct_change(60).values,
                "vs_sma200": ((price / price.rolling(200).mean()) - 1.0).values,
                "vol_20": price.pct_change().rolling(20).std().values,
            },
            index=df["Date"],
        )

    if not sector_ts:
        for col in SECTOR_FEATURE_NAMES:
            features[col] = float("nan")
        return features

    # ── Build cross-sector rank lookup: (date, etf) -> rank_pct ────────────
    rank_etfs = [e for e in SECTOR_ETFS if e in sector_ts]
    if len(rank_etfs) >= 2:
        rank_dfs = []
        for etf in rank_etfs:
            tmp = sector_ts[etf][["ret_20"]].copy()
            tmp.index.name = "Date"
            tmp = tmp.reset_index()
            tmp["etf"] = etf
            rank_dfs.append(tmp)
        rank_long = pd.concat(rank_dfs, ignore_index=True)
        rank_long["sector_rank_pct"] = rank_long.groupby("Date")["ret_20"].rank(pct=True)
        # Build dict: (date, etf) -> rank_pct for fast lookup
        sector_rank_lookup: dict[tuple, float] = {
            (row.Date, row.etf): row.sector_rank_pct
            for row in rank_long.itertuples(index=False)
        }
    else:
        rank_etfs = []
        sector_rank_lookup = {}

    # ── Map each ticker to its ETF key (fall back to SPY) ──────────────────
    out = features.copy()
    out["_sector_etf"] = out["ticker"].map(TICKER_TO_SECTOR_ETF).fillna("SPY")
    # Resolve unmapped ETFs to SPY if SPY is available
    if "SPY" in sector_ts:
        out["_sector_etf"] = out["_sector_etf"].apply(
            lambda e: e if e in sector_ts else "SPY"
        )

    # ── Initialise output columns ───────────────────────────────────────────
    for col in ["sector_ret_20", "sector_ret_60", "sector_vs_sma200", "sector_vol_20", "sector_rank_pct"]:
        out[col] = np.nan

    # ── Fill sector features row-by-row via vectorised date lookup ──────────
    for etf_key, grp_mask in out.groupby("_sector_etf").groups.items():
        ts = sector_ts.get(etf_key)
        if ts is None:
            continue

        grp = out.loc[grp_mask]
        dates = grp["Date"]

        # Align ETF time series to the dates present in this group
        aligned = ts.reindex(dates)

        out.loc[grp_mask, "sector_ret_20"] = aligned["ret_20"].values
        out.loc[grp_mask, "sector_ret_60"] = aligned["ret_60"].values
        out.loc[grp_mask, "sector_vs_sma200"] = aligned["vs_sma200"].values
        out.loc[grp_mask, "sector_vol_20"] = aligned["vol_20"].values

        # sector_rank_pct
        if sector_rank_lookup and etf_key in rank_etfs:
            rank_vals = np.array(
                [sector_rank_lookup.get((d, etf_key), np.nan) for d in dates]
            )
            out.loc[grp_mask, "sector_rank_pct"] = rank_vals
        elif sector_rank_lookup:
            # SPY / fallback tickers get median rank
            out.loc[grp_mask, "sector_rank_pct"] = 0.5

    out["rel_ret20_vs_sector"] = out["ret_20"] - out["sector_ret_20"]
    out["rel_ret60_vs_sector"] = out["ret_60"] - out["sector_ret_60"]
    out = out.drop(columns=["_sector_etf"])
    return out


def build_feature_columns(features: pd.DataFrame) -> list[str]:
    ticker_dummies = sorted([column for column in features.columns if column.startswith("ticker_")])
    return BASE_NUMERIC_FEATURES + SECTOR_FEATURE_NAMES + ticker_dummies


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
    features_all = build_sector_features(features_all, data_all)

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
    features = build_sector_features(features, data_all)
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
