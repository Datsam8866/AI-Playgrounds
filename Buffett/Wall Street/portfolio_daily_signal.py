# -*- coding: utf-8 -*-
"""
portfolio_daily_signal.py
─────────────────────────
16 檔 portfolio 宇宙的每日持倉訊號。

邏輯：
1. 從既有 predictions CSV 計算 rolling IC（w=40, thr=0.00）
2. 下載最新市場資料；若線上下載失敗，退回本地 sqlite 歷史資料
3. 優先用最新資料重訓模型後預測；若失敗則改用 CSV 最後一期 predicted_return 排名
4. 依 IC-Adaptive + VIX 熔斷決定 top-k
5. 以 defensive overlay 建立 base weights（50% core + 50% satellite + 單股上限 12%）
6. 套用驗證最佳的風控規則：EXP50:ANY(VIX>25 OR SPY<SMA200)

執行：
  python portfolio_daily_signal.py
"""
from __future__ import annotations

import contextlib
import io
import os
import sqlite3
import sys
import warnings
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent
_CACHE_DIR = ROOT / ".sandbox_cache"
_CACHE_DIR.mkdir(exist_ok=True)
os.environ.setdefault("LOCALAPPDATA", str(_CACHE_DIR))
os.environ.setdefault("TEMP", str(_CACHE_DIR))
os.environ.setdefault("TMP", str(_CACHE_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

DB_PATH = ROOT / "stock_forecast.sqlite"
PREDICTIONS_CSV = ROOT / "portfolio_universe_xgboost_regression_predictions.csv"
QUARTERLY_MAINLINE_CSV = ROOT / "portfolio_pool_probability_quarterly_walkforward_combined_z_regime_constrained.csv"
TRAIN_START = "2000-01-01"

IC_WINDOW = 40
IC_THRESHOLD = 0.00
K_HIGH = 5
K_LOW = 16
VIX_CIRCUIT_BREAKER = 30.0
CORE_WEIGHT = 0.50
MAX_SINGLE_WEIGHT = 0.12
RISK_OFF_EXPOSURE = 0.50
RISK_OFF_VIX_THRESHOLD = 25.0

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
REFERENCE_TICKERS = ["SPY", "QQQ"]
ALL_TICKERS = list(dict.fromkeys(PORTFOLIO_TICKERS + REFERENCE_TICKERS))


def load_latest_quarterly_calibrated_context() -> dict[str, object] | None:
    if not QUARTERLY_MAINLINE_CSV.exists():
        return None

    frame = pd.read_csv(QUARTERLY_MAINLINE_CSV)
    if frame.empty:
        return None

    current = frame[frame["period"].astype(str).str.endswith("_current")].copy()
    if current.empty:
        current = frame.sort_values("forecast_start").tail(1).copy()
    row = current.iloc[-1]
    return {
        "quarterly_period": str(row.get("period", "")),
        "quarterly_regime": str(row.get("regime_label", "")),
        "quarterly_selection_stage": str(row.get("selection_stage", "")),
        "quarterly_score_combined_z": float(row.get("score_combined_z", np.nan)),
        "quarterly_p_gt_5": float(row.get("p_gt_5", np.nan)),
        "quarterly_p_lt_0": float(row.get("p_lt_0", np.nan)),
        "quarterly_raw_p_gt_5": float(row.get("raw_p_gt_5", np.nan)),
        "quarterly_raw_p_lt_0": float(row.get("raw_p_lt_0", np.nan)),
    }


def compute_rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def configure_runtime_dirs() -> None:
    cache_dir = ROOT / ".sandbox_cache"
    cache_dir.mkdir(exist_ok=True)
    os.environ.setdefault("LOCALAPPDATA", str(cache_dir))
    os.environ.setdefault("TEMP", str(cache_dir))
    os.environ.setdefault("TMP", str(cache_dir))


def open_sqlite_connection(*, writable: bool = False) -> sqlite3.Connection:
    mode = "rwc" if writable else "rw"
    uri = f"file:{DB_PATH.as_posix()}?mode={mode}"
    return sqlite3.connect(uri, uri=True, timeout=30)


def validate_market_data(data_by_ticker: dict[str, pd.DataFrame], tickers: list[str]) -> None:
    missing = []
    for ticker in tickers:
        frame = data_by_ticker.get(ticker)
        if frame is None or frame.empty:
            missing.append(ticker)
            continue
        required = {"Date", "Open", "High", "Low", "Close", "Adj Close", "Volume", "ticker"}
        if not required.issubset(frame.columns):
            missing.append(ticker)

    if missing:
        raise ValueError(f"市場資料缺少 ticker/欄位: {missing}")


def load_market_data_from_sqlite(tickers: list[str]) -> dict[str, pd.DataFrame]:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"找不到 sqlite 歷史資料庫：{DB_PATH}")

    placeholders = ",".join("?" for _ in tickers)
    query = f"""
        SELECT
            date AS Date,
            ticker,
            open AS [Open],
            high AS [High],
            low AS [Low],
            close AS [Close],
            adj_close AS [Adj Close],
            volume AS [Volume]
        FROM price_history
        WHERE ticker IN ({placeholders})
        ORDER BY ticker, Date
    """
    with open_sqlite_connection() as connection:
        frame = pd.read_sql_query(query, connection, params=tickers, parse_dates=["Date"])

    if frame.empty:
        raise ValueError("sqlite price_history 無資料。")

    data_by_ticker: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        local = frame[frame["ticker"] == ticker].copy().reset_index(drop=True)
        if not local.empty:
            data_by_ticker[ticker] = local

    validate_market_data(data_by_ticker, tickers)
    return data_by_ticker


def load_macro_data_from_sqlite() -> pd.DataFrame:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"找不到 sqlite 歷史資料庫：{DB_PATH}")

    query = """
        SELECT
            date AS Date,
            ticker,
            close AS close
        FROM macro_history
        WHERE ticker IN ('^VIX', '^TNX')
        ORDER BY Date
    """
    with open_sqlite_connection() as connection:
        raw = pd.read_sql_query(query, connection, parse_dates=["Date"])

    if raw.empty:
        raise ValueError("sqlite macro_history 無資料。")

    vix = raw[raw["ticker"] == "^VIX"].copy().sort_values("Date")
    tnx = raw[raw["ticker"] == "^TNX"].copy().sort_values("Date")

    vix_price = vix["close"].astype(float)
    tnx_price = tnx["close"].astype(float)
    macro = pd.DataFrame(
        {
            "Date": vix["Date"],
            "vix_close": vix_price,
            "vix_ret_5": vix_price.pct_change(5),
            "vix_vs_sma20": (vix_price / vix_price.rolling(20).mean()) - 1.0,
            "vix_rsi_14": compute_rsi(vix_price, 14),
        }
    )
    macro = macro.merge(
        pd.DataFrame(
            {
                "Date": tnx["Date"],
                "tnx_close": tnx_price,
                "tnx_change_5": tnx_price.diff(5),
                "tnx_vs_sma20": (tnx_price / tnx_price.rolling(20).mean()) - 1.0,
            }
        ),
        on="Date",
        how="outer",
    )
    return macro.sort_values("Date").reset_index(drop=True)


def download_or_load_market_data(start_date: str, end_date: str) -> tuple[dict[str, pd.DataFrame], str]:
    try:
        import yfinance as yf

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            raw = yf.download(
                tickers=ALL_TICKERS,
                start=pd.Timestamp(start_date),
                end=pd.Timestamp(end_date),
                auto_adjust=False,
                progress=False,
                group_by="ticker",
            )
        data_by_ticker = {}
        for ticker in ALL_TICKERS:
            if ticker in raw.columns.get_level_values(0):
                ticker_frame = raw[ticker].copy()
            else:
                ticker_frame = raw.copy()
            ticker_frame = ticker_frame.rename_axis("Date").reset_index()
            ticker_frame["Date"] = pd.to_datetime(ticker_frame["Date"])
            ticker_frame["ticker"] = ticker
            ticker_frame = ticker_frame.dropna(subset=["Close"])
            data_by_ticker[ticker] = ticker_frame
        validate_market_data(data_by_ticker, ALL_TICKERS)
        validate_market_data(data_by_ticker, ALL_TICKERS)
        return data_by_ticker, "yfinance"
    except Exception as exc:
        print(f"  [警告] 線上價格下載失敗，改用本地 sqlite：{exc}")
        return load_market_data_from_sqlite(ALL_TICKERS), "sqlite"


def download_or_load_macro_data(start_date: str, end_date: str) -> tuple[pd.DataFrame, str]:
    try:
        import yfinance as yf

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            raw = yf.download(
                tickers=["^VIX", "^TNX"],
                start=pd.Timestamp(start_date),
                end=pd.Timestamp(end_date),
                auto_adjust=False,
                progress=False,
                group_by="ticker",
            )
        rows = []
        for ticker in ["^VIX", "^TNX"]:
            ticker_frame = raw[ticker].copy().rename_axis("Date").reset_index()
            ticker_frame["ticker"] = ticker
            ticker_frame.columns = [column if column in {"Date", "ticker"} else str(column).lower().replace(" ", "_") for column in ticker_frame.columns]
            rows.append(ticker_frame[["Date", "ticker", "close"]])
        macro_raw = pd.concat(rows, ignore_index=True).sort_values(["Date", "ticker"]).reset_index(drop=True)
        if macro_raw.empty:
            raise ValueError("macro yfinance 回傳空值")

        vix = macro_raw[macro_raw["ticker"] == "^VIX"].copy().sort_values("Date")
        tnx = macro_raw[macro_raw["ticker"] == "^TNX"].copy().sort_values("Date")
        vix_price = vix["close"].astype(float)
        tnx_price = tnx["close"].astype(float)
        macro_data = pd.DataFrame(
            {
                "Date": vix["Date"],
                "vix_close": vix_price,
                "vix_ret_5": vix_price.pct_change(5),
                "vix_vs_sma20": (vix_price / vix_price.rolling(20).mean()) - 1.0,
                "vix_rsi_14": compute_rsi(vix_price, 14),
            }
        )
        macro_data = macro_data.merge(
            pd.DataFrame(
                {
                    "Date": tnx["Date"],
                    "tnx_close": tnx_price,
                    "tnx_change_5": tnx_price.diff(5),
                    "tnx_vs_sma20": (tnx_price / tnx_price.rolling(20).mean()) - 1.0,
                }
            ),
            on="Date",
            how="outer",
        )
        return macro_data.sort_values("Date").reset_index(drop=True), "yfinance"
    except Exception as exc:
        print(f"  [警告] 線上宏觀資料下載失敗，改用本地 sqlite：{exc}")
        return load_macro_data_from_sqlite(), "sqlite"


def compute_rolling_ic_series(predictions: pd.DataFrame, ic_window: int) -> pd.Series:
    daily_ic = compute_daily_ic_local(predictions.dropna(subset=["forward_20d_return"]).copy())
    lagged_ic = daily_ic.shift(20)
    return lagged_ic.rolling(window=ic_window, min_periods=ic_window // 2).mean()


def compute_daily_ic_local(predictions: pd.DataFrame) -> pd.Series:
    rows = []
    for date_value, group in predictions.groupby("Date"):
        valid = group.dropna(subset=["predicted_return", "forward_20d_return"])
        if len(valid) < 3:
            continue
        pred_rank = valid["predicted_return"].rank(method="average")
        fwd_rank = valid["forward_20d_return"].rank(method="average")
        if pred_rank.nunique() < 2 or fwd_rank.nunique() < 2:
            continue
        rows.append({"Date": date_value, "IC": pred_rank.corr(fwd_rank)})
    if not rows:
        return pd.Series(dtype=float)
    return pd.DataFrame(rows).set_index("Date")["IC"].sort_index()


def compute_latest_rolling_ic_local(predictions: pd.DataFrame, ic_window: int, ic_threshold: float) -> tuple[float, str]:
    rolling_mean_ic = compute_rolling_ic_series(predictions, ic_window)
    latest_ic = float(rolling_mean_ic.dropna().iloc[-1]) if not rolling_mean_ic.dropna().empty else np.nan
    mode = "high_confidence" if (not np.isnan(latest_ic) and latest_ic > ic_threshold) else "low_confidence"
    return latest_ic, mode


def load_historical_predictions() -> pd.DataFrame:
    if not PREDICTIONS_CSV.exists():
        raise FileNotFoundError(f"找不到歷史 predictions CSV：{PREDICTIONS_CSV}")
    return pd.read_csv(PREDICTIONS_CSV, parse_dates=["Date"])


def get_latest_csv_rankings(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.Timestamp]:
    latest_prediction_date = predictions["Date"].max()
    latest = (
        predictions.loc[predictions["Date"] == latest_prediction_date, ["Date", "ticker", "predicted_return"]]
        .sort_values("predicted_return", ascending=False)
        .reset_index(drop=True)
    )
    return latest, latest_prediction_date


def get_latest_vix_status(macro_data: pd.DataFrame) -> tuple[float, str, bool]:
    latest_vix = macro_data.sort_values("Date")["vix_close"].dropna()
    if latest_vix.empty:
        return np.nan, "資料不可用", False
    value = float(latest_vix.iloc[-1])
    if value > VIX_CIRCUIT_BREAKER:
        return value, "熔斷", True
    return value, "正常", False


def format_mode(k_value: int) -> str:
    return "高信心" if k_value == K_HIGH else "低信心"


def cap_weights(row: pd.Series, max_single_weight: float) -> pd.Series:
    capped = row.copy().astype(float)
    total = float(capped.sum())
    if total <= 0:
        return capped

    iterations = 0
    while capped.max() > max_single_weight + 1e-12 and iterations < 20:
        over_mask = capped > max_single_weight
        excess = float((capped[over_mask] - max_single_weight).sum())
        capped[over_mask] = max_single_weight

        under_mask = capped < max_single_weight - 1e-12
        under_total = float(capped[under_mask].sum())
        if excess <= 0 or under_total <= 0:
            break

        capped.loc[under_mask] += capped.loc[under_mask] / under_total * excess
        iterations += 1

    final_total = float(capped.sum())
    if final_total > 0:
        capped *= total / final_total
    return capped


def build_overlay_portfolio(predictions_today: pd.DataFrame, k_value: int) -> pd.DataFrame:
    ranked = predictions_today.sort_values("predicted_return", ascending=False).reset_index(drop=True).copy()
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    ranked["selected"] = ranked["rank"] <= k_value
    ranked["core_weight"] = CORE_WEIGHT / len(PORTFOLIO_TICKERS)
    ranked["sat_weight"] = np.where(ranked["selected"], (1.0 - CORE_WEIGHT) / k_value, 0.0)
    ranked["base_weight_pre_cap"] = ranked["core_weight"] + ranked["sat_weight"]
    capped = cap_weights(ranked.set_index("ticker")["base_weight_pre_cap"], MAX_SINGLE_WEIGHT)
    ranked["base_weight"] = ranked["ticker"].map(capped)
    return ranked


def get_latest_spy_trend(data_by_ticker: dict[str, pd.DataFrame]) -> tuple[float, str, bool]:
    spy_frame = data_by_ticker.get("SPY")
    if spy_frame is None or spy_frame.empty:
        return np.nan, "資料不可用", False
    spy_frame = spy_frame.sort_values("Date").copy()
    price = spy_frame["Adj Close"].astype(float)
    sma200 = price.rolling(200).mean()
    latest_ratio = ((price / sma200) - 1.0).dropna()
    if latest_ratio.empty:
        return np.nan, "資料不足", False
    value = float(latest_ratio.iloc[-1])
    return value, "站上 SMA200" if value > 0 else "跌破 SMA200", value <= 0


def apply_market_risk_overlay(
    portfolio: pd.DataFrame,
    latest_vix: float,
    spy_vs_sma200: float,
) -> tuple[pd.DataFrame, bool, float, list[str]]:
    triggers = []
    if not np.isnan(latest_vix) and latest_vix > RISK_OFF_VIX_THRESHOLD:
        triggers.append(f"VIX>{RISK_OFF_VIX_THRESHOLD:.0f}")
    if not np.isnan(spy_vs_sma200) and spy_vs_sma200 <= 0:
        triggers.append("SPY<SMA200")

    net_exposure = RISK_OFF_EXPOSURE if triggers else 1.0
    result = portfolio.copy()
    result["final_weight"] = result["base_weight"] * net_exposure
    return result, bool(triggers), net_exposure, triggers


def try_build_live_predictions(
    data_by_ticker: dict[str, pd.DataFrame],
    macro_data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    from daily_signal import BASE_NUMERIC_FEATURES, build_and_predict_today
    from multi_asset_logistic_baseline import build_feature_frame
    from multi_asset_xgboost_regime_baseline import build_regime_features

    portfolio_data = {ticker: data_by_ticker[ticker] for ticker in PORTFOLIO_TICKERS}
    base_features_all = build_feature_frame(portfolio_data, keep_all_rows=True)
    features_all = build_regime_features(base_features_all, data_by_ticker, macro_data, keep_all_rows=True)
    ticker_dummies = sorted(column for column in features_all.columns if column.startswith("ticker_"))
    feature_columns = BASE_NUMERIC_FEATURES + ticker_dummies
    feature_latest_date = pd.Timestamp(features_all["Date"].max())
    predictions_today = build_and_predict_today(features_all, feature_columns, feature_latest_date)
    return predictions_today, feature_latest_date


def upsert_portfolio_signal_snapshot(
    portfolio: pd.DataFrame,
    signal_date: pd.Timestamp,
    rolling_ic: float,
    k_value: int,
    latest_vix: float,
    vix_status: str,
    spy_vs_sma200: float,
    spy_status: str,
    risk_overlay_on: bool,
    net_exposure: float,
    risk_triggers: list[str],
    prediction_note: str,
    quarterly_context: dict[str, object] | None,
) -> None:
    if not DB_PATH.exists():
        return

    snapshot = portfolio.copy()
    snapshot["date"] = pd.Timestamp(signal_date).strftime("%Y-%m-%d")
    snapshot["rolling_ic"] = rolling_ic
    snapshot["k_value"] = int(k_value)
    snapshot["signal_mode"] = format_mode(k_value)
    snapshot["vix_value"] = latest_vix
    snapshot["vix_status"] = vix_status
    snapshot["spy_vs_sma200"] = spy_vs_sma200
    snapshot["spy_status"] = spy_status
    snapshot["risk_overlay_on"] = int(risk_overlay_on)
    snapshot["net_exposure"] = net_exposure
    snapshot["risk_rule"] = "EXP50:ANY(VIX>25 OR SPY<SMA200)"
    snapshot["risk_triggers"] = ", ".join(risk_triggers)
    snapshot["prediction_note"] = prediction_note
    snapshot["quarterly_period"] = quarterly_context.get("quarterly_period") if quarterly_context else None
    snapshot["quarterly_regime"] = quarterly_context.get("quarterly_regime") if quarterly_context else None
    snapshot["quarterly_selection_stage"] = quarterly_context.get("quarterly_selection_stage") if quarterly_context else None
    snapshot["quarterly_score_combined_z"] = quarterly_context.get("quarterly_score_combined_z") if quarterly_context else np.nan
    snapshot["quarterly_p_gt_5"] = quarterly_context.get("quarterly_p_gt_5") if quarterly_context else np.nan
    snapshot["quarterly_p_lt_0"] = quarterly_context.get("quarterly_p_lt_0") if quarterly_context else np.nan
    snapshot["quarterly_raw_p_gt_5"] = quarterly_context.get("quarterly_raw_p_gt_5") if quarterly_context else np.nan
    snapshot["quarterly_raw_p_lt_0"] = quarterly_context.get("quarterly_raw_p_lt_0") if quarterly_context else np.nan
    snapshot["generated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    snapshot = snapshot.rename(columns={"Date": "prediction_date"})
    ordered = snapshot[
        [
            "date",
            "prediction_date",
            "ticker",
            "predicted_return",
            "rank",
            "selected",
            "core_weight",
            "sat_weight",
            "base_weight_pre_cap",
            "base_weight",
            "final_weight",
            "rolling_ic",
            "k_value",
            "signal_mode",
            "vix_value",
            "vix_status",
            "spy_vs_sma200",
            "spy_status",
            "risk_overlay_on",
            "net_exposure",
            "risk_rule",
            "risk_triggers",
            "prediction_note",
            "quarterly_period",
            "quarterly_regime",
            "quarterly_selection_stage",
            "quarterly_score_combined_z",
            "quarterly_p_gt_5",
            "quarterly_p_lt_0",
            "quarterly_raw_p_gt_5",
            "quarterly_raw_p_lt_0",
            "generated_at",
        ]
    ].copy()

    with open_sqlite_connection(writable=True) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS portfolio_daily_signals (
                date TEXT NOT NULL,
                prediction_date TEXT,
                ticker TEXT NOT NULL,
                predicted_return REAL,
                rank INTEGER,
                selected INTEGER,
                core_weight REAL,
                sat_weight REAL,
                base_weight_pre_cap REAL,
                base_weight REAL,
                final_weight REAL,
                rolling_ic REAL,
                k_value INTEGER,
                signal_mode TEXT,
                vix_value REAL,
                vix_status TEXT,
                spy_vs_sma200 REAL,
                spy_status TEXT,
                risk_overlay_on INTEGER,
                net_exposure REAL,
                risk_rule TEXT,
                risk_triggers TEXT,
                prediction_note TEXT,
                quarterly_period TEXT,
                quarterly_regime TEXT,
                quarterly_selection_stage TEXT,
                quarterly_score_combined_z REAL,
                quarterly_p_gt_5 REAL,
                quarterly_p_lt_0 REAL,
                quarterly_raw_p_gt_5 REAL,
                quarterly_raw_p_lt_0 REAL,
                generated_at TEXT
            )
            """
        )
        existing_columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(portfolio_daily_signals)").fetchall()
        }
        required_columns = {
            "quarterly_period": "TEXT",
            "quarterly_regime": "TEXT",
            "quarterly_selection_stage": "TEXT",
            "quarterly_score_combined_z": "REAL",
            "quarterly_p_gt_5": "REAL",
            "quarterly_p_lt_0": "REAL",
            "quarterly_raw_p_gt_5": "REAL",
            "quarterly_raw_p_lt_0": "REAL",
        }
        for column_name, column_type in required_columns.items():
            if column_name not in existing_columns:
                connection.execute(
                    f"ALTER TABLE portfolio_daily_signals ADD COLUMN {column_name} {column_type}"
                )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_portfolio_daily_signals_date_ticker
            ON portfolio_daily_signals(date, ticker)
            """
        )
        connection.execute("DELETE FROM portfolio_daily_signals WHERE date = ?", (ordered["date"].iloc[0],))
        ordered.to_sql("portfolio_daily_signals", connection, if_exists="append", index=False)
        connection.commit()


def print_signal_report(
    portfolio: pd.DataFrame,
    signal_date: pd.Timestamp,
    rolling_ic: float,
    k_value: int,
    ic_series: pd.Series,
    vix_value: float,
    vix_status: str,
    spy_vs_sma200: float,
    spy_status: str,
    net_exposure: float,
    risk_overlay_on: bool,
    risk_triggers: list[str],
    prediction_note: str,
    quarterly_context: dict[str, object] | None,
) -> None:
    ranked = portfolio.sort_values(["final_weight", "predicted_return"], ascending=[False, False]).reset_index(drop=True)

    print()
    print(f"Portfolio Daily Signal  [{signal_date.date()}]")
    print()
    print(f"Step 1  Rolling IC (w={IC_WINDOW}, thr={IC_THRESHOLD:.2f}) = {rolling_ic:+.4f}")
    print(f"  → 今日 k = {k_value}（{format_mode(k_value)}）")
    print()

    if np.isnan(vix_value):
        print(f"Step 2  VIX = N/A（{vix_status}）")
    else:
        print(f"Step 2  VIX = {vix_value:.1f}（{vix_status}）")
    print()

    if np.isnan(spy_vs_sma200):
        print(f"Step 3  SPY vs SMA200 = N/A（{spy_status}）")
    else:
        print(f"Step 3  SPY vs SMA200 = {spy_vs_sma200:+.2%}（{spy_status}）")
    if risk_overlay_on:
        print(f"  → 套用風控規則 EXP50:ANY(VIX>25 OR SPY<SMA200)，淨曝險降至 {net_exposure:.0%}")
        print(f"  → 觸發條件：{', '.join(risk_triggers)}")
    else:
        print(f"  → 未觸發風控規則，維持 {net_exposure:.0%} 淨曝險")
    print()

    if quarterly_context is not None:
        print("Step 4  季度主線校準機率（Combined Z + constraints）：")
        print(
            f"  {quarterly_context['quarterly_period']}  |  regime={quarterly_context['quarterly_regime']}  |  "
            f"stage={quarterly_context['quarterly_selection_stage']}"
        )
        print(
            f"  Calibrated P(>5%) = {float(quarterly_context['quarterly_p_gt_5']):.1%}  |  "
            f"Calibrated P(<0%) = {float(quarterly_context['quarterly_p_lt_0']):.1%}"
        )
        print(
            f"  Raw P(>5%) = {float(quarterly_context['quarterly_raw_p_gt_5']):.1%}  |  "
            f"Raw P(<0%) = {float(quarterly_context['quarterly_raw_p_lt_0']):.1%}  |  "
            f"Combined Z = {float(quarterly_context['quarterly_score_combined_z']):.3f}"
        )
        print()

    print("今日配置建議（defensive overlay + market overlay）：")
    print("  標記 代號    排名   預測20日報酬    Base權重    Final權重")
    for _, row in ranked.iterrows():
        marker = "*" if row["selected"] else " "
        print(
            f"  {marker}   {row['ticker']:<6} {int(row['rank']):>2}    {row['predicted_return']:+.2%}      "
            f"{row['base_weight']:.2%}      {row['final_weight']:.2%}"
        )
    print()
    print(f"總淨曝險：{portfolio['final_weight'].sum():.2%}  |  現金/保留：{1.0 - portfolio['final_weight'].sum():.2%}")

    print()
    print("近期 Rolling IC 趨勢（最近 10 筆）：")
    recent_ic = ic_series.dropna().tail(10)
    if recent_ic.empty:
        print("  無可用 IC 歷史。")
    else:
        print("  日期        IC值")
        for ic_date, value in recent_ic.items():
            print(f"  {ic_date.date()}  {value:+.4f}")

    print()
    print(prediction_note)
    print()


def main() -> None:
    configure_runtime_dirs()
    run_date = pd.Timestamp(date.today())
    end_date = (date.today() + timedelta(days=1)).strftime("%Y-%m-%d")

    print(f"Portfolio Daily Signal Generator  [{run_date.date()}]")
    print()

    print(f"Step 1 / 4  計算 rolling IC（從 {PREDICTIONS_CSV.name}）...")
    historical_predictions = load_historical_predictions()
    rolling_ic, ic_mode = compute_latest_rolling_ic_local(historical_predictions, ic_window=IC_WINDOW, ic_threshold=IC_THRESHOLD)
    ic_series = compute_rolling_ic_series(historical_predictions, ic_window=IC_WINDOW)
    k_value = K_HIGH if ic_mode == "high_confidence" else K_LOW
    print(f"  Rolling IC = {rolling_ic:+.4f}")
    print(f"  初始 k = {k_value}（{format_mode(k_value)}）")
    print()

    print(f"Step 2 / 4  下載最新市場資料（{TRAIN_START} ~ {end_date}）...")
    data_by_ticker, price_source = download_or_load_market_data(TRAIN_START, end_date)
    price_latest_date = min(frame["Date"].max() for frame in data_by_ticker.values())
    print(f"  價格資料來源：{price_source}")
    print(f"  最新價格日期：{price_latest_date.date()}")
    print()

    print("Step 3 / 4  讀取宏觀條件與風控訊號...")
    macro_data, macro_source = download_or_load_macro_data(TRAIN_START, end_date)
    print(f"  宏觀資料來源：{macro_source}")

    latest_vix, vix_status, vix_breaker = get_latest_vix_status(macro_data)
    if vix_breaker:
        k_value = K_LOW
        print(f"  [VIX熔斷] VIX={latest_vix:.1f} > {VIX_CIRCUIT_BREAKER:.0f}，強制 k={K_LOW}")
    elif np.isnan(latest_vix):
        print("  [VIX] 無資料，維持 IC-Adaptive k")
    else:
        print(f"  [VIX正常] VIX={latest_vix:.1f} <= {VIX_CIRCUIT_BREAKER:.0f}，維持 k={k_value}")
    spy_vs_sma200, spy_status, spy_risk_off = get_latest_spy_trend(data_by_ticker)
    if np.isnan(spy_vs_sma200):
        print(f"  [SPY趨勢] {spy_status}")
    else:
        direction = "觸發風控" if spy_risk_off else "維持 risk-on"
        print(f"  [SPY趨勢] SPY vs SMA200 = {spy_vs_sma200:+.2%}，{direction}")
    print()

    print("Step 4 / 4  訓練模型並預測今日...")
    prediction_note = ""
    try:
        predictions_today, feature_latest_date = try_build_live_predictions(data_by_ticker, macro_data)
        signal_date = pd.Timestamp(feature_latest_date)
        prediction_note = (
            f"註：預測值由 expanding-window 模型以 {signal_date.date()} 最新可用市場資料即時重訓得出。"
        )
        print(f"  使用即時重訓模型，預測日期：{signal_date.date()}")
    except Exception as exc:
        predictions_today, csv_prediction_date = get_latest_csv_rankings(historical_predictions)
        signal_date = pd.Timestamp(csv_prediction_date)
        prediction_note = (
            f"註：即時重訓失敗（{exc}），改用 {csv_prediction_date.date()} 的 predictions CSV 排名作為今日近似訊號。"
        )
        print(f"  [警告] 即時重訓失敗，改用 CSV 最後一期排名：{csv_prediction_date.date()}")
    print()

    portfolio = build_overlay_portfolio(predictions_today, k_value)
    portfolio, risk_overlay_on, net_exposure, risk_triggers = apply_market_risk_overlay(
        portfolio=portfolio,
        latest_vix=latest_vix,
        spy_vs_sma200=spy_vs_sma200,
    )
    quarterly_context = load_latest_quarterly_calibrated_context()
    upsert_portfolio_signal_snapshot(
        portfolio=portfolio,
        signal_date=signal_date,
        rolling_ic=rolling_ic,
        k_value=k_value,
        latest_vix=latest_vix,
        vix_status=vix_status,
        spy_vs_sma200=spy_vs_sma200,
        spy_status=spy_status,
        risk_overlay_on=risk_overlay_on,
        net_exposure=net_exposure,
        risk_triggers=risk_triggers,
        prediction_note=prediction_note,
        quarterly_context=quarterly_context,
    )

    print_signal_report(
        portfolio=portfolio,
        signal_date=signal_date,
        rolling_ic=rolling_ic,
        k_value=k_value,
        ic_series=ic_series,
        vix_value=latest_vix,
        vix_status=vix_status,
        spy_vs_sma200=spy_vs_sma200,
        spy_status=spy_status,
        net_exposure=net_exposure,
        risk_overlay_on=risk_overlay_on,
        risk_triggers=risk_triggers,
        prediction_note=prediction_note,
        quarterly_context=quarterly_context,
    )


if __name__ == "__main__":
    main()
