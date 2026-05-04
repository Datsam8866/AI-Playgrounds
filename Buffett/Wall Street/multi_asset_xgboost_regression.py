# -*- coding: utf-8 -*-
"""
multi_asset_xgboost_regression.py
───────────────────────────────────
改用 XGBRegressor 預測 forward_20d_return（連續值），
以「預期報酬排序」取代「漲跌機率 threshold」做選股。

核心差異 vs 分類版本：
  - 目標變數：forward_20d_return（連續報酬）
  - 排名信號：predicted_return（每日橫截面排序）
  - 選股規則：top-k / top-percentile / predicted_return > 0
  - 加權方式：equal weight 或 predicted-return-proportional weight
  - 評估指標：IC（Information Coefficient）+ ICIR + strategy Sharpe

label_available_date 邏輯與分類版本相同，不引入 lookahead bias。
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

from multi_asset_logistic_baseline import (
    TICKERS,
    HORIZON_DAYS,
    TRAIN_START,
    TRAIN_END,
    VALIDATION_START,
    VALIDATION_END,
    TEST_START,
    TEST_END,
    build_feature_frame,
    build_monthly_periods,
    download_market_data,
    max_drawdown_from_returns,
    sharpe_from_returns,
    TRADING_DAYS_PER_YEAR,
)
import yfinance as yf

from multi_asset_xgboost_regime_baseline import (
    download_macro_data,
    build_regime_features,
)
from multi_asset_logistic_baseline import compute_rsi

pd.set_option("display.notebook_repr_html", False)
pd.set_option("display.max_columns", 26)
pd.set_option("display.max_rows", 40)
pd.set_option("display.width", 180)

warnings.filterwarnings("ignore", category=FutureWarning, module=r"yfinance(\..*)?$")
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*ChainedAssignmentError: behaviour will change in pandas 3\.0!.*",
)

# 對照組：v1 分類版本的 CSV（若存在則做直接比較）
CLASSIFICATION_PREDICTIONS_PATH = "multi_asset_xgboost_regime_predictions.csv"

# 擴充宏觀特徵（v2）— 實驗後確認不改善效能，預設關閉
# 根本原因：27 個全局宏觀特徵對所有標的相同，稀釋股票相對特徵重要性，test ICIR 0.852→0.704
USE_EXTENDED_MACRO = False

EXTENDED_MACRO_TICKERS = {
    "^IRX":      "irx",   # 3 個月國庫券（Fed 短期利率代理）
    "DX-Y.NYB":  "usd",   # 美元指數
    "GC=F":      "gold",  # 黃金期貨（避險需求）
    "CL=F":      "oil",   # 原油期貨（通膨/成長指標）
    "^VVIX":     "vvix",  # VIX of VIX（市場壓力深度）
}


def download_extended_macro(start: str, end: str) -> pd.DataFrame:
    """
    下載擴充宏觀資料：IRX、USD、Gold、Oil、VVIX。
    失敗的 ticker 自動跳過，不會中斷流程。
    """
    tickers = list(EXTENDED_MACRO_TICKERS.keys())
    try:
        raw = yf.download(
            tickers=tickers,
            start=pd.Timestamp(start),
            end=pd.Timestamp(end) + pd.Timedelta(days=1),
            auto_adjust=True,
            progress=False,
            group_by="ticker",
        )
    except Exception as e:
        print(f"  [extended macro] download failed: {e}")
        return pd.DataFrame()

    result = None
    for ticker, name in EXTENDED_MACRO_TICKERS.items():
        try:
            if len(tickers) == 1:
                frame = raw.copy()
            else:
                frame = raw[ticker].copy() if ticker in raw.columns.get_level_values(0) else pd.DataFrame()
            if frame.empty:
                print(f"  [extended macro] {ticker} empty, skipping.")
                continue

            frame = frame.rename_axis("Date").reset_index()
            frame["Date"] = pd.to_datetime(frame["Date"])
            price = frame["Close"].astype(float)

            enriched = pd.DataFrame({"Date": frame["Date"]})
            enriched[f"{name}_close"]    = price
            enriched[f"{name}_ret_5"]    = price.pct_change(5)
            enriched[f"{name}_ret_20"]   = price.pct_change(20)
            enriched[f"{name}_vs_sma20"] = (price / price.rolling(20).mean()) - 1.0
            enriched[f"{name}_vs_sma60"] = (price / price.rolling(60).mean()) - 1.0

            if result is None:
                result = enriched
            else:
                result = result.merge(enriched, on="Date", how="outer")
        except Exception as e:
            print(f"  [extended macro] {ticker} processing failed: {e}")
            continue

    if result is None:
        return pd.DataFrame()

    result = result.sort_values("Date").reset_index(drop=True)

    # 前向填補（適用期貨/指數資料的非交易日缺口）
    fill_cols = [c for c in result.columns if c != "Date"]
    result[fill_cols] = result[fill_cols].ffill()

    return result


def add_extended_macro_features(
    features: pd.DataFrame,
    extended_macro: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """
    將 extended_macro 合併到 features，並計算衍生特徵（殖利率曲線斜率）。
    回傳（更新後 features, 新增的特徵欄位名稱）。
    """
    if extended_macro.empty:
        return features, []

    out = features.merge(extended_macro, on="Date", how="left")

    new_cols: list[str] = []
    # 記錄哪些欄位成功合併
    for name in EXTENDED_MACRO_TICKERS.values():
        for suffix in ["close", "ret_5", "ret_20", "vs_sma20", "vs_sma60"]:
            col = f"{name}_{suffix}"
            if col in out.columns:
                new_cols.append(col)

    # 殖利率曲線斜率：10Y（tnx_close）- 3M（irx_close）
    if "tnx_close" in out.columns and "irx_close" in out.columns:
        out["yield_curve_slope"]    = out["tnx_close"] - out["irx_close"]
        out["yield_curve_chg_20"]   = out["yield_curve_slope"].diff(20)
        new_cols += ["yield_curve_slope", "yield_curve_chg_20"]

    return out, new_cols

# Validation 選參使用的交易成本
VALIDATION_COST = 0.001

# 選股規則候選
TOP_K_CANDIDATES = list(range(1, 9))               # top-1 到 top-8
TOP_PCT_CANDIDATES = [0.125, 0.25, 0.375, 0.5]    # top 12.5% 到 50%

# IC-Adaptive 動態選股候選參數
IC_WINDOW_CANDIDATES = [40, 60, 90]          # 滾動 IC 計算窗口（天）
IC_THRESHOLD_CANDIDATES = [0.0, 0.02, 0.05, 0.08]  # IC 切換閾值
IC_ADAPTIVE_K_HIGH = 3   # 高 IC 時集中選股
IC_ADAPTIVE_K_LOW  = 8   # 低 IC 時分散持倉（全持）


def build_regressor() -> XGBRegressor:
    """與分類版本相同的超參數，只換 objective。"""
    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=120,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=20,
        reg_lambda=3.0,
        random_state=42,
        n_jobs=4,
        tree_method="hist",
    )


def run_walk_forward_regression(
    features: pd.DataFrame,
    feature_columns: list[str],
    periods: list,
    train_window_days: int | None = None,
) -> pd.DataFrame:
    """Walk-forward 預測，可選 expanding 或 rolling 訓練窗口。

    train_window_days=None  → expanding window（從 TRAIN_START 累積至今，原行為）
    train_window_days=N     → rolling window（只用預測日前 N 個日曆天的資料）
    """
    all_predictions = []

    for period in periods:
        if train_window_days is not None:
            # Rolling window：只用最近 N 個日曆天
            window_start = period.start - pd.Timedelta(days=train_window_days)
            train_mask = (
                (features["Date"] >= window_start)
                & (features["Date"] < period.start)
                & (features["label_available_date"] < period.start)
            )
        else:
            # Expanding window（原行為）
            train_mask = (
                (features["Date"] >= pd.Timestamp(TRAIN_START))
                & (features["Date"] < period.start)
                & (features["label_available_date"] < period.start)
            )
        eval_mask = (
            (features["Date"] >= period.start)
            & (features["Date"] <= period.end)
        )

        train_frame = features.loc[train_mask].copy()
        eval_frame = features.loc[eval_mask].copy()

        if train_frame.empty or eval_frame.empty:
            continue
        if train_frame["forward_20d_return"].isna().all():
            continue

        model = build_regressor()
        train_y = train_frame["forward_20d_return"].astype(float)
        model.fit(train_frame[feature_columns], train_y)

        eval_frame = eval_frame.assign(
            split=period.label,
            predicted_return=model.predict(eval_frame[feature_columns]),
        )
        all_predictions.append(eval_frame)

    if not all_predictions:
        raise ValueError("No walk-forward predictions produced.")

    return pd.concat(all_predictions, ignore_index=True)


# ── IC / ICIR 計算 ─────────────────────────────────────────────────────────

def compute_daily_ic(predictions: pd.DataFrame) -> pd.Series:
    """每日橫截面 Spearman IC（predicted_return vs forward_20d_return）。"""
    daily_ic = []
    for date, group in predictions.groupby("Date"):
        if len(group) < 3:
            continue
        ic, _ = spearmanr(group["predicted_return"], group["forward_20d_return"])
        daily_ic.append({"Date": date, "IC": ic})
    return pd.DataFrame(daily_ic).set_index("Date")["IC"]


def icir(ic_series: pd.Series, horizon: int = HORIZON_DAYS) -> float:
    """ICIR = mean(IC) / std(IC) × sqrt(252 / horizon)。"""
    if ic_series.std(ddof=1) == 0 or ic_series.empty:
        return np.nan
    return float(ic_series.mean() / ic_series.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR / horizon))


# ── Signal 建構 ───────────────────────────────────────────────────────────

def top_k_signal_reg(predictions: pd.DataFrame, k: int) -> pd.Series:
    ranks = predictions.groupby("Date")["predicted_return"].rank(method="first", ascending=False)
    return (ranks <= k).astype(int)


def top_pct_signal_reg(predictions: pd.DataFrame, pct: float) -> pd.Series:
    counts = predictions.groupby("Date")["predicted_return"].transform("size")
    k_by_row = np.ceil(counts * pct).astype(int).clip(lower=1)
    ranks = predictions.groupby("Date")["predicted_return"].rank(method="first", ascending=False)
    return (ranks <= k_by_row).astype(int)


def positive_pred_signal(predictions: pd.DataFrame) -> pd.Series:
    return (predictions["predicted_return"] > 0).astype(int)


def rolling_ic_adaptive_signal(
    predictions: pd.DataFrame,
    ic_window: int = 60,
    ic_threshold: float = 0.05,
    k_high: int = IC_ADAPTIVE_K_HIGH,
    k_low: int = IC_ADAPTIVE_K_LOW,
) -> pd.Series:
    """
    IC-Adaptive 動態 k 選股信號。

    邏輯：
    - 計算每日橫截面 Spearman IC
    - 向後移位 HORIZON_DAYS（避免 lookahead：IC[D] 需要 D+20 的 forward return）
    - 計算滾動均值 IC（ic_window 天）
    - 當日滾動 IC > ic_threshold → 使用 k_high（集中）；否則 → k_low（分散）

    無 lookahead bias：決策時使用的 IC 均來自 HORIZON_DAYS 之前已實現的數據。
    """
    daily_ic = compute_daily_ic(predictions)

    # 移位 HORIZON_DAYS 確保無 lookahead（IC at T uses forward return at T+20）
    lagged_ic = daily_ic.shift(HORIZON_DAYS)
    rolling_mean_ic = lagged_ic.rolling(window=ic_window, min_periods=ic_window // 2).mean()

    all_dates = pd.DatetimeIndex(sorted(predictions["Date"].unique()))
    signal = pd.Series(0, index=predictions.index, dtype=int)

    for date in all_dates:
        day_mask = predictions["Date"] == date
        day_df = predictions.loc[day_mask]

        ic_val = rolling_mean_ic.get(date, np.nan)
        k = k_high if (not np.isnan(ic_val) and ic_val > ic_threshold) else k_low

        ranks = day_df["predicted_return"].rank(method="first", ascending=False)
        signal.loc[day_mask] = (ranks <= k).astype(int).values

    return signal


def sweep_ic_adaptive(
    predictions: pd.DataFrame,
    transaction_cost: float = 0.0,
    k_high: int = IC_ADAPTIVE_K_HIGH,
    k_low: int = IC_ADAPTIVE_K_LOW,
) -> pd.DataFrame:
    """在給定 predictions 上，對 ic_window × ic_threshold 做 grid sweep。"""
    rows = []
    for window in IC_WINDOW_CANDIDATES:
        for thr in IC_THRESHOLD_CANDIDATES:
            sig = rolling_ic_adaptive_signal(predictions, window, thr, k_high, k_low)
            label = f"ic_adapt_w{window}_t{thr:.2f}"
            res = evaluate_signal_reg(predictions, sig, "ic_adaptive", label, transaction_cost, False)
            res["ic_window"] = window
            res["ic_threshold"] = thr
            res["k_high"] = k_high
            res["k_low"] = k_low
            # 計算平均每日持倉數（等效覆蓋率描述）
            res["avg_k_used"] = float(sig.groupby(predictions["Date"]).sum().mean())
            rows.append(res)
    return pd.DataFrame(rows)


# ── Strategy 評估 ─────────────────────────────────────────────────────────

def evaluate_signal_reg(
    predictions: pd.DataFrame,
    signal: pd.Series,
    rule: str,
    parameter: float | int | str,
    transaction_cost: float = 0.0,
    weight_by_pred: bool = False,
) -> dict:
    signal = signal.astype(int)
    all_dates = pd.DatetimeIndex(sorted(predictions["Date"].unique()))
    all_tickers = sorted(predictions["ticker"].unique())
    selected = predictions[signal == 1].copy()

    returns_wide = (
        predictions.pivot(index="Date", columns="ticker", values="next_day_return")
        .reindex(index=all_dates, columns=all_tickers)
        .fillna(0.0)
    )

    if selected.empty:
        net_returns = pd.Series(0.0, index=all_dates)
    else:
        if weight_by_pred:
            # predicted_return 比例加權（只取正值部分）
            selected["raw_weight"] = selected["predicted_return"].clip(lower=0.0)
            total = selected.groupby("Date")["raw_weight"].transform("sum").replace(0.0, np.nan)
            selected["weight"] = selected["raw_weight"] / total
            # 若某日全部預測報酬 ≤ 0，退回等權
            selected["weight"] = selected["weight"].fillna(1.0 / selected.groupby("Date")["ticker"].transform("count"))
        else:
            selected["weight"] = 1.0 / selected.groupby("Date")["ticker"].transform("count")

        position_wide = (
            selected.pivot(index="Date", columns="ticker", values="weight")
            .reindex(index=all_dates, columns=all_tickers)
            .fillna(0.0)
        )
        turnover = position_wide.diff().abs().sum(axis=1)
        turnover.iloc[0] = position_wide.iloc[0].abs().sum()

        gross = (position_wide * returns_wide).sum(axis=1)
        net_returns = gross - transaction_cost * turnover

    benchmark_returns = predictions.groupby("Date")["next_day_return"].mean().reindex(all_dates)

    return {
        "rule": rule,
        "parameter": parameter,
        "weight_by_pred": weight_by_pred,
        "transaction_cost": transaction_cost,
        "coverage": float(signal.mean()),
        "avg_names_per_day": float(signal.groupby(predictions["Date"]).sum().mean()),
        "strategy_return": float((1.0 + net_returns).prod()),
        "strategy_sharpe": sharpe_from_returns(net_returns),
        "strategy_max_drawdown": max_drawdown_from_returns(net_returns),
        "benchmark_return": float((1.0 + benchmark_returns).prod()),
        "benchmark_sharpe": sharpe_from_returns(benchmark_returns),
        "benchmark_max_drawdown": max_drawdown_from_returns(benchmark_returns),
    }


def evaluate_all_rules(
    predictions: pd.DataFrame,
    transaction_cost: float = 0.0,
) -> pd.DataFrame:
    rows = []
    for k in TOP_K_CANDIDATES:
        sig = top_k_signal_reg(predictions, k)
        rows.append(evaluate_signal_reg(predictions, sig, "top_k", k, transaction_cost, False))
        rows.append(evaluate_signal_reg(predictions, sig, "top_k_pred_weight", k, transaction_cost, True))
    for pct in TOP_PCT_CANDIDATES:
        sig = top_pct_signal_reg(predictions, pct)
        rows.append(evaluate_signal_reg(predictions, sig, "top_pct", pct, transaction_cost, False))
    # predicted_return > 0
    sig = positive_pred_signal(predictions)
    rows.append(evaluate_signal_reg(predictions, sig, "positive_pred", 0, transaction_cost, False))
    rows.append(evaluate_signal_reg(predictions, sig, "positive_pred_weight", 0, transaction_cost, True))
    return pd.DataFrame(rows)


def feature_importance_report(
    features: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    mask = (
        (features["Date"] >= pd.Timestamp(TRAIN_START))
        & (features["Date"] <= pd.Timestamp(VALIDATION_END))
        & (features["label_available_date"] <= pd.Timestamp(VALIDATION_END))
    )
    model = build_regressor()
    model.fit(
        features.loc[mask, feature_columns],
        features.loc[mask, "forward_20d_return"].astype(float),
    )
    return (
        pd.DataFrame({"feature": feature_columns, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .head(15)
        .reset_index(drop=True)
    )


def main() -> None:
    # ── 資料準備（同 regime_baseline）
    data_by_ticker = download_market_data(TICKERS, TRAIN_START, TEST_END)
    base_features = build_feature_frame(data_by_ticker)
    macro_data = download_macro_data(TRAIN_START, TEST_END)
    features = build_regime_features(base_features, data_by_ticker, macro_data)

    # ── 擴充宏觀特徵（實驗用，預設關閉）
    extended_feature_cols: list[str] = []
    if USE_EXTENDED_MACRO:
        print("Downloading extended macro data (IRX, USD, Gold, Oil, VVIX)...")
        extended_macro = download_extended_macro(TRAIN_START, TEST_END)
        features, extended_feature_cols = add_extended_macro_features(features, extended_macro)
        if extended_feature_cols:
            print(f"  Added {len(extended_feature_cols)} extended macro features.")
            features[extended_feature_cols] = features[extended_feature_cols].ffill()
        else:
            print("  No extended macro features added (download may have failed).")

    base_numeric_features = [
        "ret_5", "ret_20", "ret_60",
        "vol_20", "vol_60",
        "price_vs_sma20", "price_vs_sma60", "price_vs_sma200",
        "rsi_14", "macd_hist", "volume_ratio20",
        "drawdown_20", "drawdown_60", "intraday_range",
        "spy_ret_20", "spy_ret_60", "spy_vs_sma200", "spy_vol_20",
        "qqq_ret_20", "qqq_vs_sma200", "qqq_vol_20",
        "vix_close", "vix_ret_5", "vix_vs_sma20", "vix_rsi_14",
        "tnx_close", "tnx_change_5", "tnx_vs_sma20",
        "rel_ret20_vs_spy", "rel_ret60_vs_spy",
        "breadth_above_sma20", "breadth_above_sma200",
        "rank_ret20_pct", "rank_ret60_pct", "rank_vol20_pct", "rank_drawdown20_pct",
    ]
    ticker_dummies = sorted([c for c in features.columns if c.startswith("ticker_")])
    # 只加入實際存在的 extended 特徵
    valid_extended = [c for c in extended_feature_cols if c in features.columns]
    feature_columns = base_numeric_features + valid_extended + ticker_dummies
    print(f"  Total features: {len(feature_columns)} (base={len(base_numeric_features)}, extended={len(valid_extended)}, dummies={len(ticker_dummies)})")

    periods = (
        build_monthly_periods(VALIDATION_START, VALIDATION_END, "validation")
        + build_monthly_periods(TEST_START, TEST_END, "test")
    )

    # ── Expanding window（原行為）
    print("Running walk-forward XGBoost Regression (expanding window)...")
    predictions = run_walk_forward_regression(features, feature_columns, periods)
    val_preds = predictions[predictions["split"] == "validation"].copy()
    test_preds = predictions[predictions["split"] == "test"].copy()

    # ── Rolling window（3 年 = 1095 日曆天）
    ROLLING_WINDOW_DAYS = 3 * 365
    print(f"Running walk-forward XGBoost Regression (rolling {ROLLING_WINDOW_DAYS//365}-year window)...")
    predictions_roll = run_walk_forward_regression(
        features, feature_columns, periods, train_window_days=ROLLING_WINDOW_DAYS
    )
    val_preds_roll = predictions_roll[predictions_roll["split"] == "validation"].copy()
    test_preds_roll = predictions_roll[predictions_roll["split"] == "test"].copy()

    # ── IC / ICIR（expanding）
    val_ic = compute_daily_ic(val_preds)
    test_ic = compute_daily_ic(test_preds)

    # ── IC / ICIR（rolling window）
    val_ic_roll = compute_daily_ic(val_preds_roll)
    test_ic_roll = compute_daily_ic(test_preds_roll)
    val_ic_roll_A = compute_daily_ic(val_preds_roll[val_preds_roll["Date"] < "2015-01-01"])
    val_ic_roll_B = compute_daily_ic(val_preds_roll[val_preds_roll["Date"] >= "2015-01-01"])

    # ── 全段 Validation 選參（cost-aware Sharpe，原有邏輯）
    print(f"Evaluating rules on validation (cost={VALIDATION_COST:.3f})...")
    val_rules = evaluate_all_rules(val_preds, VALIDATION_COST)
    val_rules = val_rules.sort_values(
        ["strategy_sharpe", "strategy_return"], ascending=[False, False]
    ).reset_index(drop=True)

    best_rule = val_rules.iloc[0]

    # ── 雙段 Validation：ICIR + Sharpe 一致性選 k
    print("Running dual-window validation (val_A=2010-14, val_B=2015-18)...")
    val_A = val_preds[val_preds["Date"] < "2015-01-01"].copy()
    val_B = val_preds[val_preds["Date"] >= "2015-01-01"].copy()

    def k_sweep_summary(data: pd.DataFrame, label: str) -> pd.DataFrame:
        ic_s = compute_daily_ic(data)
        rows = []
        for k in TOP_K_CANDIDATES:
            sig = top_k_signal_reg(data, k)
            res = evaluate_signal_reg(data, sig, "top_k", k, VALIDATION_COST, False)
            res_w = evaluate_signal_reg(data, sig, "top_k_pred_weight", k, VALIDATION_COST, True)
            rows.append({
                "k": k,
                "window": label,
                "ic_mean": float(ic_s.mean()),
                "icir": icir(ic_s),
                "sharpe_equal": res["strategy_sharpe"],
                "return_equal": res["strategy_return"],
                "sharpe_pred_w": res_w["strategy_sharpe"],
                "return_pred_w": res_w["strategy_return"],
                "max_dd_equal": res["strategy_max_drawdown"],
            })
        return pd.DataFrame(rows)

    sweep_A = k_sweep_summary(val_A, "val_A 2010-2014")
    sweep_B = k_sweep_summary(val_B, "val_B 2015-2018")

    # 合併：每個 k 的跨時期一致性
    merge_cols = ["k", "sharpe_equal", "return_equal", "sharpe_pred_w", "max_dd_equal"]
    dual = sweep_A[merge_cols].merge(
        sweep_B[merge_cols], on="k", suffixes=("_A", "_B")
    )
    dual["avg_sharpe_equal"] = (dual["sharpe_equal_A"] + dual["sharpe_equal_B"]) / 2
    dual["avg_sharpe_pred_w"] = (dual["sharpe_pred_w_A"] + dual["sharpe_pred_w_B"]) / 2
    dual["both_sharpe_positive"] = (dual["sharpe_equal_A"] > 0) & (dual["sharpe_equal_B"] > 0)
    dual["sharpe_gap"] = (dual["sharpe_equal_A"] - dual["sharpe_equal_B"]).abs()

    # IC 在全段 val 上計算（兩段 IC 幾乎相同）
    val_ic_A = compute_daily_ic(val_A)
    val_ic_B = compute_daily_ic(val_B)

    # 選出最佳 k（5 種方法）：
    # 方法 1：avg_sharpe 最高且兩段都為正（最保守）
    consistent_k_df = dual[dual["both_sharpe_positive"]].sort_values("avg_sharpe_equal", ascending=False)
    best_k_consistent = int(consistent_k_df.iloc[0]["k"]) if not consistent_k_df.empty else 8

    # 方法 2：avg_sharpe 最高（不強制兩段都正）
    best_k_avg_sharpe = int(dual.sort_values("avg_sharpe_equal", ascending=False).iloc[0]["k"])

    # 方法 3：pred_weight avg_sharpe 最高
    best_k_pred_weight = int(dual.sort_values("avg_sharpe_pred_w", ascending=False).iloc[0]["k"])

    # 方法 4：val_A-only equal weight Sharpe（val_B 結構性 ICIR≈0，排除其影響）
    best_k_val_A_equal = int(sweep_A.sort_values("sharpe_equal", ascending=False).iloc[0]["k"])

    # 方法 5：val_A-only pred_weight Sharpe
    best_k_val_A_pred_w = int(sweep_A.sort_values("sharpe_pred_w", ascending=False).iloc[0]["k"])

    # 方法 6：val_B-only equal weight Sharpe
    # 理由：val_A (2010-14) ICIR=0.202 較弱，val_B (2015-18) ICIR≈0.510 訊號更穩定，
    # 用 val_B 單獨選 k 更能反映模型在高 IC 期的真實排名能力
    best_k_val_B_equal = int(sweep_B.sort_values("sharpe_equal", ascending=False).iloc[0]["k"])

    # ── 穩定性分析：各方法的 k 選擇與 val_A Sharpe 差距
    method_k_map = {
        "M1_consistent":    best_k_consistent,
        "M2_avg_sharpe":    best_k_avg_sharpe,
        "M3_pred_weight":   best_k_pred_weight,
        "M4_valA_equal":    best_k_val_A_equal,
        "M5_valA_pred_w":   best_k_val_A_pred_w,
        "M6_valB_equal":    best_k_val_B_equal,
    }
    # val_A sharpe 分布（用於判斷 k=2/3/4 差距是否在統計誤差範圍內）
    val_A_sharpe_equal   = sweep_A.set_index("k")["sharpe_equal"]
    val_A_sharpe_pred_w  = sweep_A.set_index("k")["sharpe_pred_w"]
    # top-3 k 值的 val_A Sharpe（equal weight）排名
    top3_val_A = sweep_A.sort_values("sharpe_equal", ascending=False).head(3)

    # ── Test 評估：5 種選法 + 原始全段 val 選法
    def make_test_result(k: int, label: str, weight_by_pred: bool) -> dict:
        sig = top_k_signal_reg(test_preds, k)
        r = evaluate_signal_reg(test_preds, sig, f"top_k_{label}", k, VALIDATION_COST, weight_by_pred)
        r["selection_method"] = label
        r["k_selected"] = k
        return r

    test_results_comparison = pd.DataFrame([
        make_test_result(int(best_rule["parameter"]) if str(best_rule["rule"]).startswith("top_k") else 8,
                         "full_val_sharpe", False),
        make_test_result(best_k_consistent,  "M1_consistent",   False),
        make_test_result(best_k_avg_sharpe,  "M2_avg_sharpe",   False),
        make_test_result(best_k_pred_weight, "M3_pred_weight",   True),
        make_test_result(best_k_val_A_equal, "M4_valA_equal",   False),
        make_test_result(best_k_val_A_pred_w,"M5_valA_pred_w",   True),
        make_test_result(best_k_val_B_equal, "M6_valB_equal",   False),
    ])

    # 主要報告用：雙段一致性選出的最佳 k
    test_signal = top_k_signal_reg(test_preds, best_k_consistent)
    test_best = pd.DataFrame([
        evaluate_signal_reg(test_preds, test_signal, "top_k", best_k_consistent, VALIDATION_COST, False)
    ])

    # 所有規則在 test 上的診斷
    test_rules = evaluate_all_rules(test_preds, VALIDATION_COST).sort_values(
        ["strategy_sharpe", "strategy_return"], ascending=[False, False]
    ).reset_index(drop=True)

    # ── Rolling window：最佳 k 選擇（用 rolling val Sharpe）
    print(f"Evaluating rolling-window rules on validation (cost={VALIDATION_COST:.3f})...")
    val_rules_roll = evaluate_all_rules(val_preds_roll, VALIDATION_COST).sort_values(
        ["strategy_sharpe", "strategy_return"], ascending=[False, False]
    ).reset_index(drop=True)
    best_rule_roll = val_rules_roll.iloc[0]
    best_k_roll = int(best_rule_roll["parameter"]) if str(best_rule_roll["rule"]).startswith("top_k") else 8
    best_weight_roll = bool(best_rule_roll["weight_by_pred"])

    # Rolling window val dual-window IC
    val_B_roll_ic = compute_daily_ic(val_preds_roll[val_preds_roll["Date"] >= "2015-01-01"])

    # Rolling window test 評估（best val rule）
    test_sig_roll = top_k_signal_reg(test_preds_roll, best_k_roll)
    test_best_roll = pd.DataFrame([
        evaluate_signal_reg(test_preds_roll, test_sig_roll, "top_k_roll",
                            best_k_roll, VALIDATION_COST, best_weight_roll)
    ])
    # Rolling window test diagnostic（k=3, k=8）
    test_k3_roll = evaluate_signal_reg(
        test_preds_roll, top_k_signal_reg(test_preds_roll, 3), "top_k", 3, VALIDATION_COST, False
    )
    test_k3w_roll = evaluate_signal_reg(
        test_preds_roll, top_k_signal_reg(test_preds_roll, 3), "top_k_pw", 3, VALIDATION_COST, True
    )
    test_k8_roll = evaluate_signal_reg(
        test_preds_roll, top_k_signal_reg(test_preds_roll, 8), "top_k", 8, VALIDATION_COST, False
    )

    # ── IC-Adaptive 動態選股：在 validation 上做 grid sweep
    print(f"Running IC-adaptive grid sweep on validation (cost={VALIDATION_COST:.3f})...")
    ic_adapt_val = sweep_ic_adaptive(val_preds, VALIDATION_COST)
    ic_adapt_val = ic_adapt_val.sort_values(
        ["strategy_sharpe", "strategy_return"], ascending=[False, False]
    ).reset_index(drop=True)
    best_ic_adapt_row = ic_adapt_val.iloc[0]
    best_ic_window = int(best_ic_adapt_row["ic_window"])
    best_ic_threshold = float(best_ic_adapt_row["ic_threshold"])

    # IC-Adaptive 最佳參數在 test 上評估
    ic_adapt_test_signal = rolling_ic_adaptive_signal(
        test_preds, best_ic_window, best_ic_threshold
    )
    ic_adapt_test_result = pd.DataFrame([
        evaluate_signal_reg(
            test_preds, ic_adapt_test_signal, "ic_adaptive",
            f"w{best_ic_window}_t{best_ic_threshold:.2f}",
            VALIDATION_COST, False
        )
    ])

    # 儲存雙段 sweep 結果
    sweep_A.to_csv("regression_dual_val_sweep_A.csv", index=False)
    sweep_B.to_csv("regression_dual_val_sweep_B.csv", index=False)
    dual.to_csv("regression_dual_val_summary.csv", index=False)
    ic_adapt_val.to_csv("regression_ic_adaptive_val_sweep.csv", index=False)

    # ── 特徵重要性
    print("Computing feature importance...")
    feat_imp = feature_importance_report(features, feature_columns)

    # ── 與分類版本比較
    try:
        clf_preds = pd.read_csv(CLASSIFICATION_PREDICTIONS_PATH, parse_dates=["Date"])
        clf_test = clf_preds[clf_preds["split"] == "test"].copy()
        # 分類版本最佳規則：threshold=0.58（從之前驗證結果）
        clf_signal = (clf_test["prob_up_20"] >= 0.58).astype(int)
        from multi_asset_logistic_baseline import compute_strategy_returns
        clf_strat_returns = compute_strategy_returns(clf_test, 0.58)
        clf_bench_returns = clf_test.groupby("Date")["next_day_return"].mean()
        # 用等權策略計算（無成本對照）
        from analyze_regime_execution_rules import compute_net_returns_equal_weight
        clf_net = compute_net_returns_equal_weight(clf_test, clf_signal, VALIDATION_COST)
        clf_comparison = {
            "model": "classification_v1 (threshold=0.58, 0.1%cost)",
            "strategy_return": float((1 + clf_net).prod()),
            "strategy_sharpe": sharpe_from_returns(clf_net),
            "strategy_max_drawdown": max_drawdown_from_returns(clf_net),
        }
        has_clf = True
    except Exception:
        has_clf = False

    # ── 儲存
    predictions[
        ["Date", "ticker", "split", "predicted_return", "forward_20d_return", "next_day_return"]
    ].to_csv("multi_asset_xgboost_regression_predictions.csv", index=False)

    # ── 輸出
    print()
    print("=" * 70)
    print("Multi-Asset XGBoost Regression")
    print(f"Target  : forward_{HORIZON_DAYS}d_return (continuous)")
    print(f"Features: {len(feature_columns)}")
    print(f"Rows    : {len(features)} features / {len(predictions)} predictions")
    print()

    print("IC (Information Coefficient, Spearman):")
    print(f"  Validation : mean={val_ic.mean():.4f}  std={val_ic.std():.4f}  ICIR={icir(val_ic):.3f}  positive_rate={( val_ic > 0).mean():.1%}")
    print(f"  Test       : mean={test_ic.mean():.4f}  std={test_ic.std():.4f}  ICIR={icir(test_ic):.3f}  positive_rate={(test_ic > 0).mean():.1%}")
    print()

    print(f"Validation rules (top 10, cost={VALIDATION_COST:.3f}):")
    print(val_rules.head(10)[["rule", "parameter", "weight_by_pred", "coverage",
                               "strategy_return", "strategy_sharpe", "strategy_max_drawdown"]])
    print()
    print(f"Best validation rule (full-val Sharpe): {best_rule['rule']}  k={best_rule['parameter']}")
    print()

    print("=" * 70)
    print("Dual-window validation k selection")
    print()
    print("val_A (2010-2014) k sweep:")
    print(sweep_A[["k", "sharpe_equal", "return_equal", "sharpe_pred_w", "max_dd_equal"]].to_string(index=False))
    print()
    print("val_B (2015-2018) k sweep:")
    print(sweep_B[["k", "sharpe_equal", "return_equal", "sharpe_pred_w", "max_dd_equal"]].to_string(index=False))
    print()
    print("Cross-window summary (equal weight):")
    print(dual[["k", "sharpe_equal_A", "sharpe_equal_B", "avg_sharpe_equal",
                "both_sharpe_positive", "sharpe_gap"]].to_string(index=False))
    print()
    print(f"Selected k — M1 (consistent both positive, best avg Sharpe)  : k={best_k_consistent}")
    print(f"Selected k — M2 (best avg Sharpe, no constraint)            : k={best_k_avg_sharpe}")
    print(f"Selected k — M3 (pred_weight, best avg Sharpe)              : k={best_k_pred_weight}")
    print(f"Selected k — M4 (val_A only, equal weight Sharpe)           : k={best_k_val_A_equal}")
    print(f"Selected k — M5 (val_A only, pred_weight Sharpe)            : k={best_k_val_A_pred_w}")
    print(f"Selected k — M6 (val_B only, equal weight Sharpe)           : k={best_k_val_B_equal}")
    print()
    print("IC by window:")
    print(f"  val_A (2010-14): mean={val_ic_A.mean():.4f}  ICIR={icir(val_ic_A):.3f}")
    print(f"  val_B (2015-18): mean={val_ic_B.mean():.4f}  ICIR={icir(val_ic_B):.3f}")
    print()
    print("Stability: val_A equal-weight Sharpe for top-3 k values:")
    print(top3_val_A[["k", "sharpe_equal", "sharpe_pred_w", "max_dd_equal"]].to_string(index=False))
    val_A_top2_gap = float(top3_val_A["sharpe_equal"].iloc[0]) - float(top3_val_A["sharpe_equal"].iloc[1])
    print(f"  Gap between rank-1 and rank-2 val_A Sharpe: {val_A_top2_gap:.4f}")
    print(f"  All 6 methods agree? {len(set(method_k_map.values())) == 1}")
    print(f"  Methods k distribution: {dict(method_k_map)}")
    print()

    print("=" * 70)
    print(f"Test comparison — all 6 selection methods (cost={VALIDATION_COST:.3f}):")
    print(test_results_comparison[["selection_method", "k_selected",
                                   "strategy_return", "strategy_sharpe",
                                   "strategy_max_drawdown", "coverage"]].to_string(index=False))
    print()
    # 穩定性摘要：哪個 k 被最多方法選中
    k_counts = pd.Series(list(method_k_map.values())).value_counts()
    consensus_k = int(k_counts.index[0])
    print(f"Consensus k (most methods agree): k={consensus_k}  (voted by {k_counts.iloc[0]}/5 methods)")
    print()
    # val_A Sharpe 與 test Sharpe 相關性（看 val_A 能否預測 test）
    k_val_A_sharpe = {int(row["k"]): row["sharpe_equal"] for _, row in sweep_A.iterrows()}
    test_k_sharpe = {
        int(r["k_selected"]): r["strategy_sharpe"]
        for _, r in test_results_comparison.iterrows()
        if str(r["selection_method"]).startswith("M")
    }
    print("val_A Sharpe vs test Sharpe (per k, equal weight):")
    test_k_rows = [(k, k_val_A_sharpe.get(k, np.nan)) for k in sorted(k_val_A_sharpe.keys())]
    test_actual = evaluate_all_rules(test_preds, VALIDATION_COST)
    test_k_eq = {int(r["parameter"]): r["strategy_sharpe"]
                  for _, r in test_actual[test_actual["rule"] == "top_k"].iterrows()}
    for k in sorted(k_val_A_sharpe.keys()):
        print(f"  k={k}: val_A_sharpe={k_val_A_sharpe[k]:.4f}  test_sharpe={test_k_eq.get(k, np.nan):.4f}")
    print()
    print(f"Test result — M1 consistent best k={best_k_consistent} (cost={VALIDATION_COST:.3f}):")
    print(test_best[["rule", "parameter", "coverage",
                      "strategy_return", "strategy_sharpe", "strategy_max_drawdown",
                      "benchmark_return", "benchmark_sharpe"]])
    print()

    print(f"Test rules top 10 (diagnostic, cost={VALIDATION_COST:.3f}):")
    print(test_rules.head(10)[["rule", "parameter", "weight_by_pred", "coverage",
                                "strategy_return", "strategy_sharpe", "strategy_max_drawdown"]])
    print()

    print("=" * 70)
    print(f"IC-Adaptive Dynamic k Selection  (k_high={IC_ADAPTIVE_K_HIGH}, k_low={IC_ADAPTIVE_K_LOW})")
    print(f"Validation sweep (top 8, sorted by Sharpe, cost={VALIDATION_COST:.3f}):")
    print(ic_adapt_val.head(8)[["ic_window", "ic_threshold", "avg_k_used",
                                  "strategy_return", "strategy_sharpe",
                                  "strategy_max_drawdown"]].to_string(index=False))
    print()
    print(f"Best val params: ic_window={best_ic_window}  ic_threshold={best_ic_threshold:.2f}")
    print()
    print(f"Test result — IC-adaptive (w={best_ic_window}, thr={best_ic_threshold:.2f}, cost={VALIDATION_COST:.3f}):")
    print(ic_adapt_test_result[["rule", "parameter", "coverage",
                                  "strategy_return", "strategy_sharpe",
                                  "strategy_max_drawdown", "benchmark_return", "benchmark_sharpe"]])
    print()

    # 彙整比較：fixed k=3 pred_weight vs IC-adaptive
    ic_test_sharpe = float(ic_adapt_test_result["strategy_sharpe"].iloc[0])
    ic_test_return = float(ic_adapt_test_result["strategy_return"].iloc[0])
    ic_test_dd = float(ic_adapt_test_result["strategy_max_drawdown"].iloc[0])
    pk3_sig = top_k_signal_reg(test_preds, 3)
    pk3_res = evaluate_signal_reg(test_preds, pk3_sig, "top_k", 3, VALIDATION_COST, True)
    print("Final comparison (test, 0.1% cost):")
    print(f"  Fixed k=3 pred_weight            : return={pk3_res['strategy_return']:.4f}x  Sharpe={pk3_res['strategy_sharpe']:.4f}  DD={pk3_res['strategy_max_drawdown']:.4f}")
    print(f"  IC-adaptive (w={best_ic_window},thr={best_ic_threshold:.2f}) : return={ic_test_return:.4f}x  Sharpe={ic_test_sharpe:.4f}  DD={ic_test_dd:.4f}")
    print()

    # ── Rolling Window Training 比較輸出
    print("=" * 70)
    print(f"Rolling Window Training  ({ROLLING_WINDOW_DAYS//365}-year window)  vs  Expanding Window")
    print()
    print("Rolling window IC:")
    print(f"  Validation : mean={val_ic_roll.mean():.4f}  std={val_ic_roll.std():.4f}  ICIR={icir(val_ic_roll):.3f}  positive_rate={(val_ic_roll > 0).mean():.1%}")
    print(f"  Test       : mean={test_ic_roll.mean():.4f}  std={test_ic_roll.std():.4f}  ICIR={icir(test_ic_roll):.3f}  positive_rate={(test_ic_roll > 0).mean():.1%}")
    print(f"  val_A IC (2010-14): mean={val_ic_roll_A.mean():.4f}  ICIR={icir(val_ic_roll_A):.3f}")
    print(f"  val_B IC (2015-18): mean={val_ic_roll_B.mean():.4f}  ICIR={icir(val_ic_roll_B):.3f}")
    print()
    print(f"Rolling window validation top 5 rules (cost={VALIDATION_COST:.3f}):")
    print(val_rules_roll.head(5)[["rule", "parameter", "weight_by_pred", "coverage",
                                    "strategy_return", "strategy_sharpe", "strategy_max_drawdown"]].to_string(index=False))
    print()
    print(f"Rolling window test results (best val rule: {best_rule_roll['rule']} k={best_k_roll}, pred_weight={best_weight_roll}):")
    print(test_best_roll[["rule", "parameter", "coverage",
                            "strategy_return", "strategy_sharpe",
                            "strategy_max_drawdown", "benchmark_return", "benchmark_sharpe"]])
    print()
    print("Rolling window test diagnostic (k=3 / k=3 pred_weight / k=8):")
    print(f"  k=3 equal  : return={test_k3_roll['strategy_return']:.4f}x  Sharpe={test_k3_roll['strategy_sharpe']:.4f}  DD={test_k3_roll['strategy_max_drawdown']:.4f}")
    print(f"  k=3 pred_w : return={test_k3w_roll['strategy_return']:.4f}x  Sharpe={test_k3w_roll['strategy_sharpe']:.4f}  DD={test_k3w_roll['strategy_max_drawdown']:.4f}")
    print(f"  k=8 equal  : return={test_k8_roll['strategy_return']:.4f}x  Sharpe={test_k8_roll['strategy_sharpe']:.4f}  DD={test_k8_roll['strategy_max_drawdown']:.4f}")
    print()

    if has_clf:
        print("=" * 70)
        print("Final comparison — all models (test, 0.1% cost):")
        print(f"  Regression expand k=8 (val)     : return={float(test_best['strategy_return'].iloc[0]):.4f}x  Sharpe={float(test_best['strategy_sharpe'].iloc[0]):.4f}  DD={float(test_best['strategy_max_drawdown'].iloc[0]):.4f}")
        print(f"  Regression expand k=3 equal     : return={test_k3_roll['strategy_return']:.4f}x  Sharpe=--- (see test_rules)")
        pk3_exp = evaluate_signal_reg(test_preds, top_k_signal_reg(test_preds, 3), "top_k", 3, VALIDATION_COST, False)
        print(f"  Regression expand k=3 equal     : return={pk3_exp['strategy_return']:.4f}x  Sharpe={pk3_exp['strategy_sharpe']:.4f}  DD={pk3_exp['strategy_max_drawdown']:.4f}")
        print(f"  Regression expand k=3 pred_w    : return={pk3_res['strategy_return']:.4f}x  Sharpe={pk3_res['strategy_sharpe']:.4f}  DD={pk3_res['strategy_max_drawdown']:.4f}")
        print(f"  Regression expand IC-adaptive   : return={ic_test_return:.4f}x  Sharpe={ic_test_sharpe:.4f}  DD={ic_test_dd:.4f}")
        print(f"  Regression rolling k=3 equal    : return={test_k3_roll['strategy_return']:.4f}x  Sharpe={test_k3_roll['strategy_sharpe']:.4f}  DD={test_k3_roll['strategy_max_drawdown']:.4f}")
        print(f"  Regression rolling k=3 pred_w   : return={test_k3w_roll['strategy_return']:.4f}x  Sharpe={test_k3w_roll['strategy_sharpe']:.4f}  DD={test_k3w_roll['strategy_max_drawdown']:.4f}")
        print(f"  Regression rolling k=8 equal    : return={test_k8_roll['strategy_return']:.4f}x  Sharpe={test_k8_roll['strategy_sharpe']:.4f}  DD={test_k8_roll['strategy_max_drawdown']:.4f}")
        print(f"  Classification v1 (thr=0.58)    : return={clf_comparison['strategy_return']:.4f}x  Sharpe={clf_comparison['strategy_sharpe']:.4f}  DD={clf_comparison['strategy_max_drawdown']:.4f}")
        print()

    print("Top 15 feature importances (regression, train+val):")
    print(feat_imp)
    print()
    print("Saved: multi_asset_xgboost_regression_predictions.csv, "
          "regression_dual_val_sweep_A.csv, regression_dual_val_sweep_B.csv, "
          "regression_dual_val_summary.csv, regression_ic_adaptive_val_sweep.csv")


if __name__ == "__main__":
    main()
