# -*- coding: utf-8 -*-
"""
daily_signal.py
───────────────
每日選股訊號輸出：今日應持有哪幾檔股票。

策略：IC-Adaptive (w=90, thr=0.08)
  - rolling_mean_IC (過去 90 天，滯後 20 天) > 0.08 → 選 top-3 (集中)
  - rolling_mean_IC ≤ 0.08                            → 選 top-8 (分散，等同 benchmark)

流程：
  1. 下載最新市場資料（到今日）
  2. 建立特徵表（與訓練時完全相同的 44 個特徵）
  3. 以全部可用歷史資料訓練 XGBRegressor（expanding window）
  4. 預測今日每檔股票的未來 20 日報酬（cross-sectional ranking 用）
  5. 從歷史預測 CSV 計算 rolling IC → 決定今日用 k=3 還是 k=8
  6. 輸出今日持倉建議

執行：
  python daily_signal.py
"""
from __future__ import annotations

import sys
import io
import warnings
from datetime import date, timedelta
from pathlib import Path

# Windows 終端機 UTF-8 輸出
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from xgboost import XGBRegressor

from multi_asset_logistic_baseline import (
    TICKERS,
    HORIZON_DAYS,
    TRAIN_START,
    build_feature_frame,   # keep_all_rows=True 供預測用
    download_market_data,
    TRADING_DAYS_PER_YEAR,
)
from multi_asset_xgboost_regime_baseline import (
    download_macro_data,
    build_regime_features,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ── 策略參數（與 validation 選出的最佳參數一致）──────────────────────────────
IC_WINDOW      = 90     # 滾動 IC 窗口（天）
IC_THRESHOLD   = 0.08   # IC 閾值：> 此值 → 集中，否則分散
K_HIGH         = 3      # 高 IC 時選幾檔
K_LOW          = 8      # 低 IC 時選幾檔

# 歷史預測 CSV 路徑（walk-forward 回測輸出，用於計算 rolling IC）
PREDICTIONS_CSV = "multi_asset_xgboost_regression_predictions.csv"

# 特徵欄位（與訓練時完全一致）
BASE_NUMERIC_FEATURES = [
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


def build_regressor() -> XGBRegressor:
    """與回測訓練時完全相同的超參數。"""
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


def compute_rolling_ic(predictions_csv: str, ic_window: int, horizon: int) -> tuple[float, pd.Series]:
    """
    從歷史預測 CSV 計算 rolling IC，回傳：
      - latest_ic: 最新一天的 rolling_mean_IC（用於今日 k 決策）
      - rolling_mean_ic: 完整時間序列（供展示）

    IC 計算邏輯（與回測一致，無 lookahead）：
      daily_IC[T] = Spearman(predicted_return[T], forward_20d_return[T])
      rolling_mean_IC[T] = mean(daily_IC[T-horizon : T-horizon+window])
      決策時使用 rolling_mean_IC[T-horizon]（滯後 horizon 天）
    """
    if not Path(predictions_csv).exists():
        print(f"  [警告] 找不到 {predictions_csv}，無法計算 rolling IC，使用 k={K_LOW} (分散)")
        return np.nan, pd.Series(dtype=float)

    df = pd.read_csv(predictions_csv, parse_dates=["Date"])

    # 只保留有 forward_20d_return 的行（最後 HORIZON_DAYS 行為 NaN，排除）
    df = df.dropna(subset=["predicted_return", "forward_20d_return"])

    # 每日橫截面 Spearman IC
    daily_ic_rows = []
    for date_val, group in df.groupby("Date"):
        if len(group) < 3:
            continue
        ic, _ = spearmanr(group["predicted_return"], group["forward_20d_return"])
        daily_ic_rows.append({"Date": date_val, "IC": ic})

    if not daily_ic_rows:
        return np.nan, pd.Series(dtype=float)

    daily_ic = pd.DataFrame(daily_ic_rows).set_index("Date")["IC"].sort_index()

    # 滯後 horizon 天（避免 lookahead）再做滾動均值
    lagged_ic = daily_ic.shift(horizon)
    rolling_mean_ic = lagged_ic.rolling(window=ic_window, min_periods=ic_window // 2).mean()

    latest_ic = float(rolling_mean_ic.dropna().iloc[-1]) if not rolling_mean_ic.dropna().empty else np.nan
    return latest_ic, rolling_mean_ic


def get_feature_columns(features: pd.DataFrame) -> list[str]:
    """回傳與回測一致的特徵欄位列表。"""
    ticker_dummies = sorted([c for c in features.columns if c.startswith("ticker_")])
    return BASE_NUMERIC_FEATURES + ticker_dummies


def patch_with_recent_data(
    data_by_ticker: dict,
    end_date: str,
    lookback_days: int = 90,
) -> dict:
    """
    補丁：下載最近 lookback_days 個日曆天的新資料，合併進 data_by_ticker。
    解決 yfinance 對大範圍下載返回舊快取的問題。
    """
    import yfinance as yf

    recent_start = (pd.Timestamp(end_date) - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    tickers = list(data_by_ticker.keys())

    try:
        raw = yf.download(
            tickers=tickers,
            start=recent_start,
            end=end_date,
            auto_adjust=False,
            progress=False,
            group_by="ticker",
        )
    except Exception as e:
        print(f"  [警告] 補丁下載失敗：{e}，使用原始資料")
        return data_by_ticker

    patched = {}
    for ticker in tickers:
        try:
            if ticker in raw.columns.get_level_values(0):
                new_df = raw[ticker].copy()
            else:
                new_df = raw.copy()
            new_df = new_df.rename_axis("Date").reset_index()
            new_df["Date"] = pd.to_datetime(new_df["Date"])
            new_df["ticker"] = ticker
            new_df = new_df.dropna(subset=["Close"])

            old_df = data_by_ticker[ticker]
            # 合併：取新資料（優先）+ 舊資料中不重複的日期
            combined = pd.concat(
                [old_df[~old_df["Date"].isin(new_df["Date"])], new_df],
                ignore_index=True,
            ).sort_values("Date").reset_index(drop=True)
            patched[ticker] = combined
        except Exception:
            patched[ticker] = data_by_ticker[ticker]

    return patched


def build_and_predict_today(
    features_all: pd.DataFrame,
    feature_columns: list[str],
    today: pd.Timestamp,
) -> pd.DataFrame:
    """
    用全部有標籤的歷史資料訓練模型，預測最新交易日每檔股票的 predicted_return。

    features_all：包含全部日期（含 forward_20d_return 為 NaN 的最新日期）的特徵表。
    訓練只用 forward_20d_return 不為 NaN 的行（有已知標籤）。
    預測使用最新交易日的行（forward_20d_return 可以為 NaN）。
    """
    # 訓練集：有已知標籤的行
    train_mask = (
        (features_all["Date"] >= pd.Timestamp(TRAIN_START))
        & features_all["forward_20d_return"].notna()
    )
    train_frame = features_all.loc[train_mask].copy()

    if train_frame.empty:
        raise ValueError("訓練資料為空，無法建立模型。")

    print(f"  訓練樣本數：{len(train_frame):,}  (截至 {train_frame['Date'].max().date()})")

    model = build_regressor()
    model.fit(train_frame[feature_columns], train_frame["forward_20d_return"].astype(float))

    # 預測行：最新交易日（不需要 forward_20d_return）
    latest_date = features_all["Date"].max()
    today_mask = features_all["Date"] == latest_date
    today_frame = features_all.loc[today_mask].copy()

    if today_frame.empty:
        raise ValueError(f"找不到最新交易日 {latest_date.date()} 的特徵行。")

    # 確認特徵無 NaN
    missing_feat = [c for c in feature_columns if today_frame[c].isna().any()]
    if missing_feat:
        print(f"  [警告] 今日特徵有 NaN：{missing_feat}（共 {len(missing_feat)} 個），用 0 填補")
        today_frame[missing_feat] = today_frame[missing_feat].fillna(0)

    today_frame = today_frame.assign(
        predicted_return=model.predict(today_frame[feature_columns])
    )
    return today_frame[["Date", "ticker", "predicted_return"]].copy()


def print_signal(predictions_today: pd.DataFrame, rolling_ic: float, k: int) -> None:
    """格式化輸出今日持倉建議。"""
    pred = predictions_today.sort_values("predicted_return", ascending=False).reset_index(drop=True)
    signal_date = pred["Date"].iloc[0].date()

    print()
    print("=" * 60)
    print(f"  每日選股訊號 — {signal_date}")
    print("=" * 60)
    print()

    # IC 狀態
    if np.isnan(rolling_ic):
        ic_status = "無資料（使用預設 k_low）"
        ic_flag = "⚠"
    elif rolling_ic > IC_THRESHOLD:
        ic_status = f"高信心模式（IC={rolling_ic:.4f} > {IC_THRESHOLD}）"
        ic_flag = "HIGH"
    else:
        ic_status = f"低信心模式（IC={rolling_ic:.4f} <= {IC_THRESHOLD}）"
        ic_flag = "LOW"

    print(f"  模型信心  [{ic_flag}]  {ic_status}")
    print(f"  今日 k    →  {k} 檔")
    print()

    # 持倉建議
    selected = pred.head(k)
    others   = pred.iloc[k:]

    print(f"  今日持倉建議（k={k}，等權重各 {100/k:.1f}%）：")
    for i, row in selected.iterrows():
        bar = "#" * int(max(0, row["predicted_return"]) * 300)
        print(f"    #{i+1}  {row['ticker']:<6}  預測 20 日報酬 = {row['predicted_return']:+.2%}  {bar}")

    if not others.empty:
        print()
        print(f"  觀望（不在 top-{k}）：")
        other_str = "  ".join(
            f"{r['ticker']}({r['predicted_return']:+.2%})"
            for _, r in others.iterrows()
        )
        print(f"    {other_str}")

    print()
    print("  注意：predicted_return 為模型預測的未來 20 交易日報酬，非今日漲跌。")
    print("        建議每個交易日收盤後重跑一次，取得最新訊號。")
    print("=" * 60)
    print()


def main() -> None:
    today = pd.Timestamp(date.today())
    # 用稍晚一天的日期下載（確保抓到最新收盤價）
    end_date = (date.today() + timedelta(days=1)).strftime("%Y-%m-%d")

    print(f"Daily Signal Generator  [{today.date()}]")
    print()

    # ── Step 1：計算 Rolling IC（從歷史預測 CSV）
    print(f"Step 1 / 4  計算 rolling IC（從 {PREDICTIONS_CSV}）...")
    rolling_ic, rolling_ic_series = compute_rolling_ic(PREDICTIONS_CSV, IC_WINDOW, HORIZON_DAYS)

    if not np.isnan(rolling_ic):
        print(f"  最新 rolling_mean_IC (w={IC_WINDOW}, lag={HORIZON_DAYS}d) = {rolling_ic:.4f}")
    else:
        print(f"  無法取得 rolling IC")

    # ── Step 2：決定今日 k
    k = K_HIGH if (not np.isnan(rolling_ic) and rolling_ic > IC_THRESHOLD) else K_LOW
    print(f"  → 今日 k = {k}  ({'集中' if k == K_HIGH else '分散'})")
    print()

    # ── Step 3：下載最新資料並建立特徵表
    print(f"Step 2 / 4  下載市場資料（{TRAIN_START} ~ {end_date}）...")
    data_by_ticker = download_market_data(TICKERS, TRAIN_START, end_date)
    # 補丁：yfinance 大範圍下載常返回舊快取，用短窗口下載補足最新 90 天
    print(f"  補丁：更新最近 90 天資料...")
    data_by_ticker = patch_with_recent_data(data_by_ticker, end_date, lookback_days=90)
    base_features = build_feature_frame(data_by_ticker)
    latest_date = base_features["Date"].max().date()
    print(f"  下載完成，最新交易日：{latest_date}")
    print()

    print("Step 3 / 4  建立特徵表（加入宏觀特徵）...")
    macro_data = download_macro_data(TRAIN_START, end_date)
    # keep_all_rows=True：保留最新日期（forward_20d_return=NaN）的行，供預測使用
    base_features_all = build_feature_frame(data_by_ticker, keep_all_rows=True)
    features_all = build_regime_features(base_features_all, data_by_ticker, macro_data, keep_all_rows=True)

    feature_columns = get_feature_columns(features_all)
    latest_date = features_all["Date"].max().date()
    n_train = features_all["forward_20d_return"].notna().sum()
    print(f"  特徵數：{len(feature_columns)}（{len(BASE_NUMERIC_FEATURES)} 技術/宏觀 + {len(feature_columns)-len(BASE_NUMERIC_FEATURES)} ticker dummies）")
    print(f"  最新交易日：{latest_date}  訓練用行數：{n_train:,}")
    print()

    # ── VIX 熔斷機制：VIX > 30 強制分散（覆蓋 Step 2 的 IC-Adaptive 決定）
    VIX_CIRCUIT_BREAKER = 30.0
    latest_vix_series = macro_data.sort_values("Date")["vix_close"].dropna()
    if not latest_vix_series.empty:
        latest_vix = float(latest_vix_series.iloc[-1])
        if latest_vix > VIX_CIRCUIT_BREAKER:
            k = K_LOW
            print(f"  [VIX熔斷] VIX={latest_vix:.1f} > {VIX_CIRCUIT_BREAKER:.0f}，強制 k={K_LOW}（分散，覆蓋 IC-Adaptive）")
        else:
            print(f"  [VIX正常] VIX={latest_vix:.1f} <= {VIX_CIRCUIT_BREAKER:.0f}，維持 IC-Adaptive k={k}")
    else:
        print(f"  [VIX] 無法取得 VIX 資料，維持 k={k}")
    print()

    # ── Step 4：訓練模型並預測今日
    print("Step 4 / 4  訓練模型並預測今日...")
    predictions_today = build_and_predict_today(features_all, feature_columns, today)
    print()

    # ── 輸出結果
    print_signal(predictions_today, rolling_ic, k)

    # ── 補充：recent IC trend（最近 10 天）
    if not rolling_ic_series.dropna().empty:
        recent = rolling_ic_series.dropna().tail(10)
        print("  近期 Rolling IC 趨勢（最近 10 筆）：")
        for d, v in recent.items():
            bar = "#" * int(max(0, v) * 100)
            flag = " ← 集中" if v > IC_THRESHOLD else ""
            print(f"    {str(d)[:10]}  {v:+.4f}  {bar}{flag}")
        print()


if __name__ == "__main__":
    main()
