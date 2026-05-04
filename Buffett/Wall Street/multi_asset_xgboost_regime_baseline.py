# -*- coding: utf-8 -*-
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from xgboost import XGBClassifier
import yfinance as yf

from multi_asset_logistic_baseline import (
    TICKERS,
    HORIZON_DAYS,
    TRAIN_START,
    TRAIN_END,
    VALIDATION_START,
    VALIDATION_END,
    TEST_START,
    TEST_END,
    THRESHOLDS,
    build_feature_frame,
    build_monthly_periods,
    compute_benchmark_returns,
    compute_rsi,
    compute_strategy_returns,
    download_market_data,
    max_drawdown_from_returns,
    sharpe_from_returns,
    summarize_by_ticker,
)

pd.set_option("display.notebook_repr_html", False)
pd.set_option("display.max_columns", 24)
pd.set_option("display.max_rows", 30)
pd.set_option("display.width", 160)

warnings.filterwarnings("ignore", category=FutureWarning, module=r"yfinance(\..*)?$")
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*ChainedAssignmentError: behaviour will change in pandas 3\.0!.*",
)

MACRO_TICKERS = ["^VIX", "^TNX"]


def build_model() -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
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


def download_macro_data(start: str, end: str) -> pd.DataFrame:
    raw = yf.download(
        tickers=MACRO_TICKERS,
        start=pd.Timestamp(start),
        end=pd.Timestamp(end) + pd.Timedelta(days=1),
        auto_adjust=False,
        progress=False,
        group_by="ticker",
    )

    frames = []

    for ticker in MACRO_TICKERS:
        frame = raw[ticker].copy().rename_axis("Date").reset_index()
        frame["Date"] = pd.to_datetime(frame["Date"])
        price = frame["Close"].astype(float)

        if ticker == "^VIX":
            enriched = pd.DataFrame(
                {
                    "Date": frame["Date"],
                    "vix_close": price,
                    "vix_ret_5": price.pct_change(5),
                    "vix_vs_sma20": (price / price.rolling(20).mean()) - 1.0,
                    "vix_rsi_14": compute_rsi(price, 14),
                }
            )
        else:
            enriched = pd.DataFrame(
                {
                    "Date": frame["Date"],
                    "tnx_close": price,
                    "tnx_change_5": price.diff(5),
                    "tnx_vs_sma20": (price / price.rolling(20).mean()) - 1.0,
                }
            )

        frames.append(enriched)

    macro = frames[0]
    for frame in frames[1:]:
        macro = macro.merge(frame, on="Date", how="outer")

    return macro.sort_values("Date").reset_index(drop=True)


def build_regime_features(
    base_features: pd.DataFrame,
    data_by_ticker: dict[str, pd.DataFrame],
    macro_data: pd.DataFrame,
    keep_all_rows: bool = False,
) -> pd.DataFrame:
    features = base_features.copy()

    spy = data_by_ticker["SPY"].sort_values("Date").copy()
    qqq = data_by_ticker["QQQ"].sort_values("Date").copy()

    spy_price = spy["Adj Close"].astype(float)
    qqq_price = qqq["Adj Close"].astype(float)

    regime = pd.DataFrame(
        {
            "Date": spy["Date"],
            "spy_ret_20": spy_price.pct_change(20),
            "spy_ret_60": spy_price.pct_change(60),
            "spy_vs_sma200": (spy_price / spy_price.rolling(200).mean()) - 1.0,
            "spy_vol_20": spy_price.pct_change().rolling(20).std(),
            "qqq_ret_20": qqq_price.pct_change(20),
            "qqq_vs_sma200": (qqq_price / qqq_price.rolling(200).mean()) - 1.0,
            "qqq_vol_20": qqq_price.pct_change().rolling(20).std(),
        }
    )

    features = features.merge(regime, on="Date", how="left")
    features = features.merge(macro_data, on="Date", how="left")

    features["rel_ret20_vs_spy"] = features["ret_20"] - features["spy_ret_20"]
    features["rel_ret60_vs_spy"] = features["ret_60"] - features["spy_ret_60"]
    features["breadth_above_sma20"] = features.groupby("Date")["price_vs_sma20"].transform(lambda x: (x > 0).mean())
    features["breadth_above_sma200"] = features.groupby("Date")["price_vs_sma200"].transform(lambda x: (x > 0).mean())
    features["rank_ret20_pct"] = features.groupby("Date")["ret_20"].rank(pct=True)
    features["rank_ret60_pct"] = features.groupby("Date")["ret_60"].rank(pct=True)
    features["rank_vol20_pct"] = features.groupby("Date")["vol_20"].rank(pct=True)
    features["rank_drawdown20_pct"] = features.groupby("Date")["drawdown_20"].rank(pct=True)

    if keep_all_rows:
        # 只 drop 技術/宏觀特徵為 NaN 的行，保留 forward_20d_return=NaN 的最新日期（供預測用）
        regime_cols = ["spy_ret_20", "spy_ret_60", "spy_vs_sma200", "spy_vol_20",
                       "qqq_ret_20", "qqq_vs_sma200", "qqq_vol_20",
                       "vix_close", "tnx_close"]
        return features.dropna(subset=regime_cols).reset_index(drop=True)
    return features.dropna().reset_index(drop=True)


def run_walk_forward_predictions(
    features: pd.DataFrame,
    feature_columns: list[str],
    periods: list,
) -> pd.DataFrame:
    all_predictions = []

    for period in periods:
        train_mask = (
            (features["Date"] >= pd.Timestamp(TRAIN_START))
            & (features["Date"] < period.start)
            & (features["label_available_date"] < period.start)
        )
        eval_mask = (features["Date"] >= period.start) & (features["Date"] <= period.end)

        train_frame = features.loc[train_mask].copy()
        eval_frame = features.loc[eval_mask].copy()

        if train_frame.empty or eval_frame.empty:
            continue

        if train_frame["target_up_20"].nunique() < 2:
            continue

        model = build_model()
        model.fit(train_frame[feature_columns], train_frame["target_up_20"].astype(int))
        probabilities = model.predict_proba(eval_frame[feature_columns])[:, 1]

        eval_frame = eval_frame.assign(
            split=period.label,
            prob_up_20=probabilities,
        )
        all_predictions.append(eval_frame)

    if not all_predictions:
        raise ValueError("No walk-forward predictions were produced.")

    return pd.concat(all_predictions, ignore_index=True)


def evaluate_threshold(predictions: pd.DataFrame, threshold: float) -> dict[str, float]:
    predicted_up = (predictions["prob_up_20"] >= threshold).astype(int)
    actual = predictions["target_up_20"].astype(int)

    strategy_returns = compute_strategy_returns(predictions, threshold)
    benchmark_returns = compute_benchmark_returns(predictions).reindex(strategy_returns.index)
    selected = predictions[predicted_up == 1]

    return {
        "threshold": threshold,
        "rows": int(len(predictions)),
        "coverage": float(predicted_up.mean()),
        "accuracy": float(accuracy_score(actual, predicted_up)),
        "precision": float(precision_score(actual, predicted_up, zero_division=0)),
        "recall": float(recall_score(actual, predicted_up, zero_division=0)),
        "roc_auc": float(roc_auc_score(actual, predictions["prob_up_20"])),
        "selected_avg_forward_20d_return": float(selected["forward_20d_return"].mean()) if not selected.empty else np.nan,
        "selected_hit_rate": float((selected["target_up_20"] == 1).mean()) if not selected.empty else np.nan,
        "strategy_return": float((1.0 + strategy_returns).prod()),
        "strategy_sharpe": sharpe_from_returns(strategy_returns),
        "strategy_max_drawdown": max_drawdown_from_returns(strategy_returns),
        "benchmark_return": float((1.0 + benchmark_returns).prod()),
        "benchmark_sharpe": sharpe_from_returns(benchmark_returns),
        "benchmark_max_drawdown": max_drawdown_from_returns(benchmark_returns),
    }


def main() -> None:
    data_by_ticker = download_market_data(TICKERS, TRAIN_START, TEST_END)
    base_features = build_feature_frame(data_by_ticker)
    macro_data = download_macro_data(TRAIN_START, TEST_END)
    features = build_regime_features(base_features, data_by_ticker, macro_data)

    numeric_features = [
        "ret_5",
        "ret_20",
        "ret_60",
        "vol_20",
        "vol_60",
        "price_vs_sma20",
        "price_vs_sma60",
        "price_vs_sma200",
        "rsi_14",
        "macd_hist",
        "volume_ratio20",
        "drawdown_20",
        "drawdown_60",
        "intraday_range",
        "spy_ret_20",
        "spy_ret_60",
        "spy_vs_sma200",
        "spy_vol_20",
        "qqq_ret_20",
        "qqq_vs_sma200",
        "qqq_vol_20",
        "vix_close",
        "vix_ret_5",
        "vix_vs_sma20",
        "vix_rsi_14",
        "tnx_close",
        "tnx_change_5",
        "tnx_vs_sma20",
        "rel_ret20_vs_spy",
        "rel_ret60_vs_spy",
        "breadth_above_sma20",
        "breadth_above_sma200",
        "rank_ret20_pct",
        "rank_ret60_pct",
        "rank_vol20_pct",
        "rank_drawdown20_pct",
    ]
    dummy_features = sorted([column for column in features.columns if column.startswith("ticker_")])
    feature_columns = numeric_features + dummy_features

    periods = (
        build_monthly_periods(VALIDATION_START, VALIDATION_END, "validation")
        + build_monthly_periods(TEST_START, TEST_END, "test")
    )

    predictions = run_walk_forward_predictions(features, feature_columns, periods)
    validation_predictions = predictions[predictions["split"] == "validation"].copy()
    test_predictions = predictions[predictions["split"] == "test"].copy()

    validation_thresholds = pd.DataFrame(
        [evaluate_threshold(validation_predictions, threshold) for threshold in THRESHOLDS]
    )
    best_threshold_row = validation_thresholds.sort_values(
        ["strategy_sharpe", "strategy_return", "accuracy"],
        ascending=[False, False, False],
    ).iloc[0]
    best_threshold = float(best_threshold_row["threshold"])

    test_metrics = pd.DataFrame([evaluate_threshold(test_predictions, best_threshold)])
    test_by_ticker = summarize_by_ticker(test_predictions, best_threshold)

    feature_gain_model = build_model()
    train_mask = (
        (features["Date"] >= pd.Timestamp(TRAIN_START))
        & (features["Date"] <= pd.Timestamp(VALIDATION_END))
        & (features["label_available_date"] <= pd.Timestamp(VALIDATION_END))
    )
    feature_gain_model.fit(
        features.loc[train_mask, feature_columns],
        features.loc[train_mask, "target_up_20"].astype(int),
    )
    feature_importance = (
        pd.DataFrame(
            {
                "feature": feature_columns,
                "importance": feature_gain_model.feature_importances_,
            }
        )
        .sort_values("importance", ascending=False)
        .head(15)
        .reset_index(drop=True)
    )

    predictions[
        [
            "Date",
            "ticker",
            "split",
            "prob_up_20",
            "target_up_20",
            "forward_20d_return",
            "next_day_return",
        ]
    ].to_csv("multi_asset_xgboost_regime_predictions.csv", index=False)

    print("Multi-Asset XGBoost Regime Baseline")
    print(f"Tickers: {', '.join(TICKERS)}")
    print(f"Horizon: {HORIZON_DAYS} trading days")
    print(f"Train:      {TRAIN_START} to {TRAIN_END}")
    print(f"Validation: {VALIDATION_START} to {VALIDATION_END}")
    print(f"Test:       {TEST_START} to {TEST_END}")
    print(f"Feature rows: {len(features)}")
    print(f"Prediction rows: {len(predictions)}")
    print(f"Feature count: {len(feature_columns)}")
    print()
    print("Validation threshold search")
    print(validation_thresholds)
    print()
    print(f"Chosen threshold: {best_threshold:.2f}")
    print()
    print("Test metrics")
    print(test_metrics)
    print()
    print("Test metrics by ticker")
    print(test_by_ticker)
    print()
    print("Top feature importances (train through validation end)")
    print(feature_importance)
    print()
    print("Predictions saved to multi_asset_xgboost_regime_predictions.csv")


if __name__ == "__main__":
    main()
