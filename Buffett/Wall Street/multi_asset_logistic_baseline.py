# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import yfinance as yf

pd.set_option("display.notebook_repr_html", False)
pd.set_option("display.max_columns", 20)
pd.set_option("display.max_rows", 30)
pd.set_option("display.width", 140)

warnings.filterwarnings("ignore", category=FutureWarning, module=r"yfinance(\..*)?$")
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*ChainedAssignmentError: behaviour will change in pandas 3\.0!.*",
)

TICKERS = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMZN", "META", "TSLA"]
HORIZON_DAYS = 20
TRAIN_START = "2000-01-01"
TRAIN_END = "2009-12-31"
VALIDATION_START = "2010-01-01"
VALIDATION_END = "2018-12-31"
TEST_START = "2019-01-01"
TEST_END = "2026-04-10"
THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70]
TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class Period:
    label: str
    start: pd.Timestamp
    end: pd.Timestamp


def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def max_drawdown_from_returns(returns: pd.Series) -> float:
    if returns.empty:
        return np.nan

    equity = (1.0 + returns).cumprod()
    running_peak = equity.cummax()
    drawdown = (equity / running_peak) - 1.0
    return float(-drawdown.min())


def sharpe_from_returns(returns: pd.Series) -> float:
    if returns.empty or returns.std(ddof=1) == 0 or np.isnan(returns.std(ddof=1)):
        return np.nan

    return float(np.sqrt(TRADING_DAYS_PER_YEAR) * returns.mean() / returns.std(ddof=1))


def download_market_data(tickers: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    raw = yf.download(
        tickers=tickers,
        start=pd.Timestamp(start),
        end=pd.Timestamp(end) + pd.Timedelta(days=1),
        auto_adjust=False,
        progress=False,
        group_by="ticker",
    )

    data_by_ticker: dict[str, pd.DataFrame] = {}

    for ticker in tickers:
        ticker_frame = raw[ticker].copy()
        ticker_frame = ticker_frame.rename_axis("Date").reset_index()
        ticker_frame["Date"] = pd.to_datetime(ticker_frame["Date"])
        ticker_frame["ticker"] = ticker
        data_by_ticker[ticker] = ticker_frame

    return data_by_ticker


def build_feature_frame(data_by_ticker: dict[str, pd.DataFrame], keep_all_rows: bool = False) -> pd.DataFrame:
    frames = []

    for ticker, frame in data_by_ticker.items():
        local = frame.sort_values("Date").copy()
        price = local["Adj Close"].astype(float)
        volume = local["Volume"].astype(float)
        daily_return = price.pct_change()

        sma20 = price.rolling(20).mean()
        sma60 = price.rolling(60).mean()
        sma200 = price.rolling(200).mean()
        ema12 = price.ewm(span=12, adjust=False).mean()
        ema26 = price.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        rolling_high_20 = price.rolling(20).max()
        rolling_high_60 = price.rolling(60).max()

        local = local.assign(
            ret_5=price.pct_change(5),
            ret_20=price.pct_change(20),
            ret_60=price.pct_change(60),
            vol_20=daily_return.rolling(20).std(),
            vol_60=daily_return.rolling(60).std(),
            price_vs_sma20=(price / sma20) - 1.0,
            price_vs_sma60=(price / sma60) - 1.0,
            price_vs_sma200=(price / sma200) - 1.0,
            rsi_14=compute_rsi(price, 14),
            macd_hist=macd - macd_signal,
            volume_ratio20=volume / volume.rolling(20).mean(),
            drawdown_20=(price / rolling_high_20) - 1.0,
            drawdown_60=(price / rolling_high_60) - 1.0,
            intraday_range=(local["High"] - local["Low"]) / local["Close"].replace(0.0, np.nan),
            next_day_return=price.shift(-1) / price - 1.0,
            forward_20d_return=price.shift(-HORIZON_DAYS) / price - 1.0,
            label_available_date=local["Date"].shift(-HORIZON_DAYS),
            target_up_20=(price.shift(-HORIZON_DAYS) / price - 1.0 > 0.0).astype(float),
        )
        frames.append(local)

    features = pd.concat(frames, ignore_index=True).sort_values(["Date", "ticker"]).reset_index(drop=True)
    features["target_up_20"] = features["target_up_20"].astype("Int64")

    feature_columns = [
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
    ]

    if keep_all_rows:
        # 只 drop 技術特徵為 NaN 的行（保留 forward_20d_return 為 NaN 的最新日期，供預測用）
        features = features.dropna(subset=feature_columns).copy()
    else:
        features = features.dropna(subset=feature_columns + ["forward_20d_return", "label_available_date", "next_day_return"]).copy()
    ticker_dummies = pd.get_dummies(features["ticker"], prefix="ticker", dtype=float)
    features = pd.concat([features, ticker_dummies], axis=1)
    return features


def build_monthly_periods(start: str, end: str, label: str) -> list[Period]:
    month_starts = pd.period_range(start=start, end=end, freq="M")
    periods = []

    for period in month_starts:
        periods.append(
            Period(
                label=label,
                start=period.start_time.normalize(),
                end=period.end_time.normalize(),
            )
        )

    return periods


def build_model() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, solver="lbfgs")),
        ]
    )


def run_walk_forward_predictions(
    features: pd.DataFrame,
    feature_columns: list[str],
    periods: list[Period],
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


def compute_strategy_returns(predictions: pd.DataFrame, threshold: float) -> pd.Series:
    signals = predictions.assign(signal=(predictions["prob_up_20"] >= threshold).astype(int))
    signal_rows = signals[signals["signal"] == 1].copy()

    if signal_rows.empty:
        grouped = predictions.groupby("Date").size().index
        return pd.Series(0.0, index=grouped, name="strategy_return")

    daily = signal_rows.groupby("Date")["next_day_return"].mean()
    full_index = predictions.groupby("Date").size().index
    return daily.reindex(full_index, fill_value=0.0).rename("strategy_return")


def compute_benchmark_returns(predictions: pd.DataFrame) -> pd.Series:
    return predictions.groupby("Date")["next_day_return"].mean().rename("benchmark_return")


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


def summarize_by_ticker(predictions: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows = []

    for ticker, ticker_frame in predictions.groupby("ticker"):
        predicted_up = (ticker_frame["prob_up_20"] >= threshold).astype(int)
        actual = ticker_frame["target_up_20"].astype(int)
        selected = ticker_frame[predicted_up == 1]
        rows.append(
            {
                "ticker": ticker,
                "rows": int(len(ticker_frame)),
                "coverage": float(predicted_up.mean()),
                "accuracy": float(accuracy_score(actual, predicted_up)),
                "selected_avg_forward_20d_return": float(selected["forward_20d_return"].mean()) if not selected.empty else np.nan,
                "selected_hit_rate": float((selected["target_up_20"] == 1).mean()) if not selected.empty else np.nan,
            }
        )

    return pd.DataFrame(rows).sort_values("ticker").reset_index(drop=True)


def main() -> None:
    data_by_ticker = download_market_data(TICKERS, TRAIN_START, TEST_END)
    features = build_feature_frame(data_by_ticker)

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

    prediction_output = predictions[
        [
            "Date",
            "ticker",
            "split",
            "prob_up_20",
            "target_up_20",
            "forward_20d_return",
            "next_day_return",
        ]
    ].copy()
    prediction_output.to_csv("multi_asset_logistic_predictions.csv", index=False)

    print("Multi-Asset Logistic Baseline")
    print(f"Tickers: {', '.join(TICKERS)}")
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
    print("Predictions saved to multi_asset_logistic_predictions.csv")


if __name__ == "__main__":
    main()
