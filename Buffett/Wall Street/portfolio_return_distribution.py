# -*- coding: utf-8 -*-
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "stock_forecast.sqlite"
SUMMARY_OUT = ROOT / "portfolio_return_distribution_summary.csv"
PATHS_OUT = ROOT / "portfolio_return_distribution_paths.csv"

END_DATE = pd.Timestamp("2026-12-31")
CURRENT_DATE = pd.Timestamp("2026-04-11")
START_HISTORY = pd.Timestamp("2019-01-01")
N_SIMS = 20000
BOOTSTRAP_BLOCK = 20
THRESHOLDS = [0.10, 0.20, 0.30, 0.40, 0.50]

# 依照前面已確認的建議配置；會在程式內重新 normalize
TARGET_WEIGHTS = {
    "VOO": 0.5483,
    "GOOGL": 0.0608,
    "NVDA": 0.0606,
    "AVGO": 0.0512,
    "TSLA": 0.0403,
    "TSM": 0.0398,
    "SSO": 0.0000,
    "SOXX": 0.0089,
    "QQQ": 0.1008,
    "PLTR": 0.0198,
    "MSTR": 0.0000,
    "VRT": 0.0193,
    "AMD": 0.0197,
    "DELL": 0.0095,
    "MRVL": 0.0000,
    "QCOM": 0.0196,
}


def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    filtered = {ticker: weight for ticker, weight in weights.items() if weight > 0}
    total = sum(filtered.values())
    return {ticker: weight / total for ticker, weight in filtered.items()}


def estimate_horizon_days(current_date: pd.Timestamp, end_date: pd.Timestamp) -> int:
    remaining_days = (end_date - current_date).days
    return int(round(252 * remaining_days / 365.25))


def load_returns(
    connection: sqlite3.Connection,
    tickers: list[str],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    placeholders = ",".join(["?"] * len(tickers))
    query = f"""
    SELECT date, ticker, adj_close
    FROM price_history
    WHERE ticker IN ({placeholders})
      AND date >= ?
      AND (? IS NULL OR date <= ?)
    ORDER BY date, ticker
    """
    end_date_str = end_date.strftime("%Y-%m-%d") if end_date is not None else None
    params = tickers + [start_date.strftime("%Y-%m-%d"), end_date_str, end_date_str]
    prices = pd.read_sql_query(query, connection, params=params, parse_dates=["date"])
    prices["adj_close"] = prices["adj_close"].astype(float)

    wide = prices.pivot(index="date", columns="ticker", values="adj_close").sort_index()
    returns = wide.pct_change().dropna(how="all")
    returns = returns.dropna(axis=0, how="any")
    return returns


def block_bootstrap_portfolio(
    asset_returns: pd.DataFrame,
    weights: pd.Series,
    horizon_days: int,
    n_sims: int,
    block_size: int,
    seed: int = 42,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    matrix = asset_returns[weights.index].to_numpy()
    n_obs = matrix.shape[0]
    max_start = n_obs - block_size
    if max_start < 0:
        raise ValueError("Not enough observations for block bootstrap.")

    results = np.empty(n_sims, dtype=float)
    for sim_idx in range(n_sims):
        sampled_blocks = []
        total_days = 0
        while total_days < horizon_days:
            start = int(rng.integers(0, max_start + 1))
            block = matrix[start : start + block_size]
            sampled_blocks.append(block)
            total_days += len(block)
        sampled = np.vstack(sampled_blocks)[:horizon_days]
        portfolio_daily = sampled @ weights.to_numpy()
        results[sim_idx] = float(np.prod(1.0 + portfolio_daily) - 1.0)
    return results


def monte_carlo_portfolio(
    asset_returns: pd.DataFrame,
    weights: pd.Series,
    horizon_days: int,
    n_sims: int,
    seed: int = 42,
) -> np.ndarray:
    rng = np.random.default_rng(seed + 1)
    matrix = asset_returns[weights.index]
    mean_vector = matrix.mean().to_numpy()
    cov_matrix = matrix.cov().to_numpy()
    simulated = rng.multivariate_normal(mean_vector, cov_matrix, size=(n_sims, horizon_days), method="svd")
    portfolio_daily = simulated @ weights.to_numpy()
    return np.prod(1.0 + portfolio_daily, axis=1) - 1.0


def summarize_distribution(name: str, outcomes: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    probs = []
    for threshold in THRESHOLDS:
        probs.append(
            {
                "method": name,
                "threshold_return": threshold,
                "probability": float(np.mean(outcomes > threshold)),
            }
        )

    quantiles = pd.DataFrame(
        [
            {
                "method": name,
                "mean_return": float(np.mean(outcomes)),
                "median_return": float(np.median(outcomes)),
                "p05_return": float(np.quantile(outcomes, 0.05)),
                "p25_return": float(np.quantile(outcomes, 0.25)),
                "p75_return": float(np.quantile(outcomes, 0.75)),
                "p95_return": float(np.quantile(outcomes, 0.95)),
            }
        ]
    )
    return pd.DataFrame(probs), quantiles


def main() -> None:
    weights = normalize_weights(TARGET_WEIGHTS)
    weight_series = pd.Series(weights).sort_index()
    horizon_days = estimate_horizon_days(CURRENT_DATE, END_DATE)

    with sqlite3.connect(DB_PATH) as connection:
        asset_returns = load_returns(connection, list(weight_series.index), START_HISTORY, end_date=CURRENT_DATE)

    bootstrap_outcomes = block_bootstrap_portfolio(
        asset_returns,
        weight_series,
        horizon_days,
        N_SIMS,
        BOOTSTRAP_BLOCK,
    )
    mc_outcomes = monte_carlo_portfolio(
        asset_returns,
        weight_series,
        horizon_days,
        N_SIMS,
    )

    bootstrap_probs, bootstrap_stats = summarize_distribution("block_bootstrap", bootstrap_outcomes)
    mc_probs, mc_stats = summarize_distribution("monte_carlo", mc_outcomes)

    summary = pd.concat([bootstrap_probs, mc_probs, bootstrap_stats, mc_stats], ignore_index=True)
    summary.to_csv(SUMMARY_OUT, index=False)

    paths = pd.DataFrame(
        {
            "block_bootstrap_return": bootstrap_outcomes,
            "monte_carlo_return": mc_outcomes,
        }
    )
    paths.to_csv(PATHS_OUT, index=False)

    print("Portfolio return distribution summary")
    print(f"Current date assumption : {CURRENT_DATE.date()}")
    print(f"End date               : {END_DATE.date()}")
    print(f"Estimated horizon days : {horizon_days}")
    print()
    print("Target weights:")
    print((weight_series * 100).round(2).astype(str) + "%")
    print()
    print("Threshold probabilities:")
    print(pd.concat([bootstrap_probs, mc_probs], ignore_index=True).to_string(index=False))
    print()
    print("Distribution stats:")
    print(pd.concat([bootstrap_stats, mc_stats], ignore_index=True).to_string(index=False))
    print()
    print(f"Saved summary: {SUMMARY_OUT.name}")
    print(f"Saved paths  : {PATHS_OUT.name}")


if __name__ == "__main__":
    main()
