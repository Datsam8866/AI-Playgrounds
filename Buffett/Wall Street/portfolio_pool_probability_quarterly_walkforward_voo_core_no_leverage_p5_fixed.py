# -*- coding: utf-8 -*-
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

from expanded_pool_config import CORE_TICKER, DB_PATH, NON_LEVERAGED_EXPANDED_POOL_TICKERS, ROOT
from portfolio_pool_probability_optimizer import N_SIMS, SATELLITE_MODES, THRESHOLDS, UTILITY_EPS
from portfolio_return_distribution import BOOTSTRAP_BLOCK, block_bootstrap_portfolio, estimate_horizon_days


PREDICTIONS_PATH = ROOT / "expanded_pool_xgboost_regression_no_leverage_predictions_v2.csv"
OUTPUT_PATH = ROOT / "portfolio_pool_probability_quarterly_walkforward_voo_core_no_leverage_p5_fixed_v2.csv"
START_QUARTER = "2016Q1"
CURRENT_DATE = pd.Timestamp("2026-05-01")
UNIVERSE = NON_LEVERAGED_EXPANDED_POOL_TICKERS
VOO_CORE_WEIGHT_CANDIDATES = [0.50, 0.60, 0.70, 0.80]
TOP_K_CANDIDATES = [2, 4, 6, 8, 10, 12]
SATELLITE_CAP_CANDIDATES = [0.10, 0.12, 0.15]
MIN_HISTORY_DAYS = 60
UTILITY_THRESHOLD = 0.05


def load_predictions() -> pd.DataFrame:
    predictions = pd.read_csv(PREDICTIONS_PATH, parse_dates=["Date"])
    return predictions[predictions["ticker"].isin(UNIVERSE)].sort_values(["Date", "ticker"]).reset_index(drop=True)


def load_prices(connection: sqlite3.Connection) -> pd.DataFrame:
    placeholders = ",".join(["?"] * len(UNIVERSE))
    query = f"""
    SELECT date, ticker, adj_close
    FROM price_history
    WHERE ticker IN ({placeholders})
    ORDER BY date, ticker
    """
    prices = pd.read_sql_query(query, connection, params=UNIVERSE, parse_dates=["date"])
    prices = prices.astype({"adj_close": float})
    return prices.pivot(index="date", columns="ticker", values="adj_close").sort_index()


def pick_snapshot_on_or_before(predictions: pd.DataFrame, cutoff: pd.Timestamp) -> tuple[pd.DataFrame, pd.Timestamp]:
    prior = predictions[predictions["Date"] <= cutoff].copy()
    if prior.empty:
        raise ValueError(f"No prediction snapshot available on or before {cutoff.date()}.")
    snapshot_date = pd.Timestamp(prior["Date"].max())
    snapshot = prior[prior["Date"] == snapshot_date].copy()
    snapshot = snapshot.sort_values(["predicted_return", "ticker"], ascending=[False, True]).reset_index(drop=True)
    return snapshot, snapshot_date


def pick_latest_inference_snapshot(predictions: pd.DataFrame, cutoff: pd.Timestamp) -> tuple[pd.DataFrame, pd.Timestamp] | None:
    inference = predictions[(predictions["split"] == "inference") & (predictions["Date"] <= cutoff)].copy()
    if inference.empty:
        return None
    snapshot_date = pd.Timestamp(inference["Date"].max())
    snapshot = inference[inference["Date"] == snapshot_date].copy()
    snapshot = snapshot.sort_values(["predicted_return", "ticker"], ascending=[False, True]).reset_index(drop=True)
    return snapshot, snapshot_date


def available_tickers(prices: pd.DataFrame, start_date: pd.Timestamp) -> list[str]:
    available_dates = prices.index[prices.index <= start_date]
    if available_dates.empty:
        return []
    last_date = pd.Timestamp(available_dates[-1])
    row = prices.loc[last_date, UNIVERSE]
    return row.index[row.notna()].tolist()


def cap_satellite_weights(weights: pd.Series, max_single_weight: float, total_weight: float) -> pd.Series:
    capped = weights.copy().astype(float)
    if len(capped) * max_single_weight + 1e-12 < total_weight:
        raise ValueError("Infeasible satellite cap: top-k capacity is below required satellite weight.")

    iterations = 0
    while capped.max() > max_single_weight + 1e-12 and iterations < 20:
        over_mask = capped > max_single_weight
        excess = float((capped[over_mask] - max_single_weight).sum())
        capped.loc[over_mask] = max_single_weight

        under_mask = capped < max_single_weight - 1e-12
        under_total = float(capped[under_mask].sum())
        if excess <= 0 or under_total <= 0:
            break
        capped.loc[under_mask] += capped.loc[under_mask] / under_total * excess
        iterations += 1

    if capped.max() > max_single_weight + 1e-9:
        raise ValueError("Unable to satisfy satellite cap constraints.")

    final_total = float(capped.sum())
    if final_total > 0:
        capped *= total_weight / final_total
    if capped.max() > max_single_weight + 1e-9:
        raise ValueError("Satellite cap violated after normalization.")
    return capped


def build_voo_core_weights(
    snapshot: pd.DataFrame,
    available: list[str],
    voo_core_weight: float,
    top_k: int,
    satellite_cap: float,
    satellite_mode: str,
) -> tuple[pd.Series, list[str]]:
    if CORE_TICKER not in available:
        raise ValueError("VOO is not available in this forecast window.")

    candidates = snapshot[snapshot["ticker"].isin([ticker for ticker in available if ticker != CORE_TICKER])].copy()
    if len(candidates) < top_k:
        raise ValueError("Not enough available satellite names.")

    selected_frame = candidates.head(top_k).copy()
    selected = selected_frame["ticker"].tolist()
    satellite_total = 1.0 - voo_core_weight

    if satellite_mode == "equal":
        satellite_weights = pd.Series(satellite_total / top_k, index=selected, dtype=float)
    elif satellite_mode == "predicted":
        pred = selected_frame.set_index("ticker")["predicted_return"].astype(float).clip(lower=0.0)
        if float(pred.sum()) <= 0:
            satellite_weights = pd.Series(satellite_total / top_k, index=selected, dtype=float)
        else:
            satellite_weights = pred / float(pred.sum()) * satellite_total
    else:
        raise ValueError(f"Unsupported satellite_mode: {satellite_mode}")

    satellite_weights = cap_satellite_weights(satellite_weights, satellite_cap, satellite_total)
    weights = pd.Series(0.0, index=available, dtype=float)
    weights.loc[CORE_TICKER] = voo_core_weight
    weights.loc[selected] = satellite_weights
    total = float(weights.sum())
    if total > 0:
        weights /= total
    return weights.sort_values(ascending=False), selected


def compute_historical_sharpe(asset_returns: pd.DataFrame, weights: pd.Series) -> float:
    portfolio_daily = asset_returns[weights.index] @ weights.to_numpy(dtype=float)
    std = float(portfolio_daily.std(ddof=1))
    if std <= 0 or np.isnan(std):
        return np.nan
    return float(np.sqrt(252.0) * portfolio_daily.mean() / std)


def realized_buyandhold_return(
    prices: pd.DataFrame,
    weights: pd.Series,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> tuple[float, int]:
    period_prices = prices.loc[(prices.index >= start_date) & (prices.index <= end_date), weights.index]
    if period_prices.empty:
        raise ValueError("No realized return series available.")

    start_prices = period_prices.iloc[0]
    if start_prices.isna().any():
        raise ValueError("Missing start prices for realized return calculation.")

    # Hold the initial basket until the end of the window; if a name stops trading,
    # keep its last observed close rather than implicitly rebalancing away from it.
    end_prices = period_prices.ffill().iloc[-1]
    if end_prices.isna().any():
        raise ValueError("Missing end prices for realized return calculation.")

    individual_returns = end_prices / start_prices - 1.0
    realized_return = float((individual_returns * weights).sum())
    return realized_return, int(max(len(period_prices.index) - 1, 0))


def build_quarter_windows(prices: pd.DataFrame) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    max_date = pd.Timestamp(prices.index.max())
    quarters = pd.period_range(start=START_QUARTER, end=max_date.to_period("Q"), freq="Q")
    windows: list[tuple[str, pd.Timestamp, pd.Timestamp]] = []

    for quarter in quarters:
        quarter_start = quarter.start_time.normalize()
        quarter_end = quarter.end_time.normalize()
        tradable = prices.loc[(prices.index >= quarter_start) & (prices.index <= quarter_end)].index
        if tradable.empty:
            continue
        windows.append((str(quarter), pd.Timestamp(tradable.min()), pd.Timestamp(tradable.max())))
    return windows


def evaluate_window(
    period_label: str,
    snapshot: pd.DataFrame,
    snapshot_date: pd.Timestamp,
    prices: pd.DataFrame,
    forecast_start: pd.Timestamp,
    forecast_end: pd.Timestamp,
    include_realized: bool,
) -> dict[str, float | int | str]:
    available = available_tickers(prices, forecast_start)
    if CORE_TICKER not in available:
        raise ValueError("VOO missing from available universe.")

    period_dates = prices.loc[(prices.index >= forecast_start) & (prices.index <= min(forecast_end, prices.index.max()))].index
    if period_dates.empty:
        raise ValueError("Insufficient forecast window.")

    if include_realized:
        if len(period_dates) < 2:
            raise ValueError("Insufficient realized forecast window.")
        horizon_days = max(len(period_dates) - 1, 1)
    else:
        horizon_days = max(int(estimate_horizon_days(forecast_start, forecast_end)), 1)

    rows = []
    cache: dict[tuple[float, int, float, str], tuple[pd.Series, list[str]]] = {}

    for voo_core_weight in VOO_CORE_WEIGHT_CANDIDATES:
        for top_k in TOP_K_CANDIDATES:
            for satellite_cap in SATELLITE_CAP_CANDIDATES:
                for satellite_mode in SATELLITE_MODES:
                    try:
                        weights, selected = build_voo_core_weights(
                            snapshot=snapshot,
                            available=available,
                            voo_core_weight=voo_core_weight,
                            top_k=top_k,
                            satellite_cap=satellite_cap,
                            satellite_mode=satellite_mode,
                        )
                    except ValueError:
                        continue

                    active_weights = weights[weights > 0].copy()
                    history_prices = prices.loc[prices.index < forecast_start, active_weights.index].dropna(how="any")
                    history_returns = history_prices.pct_change().dropna(how="any")
                    history_days = int(len(history_returns))
                    if history_days < MIN_HISTORY_DAYS:
                        continue

                    outcomes = block_bootstrap_portfolio(
                        asset_returns=history_returns,
                        weights=active_weights.sort_index(),
                        horizon_days=horizon_days,
                        n_sims=N_SIMS,
                        block_size=BOOTSTRAP_BLOCK,
                        seed=int(forecast_start.year * 10 + forecast_start.quarter),
                    )
                    p_lt_0 = float(np.mean(outcomes < 0.0))
                    p_gt_target = float(np.mean(outcomes > UTILITY_THRESHOLD))
                    utility = float((p_gt_target + UTILITY_EPS) / (p_lt_0 + UTILITY_EPS))
                    hist_sharpe = compute_historical_sharpe(history_returns, active_weights.sort_index())

                    row: dict[str, float | int | str] = {
                        "period": period_label,
                        "snapshot_date": snapshot_date.date().isoformat(),
                        "forecast_start": forecast_start.date().isoformat(),
                        "forecast_end": forecast_end.date().isoformat(),
                        "history_days": history_days,
                        "available_names": len(available),
                        "voo_core_weight": voo_core_weight,
                        "top_k": top_k,
                        "satellite_cap": satellite_cap,
                        "satellite_mode": satellite_mode,
                        "utility_p5_over_p0": utility,
                        "historical_sharpe": hist_sharpe,
                        "pred_mean": float(np.mean(outcomes)),
                        "p_gt_5": p_gt_target,
                        "p_lt_0": p_lt_0,
                    }
                    for threshold in THRESHOLDS:
                        row[f"p_gt_{int(threshold * 100)}"] = float(np.mean(outcomes > threshold))
                    rows.append(row)
                    cache[(voo_core_weight, top_k, satellite_cap, satellite_mode)] = (active_weights, selected)

    if not rows:
        raise ValueError("No feasible configurations.")

    results = pd.DataFrame(rows).sort_values(
        ["utility_p5_over_p0", "historical_sharpe", "p_gt_10", "pred_mean"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    best = results.iloc[0].to_dict()
    best_key = (
        float(best["voo_core_weight"]),
        int(best["top_k"]),
        float(best["satellite_cap"]),
        str(best["satellite_mode"]),
    )
    weights, selected = cache[best_key]

    out = best.copy()
    out["selected_satellite"] = ",".join(selected)
    out["weights"] = ", ".join(f"{ticker} {weight * 100:.1f}%" for ticker, weight in weights.items())
    if include_realized:
        realized_return, _ = realized_buyandhold_return(prices, weights.sort_index(), forecast_start, forecast_end)
        out["realized_return"] = realized_return
    else:
        out["realized_return"] = np.nan
    return out


def main() -> None:
    predictions = load_predictions()
    with sqlite3.connect(DB_PATH) as connection:
        prices = load_prices(connection)

    max_price_date = pd.Timestamp(prices.index.max())
    quarter_windows = build_quarter_windows(prices)
    rows: list[dict[str, float | int | str]] = []

    for period_label, forecast_start, forecast_end in quarter_windows:
        period = pd.Period(period_label, freq="Q")
        quarter_end = period.end_time.normalize()
        include_realized = quarter_end <= max_price_date
        if not include_realized:
            continue
        snapshot, snapshot_date = pick_snapshot_on_or_before(predictions, forecast_start)
        try:
            row = evaluate_window(
                period_label=period_label,
                snapshot=snapshot,
                snapshot_date=snapshot_date,
                prices=prices,
                forecast_start=forecast_start,
                forecast_end=forecast_end,
                include_realized=True,
            )
        except ValueError:
            continue
        rows.append(row)

    current_quarter = CURRENT_DATE.to_period("Q")
    current_start = current_quarter.start_time.normalize()
    current_end = current_quarter.end_time.normalize()
    current_tradable = prices.loc[(prices.index >= current_start) & (prices.index <= min(current_end, prices.index.max()))].index
    if not current_tradable.empty:
        inference_snapshot = pick_latest_inference_snapshot(predictions, CURRENT_DATE.normalize())
        if inference_snapshot is not None:
            current_snapshot, current_snapshot_date = inference_snapshot
            current_forecast_start = current_snapshot_date
        else:
            current_forecast_start = pd.Timestamp(current_tradable.min())
            current_snapshot, current_snapshot_date = pick_snapshot_on_or_before(predictions, current_forecast_start)
        try:
            current_row = evaluate_window(
                period_label=f"{current_quarter}_current",
                snapshot=current_snapshot,
                snapshot_date=current_snapshot_date,
                prices=prices,
                forecast_start=current_forecast_start,
                forecast_end=current_end,
                include_realized=False,
            )
            rows.append(current_row)
        except ValueError:
            pass

    report = pd.DataFrame(rows)
    report.to_csv(OUTPUT_PATH, index=False)

    display = report[
        [
            "period",
            "snapshot_date",
            "voo_core_weight",
            "top_k",
            "satellite_cap",
            "satellite_mode",
            "utility_p5_over_p0",
            "historical_sharpe",
            "p_gt_5",
            "p_gt_10",
            "p_gt_20",
            "p_gt_30",
            "p_gt_40",
            "p_gt_50",
            "p_lt_0",
            "realized_return",
        ]
    ].copy()
    for column in ["p_gt_5", "p_gt_10", "p_gt_20", "p_gt_30", "p_gt_40", "p_gt_50", "p_lt_0", "realized_return"]:
        display.loc[:, column] = display[column].map(lambda value: "" if pd.isna(value) else f"{value * 100:.1f}%")

    print("Quarterly walk-forward with VOO core >= 50% and no leveraged ETFs")
    print("Objective: maximize P(>5%) / P(<0%), then Sharpe")
    print(display.to_string(index=False))
    print()
    print(f"Saved CSV: {OUTPUT_PATH.name}")


if __name__ == "__main__":
    main()
