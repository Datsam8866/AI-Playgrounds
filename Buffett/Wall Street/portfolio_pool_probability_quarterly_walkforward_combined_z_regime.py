# -*- coding: utf-8 -*-
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

from expanded_pool_config import CORE_TICKER, DB_PATH, NON_LEVERAGED_EXPANDED_POOL_TICKERS, ROOT
from portfolio_pool_probability_optimizer import N_SIMS, SATELLITE_MODES
from portfolio_pool_probability_quarterly_walkforward_voo_core_no_leverage_p5_fixed import (
    CURRENT_DATE,
    MIN_HISTORY_DAYS,
    PREDICTIONS_PATH,
    SATELLITE_CAP_CANDIDATES,
    START_QUARTER,
    TOP_K_CANDIDATES,
    VOO_CORE_WEIGHT_CANDIDATES,
    available_tickers,
    build_quarter_windows,
    build_voo_core_weights,
    compute_historical_sharpe,
    pick_latest_inference_snapshot,
    pick_snapshot_on_or_before,
    realized_buyandhold_return,
)
from portfolio_return_distribution import BOOTSTRAP_BLOCK, block_bootstrap_portfolio, estimate_horizon_days


OUTPUT_PATH = ROOT / "portfolio_pool_probability_quarterly_walkforward_combined_z_regime.csv"
SUMMARY_PATH = ROOT / "portfolio_pool_probability_quarterly_walkforward_combined_z_regime_summary.csv"
UNIVERSE = NON_LEVERAGED_EXPANDED_POOL_TICKERS
SPY_TICKER = "SPY"
RISK_OFF_VIX_LEVEL = 25.0


def load_predictions() -> pd.DataFrame:
    predictions = pd.read_csv(PREDICTIONS_PATH, parse_dates=["Date"])
    return predictions[predictions["ticker"].isin(UNIVERSE)].sort_values(["Date", "ticker"]).reset_index(drop=True)


def load_prices_with_spy(connection: sqlite3.Connection) -> pd.DataFrame:
    tickers = sorted(set(UNIVERSE + [SPY_TICKER]))
    placeholders = ",".join(["?"] * len(tickers))
    query = f"""
    SELECT date, ticker, adj_close
    FROM price_history
    WHERE ticker IN ({placeholders})
    ORDER BY date, ticker
    """
    prices = pd.read_sql_query(query, connection, params=tickers, parse_dates=["date"])
    prices = prices.astype({"adj_close": float})
    return prices.pivot(index="date", columns="ticker", values="adj_close").sort_index()


def load_macro_history(connection: sqlite3.Connection) -> pd.DataFrame:
    query = """
    SELECT date, ticker, adj_close
    FROM macro_history
    WHERE ticker IN ('^VIX', '^TNX')
    ORDER BY date, ticker
    """
    macro = pd.read_sql_query(query, connection, parse_dates=["date"])
    macro = macro.astype({"adj_close": float})
    wide = macro.pivot(index="date", columns="ticker", values="adj_close").sort_index()
    wide = wide.rename(columns={"^VIX": "vix_close", "^TNX": "tnx_close"})
    return wide


def build_regime_frame(prices: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    spy = prices[SPY_TICKER].dropna().astype(float)
    regime = pd.DataFrame(index=spy.index)
    regime["spy_close"] = spy
    regime["spy_vs_sma200"] = (spy / spy.rolling(200).mean()) - 1.0

    regime = regime.join(macro[["vix_close", "tnx_close"]], how="left").sort_index().ffill()
    regime["vix_vs_sma20"] = (regime["vix_close"] / regime["vix_close"].rolling(20).mean()) - 1.0
    regime["tnx_vs_sma20"] = (regime["tnx_close"] / regime["tnx_close"].rolling(20).mean()) - 1.0
    regime["spy_below_sma200"] = regime["spy_vs_sma200"] < 0.0
    regime["vix_above_25"] = regime["vix_close"] > RISK_OFF_VIX_LEVEL
    regime["tnx_above_sma20"] = regime["tnx_vs_sma20"] > 0.0
    regime["risk_score"] = (
        regime["spy_below_sma200"].astype(int)
        + regime["vix_above_25"].astype(int)
        + regime["tnx_above_sma20"].astype(int)
    )
    regime["regime_label"] = np.select(
        [regime["risk_score"] >= 2, regime["risk_score"] == 1],
        ["risk_off", "caution"],
        default="risk_on",
    )
    return regime.reset_index().rename(columns={"index": "date"})


def pick_regime_on_or_before(regime: pd.DataFrame, cutoff: pd.Timestamp) -> pd.Series:
    prior = regime[regime["date"] <= cutoff].copy()
    if prior.empty:
        raise ValueError(f"No regime row available on or before {cutoff.date()}.")
    return prior.iloc[-1]


def candidate_grid_for_regime(regime_row: pd.Series) -> tuple[list[float], list[int], list[float], list[str]]:
    risk_score = int(regime_row["risk_score"])
    if risk_score >= 2:
        voo_core_weights = [weight for weight in VOO_CORE_WEIGHT_CANDIDATES if weight >= 0.80]
        top_k_values = [value for value in TOP_K_CANDIDATES if value <= 6]
        caps = [value for value in SATELLITE_CAP_CANDIDATES if value <= 0.10]
        modes = ["equal"]
    elif risk_score == 1:
        voo_core_weights = [weight for weight in VOO_CORE_WEIGHT_CANDIDATES if weight >= 0.60]
        top_k_values = [value for value in TOP_K_CANDIDATES if value <= 10]
        caps = SATELLITE_CAP_CANDIDATES
        modes = SATELLITE_MODES
    else:
        voo_core_weights = VOO_CORE_WEIGHT_CANDIDATES
        top_k_values = TOP_K_CANDIDATES
        caps = SATELLITE_CAP_CANDIDATES
        modes = SATELLITE_MODES

    return voo_core_weights, top_k_values, caps, modes


def zscore(series: pd.Series) -> pd.Series:
    std = float(series.std(ddof=0))
    if std <= 0 or np.isnan(std):
        return pd.Series(0.0, index=series.index)
    return (series - float(series.mean())) / std


def evaluate_window(
    period_label: str,
    snapshot: pd.DataFrame,
    snapshot_date: pd.Timestamp,
    prices: pd.DataFrame,
    regime: pd.DataFrame,
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

    regime_row = pick_regime_on_or_before(regime, forecast_start)
    voo_candidates, topk_candidates, cap_candidates, mode_candidates = candidate_grid_for_regime(regime_row)

    rows = []
    cache: dict[tuple[float, int, float, str], tuple[pd.Series, list[str]]] = {}
    seed = int(forecast_start.year * 10 + forecast_start.quarter)

    for voo_core_weight in voo_candidates:
        for top_k in topk_candidates:
            for satellite_cap in cap_candidates:
                for satellite_mode in mode_candidates:
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
                        seed=seed,
                    )
                    p_gt_5 = float(np.mean(outcomes > 0.05))
                    p_lt_0 = float(np.mean(outcomes < 0.0))
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
                        "historical_sharpe": hist_sharpe,
                        "pred_mean": float(np.mean(outcomes)),
                        "p_gt_5": p_gt_5,
                        "p_lt_0": p_lt_0,
                        "regime_label": str(regime_row["regime_label"]),
                        "risk_score": int(regime_row["risk_score"]),
                        "spy_vs_sma200": float(regime_row["spy_vs_sma200"]),
                        "vix_close": float(regime_row["vix_close"]),
                        "tnx_vs_sma20": float(regime_row["tnx_vs_sma20"]),
                    }
                    rows.append(row)
                    cache[(voo_core_weight, top_k, satellite_cap, satellite_mode)] = (active_weights, selected)

    if not rows:
        raise ValueError("No feasible configurations.")

    results = pd.DataFrame(rows)
    results["score_combined_z"] = (
        zscore(results["p_gt_5"]) - zscore(results["p_lt_0"]) + 0.3 * zscore(results["historical_sharpe"].fillna(0.0))
    )
    results = results.sort_values(
        ["score_combined_z", "historical_sharpe", "p_gt_5", "pred_mean"],
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


def summarize(report: pd.DataFrame) -> pd.DataFrame:
    realized = report[report["realized_return"].notna()].copy()
    summary = pd.DataFrame(
        [
            {
                "quarters": int(len(report)),
                "realized_quarters": int(len(realized)),
                "avg_realized_return": float(realized["realized_return"].mean()) if not realized.empty else np.nan,
                "median_realized_return": float(realized["realized_return"].median()) if not realized.empty else np.nan,
                "positive_quarter_rate": float((realized["realized_return"] > 0).mean()) if not realized.empty else np.nan,
                "avg_p_gt_5": float(realized["p_gt_5"].mean()) if not realized.empty else np.nan,
                "avg_p_lt_0": float(realized["p_lt_0"].mean()) if not realized.empty else np.nan,
                "avg_historical_sharpe": float(realized["historical_sharpe"].mean()) if not realized.empty else np.nan,
                "pearson_score_to_realized": float(realized["score_combined_z"].corr(realized["realized_return"], method="pearson"))
                if len(realized) >= 2
                else np.nan,
                "spearman_score_to_realized": float(realized["score_combined_z"].corr(realized["realized_return"], method="spearman"))
                if len(realized) >= 2
                else np.nan,
            }
        ]
    )
    return summary


def main() -> None:
    predictions = load_predictions()
    with sqlite3.connect(DB_PATH) as connection:
        prices = load_prices_with_spy(connection)
        macro = load_macro_history(connection)

    regime = build_regime_frame(prices, macro)
    max_price_date = pd.Timestamp(prices.index.max())
    quarter_windows = build_quarter_windows(prices[UNIVERSE])
    rows: list[dict[str, float | int | str]] = []

    for period_label, forecast_start, forecast_end in quarter_windows:
        period = pd.Period(period_label, freq="Q")
        quarter_end = period.end_time.normalize()
        if quarter_end > max_price_date:
            continue
        snapshot, snapshot_date = pick_snapshot_on_or_before(predictions, forecast_start)
        try:
            row = evaluate_window(
                period_label=period_label,
                snapshot=snapshot,
                snapshot_date=snapshot_date,
                prices=prices,
                regime=regime,
                forecast_start=forecast_start,
                forecast_end=forecast_end,
                include_realized=True,
            )
        except ValueError:
            continue
        rows.append(row)

    current_quarter = CURRENT_DATE.to_period("Q")
    current_end = current_quarter.end_time.normalize()
    inference_snapshot = pick_latest_inference_snapshot(predictions, CURRENT_DATE.normalize())
    if inference_snapshot is not None:
        current_snapshot, current_snapshot_date = inference_snapshot
        current_forecast_start = current_snapshot_date
        try:
            current_row = evaluate_window(
                period_label=f"{current_quarter}_current",
                snapshot=current_snapshot,
                snapshot_date=current_snapshot_date,
                prices=prices,
                regime=regime,
                forecast_start=current_forecast_start,
                forecast_end=current_end,
                include_realized=False,
            )
            rows.append(current_row)
        except ValueError:
            pass

    report = pd.DataFrame(rows)
    report.to_csv(OUTPUT_PATH, index=False)
    summary = summarize(report)
    summary.to_csv(SUMMARY_PATH, index=False)

    display = report[
        [
            "period",
            "regime_label",
            "risk_score",
            "voo_core_weight",
            "top_k",
            "satellite_cap",
            "satellite_mode",
            "score_combined_z",
            "historical_sharpe",
            "p_gt_5",
            "p_lt_0",
            "realized_return",
        ]
    ].copy().astype(object)
    for column in ["voo_core_weight", "satellite_cap", "p_gt_5", "p_lt_0", "realized_return"]:
        if column in ["voo_core_weight", "satellite_cap"]:
            display.loc[:, column] = display[column].map(lambda value: f"{value * 100:.0f}%")
        else:
            display.loc[:, column] = display[column].map(lambda value: "" if pd.isna(value) else f"{value * 100:.1f}%")
    for column in ["score_combined_z", "historical_sharpe"]:
        display.loc[:, column] = display[column].map(lambda value: "" if pd.isna(value) else f"{value:.3f}")

    print("Quarterly walk-forward with Combined Z objective + regime filter")
    print(display.to_string(index=False))
    print()
    print(summary.to_string(index=False))
    print()
    print(f"Saved CSV: {OUTPUT_PATH.name}")
    print(f"Saved summary CSV: {SUMMARY_PATH.name}")


if __name__ == "__main__":
    main()
