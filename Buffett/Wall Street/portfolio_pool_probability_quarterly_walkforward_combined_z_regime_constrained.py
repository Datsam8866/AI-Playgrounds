# -*- coding: utf-8 -*-
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

from expanded_pool_config import CORE_TICKER, DB_PATH, NON_LEVERAGED_EXPANDED_POOL_TICKERS, ROOT
from portfolio_pool_probability_optimizer import N_SIMS, SATELLITE_MODES
from portfolio_pool_probability_quarterly_walkforward_combined_z_regime import (
    CURRENT_DATE,
    MIN_HISTORY_DAYS,
    OUTPUT_PATH as _UNUSED_OUTPUT_PATH,
    PREDICTIONS_PATH,
    SATELLITE_CAP_CANDIDATES,
    SPY_TICKER,
    START_QUARTER,
    TOP_K_CANDIDATES,
    VOO_CORE_WEIGHT_CANDIDATES,
    build_regime_frame,
    build_voo_core_weights,
    compute_historical_sharpe,
    load_macro_history,
    load_predictions,
    load_prices_with_spy,
    pick_latest_inference_snapshot,
    pick_regime_on_or_before,
    pick_snapshot_on_or_before,
    realized_buyandhold_return,
    zscore,
)
from portfolio_pool_probability_quarterly_walkforward_voo_core_no_leverage_p5_fixed import (
    available_tickers,
    build_quarter_windows,
)
from portfolio_return_distribution import BOOTSTRAP_BLOCK, block_bootstrap_portfolio, estimate_horizon_days
from quarterly_probability_calibration import append_platt_calibration_columns


OUTPUT_PATH = ROOT / "portfolio_pool_probability_quarterly_walkforward_combined_z_regime_constrained.csv"
SUMMARY_PATH = ROOT / "portfolio_pool_probability_quarterly_walkforward_combined_z_regime_constrained_summary.csv"
RUN_N_SIMS = min(N_SIMS, 3000)

TURNOVER_CAP_BY_REGIME = {"risk_on": 0.35, "caution": 0.25, "risk_off": 0.20}
TRACKING_ERROR_CAP_BY_REGIME = {"risk_on": 0.35, "caution": 0.30, "risk_off": 0.25}
THEME_CAP_BY_REGIME = {"risk_on": 0.30, "caution": 0.25, "risk_off": 0.20}

THEME_BY_TICKER = {
    "AAPL": "hardware_platforms",
    "ACN": "services_hardware",
    "ADBE": "software",
    "AMD": "semiconductors",
    "AMZN": "internet_platforms",
    "ARM": "semiconductors",
    "AVGO": "semiconductors",
    "CFLT": "software",
    "CLS": "services_hardware",
    "CRM": "software",
    "CRWD": "software",
    "CRWV": "infra_compute",
    "DELL": "services_hardware",
    "GOOGL": "internet_platforms",
    "IBM": "services_hardware",
    "IFNNY": "semiconductors",
    "INTC": "semiconductors",
    "LOGI": "services_hardware",
    "META": "internet_platforms",
    "MRVL": "semiconductors",
    "MSFT": "software",
    "MSTR": "infra_compute",
    "MU": "semiconductors",
    "NFLX": "internet_platforms",
    "NOW": "software",
    "NTAP": "services_hardware",
    "NVDA": "semiconductors",
    "OKTA": "software",
    "ORCL": "software",
    "PLTR": "software",
    "PSTG": "software",
    "QCOM": "semiconductors",
    "QQQ": "broad_etf",
    "SMCI": "semiconductors",
    "SOXX": "broad_etf",
    "TLT": "fixed_income",
    "TSLA": "mobility",
    "TSM": "semiconductors",
    "VOO": "core_index",
    "VRT": "infra_compute",
}


def candidate_grid_for_regime(regime_label: str) -> tuple[list[float], list[int], list[float], list[str]]:
    if regime_label == "risk_off":
        voo_core_weights = [weight for weight in VOO_CORE_WEIGHT_CANDIDATES if weight >= 0.80]
        top_k_values = [value for value in TOP_K_CANDIDATES if value <= 6]
        caps = [value for value in SATELLITE_CAP_CANDIDATES if value <= 0.10]
        modes = ["equal"]
    elif regime_label == "caution":
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


def one_way_turnover(current_weights: pd.Series, previous_weights: pd.Series | None) -> float:
    if previous_weights is None or previous_weights.empty:
        return 0.0
    all_names = sorted(set(current_weights.index) | set(previous_weights.index))
    current = current_weights.reindex(all_names).fillna(0.0)
    previous = previous_weights.reindex(all_names).fillna(0.0)
    return float(0.5 * np.abs(current - previous).sum())


def tracking_error_vs_voo(history_returns: pd.DataFrame, weights: pd.Series) -> float:
    aligned = history_returns[weights.index].copy()
    portfolio_daily = aligned @ weights.to_numpy(dtype=float)
    benchmark_daily = aligned[CORE_TICKER]
    active_daily = portfolio_daily - benchmark_daily
    std = float(active_daily.std(ddof=1))
    if std <= 0 or np.isnan(std):
        return 0.0
    return float(np.sqrt(252.0) * std)


def max_theme_weight(weights: pd.Series) -> float:
    non_core = weights[weights.index != CORE_TICKER].copy()
    if non_core.empty:
        return 0.0
    themes = non_core.index.to_series().map(lambda ticker: THEME_BY_TICKER.get(ticker, "other"))
    grouped = non_core.groupby(themes).sum()
    return float(grouped.max()) if not grouped.empty else 0.0


def violation_amount(value: float, cap: float) -> float:
    return float(max(value - cap, 0.0))


def evaluate_window(
    period_label: str,
    snapshot: pd.DataFrame,
    snapshot_date: pd.Timestamp,
    prices: pd.DataFrame,
    regime: pd.DataFrame,
    forecast_start: pd.Timestamp,
    forecast_end: pd.Timestamp,
    previous_weights: pd.Series | None,
    include_realized: bool,
) -> tuple[dict[str, float | int | str], pd.Series]:
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
    regime_label = str(regime_row["regime_label"])
    voo_candidates, topk_candidates, cap_candidates, mode_candidates = candidate_grid_for_regime(regime_label)
    turnover_cap = TURNOVER_CAP_BY_REGIME[regime_label]
    tracking_error_cap = TRACKING_ERROR_CAP_BY_REGIME[regime_label]
    theme_cap = THEME_CAP_BY_REGIME[regime_label]

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
                        n_sims=RUN_N_SIMS,
                        block_size=BOOTSTRAP_BLOCK,
                        seed=seed,
                    )
                    p_gt_5 = float(np.mean(outcomes > 0.05))
                    p_lt_0 = float(np.mean(outcomes < 0.0))
                    hist_sharpe = compute_historical_sharpe(history_returns, active_weights.sort_index())
                    turnover = one_way_turnover(active_weights.sort_index(), previous_weights)
                    tracking_error = tracking_error_vs_voo(history_returns, active_weights.sort_index())
                    theme_weight = max_theme_weight(active_weights.sort_index())

                    turnover_violation = violation_amount(turnover, turnover_cap)
                    tracking_error_violation = violation_amount(tracking_error, tracking_error_cap)
                    theme_violation = violation_amount(theme_weight, theme_cap)
                    total_violation = turnover_violation + tracking_error_violation + theme_violation
                    feasible = total_violation <= 1e-12

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
                        "regime_label": regime_label,
                        "risk_score": int(regime_row["risk_score"]),
                        "spy_vs_sma200": float(regime_row["spy_vs_sma200"]),
                        "vix_close": float(regime_row["vix_close"]),
                        "tnx_vs_sma20": float(regime_row["tnx_vs_sma20"]),
                        "turnover": turnover,
                        "turnover_cap": turnover_cap,
                        "tracking_error": tracking_error,
                        "tracking_error_cap": tracking_error_cap,
                        "max_theme_weight": theme_weight,
                        "theme_cap": theme_cap,
                        "constraints_feasible": bool(feasible),
                        "total_violation": total_violation,
                        "turnover_violation": turnover_violation,
                        "tracking_error_violation": tracking_error_violation,
                        "theme_violation": theme_violation,
                    }
                    rows.append(row)
                    cache[(voo_core_weight, top_k, satellite_cap, satellite_mode)] = (active_weights, selected)

    if not rows:
        raise ValueError("No feasible configurations.")

    results = pd.DataFrame(rows)
    results["score_combined_z"] = (
        zscore(results["p_gt_5"]) - zscore(results["p_lt_0"]) + 0.3 * zscore(results["historical_sharpe"].fillna(0.0))
    )
    feasible = results[results["constraints_feasible"]].copy()
    if not feasible.empty:
        ordered = feasible.sort_values(
            ["score_combined_z", "historical_sharpe", "p_gt_5", "pred_mean"],
            ascending=[False, False, False, False],
        ).reset_index(drop=True)
        best = ordered.iloc[0].to_dict()
        best["selection_stage"] = "feasible"
    else:
        ordered = results.sort_values(
            [
                "turnover_violation",
                "total_violation",
                "tracking_error_violation",
                "theme_violation",
                "score_combined_z",
                "historical_sharpe",
                "p_gt_5",
                "pred_mean",
            ],
            ascending=[True, True, True, True, False, False, False, False],
        ).reset_index(drop=True)
        best = ordered.iloc[0].to_dict()
        best["selection_stage"] = "fallback_turnover_first"
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
    return out, weights.sort_index()


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
                "calibration_method": "platt",
                "avg_p_gt_5": float(realized["p_gt_5"].mean()) if not realized.empty else np.nan,
                "avg_p_lt_0": float(realized["p_lt_0"].mean()) if not realized.empty else np.nan,
                "avg_raw_p_gt_5": float(realized["raw_p_gt_5"].mean()) if not realized.empty else np.nan,
                "avg_raw_p_lt_0": float(realized["raw_p_lt_0"].mean()) if not realized.empty else np.nan,
                "avg_historical_sharpe": float(realized["historical_sharpe"].mean()) if not realized.empty else np.nan,
                "avg_turnover": float(realized["turnover"].mean()) if not realized.empty else np.nan,
                "avg_tracking_error": float(realized["tracking_error"].mean()) if not realized.empty else np.nan,
                "avg_max_theme_weight": float(realized["max_theme_weight"].mean()) if not realized.empty else np.nan,
                "feasible_rate": float(realized["constraints_feasible"].mean()) if not realized.empty else np.nan,
                "fallback_rate": float((realized["selection_stage"] == "fallback_turnover_first").mean())
                if not realized.empty
                else np.nan,
                "avg_turnover_violation": float(realized["turnover_violation"].mean()) if not realized.empty else np.nan,
                "avg_total_violation": float(realized["total_violation"].mean()) if not realized.empty else np.nan,
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
    _ = _UNUSED_OUTPUT_PATH
    predictions = load_predictions()
    with sqlite3.connect(DB_PATH) as connection:
        prices = load_prices_with_spy(connection)
        macro = load_macro_history(connection)

    regime = build_regime_frame(prices, macro)
    max_price_date = pd.Timestamp(prices.index.max())
    quarter_windows = build_quarter_windows(prices[NON_LEVERAGED_EXPANDED_POOL_TICKERS])
    rows: list[dict[str, float | int | str]] = []
    previous_weights: pd.Series | None = None

    for period_label, forecast_start, forecast_end in quarter_windows:
        period = pd.Period(period_label, freq="Q")
        quarter_end = period.end_time.normalize()
        if quarter_end > max_price_date:
            continue
        snapshot, snapshot_date = pick_snapshot_on_or_before(predictions, forecast_start)
        try:
            row, chosen_weights = evaluate_window(
                period_label=period_label,
                snapshot=snapshot,
                snapshot_date=snapshot_date,
                prices=prices,
                regime=regime,
                forecast_start=forecast_start,
                forecast_end=forecast_end,
                previous_weights=previous_weights,
                include_realized=True,
            )
        except ValueError:
            continue
        rows.append(row)
        previous_weights = chosen_weights

    current_quarter = CURRENT_DATE.to_period("Q")
    current_end = current_quarter.end_time.normalize()
    inference_snapshot = pick_latest_inference_snapshot(predictions, CURRENT_DATE.normalize())
    if inference_snapshot is not None:
        current_snapshot, current_snapshot_date = inference_snapshot
        current_forecast_start = current_snapshot_date
        try:
            current_row, _ = evaluate_window(
                period_label=f"{current_quarter}_current",
                snapshot=current_snapshot,
                snapshot_date=current_snapshot_date,
                prices=prices,
                regime=regime,
                forecast_start=current_forecast_start,
                forecast_end=current_end,
                previous_weights=previous_weights,
                include_realized=False,
            )
            rows.append(current_row)
        except ValueError:
            pass

    report = pd.DataFrame(rows)
    report = append_platt_calibration_columns(report)
    report["p_gt_5"] = report["p_gt_5_calibrated"]
    report["p_lt_0"] = report["p_lt_0_calibrated"]
    report.to_csv(OUTPUT_PATH, index=False)
    summary = summarize(report)
    summary.to_csv(SUMMARY_PATH, index=False)

    display = report[
        [
            "period",
            "regime_label",
            "voo_core_weight",
            "top_k",
            "selection_stage",
            "score_combined_z",
            "historical_sharpe",
            "p_gt_5",
            "p_lt_0",
            "turnover",
            "tracking_error",
            "max_theme_weight",
            "constraints_feasible",
            "realized_return",
        ]
    ].copy().astype(object)
    for column in ["voo_core_weight", "p_gt_5", "p_lt_0", "turnover", "tracking_error", "max_theme_weight", "realized_return"]:
        if column == "voo_core_weight":
            display.loc[:, column] = display[column].map(lambda value: f"{value * 100:.0f}%")
        else:
            display.loc[:, column] = display[column].map(lambda value: "" if pd.isna(value) else f"{value * 100:.1f}%")
    for column in ["score_combined_z", "historical_sharpe"]:
        display.loc[:, column] = display[column].map(lambda value: "" if pd.isna(value) else f"{value:.3f}")

    print("Quarterly walk-forward with Combined Z + regime + practical constraints")
    print("Displayed P(>5%) / P(<0%) are Platt-calibrated; raw values are saved in raw_p_gt_5 / raw_p_lt_0.")
    print(display.to_string(index=False))
    print()
    print(summary.to_string(index=False))
    print()
    print(f"Saved CSV: {OUTPUT_PATH.name}")
    print(f"Saved summary CSV: {SUMMARY_PATH.name}")


if __name__ == "__main__":
    main()
