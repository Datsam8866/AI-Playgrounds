# -*- coding: utf-8 -*-
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

from expanded_pool_config import CORE_TICKER, DB_PATH, NON_LEVERAGED_EXPANDED_POOL_TICKERS, ROOT
from portfolio_pool_probability_quarterly_walkforward_combined_z_regime import (
    CURRENT_DATE,
    MIN_HISTORY_DAYS,
    OUTPUT_PATH as _UNUSED_OUTPUT_PATH,
    PREDICTIONS_PATH,
    SPY_TICKER,
    START_QUARTER,
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
from portfolio_pool_probability_quarterly_walkforward_combined_z_regime_constrained import (
    RUN_N_SIMS,
    THEME_CAP_BY_REGIME,
    TURNOVER_CAP_BY_REGIME,
    TRACKING_ERROR_CAP_BY_REGIME,
    candidate_grid_for_regime,
    max_theme_weight,
    one_way_turnover,
    tracking_error_vs_voo,
    violation_amount,
)
from portfolio_pool_probability_quarterly_walkforward_voo_core_no_leverage_p5_fixed import (
    available_tickers,
    build_quarter_windows,
)
from portfolio_return_distribution import BOOTSTRAP_BLOCK, estimate_horizon_days
from quarterly_probability_calibration import append_platt_calibration_columns


OUTPUT_PATH = ROOT / "walkforward_portfolio_beta_constrained_voo_alpha.csv"
PREV_OUTPUT_PATH = ROOT / "walkforward_portfolio_beta_constrained.csv"
UNIVERSE = NON_LEVERAGED_EXPANDED_POOL_TICKERS
RECENT_START = pd.Period("2024Q1", freq="Q")

BETA_WINDOW = 252
PORTFOLIO_BETA_CAP = {
    "risk_on": 1.40,
    "caution": 1.20,
    "risk_off": 1.05,
}


def compute_rolling_beta(prices: pd.DataFrame, voo_col: str, window: int = 252) -> pd.DataFrame:
    """
    對每支股票，計算以 VOO 為基準的 rolling beta。
    使用過去 window 個交易日的日報酬做 OLS。
    回傳 DataFrame，index 為 date，columns 為 ticker，值為當日 beta。
    """
    ordered = prices.sort_index().astype(float)
    returns = ordered.pct_change()
    benchmark = returns[voo_col]
    rolling_var = benchmark.rolling(window=window, min_periods=60).var()

    beta = pd.DataFrame(index=ordered.index, columns=ordered.columns, dtype=float)
    beta[voo_col] = 1.0
    for ticker in ordered.columns:
        if ticker == voo_col:
            continue
        cov = returns[ticker].rolling(window=window, min_periods=60).cov(benchmark)
        beta[ticker] = cov / rolling_var
    return beta.replace([np.inf, -np.inf], np.nan)


def _beta_from_trailing_returns(history_returns: pd.DataFrame, tickers: list[str], voo_col: str) -> pd.Series:
    benchmark = history_returns[voo_col]
    benchmark_var = float(benchmark.var(ddof=1))
    values: dict[str, float] = {voo_col: 1.0}
    for ticker in tickers:
        if ticker == voo_col:
            values[ticker] = 1.0
            continue
        if benchmark_var <= 0 or np.isnan(benchmark_var):
            values[ticker] = np.nan
            continue
        values[ticker] = float(history_returns[ticker].cov(benchmark) / benchmark_var)
    return pd.Series(values, dtype=float)


def block_bootstrap_paired(
    asset_returns: pd.DataFrame,
    portfolio_weights: pd.Series,
    benchmark_ticker: str,
    horizon_days: int,
    n_sims: int,
    block_size: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Block bootstrap returning paired (portfolio, benchmark) outcome arrays.
    Both use the same random block draws so the comparison is properly paired."""
    rng = np.random.default_rng(seed)
    all_tickers = list(portfolio_weights.index)
    if benchmark_ticker not in all_tickers:
        all_tickers = [benchmark_ticker] + all_tickers
    matrix = asset_returns[all_tickers].to_numpy()
    port_w = portfolio_weights.reindex(all_tickers).fillna(0.0).to_numpy()
    bench_w = np.zeros(len(all_tickers), dtype=float)
    bench_w[all_tickers.index(benchmark_ticker)] = 1.0

    n_obs = matrix.shape[0]
    max_start = n_obs - block_size
    if max_start < 0:
        raise ValueError("Not enough observations for block bootstrap.")

    port_results = np.empty(n_sims, dtype=float)
    bench_results = np.empty(n_sims, dtype=float)
    for sim_idx in range(n_sims):
        sampled_blocks: list[np.ndarray] = []
        total_days = 0
        while total_days < horizon_days:
            start = int(rng.integers(0, max_start + 1))
            block = matrix[start : start + block_size]
            sampled_blocks.append(block)
            total_days += len(block)
        sampled = np.vstack(sampled_blocks)[:horizon_days]
        port_results[sim_idx] = float(np.prod(1.0 + sampled @ port_w) - 1.0)
        bench_results[sim_idx] = float(np.prod(1.0 + sampled @ bench_w) - 1.0)
    return port_results, bench_results


def compute_portfolio_beta(
    prices: pd.DataFrame,
    beta_frame: pd.DataFrame,
    weights: pd.Series,
    forecast_start: pd.Timestamp,
    voo_col: str,
    window: int,
) -> tuple[float, bool]:
    satellite_weights = weights[weights.index != voo_col].copy()
    if satellite_weights.empty:
        return float(weights.get(voo_col, 0.0)), False

    history_cols = [voo_col] + satellite_weights.index.tolist()
    history_prices = prices.loc[prices.index < forecast_start, history_cols].dropna(how="any")
    history_returns = history_prices.pct_change().dropna(how="any")
    if len(history_returns) < 60:
        return np.nan, True

    prior_beta = beta_frame.loc[beta_frame.index < forecast_start, satellite_weights.index]
    if prior_beta.empty:
        latest_beta = _beta_from_trailing_returns(history_returns.tail(window), satellite_weights.index.tolist(), voo_col)
        latest_beta = latest_beta.reindex(satellite_weights.index)
    else:
        latest_beta = prior_beta.iloc[-1].copy()
        if latest_beta.isna().any():
            repaired = _beta_from_trailing_returns(history_returns.tail(window), satellite_weights.index.tolist(), voo_col)
            latest_beta = latest_beta.fillna(repaired.reindex(satellite_weights.index))

    if latest_beta.isna().any():
        return np.nan, True

    voo_weight = float(weights.get(voo_col, 0.0))
    satellite_beta = float((satellite_weights * latest_beta.reindex(satellite_weights.index)).sum())
    portfolio_beta = voo_weight + satellite_beta
    return portfolio_beta, False


def evaluate_window(
    period_label: str,
    snapshot: pd.DataFrame,
    snapshot_date: pd.Timestamp,
    prices: pd.DataFrame,
    regime: pd.DataFrame,
    beta_frame: pd.DataFrame,
    forecast_start: pd.Timestamp,
    forecast_end: pd.Timestamp,
    previous_weights: pd.Series | None,
    include_realized: bool,
) -> tuple[dict[str, float | int | str | bool], pd.Series]:
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
    beta_cap = PORTFOLIO_BETA_CAP[regime_label]

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

                    port_outcomes, voo_outcomes = block_bootstrap_paired(
                        asset_returns=history_returns,
                        portfolio_weights=active_weights.sort_index(),
                        benchmark_ticker=CORE_TICKER,
                        horizon_days=horizon_days,
                        n_sims=RUN_N_SIMS,
                        block_size=BOOTSTRAP_BLOCK,
                        seed=seed,
                    )
                    p_beat_voo = float(np.mean(port_outcomes > voo_outcomes))
                    p_gt_5 = float(np.mean(port_outcomes > 0.05))
                    p_lt_0 = float(np.mean(port_outcomes < 0.0))
                    hist_sharpe = compute_historical_sharpe(history_returns, active_weights.sort_index())
                    turnover = one_way_turnover(active_weights.sort_index(), previous_weights)
                    tracking_error = tracking_error_vs_voo(history_returns, active_weights.sort_index())
                    theme_weight = max_theme_weight(active_weights.sort_index())

                    turnover_violation = violation_amount(turnover, turnover_cap)
                    tracking_error_violation = violation_amount(tracking_error, tracking_error_cap)
                    theme_violation = violation_amount(theme_weight, theme_cap)
                    total_violation = turnover_violation + tracking_error_violation + theme_violation
                    constraints_feasible = total_violation <= 1e-12

                    portfolio_beta, beta_skipped = compute_portfolio_beta(
                        prices=prices,
                        beta_frame=beta_frame,
                        weights=active_weights.sort_index(),
                        forecast_start=forecast_start,
                        voo_col=CORE_TICKER,
                        window=BETA_WINDOW,
                    )
                    beta_violation = 0.0 if beta_skipped or pd.isna(portfolio_beta) else violation_amount(portfolio_beta, beta_cap)
                    beta_feasible = True if beta_skipped else portfolio_beta <= beta_cap

                    row: dict[str, float | int | str | bool] = {
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
                        "pred_mean": float(np.mean(port_outcomes)),
                        "p_beat_voo": p_beat_voo,
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
                        "constraints_feasible": bool(constraints_feasible),
                        "total_violation": total_violation,
                        "turnover_violation": turnover_violation,
                        "tracking_error_violation": tracking_error_violation,
                        "theme_violation": theme_violation,
                        "portfolio_beta": portfolio_beta,
                        "beta_cap": beta_cap,
                        "beta_feasible": bool(beta_feasible),
                        "beta_violation": beta_violation,
                    }
                    rows.append(row)
                    cache[(voo_core_weight, top_k, satellite_cap, satellite_mode)] = (active_weights, selected)

    if not rows:
        raise ValueError("No feasible configurations.")

    results = pd.DataFrame(rows)
    results["score_combined_z"] = (
        zscore(results["p_beat_voo"])
        - zscore(results["p_lt_0"])
        + 0.3 * zscore(results["historical_sharpe"].fillna(0.0))
    )
    results["fully_feasible"] = results["constraints_feasible"] & results["beta_feasible"]
    feasible = results[results["fully_feasible"]].copy()
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
                "beta_violation",
                "tracking_error_violation",
                "theme_violation",
                "score_combined_z",
                "historical_sharpe",
                "p_gt_5",
                "pred_mean",
            ],
            ascending=[True, True, True, True, True, False, False, False, False],
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


def compute_voo_quarterly_returns(report: pd.DataFrame, prices: pd.DataFrame) -> pd.Series:
    returns: dict[str, float] = {}
    voo_weights = pd.Series({CORE_TICKER: 1.0}, dtype=float)
    realized = report[report["realized_return"].notna()].copy()
    for row in realized.itertuples(index=False):
        forecast_start = pd.Timestamp(row.forecast_start)
        forecast_end = pd.Timestamp(row.forecast_end)
        voo_return, _ = realized_buyandhold_return(prices, voo_weights, forecast_start, forecast_end)
        returns[str(row.period)] = voo_return
    return pd.Series(returns, dtype=float, name="voo_return")


def calibrate_p_beat_voo(report: pd.DataFrame, voo_returns: pd.Series) -> pd.DataFrame:
    """Expanding Platt walk-forward calibration for p_beat_voo.
    event_beat_voo = (realized_return > voo_return) for each historical quarter."""
    from sklearn.linear_model import LogisticRegression

    EPS = 1e-6
    MIN_TRAIN = 8

    out = report.copy()
    out["voo_return_for_calib"] = out["period"].map(voo_returns)
    realized_mask = out["realized_return"].notna()
    out["event_beat_voo"] = np.where(
        realized_mask,
        (out["realized_return"] > out["voo_return_for_calib"]).astype(float),
        np.nan,
    )

    realized = out[realized_mask].copy().reset_index(drop=True)
    calibrated: dict[str, float] = {}

    for idx in range(len(realized)):
        row = realized.iloc[idx]
        raw_prob = float(row["raw_p_beat_voo"])
        if idx < MIN_TRAIN:
            calibrated[str(row["period"])] = raw_prob
            continue
        x_train = np.clip(realized.iloc[:idx]["raw_p_beat_voo"].to_numpy(dtype=float), EPS, 1 - EPS)
        y_train = realized.iloc[:idx]["event_beat_voo"].astype(int).to_numpy()
        if len(np.unique(y_train)) < 2:
            calibrated[str(row["period"])] = raw_prob
            continue
        model = LogisticRegression(solver="lbfgs")
        model.fit(x_train.reshape(-1, 1), y_train)
        cal = float(model.predict_proba(np.array([[np.clip(raw_prob, EPS, 1 - EPS)]]))[0, 1])
        calibrated[str(row["period"])] = float(np.clip(cal, EPS, 1 - EPS))

    current_rows = out[~realized_mask]
    if not current_rows.empty:
        raw_prob = float(current_rows.iloc[0]["raw_p_beat_voo"])
        x_train = np.clip(realized["raw_p_beat_voo"].to_numpy(dtype=float), EPS, 1 - EPS)
        y_train = realized["event_beat_voo"].astype(int).to_numpy()
        if len(np.unique(y_train)) >= 2:
            model = LogisticRegression(solver="lbfgs")
            model.fit(x_train.reshape(-1, 1), y_train)
            cal = float(model.predict_proba(np.array([[np.clip(raw_prob, EPS, 1 - EPS)]]))[0, 1])
            calibrated[str(current_rows.iloc[0]["period"])] = float(np.clip(cal, EPS, 1 - EPS))
        else:
            calibrated[str(current_rows.iloc[0]["period"])] = raw_prob

    out["p_beat_voo_calibrated"] = out["period"].map(calibrated).fillna(out["raw_p_beat_voo"])
    out = out.drop(columns=["voo_return_for_calib", "event_beat_voo"])
    return out


def performance_summary(frame: pd.DataFrame) -> dict[str, float]:
    returns = frame["realized_return"].astype(float)
    voo_returns = frame["voo_return"].astype(float)
    if returns.empty:
        return {
            "quarters": 0,
            "sharpe": np.nan,
            "cagr": np.nan,
            "max_dd": np.nan,
            "positive_excess_rate": np.nan,
        }

    wealth = (1.0 + returns).cumprod()
    sharpe = np.nan
    std = float(returns.std(ddof=1))
    if len(returns) >= 2 and std > 0 and not np.isnan(std):
        sharpe = float(np.sqrt(4.0) * returns.mean() / std)

    cagr = float(wealth.iloc[-1] ** (4.0 / len(returns)) - 1.0)
    drawdown = wealth / wealth.cummax() - 1.0
    positive_excess_rate = float((returns > voo_returns).mean())
    return {
        "quarters": int(len(frame)),
        "sharpe": sharpe,
        "cagr": cagr,
        "max_dd": float(drawdown.min()),
        "positive_excess_rate": positive_excess_rate,
    }


def format_pct(value: float) -> str:
    return "" if pd.isna(value) else f"{value * 100:.2f}%"


def format_num(value: float) -> str:
    return "" if pd.isna(value) else f"{value:.3f}"


def summarize_block(name: str, frame: pd.DataFrame) -> dict[str, str]:
    stats = performance_summary(frame)
    return {
        "sample": name,
        "quarters": str(stats["quarters"]),
        "sharpe": format_num(stats["sharpe"]),
        "cagr": format_pct(stats["cagr"]),
        "max_dd": format_pct(stats["max_dd"]),
        "positive_excess_rate": format_pct(stats["positive_excess_rate"]),
    }


def main() -> None:
    _ = _UNUSED_OUTPUT_PATH
    predictions = load_predictions()
    with sqlite3.connect(DB_PATH) as connection:
        prices = load_prices_with_spy(connection)
        macro = load_macro_history(connection)

    beta_frame = compute_rolling_beta(prices[sorted(set(UNIVERSE + [SPY_TICKER]))], CORE_TICKER, window=BETA_WINDOW)
    regime = build_regime_frame(prices, macro)
    max_price_date = pd.Timestamp(prices.index.max())
    quarter_windows = build_quarter_windows(prices[NON_LEVERAGED_EXPANDED_POOL_TICKERS])

    rows: list[dict[str, float | int | str | bool]] = []
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
                beta_frame=beta_frame,
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
                beta_frame=beta_frame,
                forecast_start=current_forecast_start,
                forecast_end=current_end,
                previous_weights=previous_weights,
                include_realized=False,
            )
            rows.append(current_row)
        except ValueError:
            pass

    report = pd.DataFrame(rows)
    report["raw_p_beat_voo"] = report["p_beat_voo"]
    report = append_platt_calibration_columns(report)
    report["p_gt_5"] = report["p_gt_5_calibrated"]
    report["p_lt_0"] = report["p_lt_0_calibrated"]
    new_realized = report[report["realized_return"].notna()].copy()
    voo_returns = compute_voo_quarterly_returns(new_realized, prices)
    report = calibrate_p_beat_voo(report, voo_returns)
    report["p_beat_voo"] = report["p_beat_voo_calibrated"]
    report.to_csv(OUTPUT_PATH, index=False)

    if not PREV_OUTPUT_PATH.exists():
        print(f"Saved CSV: {OUTPUT_PATH.name}")
        print(f"(Skipping comparison: {PREV_OUTPUT_PATH.name} not found)")
        return

    prev_report = pd.read_csv(PREV_OUTPUT_PATH)
    new_realized = report[report["realized_return"].notna()].copy()
    prev_realized = prev_report[prev_report["realized_return"].notna()].copy()
    comparison = new_realized.merge(
        prev_realized[["period", "realized_return", "weights", "p_gt_5"]].rename(
            columns={"realized_return": "prev_realized", "weights": "prev_weights", "p_gt_5": "prev_p_gt_5"}
        ),
        on="period",
        how="left",
    )
    comparison["voo_return"] = comparison["period"].map(voo_returns)
    comparison["config_changed"] = comparison["weights"] != comparison["prev_weights"]
    comparison = comparison.rename(columns={"realized_return": "new_realized"})
    comparison = comparison[
        [
            "period",
            "prev_realized",
            "new_realized",
            "voo_return",
            "p_beat_voo",
            "prev_p_gt_5",
            "portfolio_beta",
            "beta_feasible",
            "config_changed",
        ]
    ].copy()

    comparison_display = comparison.copy()
    for column in ["prev_realized", "new_realized", "voo_return", "p_beat_voo", "prev_p_gt_5"]:
        comparison_display[column] = comparison_display[column].map(format_pct)
    comparison_display["portfolio_beta"] = comparison_display["portfolio_beta"].map(format_num)

    overall_new = new_realized.assign(voo_return=new_realized["period"].map(voo_returns))
    overall_prev = prev_realized.assign(voo_return=prev_realized["period"].map(voo_returns))
    new_periods = pd.PeriodIndex(overall_new["period"], freq="Q")
    prev_periods = pd.PeriodIndex(overall_prev["period"], freq="Q")
    recent_new = overall_new.loc[new_periods >= RECENT_START].copy()
    recent_prev = overall_prev.loc[prev_periods >= RECENT_START].copy()

    summary_rows = pd.DataFrame(
        [
            summarize_block("new_overall", overall_new),
            summarize_block("prev_overall", overall_prev),
            summarize_block("new_2024plus", recent_new),
            summarize_block("prev_2024plus", recent_prev),
        ]
    )

    overall_new_stats = performance_summary(overall_new)
    overall_prev_stats = performance_summary(overall_prev)
    recent_new_stats = performance_summary(recent_new)
    recent_prev_stats = performance_summary(recent_prev)
    improvement_2024 = pd.DataFrame(
        [
            {
                "window": "2024Q1_2026Q1",
                "delta_sharpe": format_num(recent_new_stats["sharpe"] - recent_prev_stats["sharpe"]),
                "delta_cagr": format_pct(recent_new_stats["cagr"] - recent_prev_stats["cagr"]),
                "delta_max_dd": format_pct(recent_new_stats["max_dd"] - recent_prev_stats["max_dd"]),
                "delta_positive_excess_rate": format_pct(
                    recent_new_stats["positive_excess_rate"] - recent_prev_stats["positive_excess_rate"]
                ),
            }
        ]
    )

    changed_count = int(comparison["config_changed"].sum())

    print("Quarterly comparison: previous beta-constrained vs VOO-alpha objective vs VOO")
    print(comparison_display.to_string(index=False))
    print()
    print("Performance summary")
    print(summary_rows.to_string(index=False))
    print()
    print("2024Q1-2026Q1 improvement (VOO-alpha objective - previous beta-constrained)")
    print(improvement_2024.to_string(index=False))
    print()
    print(f"Quarters with configuration changes vs previous beta-constrained objective: {changed_count}")
    print(f"Saved CSV: {OUTPUT_PATH.name}")


if __name__ == "__main__":
    main()
