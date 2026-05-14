# -*- coding: utf-8 -*-
"""
generate_portfolio_dashboard.py
Storytelling-first Portfolio Dashboard
"""
import sqlite3, sys, json
from datetime import date
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import numpy as np

ROOT    = Path(__file__).parent
DB_PATH = ROOT / "portfolio.sqlite"
WF_CSV  = ROOT / "Wall Street" / "walkforward_portfolio_beta_constrained_voo_alpha.csv"
WS_DB   = ROOT / "Wall Street" / "stock_forecast.sqlite"
TW_WF   = ROOT / "TWSE" / "tw_0050_walkforward.csv"
OUT     = ROOT / "portfolio_dashboard.html"
TODAY   = str(date.today())
QQQ_BENCHMARK = "QQQ"
BOOTSTRAP_SIMS = 3000
BOOTSTRAP_BLOCK = 20
MIN_CALIBRATION_ROWS = 8

# ── helpers ───────────────────────────────────────────────────────────────────

def fmt(v, prefix="", suffix="", d=2):
    return "—" if v is None else f"{prefix}{v:,.{d}f}{suffix}"

def pc(v):
    return "pos" if (v or 0) >= 0 else "neg"

def parse_weights(text):
    weights = {}
    for part in str(text).split(","):
        tok = part.strip().rsplit(" ", 1)
        if len(tok) == 2:
            try:
                ticker, weight = tok[0].strip(), float(tok[1].replace("%", "")) / 100
            except ValueError:
                continue
            weights[ticker] = weights.get(ticker, 0) + weight
    return pd.Series(weights, dtype=float)

def estimate_horizon_days(start, end):
    return max(int(round(252 * (end - start).days / 365.25)), 1)

def block_bootstrap_paired(asset_returns, portfolio_weights, benchmark_ticker, horizon_days, seed):
    tickers = list(portfolio_weights.index)
    if benchmark_ticker not in tickers:
        tickers = [benchmark_ticker] + tickers
    matrix = asset_returns[tickers].to_numpy(dtype=float)
    port_w = portfolio_weights.reindex(tickers).fillna(0.0).to_numpy(dtype=float)
    bench_w = np.zeros(len(tickers), dtype=float)
    bench_w[tickers.index(benchmark_ticker)] = 1.0

    max_start = matrix.shape[0] - BOOTSTRAP_BLOCK
    if max_start < 0:
        return np.array([]), np.array([])

    rng = np.random.default_rng(seed)
    port_results = np.empty(BOOTSTRAP_SIMS, dtype=float)
    bench_results = np.empty(BOOTSTRAP_SIMS, dtype=float)
    for idx in range(BOOTSTRAP_SIMS):
        blocks = []
        days = 0
        while days < horizon_days:
            start = int(rng.integers(0, max_start + 1))
            block = matrix[start : start + BOOTSTRAP_BLOCK]
            blocks.append(block)
            days += len(block)
        sampled = np.vstack(blocks)[:horizon_days]
        port_results[idx] = float(np.prod(1.0 + sampled @ port_w) - 1.0)
        bench_results[idx] = float(np.prod(1.0 + sampled @ bench_w) - 1.0)
    return port_results, bench_results

def load_wall_street_prices(tickers):
    if not WS_DB.exists():
        return pd.DataFrame()
    tickers = sorted(set(tickers))
    placeholders = ",".join(["?"] * len(tickers))
    with sqlite3.connect(WS_DB) as conn:
        rows = pd.read_sql_query(
            f"""
            SELECT date, ticker, adj_close
            FROM price_history
            WHERE ticker IN ({placeholders})
            ORDER BY date
            """,
            conn,
            params=tickers,
        )
    if rows.empty:
        return pd.DataFrame()
    rows["date"] = pd.to_datetime(rows["date"])
    return rows.pivot(index="date", columns="ticker", values="adj_close").sort_index()

def merge_benchmark_prices(prices, benchmark_ticker):
    if benchmark_ticker in prices.columns:
        return prices
    benchmark_prices = load_wall_street_prices([benchmark_ticker])
    if benchmark_prices.empty or benchmark_ticker not in benchmark_prices.columns:
        return prices
    return prices.join(benchmark_prices[[benchmark_ticker]], how="left")

def realized_single_ticker_return(prices, ticker, start, end):
    window = prices.loc[(prices.index >= start) & (prices.index <= min(end, prices.index.max())), [ticker]].dropna()
    if len(window) < 2:
        return np.nan
    return float(window.iloc[-1, 0] / window.iloc[0, 0] - 1.0)

def calibrate_probability(raw_probs, events, current_raw):
    try:
        from sklearn.linear_model import LogisticRegression
    except Exception:
        return current_raw

    raw_probs = np.asarray(raw_probs, dtype=float)
    events = np.asarray(events, dtype=int)
    if len(raw_probs) < MIN_CALIBRATION_ROWS or len(np.unique(events)) < 2:
        return current_raw
    eps = 1e-6
    model = LogisticRegression(solver="lbfgs")
    model.fit(np.clip(raw_probs, eps, 1 - eps).reshape(-1, 1), events)
    return float(model.predict_proba(np.array([[np.clip(current_raw, eps, 1 - eps)]]))[0, 1])

def compute_p_beat_qqq(frame):
    all_tickers = {QQQ_BENCHMARK}
    parsed_weights = []
    for weights_text in frame["weights"]:
        weights = parse_weights(weights_text)
        parsed_weights.append(weights)
        all_tickers.update(weights.index.tolist())

    prices = load_wall_street_prices(all_tickers)
    if prices.empty or QQQ_BENCHMARK not in prices.columns:
        frame["raw_p_beat_qqq"] = np.nan
        frame["p_beat_qqq_calibrated"] = np.nan
        return frame

    out = frame.copy()
    raw_values = []
    qqq_returns = []
    for idx, row in out.reset_index(drop=True).iterrows():
        weights = parsed_weights[idx]
        start = pd.Timestamp(row["forecast_start"])
        end = pd.Timestamp(row["forecast_end"])
        period_dates = prices.loc[(prices.index >= start) & (prices.index <= min(end, prices.index.max()))].index
        if pd.notna(row.get("realized_return")) and len(period_dates) >= 2:
            horizon_days = max(len(period_dates) - 1, 1)
        else:
            horizon_days = estimate_horizon_days(start, end)

        active_tickers = sorted(set(weights.index.tolist() + [QQQ_BENCHMARK]))
        if any(ticker not in prices.columns for ticker in active_tickers):
            raw_values.append(np.nan)
            qqq_returns.append(realized_single_ticker_return(prices, QQQ_BENCHMARK, start, end))
            continue
        history_prices = prices.loc[prices.index < start, active_tickers].dropna(how="any")
        history_returns = history_prices.pct_change().dropna(how="any")
        if len(history_returns) < BOOTSTRAP_BLOCK:
            raw_values.append(np.nan)
        else:
            seed = int(start.year * 10 + start.quarter)
            port, bench = block_bootstrap_paired(history_returns, weights.sort_index(), QQQ_BENCHMARK, horizon_days, seed)
            raw_values.append(float(np.mean(port > bench)) if len(port) else np.nan)
        qqq_returns.append(realized_single_ticker_return(prices, QQQ_BENCHMARK, start, end))

    out["raw_p_beat_qqq"] = raw_values
    out["qqq_return_for_calib"] = qqq_returns
    realized = out[out["realized_return"].notna() & out["raw_p_beat_qqq"].notna() & out["qqq_return_for_calib"].notna()].copy()
    realized["event_beat_qqq"] = (realized["realized_return"] > realized["qqq_return_for_calib"]).astype(int)

    calibrated = {}
    for idx in range(len(realized)):
        row = realized.iloc[idx]
        raw_prob = float(row["raw_p_beat_qqq"])
        if idx < MIN_CALIBRATION_ROWS:
            calibrated[str(row["period"])] = raw_prob
            continue
        train = realized.iloc[:idx]
        calibrated[str(row["period"])] = calibrate_probability(
            train["raw_p_beat_qqq"].to_numpy(dtype=float),
            train["event_beat_qqq"].to_numpy(dtype=int),
            raw_prob,
        )

    current = out[out["realized_return"].isna() & out["raw_p_beat_qqq"].notna()]
    if not current.empty:
        raw_prob = float(current.iloc[-1]["raw_p_beat_qqq"])
        calibrated[str(current.iloc[-1]["period"])] = calibrate_probability(
            realized["raw_p_beat_qqq"].to_numpy(dtype=float),
            realized["event_beat_qqq"].to_numpy(dtype=int),
            raw_prob,
        )

    out["p_beat_qqq_calibrated"] = out["period"].map(calibrated).fillna(out["raw_p_beat_qqq"])
    return out.drop(columns=["qqq_return_for_calib"], errors="ignore")

def compute_report_benchmark_calibration(frame, benchmark_ticker):
    all_tickers = {benchmark_ticker}
    parsed_weights = []
    for weights_text in frame["weights"]:
        weights = parse_weights(weights_text)
        parsed_weights.append(weights)
        all_tickers.update(weights.index.tolist())

    prices = load_wall_street_prices(all_tickers)
    if prices.empty or benchmark_ticker not in prices.columns:
        return [], [], np.nan

    raw_values = []
    benchmark_returns = []
    for idx, row in frame.reset_index(drop=True).iterrows():
        weights = parsed_weights[idx]
        start = pd.Timestamp(row["forecast_start"])
        end = pd.Timestamp(row["forecast_end"])
        period_dates = prices.loc[(prices.index >= start) & (prices.index <= min(end, prices.index.max()))].index
        if pd.notna(row.get("realized_return")) and len(period_dates) >= 2:
            horizon_days = max(len(period_dates) - 1, 1)
        else:
            horizon_days = estimate_horizon_days(start, end)

        active_tickers = sorted(set(weights.index.tolist() + [benchmark_ticker]))
        if any(ticker not in prices.columns for ticker in active_tickers):
            raw_values.append(np.nan)
            benchmark_returns.append(realized_single_ticker_return(prices, benchmark_ticker, start, end))
            continue
        history_prices = prices.loc[prices.index < start, active_tickers].dropna(how="any")
        history_returns = history_prices.pct_change().dropna(how="any")
        if len(history_returns) < BOOTSTRAP_BLOCK:
            raw_values.append(np.nan)
        else:
            seed = int(start.year * 10 + start.quarter)
            port, bench = block_bootstrap_paired(history_returns, weights.sort_index(), benchmark_ticker, horizon_days, seed)
            raw_values.append(float(np.mean(port > bench)) if len(port) else np.nan)
        benchmark_returns.append(realized_single_ticker_return(prices, benchmark_ticker, start, end))

    work = frame.copy()
    work["raw_p_beat_benchmark"] = raw_values
    work["benchmark_return_for_calib"] = benchmark_returns
    realized = work[
        work["realized_return"].notna()
        & work["raw_p_beat_benchmark"].notna()
        & work["benchmark_return_for_calib"].notna()
    ].copy()
    if realized.empty:
        return [], [], np.nan
    events = (realized["realized_return"] > realized["benchmark_return_for_calib"]).astype(int).to_numpy()
    return realized["raw_p_beat_benchmark"].to_numpy(dtype=float), events, raw_values[-1]

def local_zscore(values):
    series = pd.Series(values, dtype=float)
    std = series.std(ddof=0)
    if std == 0 or pd.isna(std):
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - series.mean()) / std

def compute_current_benchmark_scenario(frame, benchmark_key, benchmark_ticker, display_label):
    sys.path.insert(0, str((ROOT / "Wall Street").resolve()))
    import walkforward_portfolio_beta_constrained_voo_alpha as model
    from expanded_pool_config import CORE_TICKER, DB_PATH as WS_MODEL_DB_PATH
    from portfolio_pool_probability_quarterly_walkforward_combined_z_regime import (
        CURRENT_DATE,
        build_regime_frame,
        compute_historical_sharpe,
        load_macro_history,
        load_predictions,
        load_prices_with_spy,
        pick_latest_inference_snapshot,
        pick_regime_on_or_before,
    )
    from portfolio_pool_probability_quarterly_walkforward_combined_z_regime_constrained import (
        THEME_CAP_BY_REGIME,
        TRACKING_ERROR_CAP_BY_REGIME,
        TURNOVER_CAP_BY_REGIME,
        candidate_grid_for_regime,
        max_theme_weight,
        one_way_turnover,
        tracking_error_vs_voo,
        violation_amount,
    )
    from portfolio_pool_probability_quarterly_walkforward_voo_core_no_leverage_p5_fixed import available_tickers

    predictions = load_predictions()
    with sqlite3.connect(WS_MODEL_DB_PATH) as conn:
        prices = load_prices_with_spy(conn)
        macro = load_macro_history(conn)
    prices = merge_benchmark_prices(prices, benchmark_ticker)

    beta_frame = model.compute_rolling_beta(prices[sorted(set(model.UNIVERSE + [model.SPY_TICKER]))], CORE_TICKER, window=model.BETA_WINDOW)
    regime = build_regime_frame(prices, macro)
    snapshot, snapshot_date = pick_latest_inference_snapshot(predictions, CURRENT_DATE.normalize())
    forecast_start = snapshot_date
    forecast_end = CURRENT_DATE.to_period("Q").end_time.normalize()
    horizon_days = estimate_horizon_days(forecast_start, forecast_end)
    regime_row = pick_regime_on_or_before(regime, forecast_start)
    regime_label = str(regime_row["regime_label"])
    previous_weights = parse_weights(frame[frame["realized_return"].notna()].iloc[-1]["weights"]).sort_index()

    turnover_cap = TURNOVER_CAP_BY_REGIME[regime_label]
    tracking_error_cap = TRACKING_ERROR_CAP_BY_REGIME[regime_label]
    theme_cap = THEME_CAP_BY_REGIME[regime_label]
    beta_cap = model.PORTFOLIO_BETA_CAP[regime_label]
    available = available_tickers(prices, forecast_start)
    voo_candidates, topk_candidates, cap_candidates, mode_candidates = candidate_grid_for_regime(regime_label)
    seed = int(forecast_start.year * 10 + forecast_start.quarter)
    rows = []

    for voo_core_weight in voo_candidates:
        for top_k in topk_candidates:
            for satellite_cap in cap_candidates:
                for satellite_mode in mode_candidates:
                    try:
                        weights, selected = model.build_voo_core_weights(
                            snapshot, available, voo_core_weight, top_k, satellite_cap, satellite_mode
                        )
                    except ValueError:
                        continue
                    active = weights[weights > 0].copy().sort_index()
                    cols = sorted(set(active.index.tolist() + [benchmark_ticker]))
                    if any(ticker not in prices.columns for ticker in cols):
                        continue
                    history_prices = prices.loc[prices.index < forecast_start, cols].dropna(how="any")
                    history_returns = history_prices.pct_change().dropna(how="any")
                    if len(history_returns) < 60:
                        continue
                    port, bench = block_bootstrap_paired(history_returns, active, benchmark_ticker, horizon_days, seed)
                    port_voo, voo = block_bootstrap_paired(history_returns, active, CORE_TICKER, horizon_days, seed)
                    active_returns = history_returns[active.index]
                    turnover = one_way_turnover(active, previous_weights)
                    tracking_error = tracking_error_vs_voo(active_returns, active)
                    theme_weight = max_theme_weight(active)
                    total_violation = (
                        violation_amount(turnover, turnover_cap)
                        + violation_amount(tracking_error, tracking_error_cap)
                        + violation_amount(theme_weight, theme_cap)
                    )
                    beta, beta_skipped = model.compute_portfolio_beta(prices, beta_frame, active, forecast_start, CORE_TICKER, model.BETA_WINDOW)
                    beta_violation = 0.0 if beta_skipped or pd.isna(beta) else violation_amount(beta, beta_cap)
                    beta_feasible = True if beta_skipped else beta <= beta_cap
                    rows.append({
                        "weights": active.to_dict(),
                        "weights_text": ", ".join(f"{ticker} {weight * 100:.1f}%" for ticker, weight in active.items()),
                        "selected_satellite": ",".join(selected),
                        "p_beat": float(np.mean(port > bench)),
                        "p_beat_voo": float(np.mean(port_voo > voo)),
                        "p_gt_5": float(np.mean(port > 0.05)),
                        "p_lt_0": float(np.mean(port < 0.0)),
                        "pred_mean": float(np.mean(port)),
                        "historical_sharpe": compute_historical_sharpe(active_returns, active),
                        "turnover": turnover,
                        "tracking_error": tracking_error,
                        "theme_weight": theme_weight,
                        "portfolio_beta": beta,
                        "beta_cap": beta_cap,
                        "beta_feasible": bool(beta_feasible),
                        "total_violation": total_violation,
                        "turnover_violation": violation_amount(turnover, turnover_cap),
                        "tracking_error_violation": violation_amount(tracking_error, tracking_error_cap),
                        "theme_violation": violation_amount(theme_weight, theme_cap),
                        "beta_violation": beta_violation,
                        "fully_feasible": bool(total_violation <= 1e-12 and beta_feasible),
                    })

    results = pd.DataFrame(rows)
    if results.empty:
        return None
    results["score"] = local_zscore(results["p_beat"]) - local_zscore(results["p_lt_0"]) + 0.3 * local_zscore(results["historical_sharpe"].fillna(0.0))
    feasible = results[results["fully_feasible"]].copy()
    if not feasible.empty:
        best = feasible.sort_values(["score", "historical_sharpe", "p_gt_5", "pred_mean"], ascending=[False, False, False, False]).iloc[0]
        stage = "feasible"
    else:
        best = results.sort_values(
            ["turnover_violation", "total_violation", "beta_violation", "tracking_error_violation", "theme_violation", "score", "historical_sharpe", "p_gt_5", "pred_mean"],
            ascending=[True, True, True, True, True, False, False, False, False],
        ).iloc[0]
        stage = "fallback_turnover_first"

    strong_candidates = results[(results["p_beat"] >= 0.70) & (results["fully_feasible"])].copy()
    has_strong_candidate = not strong_candidates.empty
    if has_strong_candidate:
        strong = strong_candidates.sort_values(
            ["p_beat", "score", "historical_sharpe", "p_gt_5"],
            ascending=[False, False, False, False],
        ).iloc[0]
        target_status = "Strong Candidate"
        target_detail = f"{strong['p_beat'] * 100:.1f}% ｜ {strong['weights_text']}"
    else:
        target_status = "No Strong Candidate"
        target_detail = "找不到同時達到 70% 且 fully_feasible 的 portfolio"

    raw_probs, events, _current_existing_raw = compute_report_benchmark_calibration(frame, benchmark_ticker)
    calibrated = calibrate_probability(raw_probs, events, float(best["p_beat"]))
    return {
        "key": benchmark_key,
        "label": display_label,
        "ticker": benchmark_ticker,
        "pBeat": calibrated,
        "rawPBeat": float(best["p_beat"]),
        "pBeatVoo": float(best["p_beat_voo"]),
        "pGt5": float(best["p_gt_5"]),
        "pLt0": float(best["p_lt_0"]),
        "sharpe": float(best["historical_sharpe"]),
        "stage": stage,
        "weights": best["weights"],
        "weightsText": best["weights_text"],
        "selectedSatellite": best["selected_satellite"],
        "portfolioBeta": None if pd.isna(best["portfolio_beta"]) else float(best["portfolio_beta"]),
        "betaCap": float(best["beta_cap"]),
        "fullyFeasible": bool(best["fully_feasible"]),
        "hasStrongCandidate": has_strong_candidate,
        "targetStatus": target_status,
        "targetDetail": target_detail,
    }

def build_benchmark_scenarios(frame):
    scenarios = {}
    for key, ticker, label in [("VOO", "VOO", "VOO"), ("QQQ", "QQQ", "QQQ"), ("SOXX", "SOXX", "SOXX"), ("VT", "VT", "VT")]:
        scenario = compute_current_benchmark_scenario(frame, key, ticker, label)
        if scenario:
            scenarios[key] = scenario
        elif key == "VT":
            scenarios[key] = {
                "key": "VT",
                "label": "VT",
                "ticker": "VT",
                "unavailable": True,
                "pBeat": None,
                "rawPBeat": None,
                "pBeatVoo": None,
                "pGt5": None,
                "pLt0": None,
                "sharpe": None,
                "stage": "unavailable",
                "weights": {},
                "weightsText": "",
                "selectedSatellite": "",
                "portfolioBeta": None,
                "betaCap": None,
                "fullyFeasible": False,
                "hasStrongCandidate": False,
                "targetStatus": "Unavailable",
                "targetDetail": "價格庫目前沒有 VT 歷史資料，無法計算 benchmark",
                "signal": "Unavailable",
                "signalClass": "weak",
            }
    for scenario in scenarios.values():
        p_beat = scenario.get("pBeat")
        fully_feasible = bool(scenario.get("fullyFeasible"))
        if p_beat is not None and p_beat >= 0.70 and fully_feasible:
            scenario["signal"] = "Strong"
            scenario["signalClass"] = "strong"
        elif p_beat is not None and p_beat >= 0.60:
            scenario["signal"] = "Watch"
            scenario["signalClass"] = "watch"
        else:
            scenario["signal"] = "Weak"
            scenario["signalClass"] = "weak"
    return scenarios

# ── data loading ──────────────────────────────────────────────────────────────

def actual_signal_for(p_beat):
    if p_beat is not None and p_beat >= 0.70:
        return "Strong", "strong"
    if p_beat is not None and p_beat >= 0.60:
        return "Watch", "watch"
    return "Weak", "weak"

def annualized_sharpe(daily_returns):
    daily_returns = pd.Series(daily_returns, dtype=float).dropna()
    std = daily_returns.std(ddof=1)
    if len(daily_returns) < 2 or std == 0 or pd.isna(std):
        return np.nan
    return float(np.sqrt(252.0) * daily_returns.mean() / std)

def compute_actual_portfolio_beta(history_returns, portfolio_returns, benchmark_ticker="VOO"):
    if benchmark_ticker not in history_returns.columns:
        return np.nan
    joined = pd.concat(
        [portfolio_returns.rename("portfolio"), history_returns[benchmark_ticker].rename("benchmark")],
        axis=1,
    ).dropna().tail(252)
    if len(joined) < 60:
        return np.nan
    benchmark_var = float(joined["benchmark"].var(ddof=1))
    if benchmark_var <= 0 or pd.isna(benchmark_var):
        return np.nan
    return float(joined["portfolio"].cov(joined["benchmark"]) / benchmark_var)

def compute_actual_benchmark_scenarios(frame, actual_weights):
    current_rows = frame[frame["period"].str.contains("current", na=False)]
    if current_rows.empty or not actual_weights:
        return {}

    current = current_rows.iloc[-1]
    forecast_start = pd.Timestamp(current["forecast_start"])
    forecast_end = pd.Timestamp(current["forecast_end"])
    horizon_days = estimate_horizon_days(forecast_start, forecast_end)
    active_weights = pd.Series(actual_weights, dtype=float)
    active_weights = active_weights[active_weights > 0].sort_index()
    beta_cap = current.get("beta_cap", np.nan)

    benchmark_specs = [("VOO", "VOO", "VOO"), ("QQQ", "QQQ", "QQQ"), ("SOXX", "SOXX", "SOXX"), ("VT", "VT", "VT")]
    all_tickers = sorted(set(active_weights.index.tolist() + [ticker for _, ticker, _ in benchmark_specs]))
    prices = load_wall_street_prices(all_tickers)
    scenarios = {}

    for key, benchmark_ticker, label in benchmark_specs:
        required = sorted(set(active_weights.index.tolist() + [benchmark_ticker]))
        missing = [ticker for ticker in required if ticker not in prices.columns]
        if missing:
            signal, signal_class = actual_signal_for(None)
            scenarios[key] = {
                "key": key,
                "label": label,
                "ticker": benchmark_ticker,
                "unavailable": True,
                "pBeat": None,
                "rawPBeat": None,
                "pGt5": None,
                "pLt0": None,
                "sharpe": None,
                "portfolioBeta": None,
                "betaCap": None if pd.isna(beta_cap) else float(beta_cap),
                "signal": signal,
                "signalClass": signal_class,
                "detail": "Missing price history: " + ", ".join(missing),
            }
            continue

        history_prices = prices.loc[prices.index < forecast_start, required].dropna(how="any")
        history_returns = history_prices.pct_change().dropna(how="any")
        if len(history_returns) < BOOTSTRAP_BLOCK:
            signal, signal_class = actual_signal_for(None)
            scenarios[key] = {
                "key": key,
                "label": label,
                "ticker": benchmark_ticker,
                "unavailable": True,
                "pBeat": None,
                "rawPBeat": None,
                "pGt5": None,
                "pLt0": None,
                "sharpe": None,
                "portfolioBeta": None,
                "betaCap": None if pd.isna(beta_cap) else float(beta_cap),
                "signal": signal,
                "signalClass": signal_class,
                "detail": "Insufficient common price history",
            }
            continue

        seed = int(forecast_start.year * 10 + forecast_start.quarter + len(key))
        port, bench = block_bootstrap_paired(history_returns, active_weights, benchmark_ticker, horizon_days, seed)
        active_returns = history_returns[active_weights.index]
        portfolio_returns = active_returns @ active_weights.reindex(active_returns.columns).fillna(0.0)
        p_beat = float(np.mean(port > bench)) if len(port) else np.nan
        signal, signal_class = actual_signal_for(p_beat)
        scenarios[key] = {
            "key": key,
            "label": label,
            "ticker": benchmark_ticker,
            "pBeat": p_beat,
            "rawPBeat": p_beat,
            "pGt5": float(np.mean(port > 0.05)) if len(port) else np.nan,
            "pLt0": float(np.mean(port < 0.0)) if len(port) else np.nan,
            "sharpe": annualized_sharpe(portfolio_returns),
            "portfolioBeta": compute_actual_portfolio_beta(history_returns, portfolio_returns, "VOO"),
            "betaCap": None if pd.isna(beta_cap) else float(beta_cap),
            "signal": signal,
            "signalClass": signal_class,
            "detail": "目前實際持倉 raw bootstrap",
        }
    return scenarios

def load_holdings():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT h.market, h.ticker, h.name, h.shares, h.avg_cost,
               COALESCE(h.last_price,
                   (SELECT price FROM holdings_snapshot
                    WHERE ticker=h.ticker ORDER BY snapshot_date DESC LIMIT 1)) AS price,
               h.last_price_date, h.currency
        FROM holdings h ORDER BY h.market DESC, h.ticker
    """).fetchall()
    conn.close()
    out = []
    for market, ticker, name, shares, avg_cost, price, price_date, currency in rows:
        cost = shares * avg_cost if avg_cost else None
        mv   = shares * price   if price    else None
        pnl  = mv - cost        if (mv and cost) else None
        pct  = pnl / cost * 100 if (pnl is not None and cost) else None
        out.append(dict(market=market, ticker=ticker, name=name or ticker,
                        shares=shares, avg_cost=avg_cost, price=price,
                        price_date=price_date, currency=currency,
                        cost=cost, mv=mv, pnl=pnl, pct=pct))
    return out

def load_history():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT snapshot_date, market,
               SUM(market_value) as mv,
               SUM(pnl)          as pnl
        FROM holdings_snapshot
        WHERE market_value IS NOT NULL
        GROUP BY snapshot_date, market
        ORDER BY snapshot_date
    """).fetchall()
    conn.close()
    hist = {"US": [], "TW": []}
    for d, m, mv, pnl in rows:
        if m in hist:
            hist[m].append({"date": d[:10], "mv": round(mv, 2),
                             "pnl": round(pnl, 2) if pnl else 0})
    return hist

def build_benchmark_pnl_series(us_history, tickers=("VT", "VOO", "QQQ", "SOXX")):
    if not us_history:
        return {}
    dates = [pd.Timestamp(row["date"]) for row in us_history]
    initial_value = float(us_history[0]["mv"] or 0)
    if initial_value <= 0:
        return {}

    prices = load_wall_street_prices(tickers)
    if prices.empty:
        return {}

    out = {}
    for ticker in tickers:
        if ticker not in prices.columns:
            continue
        series = prices[ticker].dropna().sort_index()
        values = []
        start_price = None
        for idx, date_value in enumerate(dates):
            eligible = series.loc[series.index <= date_value]
            if eligible.empty:
                values.append(None)
                continue
            price = float(eligible.iloc[-1])
            if idx == 0:
                start_price = price
            if not start_price:
                values.append(None)
            else:
                values.append(round(initial_value * (price / start_price - 1.0), 2))
        out[ticker] = values
    return out

def load_us_model():
    df  = pd.read_csv(WF_CSV)
    if "p_beat_qqq_calibrated" not in df.columns:
        df = compute_p_beat_qqq(df)
    row = df[df["period"].str.contains("current", na=False)].iloc[-1]
    w   = parse_weights(row["weights"]).to_dict()
    return (w,
            row.get("regime_label", "unknown"),
            row.get("selection_stage", "unknown"),
            row.get("p_beat_qqq_calibrated", None))

def load_tw_model():
    try:
        df  = pd.read_csv(TW_WF)
        row = df[df["period"].str.contains("current", na=False)].iloc[-1]
        w   = {}
        for part in str(row.get("weights", "")).split(","):
            tok = part.strip().rsplit(" ", 1)
            if len(tok) == 2:
                t, v = tok[0].strip(), float(tok[1].replace("%","")) / 100
                w[t] = w.get(t, 0) + v
        p_beat = row.get("p_beat_0050", None)
        utility = row.get("utility", None)
        return w, p_beat, utility
    except Exception:
        return {}, None, None

def summarise(holdings, market):
    h  = [x for x in holdings if x["market"] == market]
    tc = sum(x["cost"] for x in h if x["cost"])
    tm = sum(x["mv"]   for x in h if x["mv"])
    tp = tm - tc if (tm and tc) else 0
    pp = tp / tc * 100 if tc else 0
    return tc, tm, tp, pp, h

# ── HTML fragments ────────────────────────────────────────────────────────────

def card(label, value, sub="", sub_class="", extra_class=""):
    return f"""<div class="card {extra_class}">
      <div class="card-label">{label}</div>
      <div class="card-value">{value}</div>
      {f'<div class="card-sub {sub_class}">{sub}</div>' if sub else ''}
    </div>"""

def holdings_table_us(hlist, total_mv):
    rows = []
    for h in sorted(hlist, key=lambda x: -(x["mv"] or 0)):
        pnl_s  = ("+" if (h["pnl"] or 0) >= 0 else "") + fmt(h["pnl"], prefix="$", d=0)
        pct_s  = fmt(h["pct"], suffix="%")
        rows.append(f"""<tr>
          <td><strong>{h['ticker']}</strong></td><td class="muted">{h['name']}</td>
          <td class="num">{fmt(h['shares'],d=0)}</td>
          <td class="num">{fmt(h['avg_cost'],prefix='$')}</td>
          <td class="num">{fmt(h['price'],prefix='$')}</td>
          <td class="num">{fmt(h['mv'],prefix='$',d=0)}</td>
          <td class="num {pc(h['pnl'])}">{pnl_s}</td>
          <td class="num {pc(h['pct'])}">{pct_s}</td>
        </tr>""")
    return "\n".join(rows)

def holdings_drawer_rows_us(hlist, total_mv):
    rows = []
    for h in sorted(hlist, key=lambda x: -(x["mv"] or 0)):
        weight = (h["mv"] or 0) / total_mv * 100 if total_mv else 0
        pnl_s = ("+" if (h["pnl"] or 0) >= 0 else "") + fmt(h["pnl"], prefix="$", d=0)
        pct_s = fmt(h["pct"], suffix="%")
        rows.append(f"""<tr>
          <td><strong>{h['ticker']}</strong><div class="drawer-name">{h['name']}</div></td>
          <td class="num">{weight:.1f}%</td>
          <td class="num">{fmt(h['shares'], d=0)}</td>
          <td class="num">{fmt(h['mv'], prefix='$', d=0)}</td>
          <td class="num {pc(h['pnl'])}">{pnl_s}<div class="drawer-name {pc(h['pct'])}">{pct_s}</div></td>
        </tr>""")
    return "\n".join(rows)

def regime_condition_row(label, current, triggered, detail):
    state = "觸發" if triggered else "未觸發"
    state_class = "neg" if triggered else "pos"
    return f"""<tr>
      <td><strong>{label}</strong><div class="drawer-name">{detail}</div></td>
      <td class="num">{current}</td>
      <td class="num {state_class}">{state}</td>
    </tr>"""

def holdings_table_tw(hlist):
    rows = []
    for h in hlist:
        pnl_s = ("+" if (h["pnl"] or 0) >= 0 else "") + fmt(h["pnl"], prefix="$", d=0)
        pct_s = fmt(h["pct"], suffix="%")
        rows.append(f"""<tr>
          <td><strong>{h['ticker']}</strong></td><td class="muted">{h['name']}</td>
          <td class="num">{fmt(h['shares'],d=0)}</td>
          <td class="num">{fmt(h['avg_cost'],prefix='$')}</td>
          <td class="num">{fmt(h['price'],prefix='$')}</td>
          <td class="num">{fmt(h['mv'],prefix='$',d=0)}</td>
          <td class="num {pc(h['pnl'])}">{pnl_s}</td>
          <td class="num {pc(h['pct'])}">{pct_s}</td>
        </tr>""")
    return "\n".join(rows)

def tw_model_grid(tw_model_w):
    items = sorted(tw_model_w.items(), key=lambda x: -x[1])
    cards = []
    for ticker, weight in items:
        is_core  = "0050" in ticker
        label    = "核心" if is_core else "衛星"
        kind     = "core" if is_core else "satellite"
        cards.append(f"""<div class="model-tile">
          <div class="model-ticker">{ticker}</div>
          <div class="model-weight {kind}">{weight*100:.0f}%</div>
          <div class="model-kind">{label}</div>
        </div>""")
    return "\n".join(cards)

# ── main builder ──────────────────────────────────────────────────────────────

def build():
    holdings                      = load_holdings()
    hist                          = load_history()
    wf_frame                      = pd.read_csv(WF_CSV)
    model_w, regime, stage, p_beat = load_us_model()
    tw_model_w, tw_p_beat, tw_util = load_tw_model()

    us_tc, us_tm, us_tp, us_pp, us_h = summarise(holdings, "US")
    tw_tc, tw_tm, tw_tp, tw_pp, tw_h = summarise(holdings, "TW")

    # regime
    rc = {"risk_on":"#22c55e","caution":"#f59e0b","risk_off":"#ef4444"}.get(regime,"#6b7280")
    rl = regime.upper().replace("_"," ")
    pb = f"{p_beat*100:.1f}%" if p_beat else "—"
    tw_pb = f"{tw_p_beat*100:.2f}%" if tw_p_beat else "—"

    banner_bg     = {"risk_on":"#052e16","caution":"#451a03","risk_off":"#1f0207"}.get(regime,"#1e293b")
    regime_badge  = f'<span class="regime-badge" style="background:{rc}22;color:{rc};border:1px solid {rc}66">{rl}</span>'
    current_regime = wf_frame[wf_frame["period"].str.contains("current", na=False)].iloc[-1]
    risk_score = int(float(current_regime.get("risk_score", 0)))
    spy_vs_sma200 = float(current_regime.get("spy_vs_sma200", np.nan))
    vix_close = float(current_regime.get("vix_close", np.nan))
    tnx_vs_sma20 = float(current_regime.get("tnx_vs_sma20", np.nan))
    regime_rows_html = "\n".join([
        regime_condition_row(
            "SPY < 200 日均線",
            "—" if pd.isna(spy_vs_sma200) else f"{spy_vs_sma200 * 100:+.2f}%",
            bool(spy_vs_sma200 < 0) if not pd.isna(spy_vs_sma200) else False,
            "SPY 低於 200 日均線時計 1 分",
        ),
        regime_condition_row(
            "VIX > 25",
            "—" if pd.isna(vix_close) else f"{vix_close:.2f}",
            bool(vix_close > 25) if not pd.isna(vix_close) else False,
            "VIX 高於 25 時計 1 分",
        ),
        regime_condition_row(
            "10Y yield (^TNX) > 20 日均線",
            "—" if pd.isna(tnx_vs_sma20) else f"{tnx_vs_sma20 * 100:+.2f}%",
            bool(tnx_vs_sma20 > 0) if not pd.isna(tnx_vs_sma20) else False,
            "10 年期殖利率高於自己的 20 日均線時計 1 分",
        ),
    ])

    # VOO gap
    us_total_mv  = sum(x["mv"] for x in us_h if x["mv"]) or 1
    actual_w     = {x["ticker"]: x["mv"]/us_total_mv for x in us_h if x["mv"]}
    actual_benchmark_scenarios = compute_actual_benchmark_scenarios(wf_frame, actual_w)
    if "QQQ" in actual_benchmark_scenarios and actual_benchmark_scenarios["QQQ"].get("pBeat") is not None:
        pb = f"{actual_benchmark_scenarios['QQQ']['pBeat']*100:.1f}%"
    actual_sharpe = actual_benchmark_scenarios.get("QQQ", {}).get("sharpe")
    actual_sharpe_text = fmt(actual_sharpe, d=2) if actual_sharpe is not None else "—"
    voo_actual   = actual_w.get("VOO", 0) * 100
    # US horizontal bar: all tickers sorted by actual weight desc
    bar_tickers = sorted(actual_w, key=lambda t: -actual_w[t])
    bar_actual  = [round(actual_w.get(t, 0) * 100, 1) for t in bar_tickers]
    # Color: VOO = blue, other current holdings = green.
    bar_colors  = []
    for t in bar_tickers:
        if t == "VOO":
            bar_colors.append("#38bdf8")
        else:
            bar_colors.append("#22c55e")

    # History JSON
    benchmark_pnls = build_benchmark_pnl_series(hist["US"])
    us_dates = json.dumps([d["date"] for d in hist["US"]])
    us_pnls  = json.dumps([d["pnl"]  for d in hist["US"]])
    benchmark_pnls_js = json.dumps(benchmark_pnls)
    tw_dates = json.dumps([d["date"] for d in hist["TW"]])
    tw_mvs   = json.dumps([d["mv"]   for d in hist["TW"]])

    # Overview combined
    all_dates = sorted(set(d["date"] for d in hist["US"] + hist["TW"]))
    us_mv_map = {d["date"]: d["mv"] for d in hist["US"]}
    tw_mv_map = {d["date"]: d["mv"] for d in hist["TW"]}
    ov_dates  = json.dumps(all_dates)
    ov_us_mvs = json.dumps([us_mv_map.get(d) for d in all_dates])
    ov_tw_mvs = json.dumps([tw_mv_map.get(d) for d in all_dates])

    alert_html = ""

    # TW model grid
    if tw_model_w:
        tw_grid = tw_model_grid(tw_model_w)
        tw_util_str = f"{tw_util:.3f}" if tw_util else "—"
    else:
        tw_grid     = '<div style="color:#94a3b8;padding:12px">模型配置資料未找到</div>'
        tw_util_str = "—"

    us_table_html = holdings_table_us(us_h, us_total_mv)
    us_drawer_rows_html = holdings_drawer_rows_us(us_h, us_total_mv)
    tw_table_html = holdings_table_tw(tw_h)

    hbar_labels_js = json.dumps(bar_tickers)
    hbar_actual_js = json.dumps(bar_actual)
    hbar_colors_js = json.dumps(bar_colors)
    html = f"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Portfolio Dashboard</title>
<script>
const savedTheme = localStorage.getItem('dashboard-theme') || 'dark';
document.documentElement.dataset.theme = savedTheme;
</script>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
:root{{--bg:#0f172a;--sf:#1e293b;--bd:#334155;--tx:#e2e8f0;--mu:#94a3b8;--ac:#38bdf8;--pos:#22c55e;--neg:#ef4444;--warn:#f59e0b;--warn-bg:#1c1507;--warn-tx:#fde68a;--warn-sub:#fcd34d;--table-head:#1a2942;--row-bd:#192233;--row-hover:#1e2d45;--code-bg:#0f172a;--code-tx:#a5f3fc;}}
:root[data-theme="light"]{{--bg:#f8fafc;--sf:#ffffff;--bd:#cbd5e1;--tx:#0f172a;--mu:#475569;--ac:#0369a1;--pos:#15803d;--neg:#dc2626;--warn:#b45309;--warn-bg:#fffbeb;--warn-tx:#78350f;--warn-sub:#92400e;--table-head:#e2e8f0;--row-bd:#e2e8f0;--row-hover:#f1f5f9;--code-bg:#f1f5f9;--code-tx:#0f172a;}}
*{{box-sizing:border-box;margin:0;padding:0;}}
body{{background:var(--bg);color:var(--tx);font-family:'Segoe UI',system-ui,sans-serif;font-size:14px;min-height:100vh;}}
header{{padding:12px 24px;border-bottom:1px solid var(--bd);display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap;}}
header h1{{font-size:17px;font-weight:700;color:var(--tx);}}
header .subtitle{{font-size:12px;color:var(--mu);margin-top:2px;}}
.header-right{{display:flex;align-items:center;gap:10px;}}
.updated{{color:var(--mu);font-size:12px;}}
.btn-update{{background:transparent;border:1px solid var(--bd);color:var(--mu);padding:6px 14px;border-radius:6px;cursor:pointer;font-size:12px;font-weight:600;}}
.btn-update:hover{{border-color:var(--ac);color:var(--ac);}}
.btn-icon{{width:32px;height:32px;border-radius:6px;border:1px solid var(--bd);background:transparent;color:var(--tx);cursor:pointer;font-size:15px;display:inline-flex;align-items:center;justify-content:center;}}
.btn-icon:hover{{border-color:var(--ac);color:var(--ac);}}
.tabs{{display:flex;border-bottom:1px solid var(--bd);background:var(--sf);}}
.tab{{padding:12px 24px;cursor:pointer;color:var(--mu);font-size:13px;font-weight:600;border-bottom:2px solid transparent;transition:.15s;user-select:none;}}
.tab.active{{color:var(--ac);border-bottom-color:var(--ac);}}
.tab:hover:not(.active){{color:var(--tx);}}
.panel{{display:none;padding:20px 24px;max-width:1200px;margin:0 auto;}}
.panel.active{{display:block;}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin-bottom:20px;}}
.card{{background:var(--sf);border:1px solid var(--bd);border-radius:10px;padding:14px 18px;}}
.card.warn{{border-color:var(--warn);background:var(--warn-bg);}}
.card-label{{color:var(--mu);font-size:11px;text-transform:uppercase;letter-spacing:.7px;margin-bottom:5px;}}
.card-value{{font-size:21px;font-weight:700;}}
.card-sub{{font-size:12px;color:var(--mu);margin-top:3px;}}
.alert-box{{background:var(--warn-bg);border:1px solid var(--warn);border-radius:10px;padding:14px 18px;margin-bottom:20px;display:flex;align-items:flex-start;gap:12px;color:var(--warn-tx);}}
.alert-title{{font-size:13px;font-weight:700;color:var(--warn-tx);margin-bottom:4px;}}
.alert-body{{font-size:12px;color:var(--warn-sub);line-height:1.6;}}
.regime-badge{{display:inline-block;padding:3px 12px;border-radius:16px;font-weight:700;font-size:12px;}}
.signal-badge{{display:inline-block;padding:3px 10px;border-radius:14px;font-weight:700;font-size:12px;}}
.signal-badge.strong{{background:#14532d;color:#86efac;border:1px solid #22c55e66;}}
.signal-badge.watch{{background:#451a03;color:#fde68a;border:1px solid #f59e0b66;}}
.signal-badge.weak{{background:#450a0a;color:#fecaca;border:1px solid #ef444466;}}
:root[data-theme="light"] .signal-badge.strong{{background:#dcfce7;color:#166534;border-color:#16a34a66;}}
:root[data-theme="light"] .signal-badge.watch{{background:#fef3c7;color:#92400e;border-color:#d9770666;}}
:root[data-theme="light"] .signal-badge.weak{{background:#fee2e2;color:#991b1b;border-color:#dc262666;}}
.charts-row{{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:20px;}}
.charts-row.wide{{grid-template-columns:1fr;}}
.chart-box{{background:var(--sf);border:1px solid var(--bd);border-radius:10px;padding:16px;}}
.chart-box h3{{font-size:12px;color:var(--mu);text-transform:uppercase;letter-spacing:.6px;margin-bottom:12px;}}
.chart-wrap{{position:relative;height:220px;}}
.chart-wrap.tall{{height:300px;}}
.section-label{{font-size:11px;font-weight:600;color:var(--mu);text-transform:uppercase;letter-spacing:.6px;margin-bottom:10px;}}
table{{width:100%;border-collapse:collapse;background:var(--sf);border-radius:10px;overflow:hidden;margin-bottom:20px;}}
thead th{{background:var(--table-head);color:var(--mu);font-size:11px;text-transform:uppercase;letter-spacing:.5px;padding:9px 12px;text-align:left;border-bottom:1px solid var(--bd);}}
tbody td{{padding:9px 12px;border-bottom:1px solid var(--row-bd);}}
tbody tr:last-child td{{border-bottom:none;}}
tbody tr:hover{{background:var(--row-hover);}}
.num{{text-align:right;font-variant-numeric:tabular-nums;}}
.muted{{color:var(--mu);font-size:12px;}}
.pos{{color:var(--pos);}}
.neg{{color:var(--neg);}}
.modal-overlay{{display:none;position:fixed;inset:0;background:#000a;z-index:100;align-items:center;justify-content:center;}}
.modal-overlay.open{{display:flex;}}
.modal{{background:var(--sf);border:1px solid var(--bd);border-radius:12px;padding:24px;max-width:520px;width:90%;}}
.modal h3{{color:var(--ac);margin-bottom:12px;}}
.modal code{{display:block;background:var(--code-bg);border:1px solid var(--bd);border-radius:6px;padding:12px;font-size:12px;color:var(--code-tx);margin:10px 0;line-height:1.6;word-break:break-all;}}
.modal-close{{margin-top:12px;background:var(--bd);border:none;color:var(--tx);padding:7px 18px;border-radius:6px;cursor:pointer;}}
.modal-action{{margin-top:12px;background:var(--ac);border:none;color:#fff;padding:8px 18px;border-radius:6px;cursor:pointer;font-weight:700;}}
.modal-action:disabled{{opacity:.55;cursor:not-allowed;}}
.update-status{{margin-top:12px;background:var(--code-bg);border:1px solid var(--bd);border-radius:6px;padding:10px;font-size:12px;color:var(--code-tx);line-height:1.55;white-space:pre-wrap;max-height:220px;overflow:auto;}}
.drawer-backdrop{{display:none;position:fixed;inset:0;background:#0008;z-index:90;}}
.drawer-backdrop.open{{display:block;}}
.drawer{{position:fixed;top:0;right:0;width:min(520px,94vw);height:100vh;background:var(--sf);border-left:1px solid var(--bd);z-index:91;transform:translateX(105%);transition:transform .2s ease;box-shadow:-20px 0 45px #0008;display:flex;flex-direction:column;}}
.drawer.open{{transform:translateX(0);}}
.drawer-head{{padding:18px 20px;border-bottom:1px solid var(--bd);display:flex;justify-content:space-between;gap:12px;align-items:flex-start;}}
.drawer-title{{font-size:16px;font-weight:700;color:var(--tx);}}
.drawer-sub{{font-size:12px;color:var(--mu);margin-top:4px;}}
.drawer-close{{width:30px;height:30px;border-radius:6px;border:1px solid var(--bd);background:transparent;color:var(--tx);cursor:pointer;font-size:18px;}}
.drawer-body{{padding:16px 18px;overflow:auto;}}
.drawer-body table{{font-size:12px;margin-bottom:0;}}
.drawer-body th,.drawer-body td{{padding:8px 9px;}}
.drawer-name{{font-size:11px;color:var(--mu);margin-top:3px;font-weight:400;}}
.drawer-note{{margin-top:12px;color:var(--mu);font-size:12px;line-height:1.6;background:var(--table-head);border:1px solid var(--bd);border-radius:8px;padding:10px 12px;}}
.clickable-card{{cursor:pointer;position:relative;}}
.clickable-card:hover{{border-color:var(--ac);}}
.chart-action{{font-size:11px;color:var(--ac);font-weight:600;text-transform:none;letter-spacing:0;margin-left:8px;}}
.model-tile{{text-align:center;padding:14px 10px;background:var(--table-head);border:1px solid var(--bd);border-radius:8px;}}
.model-ticker{{font-size:11px;color:var(--mu);margin-bottom:6px;}}
.model-weight{{font-size:22px;font-weight:700;}}
.model-weight.core{{color:var(--ac);}}
.model-weight.satellite{{color:var(--tx);}}
.model-kind{{font-size:11px;color:var(--mu);margin-top:4px;}}
@media(max-width:640px){{
  .charts-row{{grid-template-columns:1fr;}}
  .tab{{padding:10px 14px;font-size:12px;}}
  header h1{{font-size:15px;}}
}}
</style>
</head>
<body>

<header>
  <div>
    <h1>Portfolio Dashboard</h1>
    <div class="subtitle">Buffett 量化配置觀測系統</div>
  </div>
  <div class="header-right">
    <span class="updated">更新：{TODAY}</span>
    <button class="btn-icon" id="themeToggle" onclick="toggleTheme()" title="切換亮暗模式" aria-label="切換亮暗模式">&#9790;</button>
    <button class="btn-update" onclick="document.getElementById('modal').classList.add('open')">&#8635; 更新資料</button>
  </div>
</header>

<div class="tabs">
  <div class="tab active" onclick="switchTab(0)">總覽</div>
  <div class="tab" onclick="switchTab(1)">&#127482;&#127480; 美股</div>
  <div class="tab" onclick="switchTab(2)">&#127481;&#127484; 台股</div>
</div>

<!-- ══ Tab 0: 總覽 ══ -->
<div class="panel active" id="tab0">
  <div style="height:16px"></div>
  {alert_html}
  <div class="cards">
    {card("美股市值 (USD)", f"${us_tm:,.0f}", f"成本 ${us_tc:,.0f} ｜ {us_pp:+.1f}%", pc(us_pp))}
    {card("Sharpe Ratio", actual_sharpe_text, "目前實際持倉")}
    {card("台股市值 (TWD)", f"NT${tw_tm:,.0f}", f"成本 NT${tw_tc:,.0f} ｜ {tw_pp:+.1f}%", pc(tw_pp))}
    {card("台股模型信心", tw_pb, "P(&gt;0050) ｜ 2026Q2")}
  </div>
  <div class="charts-row">
    <div class="chart-box">
      <h3>美股損益走勢 (USD)</h3>
      <div class="chart-wrap"><canvas id="ovUsChart"></canvas></div>
    </div>
    <div class="chart-box">
      <h3>台股市值走勢 (TWD)</h3>
      <div class="chart-wrap"><canvas id="ovTwChart"></canvas></div>
    </div>
  </div>
</div>

<!-- ══ Tab 1: 美股 ══ -->
<div class="panel" id="tab1">
  <div style="height:16px"></div>
  <div class="cards">
    {card("美股市值", f"${us_tm:,.0f}", f"成本 ${us_tc:,.0f}")}
    {card("損益", f"${us_tp:+,.0f}", f"{us_pp:+.1f}%", pc(us_pp))}
    {card("Sharpe Ratio", actual_sharpe_text, "目前實際持倉")}
    <div class="clickable-card" onclick="openRegimeDrawer()">
      {card("市場狀態", regime_badge, f'<span id="usStatusSub">VOO 實際 {voo_actual:.1f}% ｜ 點選查看判斷</span>')}
    </div>
  </div>
  {alert_html}
  <div class="charts-row">
    <div class="chart-box clickable-card" id="allocationCard" onclick="openHoldingsDrawer()">
      <h3>持倉配置：實際持倉 <span class="chart-action">點選查看明細</span></h3>
      <div class="chart-wrap tall"><canvas id="usHBar"></canvas></div>
    </div>
    <div class="chart-box">
      <h3>損益走勢（近期，USD）</h3>
      <div class="chart-wrap tall"><canvas id="usLine"></canvas></div>
    </div>
  </div>
</div>

<!-- ══ Tab 2: 台股 ══ -->
<div class="panel" id="tab2">
  <div style="height:16px"></div>
  <div class="cards">
    {card("台股市值", f"NT${tw_tm:,.0f}", f"成本 NT${tw_tc:,.0f}")}
    {card("損益", f"NT${tw_tp:+,.0f}", f"{tw_pp:+.1f}%", pc(tw_pp))}
    {card("模型信心", tw_pb, f"P(&gt;0050) ｜ Utility {tw_util_str}")}
  </div>
  <div class="chart-box" style="margin-bottom:20px">
    <h3>2026Q2 模型建議配置</h3>
    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px;margin-top:12px;">
      {tw_grid}
    </div>
    <div style="margin-top:12px;padding:8px 12px;background:#0f1a2e;border-radius:6px;font-size:12px;color:var(--mu)">
      P(&gt;0050) {tw_pb} &nbsp;|&nbsp; Utility {tw_util_str} &nbsp;|&nbsp; Snapshot 2026-04-17
    </div>
  </div>
  <div class="chart-box" style="margin-bottom:20px">
    <h3>台股市值走勢 (TWD)</h3>
    <div class="chart-wrap"><canvas id="twLine"></canvas></div>
  </div>
  <div class="section-label">實際持倉</div>
  <table>
    <thead><tr>
      <th>代號</th><th>名稱</th>
      <th class="num">股數</th><th class="num">均價</th><th class="num">現價</th>
      <th class="num">市值</th><th class="num">損益</th><th class="num">損益%</th>
    </tr></thead>
    <tbody>{tw_table_html}</tbody>
  </table>
</div>

<div class="drawer-backdrop" id="holdingsDrawerBackdrop" onclick="closeHoldingsDrawer()"></div>
<aside class="drawer" id="holdingsDrawer" aria-hidden="true">
  <div class="drawer-head">
    <div>
      <div class="drawer-title">美股持倉明細</div>
      <div class="drawer-sub">依市值排序 ｜ 總市值 ${us_tm:,.0f}</div>
    </div>
    <button class="drawer-close" onclick="closeHoldingsDrawer()" aria-label="關閉持倉明細">&times;</button>
  </div>
  <div class="drawer-body">
    <table>
      <thead><tr>
        <th>代號</th><th class="num">權重</th><th class="num">股數</th><th class="num">市值</th><th class="num">損益</th>
      </tr></thead>
      <tbody>{us_drawer_rows_html}</tbody>
    </table>
  </div>
</aside>

<div class="drawer-backdrop" id="regimeDrawerBackdrop" onclick="closeRegimeDrawer()"></div>
<aside class="drawer" id="regimeDrawer" aria-hidden="true">
  <div class="drawer-head">
    <div>
      <div class="drawer-title">市場狀態判斷</div>
      <div class="drawer-sub">目前狀態：{rl} ｜ Risk score {risk_score}/3</div>
    </div>
    <button class="drawer-close" onclick="closeRegimeDrawer()" aria-label="關閉市場狀態判斷">&times;</button>
  </div>
  <div class="drawer-body">
    <table>
      <thead><tr>
        <th>條件</th><th class="num">目前值</th><th class="num">狀態</th>
      </tr></thead>
      <tbody>{regime_rows_html}</tbody>
    </table>
    <div class="drawer-note">
      0 分 = risk_on；1 分 = caution；2-3 分 = risk_off。
    </div>
  </div>
</aside>

<div class="modal-overlay" id="modal" onclick="if(event.target===this)this.classList.remove('open')">
  <div class="modal">
    <h3>&#8635; 更新資料</h3>
    <p style="color:var(--mu);font-size:13px">按下後會在本機依序更新持倉、重新產生 dashboard，並只提交與推送 dashboard HTML。</p>
    <button class="modal-action" id="runUpdateBtn" onclick="runDashboardUpdate()">一鍵更新並推送</button>
    <div class="update-status" id="updateStatus">待執行。若按鈕無法連線，請用 python -X utf8 dashboard_server.py 開啟本機 dashboard server。</div>
    <p style="color:var(--mu);font-size:12px;margin-top:8px">推送後約 30 秒 GitHub Pages 自動更新。</p>
    <button class="modal-close" onclick="document.getElementById('modal').classList.remove('open')">關閉</button>
  </div>
</div>

<script>
const tabs   = document.querySelectorAll('.tab');
const panels = document.querySelectorAll('.panel');
function switchTab(i){{
  tabs.forEach((t,j)=>t.classList.toggle('active',i===j));
  panels.forEach((p,j)=>p.classList.toggle('active',i===j));
}}

function themeValue(name){{
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}}
let gridColor = themeValue('--bd');
let tickColor = themeValue('--mu');
function syncThemeButton(){{
  const btn = document.getElementById('themeToggle');
  if (!btn) return;
  const isLight = document.documentElement.dataset.theme === 'light';
  btn.innerHTML = isLight ? '&#9788;' : '&#9790;';
  btn.title = isLight ? '切換深色模式' : '切換淺色模式';
}}
function applyChartTheme(){{
  gridColor = themeValue('--bd');
  tickColor = themeValue('--mu');
  Object.values(Chart.instances || {{}}).forEach(chart => {{
    Object.values(chart.options.scales || {{}}).forEach(scale => {{
      if (scale.ticks) scale.ticks.color = tickColor;
      if (scale.grid) scale.grid.color = gridColor;
    }});
    if (chart.options.plugins?.legend?.labels) chart.options.plugins.legend.labels.color = tickColor;
    chart.update();
  }});
}}
function toggleTheme(){{
  const next = document.documentElement.dataset.theme === 'light' ? 'dark' : 'light';
  document.documentElement.dataset.theme = next;
  localStorage.setItem('dashboard-theme', next);
  syncThemeButton();
  applyChartTheme();
}}
syncThemeButton();
async function runDashboardUpdate(){{
  const button = document.getElementById('runUpdateBtn');
  const status = document.getElementById('updateStatus');
  button.disabled = true;
  status.textContent = '更新中，請不要關閉視窗...';
  try {{
    const response = await fetch('/api/update', {{ method:'POST' }});
    if (!response.ok) throw new Error(`HTTP ${{response.status}}`);
    const result = await response.json();
    const lines = [result.ok ? '完成：' + result.message : '失敗：' + result.message];
    for (const step of result.steps || []) {{
      lines.push(`\\n[${{step.code === 0 ? 'OK' : 'ERR'}}] ${{step.label}}`);
      if (step.output) lines.push(step.output);
    }}
    status.textContent = lines.join('\\n');
    if (result.ok) setTimeout(() => window.location.reload(), 1200);
  }} catch (error) {{
    status.textContent = '無法連線到本機更新服務。請先執行：\\npython -X utf8 dashboard_server.py\\n\\n錯誤：' + error.message;
  }} finally {{
    button.disabled = false;
  }}
}}
const baseScales = {{
  x:{{ ticks:{{color:tickColor,font:{{size:10}}}}, grid:{{color:gridColor}} }},
  y:{{ ticks:{{color:tickColor}}, grid:{{color:gridColor}} }}
}};
const ACTUAL_WEIGHTS = {json.dumps(actual_w)};
const REGIME_LABEL = {json.dumps(rl)};
const VOO_ACTUAL = {voo_actual:.6f};
const US_BENCHMARK_PNLS = {benchmark_pnls_js};

function lineChart(id, labels, datasets){{
  new Chart(document.getElementById(id),{{
    type:'line',
    data:{{ labels, datasets }},
    options:{{
      responsive:true, maintainAspectRatio:false,
      plugins:{{ legend:{{ labels:{{color:tickColor}} }} }},
      scales: baseScales
    }}
  }});
}}

// Overview charts
const ovDates = {ov_dates};
lineChart('ovUsChart', ovDates, [{{
  label:'損益 USD', data:{json.dumps([us_mv_map.get(d, 0) - us_tc for d in all_dates]) if us_tc else ov_us_mvs},
  borderColor:'#22c55e', backgroundColor:'#22c55e15', fill:true, tension:.3, pointRadius:3
}}]);
lineChart('ovTwChart', ovDates, [{{
  label:'市值 TWD', data:{ov_tw_mvs},
  borderColor:'#34d399', backgroundColor:'#34d39918', fill:true, tension:.3, pointRadius:3
}}]);

// US horizontal bar: actual holdings only
const usHBarChart = new Chart(document.getElementById('usHBar'),{{
  type:'bar',
  data:{{
    labels: {hbar_labels_js},
    datasets:[
      {{
        label:'實際持倉 %',
        data: {hbar_actual_js},
        backgroundColor: {hbar_colors_js},
        borderRadius:2
      }}
    ]
  }},
  options:{{
    indexAxis:'y',
    responsive:true, maintainAspectRatio:false,
    plugins:{{
      legend:{{labels:{{color:tickColor,font:{{size:11}}}}}},
      tooltip:{{callbacks:{{label:c=>`${{c.dataset.label}}: ${{c.parsed.x}}%`}}}}
    }},
    scales:{{
      x:{{ticks:{{color:tickColor,callback:v=>v+'%'}},grid:{{color:gridColor}},max:100}},
      y:{{ticks:{{color:tickColor,font:{{size:11}}}},grid:{{color:gridColor}}}}
    }}
  }}
}});

function pct(value, digits=1){{
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '—';
  return `${{(Number(value) * 100).toFixed(digits)}}%`;
}}
function num(value, digits=2){{
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '—';
  return Number(value).toFixed(digits);
}}
function modelCellHtml(weight){{
  if (weight > 0) return `<span style="color:#22c55e">&#10003; ${{(weight * 100).toFixed(1)}}%</span>`;
  return '<span style="color:#475569">&#8212; 自選</span>';
}}
function chartColors(labels, weights){{
  return labels.map(t => t === 'VOO' ? '#38bdf8' : '#22c55e');
}}
function openHoldingsDrawer(){{
  document.getElementById('holdingsDrawer').classList.add('open');
  document.getElementById('holdingsDrawer').setAttribute('aria-hidden', 'false');
  document.getElementById('holdingsDrawerBackdrop').classList.add('open');
}}
function closeHoldingsDrawer(){{
  document.getElementById('holdingsDrawer').classList.remove('open');
  document.getElementById('holdingsDrawer').setAttribute('aria-hidden', 'true');
  document.getElementById('holdingsDrawerBackdrop').classList.remove('open');
}}
function openRegimeDrawer(){{
  document.getElementById('regimeDrawer').classList.add('open');
  document.getElementById('regimeDrawer').setAttribute('aria-hidden', 'false');
  document.getElementById('regimeDrawerBackdrop').classList.add('open');
}}
function closeRegimeDrawer(){{
  document.getElementById('regimeDrawer').classList.remove('open');
  document.getElementById('regimeDrawer').setAttribute('aria-hidden', 'true');
  document.getElementById('regimeDrawerBackdrop').classList.remove('open');
}}

// US P&L trend
const benchmarkLineColors = {{VT:'#f59e0b', VOO:'#38bdf8', QQQ:'#a78bfa', SOXX:'#ef4444'}};
const usLineDatasets = [{{
  label:'Actual Portfolio',
  data: {us_pnls},
  borderColor:'#22c55e', backgroundColor:'#22c55e12', fill:false, tension:.3, pointRadius:3
}}];
for (const ticker of ['VT', 'VOO', 'QQQ', 'SOXX']) {{
  if (US_BENCHMARK_PNLS[ticker]) {{
    usLineDatasets.push({{
      label:`100% ${{ticker}}`,
      data: US_BENCHMARK_PNLS[ticker],
      borderColor: benchmarkLineColors[ticker],
      backgroundColor: `${{benchmarkLineColors[ticker]}}12`,
      borderDash: ticker === 'VOO' ? [] : [5, 4],
      fill:false,
      tension:.3,
      pointRadius:2
    }});
  }}
}}
lineChart('usLine', {us_dates}, usLineDatasets);

// TW market value trend
lineChart('twLine', {tw_dates}, [{{
  label:'市值 TWD',
  data: {tw_mvs},
  borderColor:'#34d399', backgroundColor:'#34d39918', fill:true, tension:.3, pointRadius:3
}}]);
</script>
</body>
</html>"""
    return html


if __name__ == "__main__":
    print("讀取資料...")
    html = build()
    OUT.write_text(html, encoding="utf-8")
    print(f"✅ 已輸出：{OUT}")
