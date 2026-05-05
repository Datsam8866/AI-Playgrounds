# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import re
import sqlite3
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import urllib3
import yfinance as yf
from bs4 import BeautifulSoup
from scipy.stats import spearmanr
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "tw_stock_forecast.sqlite"
UNIVERSE_CSV = ROOT / "tw_0050_universe.csv"
PREDICTIONS_CSV = ROOT / "tw_0050_xgboost_predictions.csv"
WALKFORWARD_CSV = ROOT / "tw_0050_walkforward.csv"
SUMMARY_CSV = ROOT / "tw_0050_walkforward_summary.csv"

YUANTA_0050_RATIO_URL = "https://www.yuantaetfs.com/product/detail/0050/ratio"
CORE_TICKER = "0050.TW"
BENCHMARK_TICKER = CORE_TICKER
TRAIN_START = "2010-01-01"
WALKFORWARD_START = "2016-01-01"
CURRENT_DATE = "2026-04-17"
HORIZON_DAYS = 20
MIN_HISTORY_DAYS = 60
BOOTSTRAP_LOOKBACK_DAYS = 756
BOOTSTRAP_BLOCK_DAYS = 20
BOOTSTRAP_N_SIMS = 4000

# Extra tickers beyond 0050（僅上市 .TW）：00892 成分股 + 光通訊概念股
# tuple: (code, name)  全部為上市股，ticker = code + ".TW"
EXTRA_TICKERS: list[tuple[str, str]] = [
    # 00892 半導體 ETF 成分股（上市）
    ("1560", "中砂"),
    ("2379", "瑞昱"),
    ("2455", "全新"),
    ("2458", "義隆"),
    ("2467", "志聖"),
    ("3014", "聯陽"),
    ("3034", "聯詠"),
    ("3413", "京鼎"),
    ("3443", "創意"),
    ("3592", "瑞鼎"),
    ("4919", "新唐"),
    ("5434", "崇越"),
    ("6515", "穎崴"),
    ("6526", "達發"),
    ("6937", "天虹"),
    ("8081", "致新"),
    # 光通訊概念股（上市）
    ("3450", "聯鈞"),
    ("4977", "眾達-KY"),
    ("6442", "光聖"),
]

CORE_WEIGHT_CANDIDATES = [0.50, 0.60, 0.70, 0.80]
TOP_K_CANDIDATES = [4, 6, 8, 10]
SATELLITE_CAP = 0.10
TURNOVER_CAP = 0.35

BUY_COST = 0.001425 + 0.0005
SELL_COST = 0.001425 + 0.003 + 0.0005

FEATURE_COLUMNS = [
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


warnings.filterwarnings("ignore", category=FutureWarning, module=r"yfinance(\..*)?$")
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


@dataclass(frozen=True)
class QuarterWindow:
    label: str
    start: pd.Timestamp
    end: pd.Timestamp
    is_current: bool


def configure_runtime_dirs() -> None:
    cache_dir = ROOT / ".sqlite_cache"
    cache_dir.mkdir(exist_ok=True)
    os.environ.setdefault("LOCALAPPDATA", str(cache_dir))
    os.environ.setdefault("TEMP", str(cache_dir))
    os.environ.setdefault("TMP", str(cache_dir))


def split_top_level(value: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    quote: str | None = None
    escaped = False
    depth = 0
    for char in value:
        if quote:
            current.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in ('"', "'"):
            quote = char
            current.append(char)
        elif char in "[{(":
            depth += 1
            current.append(char)
        elif char in "]})":
            depth -= 1
            current.append(char)
        elif char == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    if current:
        parts.append("".join(current).strip())
    return parts


def parse_js_value(value: str):
    if value == "null":
        return None
    if value == "true":
        return True
    if value == "false":
        return False
    if value == "{}":
        return {}
    if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
        return json.loads(value)
    if re.fullmatch(r"-?\d+", value):
        return int(value)
    if re.fullmatch(r"-?(?:\d+\.\d*|\.\d+|\d+)(?:e[+-]?\d+)?", value, flags=re.I):
        return float(value)
    return value


def parse_nuxt_function(html: str) -> tuple[str, dict[str, object]]:
    start = html.find("window.__NUXT__=(function(")
    if start < 0:
        raise ValueError("Yuanta page does not contain window.__NUXT__ data.")
    segment = html[start:].split("</script>", 1)[0]
    head = "window.__NUXT__=(function("
    params_end = segment.find("){return")
    return_start = params_end + len("){return")
    return_end = segment.rfind("}(")
    params = segment[len(head) : params_end].split(",")
    return_expr = segment[return_start:return_end]
    args = segment[return_end + 2 :].rstrip(";")
    if args.endswith(")"):
        args = args[:-1]
    values = split_top_level(args)
    mapping = {param: parse_js_value(arg) for param, arg in zip(params, values)}
    return return_expr, mapping


def resolve_token(token: str, mapping: dict[str, object]):
    token = token.strip()
    if token in mapping:
        return mapping[token]
    return parse_js_value(token)


def extract_bracket_content(source: str, key: str) -> str:
    start = source.find(key)
    if start < 0:
        raise ValueError(f"Cannot find {key}.")
    idx = start + len(key) - 1
    quote: str | None = None
    escaped = False
    depth = 0
    for pos in range(idx, len(source)):
        char = source[pos]
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in ('"', "'"):
            quote = char
        elif char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return source[idx + 1 : pos]
    raise ValueError(f"Cannot close bracket for {key}.")


def fetch_0050_universe() -> pd.DataFrame:
    response = requests.get(
        YUANTA_0050_RATIO_URL,
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=30,
        verify=False,
    )
    response.raise_for_status()
    html = response.text
    visible_text = BeautifulSoup(html, "html.parser").get_text(" ", strip=True)
    source_date_match = re.search(r"交易日期:\s*(\d{4}/\d{2}/\d{2})", visible_text)
    source_date = (
        pd.Timestamp(source_date_match.group(1).replace("/", "-")).date().isoformat()
        if source_date_match
        else pd.Timestamp(CURRENT_DATE).date().isoformat()
    )

    return_expr, mapping = parse_nuxt_function(html)
    composition = extract_bracket_content(return_expr, "FundComposition:[")
    rows = []
    for obj in re.findall(r"\{([^{}]+)\}", composition):
        fields = {}
        for part in split_top_level(obj):
            if ":" not in part:
                continue
            key, value = part.split(":", 1)
            fields[key] = value
        code = str(resolve_token(fields["stkcd"], mapping))
        name = str(resolve_token(fields["name"], mapping))
        english_name = str(resolve_token(fields.get("ename", '""'), mapping))
        qty_value = resolve_token(fields.get("qty", "null"), mapping)
        rows.append(
            {
                "source_date": source_date,
                "code": code,
                "ticker": f"{code}.TW",
                "name": name,
                "english_name": english_name,
                "pcf_qty": pd.to_numeric(qty_value, errors="coerce"),
            }
        )

    universe = pd.DataFrame(rows).drop_duplicates("ticker").sort_values("code").reset_index(drop=True)
    if len(universe) < 40:
        raise ValueError(f"Parsed only {len(universe)} constituents; aborting to avoid a bad universe.")
    return universe


def download_price_history(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    raw = yf.download(
        tickers=tickers,
        start=pd.Timestamp(start),
        end=pd.Timestamp(end) + pd.Timedelta(days=1),
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    rows = []
    for ticker in tickers:
        try:
            frame = raw[ticker].copy() if len(tickers) > 1 else raw.copy()
        except Exception:
            continue
        if frame.empty or "Close" not in frame.columns:
            continue
        frame = frame.rename_axis("date").reset_index()
        frame["ticker"] = ticker
        frame.columns = [str(column).lower().replace(" ", "_") for column in frame.columns]
        keep = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]
        rows.append(frame[keep])
    if not rows:
        raise ValueError("No Taiwan price history was downloaded.")
    prices = pd.concat(rows, ignore_index=True)
    prices["date"] = pd.to_datetime(prices["date"]).dt.tz_localize(None).dt.normalize()
    return prices.dropna(subset=["adj_close"]).sort_values(["date", "ticker"]).reset_index(drop=True)


def write_sqlite(universe: pd.DataFrame, prices: pd.DataFrame) -> None:
    with sqlite3.connect(DB_PATH) as connection:
        universe.to_sql("tw_0050_constituents", connection, if_exists="replace", index=False)
        prices.to_sql("tw_price_history", connection, if_exists="replace", index=False)
        connection.execute("CREATE INDEX IF NOT EXISTS idx_tw_price_ticker_date ON tw_price_history(ticker, date)")
        connection.execute("CREATE INDEX IF NOT EXISTS idx_tw_price_date_ticker ON tw_price_history(date, ticker)")


def load_prices_from_sqlite(tickers: list[str]) -> pd.DataFrame:
    placeholders = ",".join(["?"] * len(tickers))
    query = f"""
        SELECT date, ticker, adj_close
        FROM tw_price_history
        WHERE ticker IN ({placeholders})
        ORDER BY date ASC, ticker ASC
    """
    with sqlite3.connect(DB_PATH) as connection:
        data = pd.read_sql_query(query, connection, params=tickers, parse_dates=["date"])
    return data.pivot(index="date", columns="ticker", values="adj_close").sort_index()


def load_ohlcv_by_ticker(tickers: list[str]) -> dict[str, pd.DataFrame]:
    placeholders = ",".join(["?"] * len(tickers))
    query = f"""
        SELECT date, ticker, open, high, low, close, adj_close, volume
        FROM tw_price_history
        WHERE ticker IN ({placeholders})
        ORDER BY date ASC, ticker ASC
    """
    with sqlite3.connect(DB_PATH) as connection:
        raw = pd.read_sql_query(query, connection, params=tickers, parse_dates=["date"])
    raw = raw.rename(
        columns={
            "date": "Date",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "adj_close": "Adj Close",
            "volume": "Volume",
        }
    )
    return {ticker: raw[raw["ticker"] == ticker].copy().reset_index(drop=True) for ticker in tickers}


def compute_rsi(price: pd.Series, window: int = 14) -> pd.Series:
    delta = price.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def build_feature_frame(data_by_ticker: dict[str, pd.DataFrame], keep_all_rows: bool = False) -> pd.DataFrame:
    frames = []
    for ticker, frame in data_by_ticker.items():
        if frame.empty:
            continue
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
            macd_hist=macd - macd.ewm(span=9, adjust=False).mean(),
            volume_ratio20=volume / volume.rolling(20).mean(),
            drawdown_20=(price / rolling_high_20) - 1.0,
            drawdown_60=(price / rolling_high_60) - 1.0,
            intraday_range=(local["High"] - local["Low"]) / local["Close"].replace(0.0, np.nan),
            next_day_return=price.shift(-1) / price - 1.0,
            forward_20d_return=price.shift(-HORIZON_DAYS) / price - 1.0,
            label_available_date=local["Date"].shift(-HORIZON_DAYS),
        )
        frames.append(local)
    features = pd.concat(frames, ignore_index=True).sort_values(["Date", "ticker"]).reset_index(drop=True)
    drop_columns = FEATURE_COLUMNS if keep_all_rows else FEATURE_COLUMNS + ["forward_20d_return", "label_available_date"]
    features = features.dropna(subset=drop_columns).copy()
    dummies = pd.get_dummies(features["ticker"], prefix="ticker", dtype=float)
    return pd.concat([features, dummies], axis=1)


def build_regressor() -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=100,
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


def build_quarter_windows(latest_date: pd.Timestamp) -> list[QuarterWindow]:
    windows = []
    for period in pd.period_range(WALKFORWARD_START, latest_date, freq="Q"):
        start = period.start_time.normalize()
        end = period.end_time.normalize()
        is_current = latest_date < end
        label = f"{period}_current" if is_current else str(period)
        windows.append(QuarterWindow(label=label, start=start, end=min(end, latest_date), is_current=is_current))
    return windows


def align_feature_columns(features: pd.DataFrame, all_feature_columns: list[str]) -> pd.DataFrame:
    aligned = features.copy()
    for column in all_feature_columns:
        if column not in aligned.columns:
            aligned[column] = 0.0
    return aligned


def run_quarterly_predictions(features: pd.DataFrame, latest_date: pd.Timestamp) -> pd.DataFrame:
    ticker_columns = sorted(column for column in features.columns if column.startswith("ticker_"))
    all_feature_columns = FEATURE_COLUMNS + ticker_columns
    rows = []
    for window in build_quarter_windows(latest_date):
        eval_date_mask = (features["Date"] >= window.start) & (features["Date"] <= window.end)
        eval_dates = features.loc[eval_date_mask, "Date"]
        if eval_dates.empty:
            continue
        snapshot_date = pd.Timestamp(eval_dates.max() if window.is_current else eval_dates.min())
        train_mask = (
            (features["Date"] >= pd.Timestamp(TRAIN_START))
            & (features["Date"] < snapshot_date)
            & (features["label_available_date"] < snapshot_date)
        )
        eval_mask = features["Date"].eq(snapshot_date)
        train_frame = align_feature_columns(features.loc[train_mask].copy(), all_feature_columns)
        eval_frame = align_feature_columns(features.loc[eval_mask].copy(), all_feature_columns)
        if train_frame.empty or eval_frame.empty:
            continue
        model = build_regressor()
        model.fit(train_frame[all_feature_columns], train_frame["forward_20d_return"].astype(float))
        predicted = model.predict(eval_frame[all_feature_columns])
        out = eval_frame[["Date", "ticker", "forward_20d_return"]].copy()
        out["period"] = window.label
        out["split"] = "current" if window.is_current else "walkforward"
        out["predicted_return"] = predicted
        rows.append(out)
    if not rows:
        raise ValueError("No quarterly predictions produced.")
    return pd.concat(rows, ignore_index=True).sort_values(["Date", "ticker"]).reset_index(drop=True)


def zscore(values: pd.Series) -> pd.Series:
    std = values.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(0.0, index=values.index)
    return (values - values.mean()) / std


def available_tickers(prices: pd.DataFrame, date: pd.Timestamp) -> list[str]:
    available = []
    for ticker in prices.columns:
        history = prices.loc[prices.index < date, ticker].dropna()
        if len(history) >= MIN_HISTORY_DAYS:
            available.append(ticker)
    return available


def build_weights(snapshot: pd.DataFrame, core_weight: float, top_k: int) -> tuple[pd.Series, list[str]]:
    satellite_total = 1.0 - core_weight
    satellite_weight = satellite_total / top_k
    if satellite_weight > SATELLITE_CAP + 1e-12:
        raise ValueError("Satellite cap violated.")
    ranked = snapshot.sort_values("predicted_return", ascending=False).head(top_k)
    selected = ranked["ticker"].tolist()
    weights = pd.Series({CORE_TICKER: core_weight, **{ticker: satellite_weight for ticker in selected}}, dtype=float)
    return weights, selected


def block_bootstrap(
    returns: pd.DataFrame,
    weights: pd.Series,
    horizon_days: int,
    n_sims: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    tickers = list(dict.fromkeys([BENCHMARK_TICKER] + weights.index.tolist()))
    matrix = returns[tickers].dropna().to_numpy()
    if len(matrix) < BOOTSTRAP_BLOCK_DAYS:
        raise ValueError("Not enough returns for bootstrap.")
    port_weights = weights.reindex(tickers).fillna(0.0).to_numpy()
    bench_weights = np.zeros(len(tickers), dtype=float)
    bench_weights[tickers.index(BENCHMARK_TICKER)] = 1.0
    rng = np.random.default_rng(seed)
    max_start = matrix.shape[0] - BOOTSTRAP_BLOCK_DAYS
    port = np.empty(n_sims, dtype=float)
    bench = np.empty(n_sims, dtype=float)
    for idx in range(n_sims):
        samples = []
        total = 0
        while total < horizon_days:
            start = int(rng.integers(0, max_start + 1))
            block = matrix[start : start + BOOTSTRAP_BLOCK_DAYS]
            samples.append(block)
            total += len(block)
        sampled = np.vstack(samples)[:horizon_days]
        port[idx] = float(np.prod(1.0 + sampled @ port_weights) - 1.0)
        bench[idx] = float(np.prod(1.0 + sampled @ bench_weights) - 1.0)
    return port, bench


def one_way_turnover(new_weights: pd.Series, old_weights: pd.Series | None) -> float:
    if old_weights is None or old_weights.empty:
        return 1.0
    tickers = sorted(set(new_weights.index) | set(old_weights.index))
    return float(np.abs(new_weights.reindex(tickers).fillna(0.0) - old_weights.reindex(tickers).fillna(0.0)).sum() / 2.0)


def trading_cost(new_weights: pd.Series, old_weights: pd.Series | None) -> float:
    if old_weights is None or old_weights.empty:
        old_weights = pd.Series(dtype=float)
    tickers = sorted(set(new_weights.index) | set(old_weights.index))
    diff = new_weights.reindex(tickers).fillna(0.0) - old_weights.reindex(tickers).fillna(0.0)
    return float(diff.clip(lower=0.0).sum() * BUY_COST + (-diff.clip(upper=0.0)).sum() * SELL_COST)


def realized_return(prices: pd.DataFrame, weights: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> float:
    period_prices = prices.loc[(prices.index >= start) & (prices.index <= end), weights.index].dropna(how="any")
    if len(period_prices) < 2:
        return float("nan")
    asset_returns = period_prices.iloc[-1] / period_prices.iloc[0] - 1.0
    return float((asset_returns * weights).sum())


def compute_max_drawdown(returns: pd.Series) -> float:
    equity = (1.0 + returns.fillna(0.0)).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    return float(drawdown.min())


def run_walkforward(predictions: pd.DataFrame, prices: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    returns = prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    name_lookup = dict(zip(universe["ticker"], universe["name"]))
    windows = {window.label: window for window in build_quarter_windows(prices.index.max())}
    previous_weights: pd.Series | None = None
    rows = []

    for period_label, snapshot in predictions.groupby("period", sort=False):
        window = windows[period_label]
        forecast_start = pd.Timestamp(snapshot["Date"].min())
        forecast_end = window.end
        horizon_days = max(1, len(prices.loc[(prices.index >= forecast_start) & (prices.index <= forecast_end)].index) - 1)
        available = set(available_tickers(prices, forecast_start))
        snapshot = snapshot[snapshot["ticker"].isin(available - {CORE_TICKER})].copy()
        if snapshot.empty or BENCHMARK_TICKER not in available:
            continue

        history_returns = returns.loc[returns.index < forecast_start].tail(BOOTSTRAP_LOOKBACK_DAYS)
        candidate_rows = []
        seed_base = int(forecast_start.year * 10 + forecast_start.quarter)
        for core_weight in CORE_WEIGHT_CANDIDATES:
            for top_k in TOP_K_CANDIDATES:
                if len(snapshot) < top_k:
                    continue
                try:
                    weights, selected = build_weights(snapshot, core_weight, top_k)
                    if not set(weights.index).issubset(available):
                        continue
                    port_sims, bench_sims = block_bootstrap(
                        history_returns,
                        weights,
                        horizon_days=max(horizon_days, HORIZON_DAYS),
                        n_sims=BOOTSTRAP_N_SIMS,
                        seed=seed_base + int(core_weight * 100) + top_k,
                    )
                except ValueError:
                    continue
                turnover = one_way_turnover(weights, previous_weights)
                p_beat = float(np.mean(port_sims > bench_sims))
                p_lt0 = float(np.mean(port_sims < 0.0))
                sharpe = float(np.mean(port_sims) / np.std(port_sims, ddof=1)) if np.std(port_sims, ddof=1) > 0 else np.nan
                candidate_rows.append(
                    {
                        "core_weight": core_weight,
                        "top_k": top_k,
                        "weights": weights,
                        "selected": selected,
                        "turnover": turnover,
                        "turnover_violation": max(0.0, turnover - TURNOVER_CAP),
                        "p_beat_0050": p_beat,
                        "p_lt0": p_lt0,
                        "sharpe": sharpe,
                    }
                )
        candidates = pd.DataFrame(candidate_rows)
        if candidates.empty:
            continue
        candidates["utility"] = (
            zscore(candidates["p_beat_0050"])
            - zscore(candidates["p_lt0"])
            + 0.3 * zscore(candidates["sharpe"].fillna(candidates["sharpe"].median()))
        )
        feasible = candidates[candidates["turnover"] <= TURNOVER_CAP].copy()
        if not feasible.empty:
            chosen = feasible.sort_values("utility", ascending=False).iloc[0]
            stage = "feasible"
        else:
            chosen = candidates.sort_values(["turnover_violation", "utility"], ascending=[True, False]).iloc[0]
            stage = "fallback_turnover"

        weights = chosen["weights"]
        selected = chosen["selected"]
        cost = trading_cost(weights, previous_weights)
        actual_return_gross = np.nan if window.is_current else realized_return(prices, weights, forecast_start, forecast_end)
        actual_return_net = np.nan if np.isnan(actual_return_gross) else actual_return_gross - cost
        benchmark_return = (
            np.nan
            if window.is_current
            else realized_return(prices, pd.Series({BENCHMARK_TICKER: 1.0}), forecast_start, forecast_end)
        )
        selected_with_weights = "; ".join(
            f"{ticker.replace('.TW', '')} {name_lookup.get(ticker, '')} {weights[ticker] * 100:.1f}%"
            for ticker in selected
        )
        rows.append(
            {
                "period": period_label,
                "snapshot_date": forecast_start.date().isoformat(),
                "start": forecast_start.date().isoformat(),
                "end": forecast_end.date().isoformat(),
                "stage": stage,
                "core_0050": float(weights.get(CORE_TICKER, 0.0)),
                "top_k": int(chosen["top_k"]),
                "selected": selected_with_weights,
                "utility": float(chosen["utility"]),
                "sharpe": float(chosen["sharpe"]),
                "p_beat_0050": float(chosen["p_beat_0050"]),
                "p_lt0": float(chosen["p_lt0"]),
                "turnover": float(chosen["turnover"]),
                "cost": cost,
                "actual_return": actual_return_net,
                "benchmark_0050_return": benchmark_return,
                "beat_0050": bool(actual_return_net > benchmark_return)
                if not np.isnan(actual_return_net) and not np.isnan(benchmark_return)
                else pd.NA,
            }
        )
        previous_weights = weights
    return pd.DataFrame(rows)


def summarize_walkforward(walkforward: pd.DataFrame) -> pd.DataFrame:
    realized = walkforward.dropna(subset=["actual_return", "benchmark_0050_return"]).copy()
    quarterly = realized["actual_return"].astype(float)
    benchmark = realized["benchmark_0050_return"].astype(float)
    sharpe = np.nan
    benchmark_sharpe = np.nan
    if quarterly.std(ddof=1) > 0:
        sharpe = float(quarterly.mean() / quarterly.std(ddof=1) * np.sqrt(4))
    if benchmark.std(ddof=1) > 0:
        benchmark_sharpe = float(benchmark.mean() / benchmark.std(ddof=1) * np.sqrt(4))
    summary = pd.DataFrame(
        [
            {
                "quarters": int(len(realized)),
                "start_period": realized["period"].iloc[0] if not realized.empty else None,
                "end_period": realized["period"].iloc[-1] if not realized.empty else None,
                "strategy_total_return": float((1.0 + quarterly).prod() - 1.0) if not realized.empty else np.nan,
                "benchmark_total_return": float((1.0 + benchmark).prod() - 1.0) if not realized.empty else np.nan,
                "strategy_quarterly_sharpe": sharpe,
                "benchmark_quarterly_sharpe": benchmark_sharpe,
                "strategy_max_drawdown": compute_max_drawdown(quarterly) if not realized.empty else np.nan,
                "benchmark_max_drawdown": compute_max_drawdown(benchmark) if not realized.empty else np.nan,
                "beat_0050_rate": float(realized["beat_0050"].astype(bool).mean()) if not realized.empty else np.nan,
                "avg_turnover": float(realized["turnover"].mean()) if not realized.empty else np.nan,
                "latest_period": walkforward["period"].iloc[-1] if not walkforward.empty else None,
            }
        ]
    )
    return summary


def compute_prediction_summary(predictions: pd.DataFrame) -> dict[str, float]:
    realized = predictions.dropna(subset=["forward_20d_return"]).copy()
    ic_rows = []
    for date, group in realized.groupby("Date"):
        if len(group) < 5:
            continue
        ic, _ = spearmanr(group["predicted_return"], group["forward_20d_return"])
        if pd.notna(ic):
            ic_rows.append(ic)
    ic_series = pd.Series(ic_rows, dtype=float)
    return {
        "prediction_rows": float(len(predictions)),
        "ic_mean": float(ic_series.mean()) if not ic_series.empty else np.nan,
        "icir": float(ic_series.mean() / ic_series.std(ddof=1) * np.sqrt(4)) if len(ic_series) > 1 and ic_series.std(ddof=1) > 0 else np.nan,
    }


def main() -> None:
    configure_runtime_dirs()
    universe = fetch_0050_universe()
    source_date = universe["source_date"].iloc[0]
    existing_codes = set(universe["code"].astype(str))
    extra_rows = [
        {"source_date": source_date, "code": code, "ticker": f"{code}.TW",
         "name": name, "english_name": "", "pcf_qty": 0}
        for code, name in EXTRA_TICKERS
        if code not in existing_codes
    ]
    if extra_rows:
        universe = pd.concat([universe, pd.DataFrame(extra_rows)], ignore_index=True).sort_values("code").reset_index(drop=True)
    tickers = [CORE_TICKER] + universe["ticker"].tolist()
    print(f"Parsed 0050 universe: {len(universe)} stocks (incl. {len(extra_rows)} extra), source date {source_date}")
    prices_long = download_price_history(tickers, TRAIN_START, CURRENT_DATE)
    downloaded_tickers = sorted(prices_long["ticker"].unique())
    missing = sorted(set(tickers) - set(downloaded_tickers))
    if missing:
        print(f"Missing yfinance price history: {', '.join(missing)}")
    write_sqlite(universe, prices_long)
    universe.to_csv(UNIVERSE_CSV, index=False)

    model_tickers = [ticker for ticker in universe["ticker"].tolist() if ticker in downloaded_tickers]
    features = build_feature_frame(load_ohlcv_by_ticker(model_tickers), keep_all_rows=True)
    prices = load_prices_from_sqlite([CORE_TICKER] + model_tickers)
    latest_date = min(pd.Timestamp(CURRENT_DATE), prices.dropna(how="all").index.max())
    predictions = run_quarterly_predictions(features, latest_date)
    predictions.to_csv(PREDICTIONS_CSV, index=False)

    walkforward = run_walkforward(predictions, prices, universe)
    summary = summarize_walkforward(walkforward)
    pred_summary = compute_prediction_summary(predictions)
    for key, value in pred_summary.items():
        summary[key] = value
    walkforward.to_csv(WALKFORWARD_CSV, index=False)
    summary.to_csv(SUMMARY_CSV, index=False)
    with sqlite3.connect(DB_PATH) as connection:
        predictions.to_sql("tw_0050_predictions", connection, if_exists="replace", index=False)
        walkforward.to_sql("tw_0050_walkforward", connection, if_exists="replace", index=False)
        summary.to_sql("tw_0050_walkforward_summary", connection, if_exists="replace", index=False)

    print("\nTaiwan 0050 walk-forward summary:")
    print(summary.to_string(index=False))
    print("\nLatest allocation:")
    print(walkforward.tail(1).T.to_string())


if __name__ == "__main__":
    main()
