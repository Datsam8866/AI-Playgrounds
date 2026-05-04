# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


ROOT = Path(__file__).resolve().parent
DEFAULT_DB_PATH = ROOT / "stock_forecast.sqlite"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query the stock_forecast SQLite database."
    )
    parser.add_argument(
        "--db",
        default=str(DEFAULT_DB_PATH),
        help="Path to SQLite database. Default: stock_forecast.sqlite in current project.",
    )
    parser.add_argument(
        "--format",
        choices=["table", "csv", "json"],
        default="table",
        help="Output format.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("tables", help="List tables and row counts.")
    subparsers.add_parser("metadata", help="Show metadata entries.")

    signal_parser = subparsers.add_parser("signal", help="Query daily IC-Adaptive signals.")
    signal_parser.add_argument("--date", help="Trading date, e.g. 2026-04-09. Default: latest available date.")
    signal_parser.add_argument("--ticker", help="Optional ticker filter.")
    signal_parser.add_argument(
        "--selected-only",
        action="store_true",
        help="Show selected holdings only.",
    )
    signal_parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Maximum rows to return.",
    )

    portfolio_signal_parser = subparsers.add_parser("portfolio-signal", help="Query portfolio daily signals.")
    portfolio_signal_parser.add_argument("--date", help="Signal date, e.g. 2026-04-09. Default: latest available date.")
    portfolio_signal_parser.add_argument("--ticker", help="Optional ticker filter.")
    portfolio_signal_parser.add_argument(
        "--selected-only",
        action="store_true",
        help="Show selected holdings only.",
    )
    portfolio_signal_parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Maximum rows to return.",
    )

    prediction_parser = subparsers.add_parser("prediction", help="Query regression predictions.")
    prediction_parser.add_argument("--date", help="Trading date. Default: latest available date.")
    prediction_parser.add_argument("--ticker", help="Optional ticker filter.")
    prediction_parser.add_argument(
        "--split",
        choices=["train", "validation", "test"],
        help="Optional split filter.",
    )
    prediction_parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Maximum rows to return.",
    )

    price_parser = subparsers.add_parser("price", help="Query price history.")
    price_parser.add_argument("--ticker", required=True, help="Ticker symbol.")
    price_parser.add_argument("--start", help="Start date, inclusive.")
    price_parser.add_argument("--end", help="End date, inclusive.")
    price_parser.add_argument(
        "--macro",
        action="store_true",
        help="Query macro_history instead of price_history.",
    )
    price_parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Maximum rows to return.",
    )
    price_parser.add_argument(
        "--desc",
        action="store_true",
        help="Sort by newest first.",
    )

    monthly_parser = subparsers.add_parser("monthly", help="Query monthly backtest summary.")
    monthly_parser.add_argument("--start", help="Start month/date, inclusive.")
    monthly_parser.add_argument("--end", help="End month/date, inclusive.")
    monthly_parser.add_argument(
        "--limit",
        type=int,
        default=120,
        help="Maximum rows to return.",
    )
    monthly_parser.add_argument(
        "--desc",
        action="store_true",
        help="Sort by newest first.",
    )

    sql_parser = subparsers.add_parser("sql", help="Run a read-only SQL query.")
    sql_parser.add_argument("query", help="Read-only SQL. Allowed starts: SELECT, WITH, PRAGMA.")

    return parser.parse_args()


def get_connection(db_path: str) -> sqlite3.Connection:
    return sqlite3.connect(db_path)


def latest_date(connection: sqlite3.Connection, table_name: str) -> str | None:
    query = f"SELECT MAX(date) FROM {table_name}"
    value = connection.execute(query).fetchone()[0]
    return value


def normalize_date_filter(date_value: str) -> tuple[str, str]:
    normalized = pd.Timestamp(date_value)
    start = normalized.strftime("%Y-%m-%d 00:00:00")
    end = (normalized + pd.Timedelta(days=1)).strftime("%Y-%m-%d 00:00:00")
    return start, end


def format_dataframe(dataframe: pd.DataFrame, output_format: str) -> str:
    if dataframe.empty:
        return "(no rows)"
    if output_format == "csv":
        return dataframe.to_csv(index=False)
    if output_format == "json":
        return json.dumps(
            json.loads(dataframe.to_json(orient="records", date_format="iso")),
            ensure_ascii=False,
            indent=2,
        )
    return dataframe.to_string(index=False)


def query_tables(connection: sqlite3.Connection) -> pd.DataFrame:
    tables = pd.read_sql_query(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
        ORDER BY name
        """,
        connection,
    )
    if tables.empty:
        return tables

    rows = []
    for table_name in tables["name"]:
        row_count = connection.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        rows.append({"table_name": table_name, "row_count": row_count})
    return pd.DataFrame(rows)


def query_metadata(connection: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT key, value FROM metadata ORDER BY key",
        connection,
    )


def query_signal(connection: sqlite3.Connection, args: argparse.Namespace) -> pd.DataFrame:
    date_value = args.date or latest_date(connection, "daily_signals_ic_adaptive")
    if date_value is None:
        return pd.DataFrame()

    start, end = normalize_date_filter(date_value)
    clauses = ["date >= ?", "date < ?"]
    params: list[object] = [start, end]
    if args.ticker:
        clauses.append("ticker = ?")
        params.append(args.ticker.upper())
    if args.selected_only:
        clauses.append("selected = 1")

    query = f"""
    SELECT
        date,
        ticker,
        split,
        predicted_return,
        forward_20d_return,
        next_day_return,
        rolling_ic,
        k_value,
        rank,
        selected,
        weight,
        signal_mode
    FROM daily_signals_ic_adaptive
    WHERE {" AND ".join(clauses)}
    ORDER BY rank ASC, ticker ASC
    LIMIT ?
    """
    params.append(args.limit)
    return pd.read_sql_query(query, connection, params=params)


def query_portfolio_signal(connection: sqlite3.Connection, args: argparse.Namespace) -> pd.DataFrame:
    date_value = args.date or latest_date(connection, "portfolio_daily_signals")
    if date_value is None:
        return pd.DataFrame()

    clauses = ["date(date) = date(?)"]
    params: list[object] = [str(date_value)]
    if args.ticker:
        clauses.append("ticker = ?")
        params.append(args.ticker.upper())
    if args.selected_only:
        clauses.append("selected = 1")

    query = f"""
    SELECT
        date,
        prediction_date,
        ticker,
        predicted_return,
        rank,
        selected,
        base_weight,
        final_weight,
        rolling_ic,
        k_value,
        signal_mode,
        vix_value,
        vix_status,
        spy_vs_sma200,
        spy_status,
        risk_overlay_on,
        net_exposure,
        risk_rule,
        risk_triggers,
        quarterly_period,
        quarterly_regime,
        quarterly_selection_stage,
        quarterly_score_combined_z,
        quarterly_p_gt_5,
        quarterly_p_lt_0,
        quarterly_raw_p_gt_5,
        quarterly_raw_p_lt_0,
        generated_at
    FROM portfolio_daily_signals
    WHERE {" AND ".join(clauses)}
    ORDER BY rank ASC, ticker ASC
    LIMIT ?
    """
    params.append(args.limit)
    return pd.read_sql_query(query, connection, params=params)


def query_prediction(connection: sqlite3.Connection, args: argparse.Namespace) -> pd.DataFrame:
    date_value = args.date or latest_date(connection, "model_predictions_regression")
    if date_value is None:
        return pd.DataFrame()

    start, end = normalize_date_filter(date_value)
    clauses = ["date >= ?", "date < ?"]
    params: list[object] = [start, end]
    if args.ticker:
        clauses.append("ticker = ?")
        params.append(args.ticker.upper())
    if args.split:
        clauses.append("split = ?")
        params.append(args.split)

    query = f"""
    SELECT
        date,
        ticker,
        split,
        predicted_return,
        forward_20d_return,
        next_day_return
    FROM model_predictions_regression
    WHERE {" AND ".join(clauses)}
    ORDER BY predicted_return DESC, ticker ASC
    LIMIT ?
    """
    params.append(args.limit)
    return pd.read_sql_query(query, connection, params=params)


def query_price(connection: sqlite3.Connection, args: argparse.Namespace) -> pd.DataFrame:
    table_name = "macro_history" if args.macro else "price_history"
    clauses = ["ticker = ?"]
    params: list[object] = [args.ticker.upper()]
    if args.start:
        clauses.append("date >= ?")
        params.append(args.start)
    if args.end:
        clauses.append("date <= ?")
        params.append(args.end)

    order = "DESC" if args.desc else "ASC"
    query = f"""
    SELECT
        date,
        ticker,
        open,
        high,
        low,
        close,
        adj_close,
        volume
    FROM {table_name}
    WHERE {" AND ".join(clauses)}
    ORDER BY date {order}
    LIMIT ?
    """
    params.append(args.limit)
    return pd.read_sql_query(query, connection, params=params)


def query_monthly(connection: sqlite3.Connection, args: argparse.Namespace) -> pd.DataFrame:
    clauses = ["1 = 1"]
    params: list[object] = []
    if args.start:
        clauses.append("date >= ?")
        params.append(args.start)
    if args.end:
        clauses.append("date <= ?")
        params.append(args.end)

    order = "DESC" if args.desc else "ASC"
    query = f"""
    SELECT
        date,
        strategy_monthly,
        benchmark_monthly,
        alpha,
        strategy_cum,
        benchmark_cum
    FROM backtest_monthly
    WHERE {" AND ".join(clauses)}
    ORDER BY date {order}
    LIMIT ?
    """
    params.append(args.limit)
    return pd.read_sql_query(query, connection, params=params)


def query_sql(connection: sqlite3.Connection, args: argparse.Namespace) -> pd.DataFrame:
    sql = args.query.strip()
    sql_head = sql.split(maxsplit=1)[0].lower() if sql else ""
    if sql_head not in {"select", "with", "pragma"}:
        raise ValueError("Only read-only SQL is allowed. Start the query with SELECT, WITH, or PRAGMA.")
    return pd.read_sql_query(sql, connection)


def main() -> None:
    args = parse_args()
    with get_connection(args.db) as connection:
        if args.command == "tables":
            dataframe = query_tables(connection)
        elif args.command == "metadata":
            dataframe = query_metadata(connection)
        elif args.command == "signal":
            dataframe = query_signal(connection, args)
        elif args.command == "portfolio-signal":
            dataframe = query_portfolio_signal(connection, args)
        elif args.command == "prediction":
            dataframe = query_prediction(connection, args)
        elif args.command == "price":
            dataframe = query_price(connection, args)
        elif args.command == "monthly":
            dataframe = query_monthly(connection, args)
        elif args.command == "sql":
            dataframe = query_sql(connection, args)
        else:
            raise ValueError(f"Unsupported command: {args.command}")

    print(format_dataframe(dataframe, args.format))


if __name__ == "__main__":
    main()
