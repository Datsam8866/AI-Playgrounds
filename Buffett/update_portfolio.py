"""
update_portfolio.py
從 yfinance 抓最新收盤價，更新 portfolio.sqlite 並顯示損益摘要。
每個交易日收盤後執行一次即可。
"""
import sqlite3
import sys
import warnings
from datetime import date

import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DB_PATH = "portfolio.sqlite"
TODAY = str(date.today())

# yfinance ticker 對照（台股加 .TW）
TICKER_MAP = {
    "0050": "0050.TW",
}


def get_prices(tickers: list[str]) -> tuple[dict[str, float], str]:
    """回傳 (prices, price_date)，price_date 為實際取得報價的交易日。"""
    yf_tickers = [TICKER_MAP.get(t, t) for t in tickers]
    raw = yf.download(yf_tickers, period="5d", progress=False, auto_adjust=True)["Close"]
    if isinstance(raw, pd.Series):
        raw = raw.to_frame(name=yf_tickers[0])
    raw = raw.dropna(how="all")
    if raw.empty:
        return {}, ""

    # 取最後一個有資料的交易日
    last_date = str(raw.index[-1].date())
    last_row = raw.iloc[-1]

    prices = {}
    for ticker in tickers:
        yf_key = TICKER_MAP.get(ticker, ticker)
        # yfinance multi-ticker 欄名可能帶 tuple，嘗試多種 key 格式
        val = last_row.get(yf_key)
        if val is None or pd.isna(val):
            # 嘗試只用 ticker 本身
            val = last_row.get(ticker)
        if val is not None and not pd.isna(val):
            prices[ticker] = float(val)

    return prices, last_date


def add_last_price_columns(conn):
    cols = [r[1] for r in conn.execute("PRAGMA table_info(holdings)").fetchall()]
    if "last_price" not in cols:
        conn.execute("ALTER TABLE holdings ADD COLUMN last_price REAL")
    if "last_price_date" not in cols:
        conn.execute("ALTER TABLE holdings ADD COLUMN last_price_date TEXT")
    conn.commit()


def update(conn, prices: dict[str, float]):
    for ticker, price in prices.items():
        conn.execute(
            "UPDATE holdings SET last_price=?, last_price_date=? WHERE ticker=?",
            (price, TODAY, ticker),
        )
    conn.commit()


def insert_snapshot(conn, prices: dict[str, float]):
    # 避免同一天重複寫入
    existing = {
        r[0]
        for r in conn.execute(
            "SELECT ticker FROM holdings_snapshot WHERE snapshot_date=?", (TODAY,)
        ).fetchall()
    }
    rows = conn.execute(
        "SELECT market, ticker, name, shares, avg_cost, currency FROM holdings"
    ).fetchall()

    for market, ticker, name, shares, avg_cost, currency in rows:
        if ticker in existing:
            continue
        price = prices.get(ticker)
        if price is None:
            continue
        mv = shares * price
        pnl = mv - shares * avg_cost if avg_cost else None
        pnl_pct = (pnl / (shares * avg_cost) * 100) if (avg_cost and pnl is not None) else None
        conn.execute(
            """INSERT INTO holdings_snapshot
               (snapshot_date, market, ticker, name, shares, price,
                market_value, avg_cost, pnl, pnl_pct, currency)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (TODAY, market, ticker, name, shares, price, mv, avg_cost, pnl, pnl_pct, currency),
        )
    conn.commit()


def print_summary(conn):
    rows = conn.execute("""
        SELECT h.market, h.ticker, h.name, h.shares, h.avg_cost,
               h.last_price, h.last_price_date, h.currency
        FROM holdings h
        ORDER BY h.market DESC, h.ticker
    """).fetchall()

    us_rows = [r for r in rows if r[0] == "US"]
    tw_rows = [r for r in rows if r[0] == "TW"]

    def section(label, data, currency):
        print(f"\n  {'─'*76}")
        print(f"  {label}")
        print(f"  {'─'*76}")
        print(f"  {'代號':<6} {'股數':>5} {'均價':>9} {'總成本':>12} {'現價':>9} {'市值':>12} {'損益':>10} {'損益%':>7}")
        print(f"  {'─'*76}")
        tc = tm = tp = 0.0
        for _, ticker, name, shares, avg_cost, price, pdate, _ in data:
            cost = shares * avg_cost if avg_cost else 0.0
            mv   = shares * price if price else 0.0
            pnl  = mv - cost
            pct  = pnl / cost * 100 if cost else 0.0
            tc += cost; tm += mv; tp += pnl
            p_str = f"{price:>9.2f}" if price else "      N/A"
            mv_str = f"{mv:>12,.2f}" if price else "           N/A"
            print(f"  {ticker:<6} {shares:>5g}  {avg_cost:>9.2f}  {cost:>12,.2f}  {p_str}  {mv_str}  {pnl:>10,.2f}  {pct:>+6.1f}%")
        print(f"  {'─'*76}")
        tpct = tp / tc * 100 if tc else 0.0
        print(f"  {'小計':<6} {'':>5}  {'':>9}  {tc:>12,.2f}  {'':>9}  {tm:>12,.2f}  {tp:>10,.2f}  {tpct:>+6.1f}%")
        return tc, tm, tp

    print(f"\n{'='*80}")
    print(f"  持倉損益摘要  [{TODAY}]")
    print(f"{'='*80}")

    tc_us, tm_us, tp_us = section("美股（USD）", us_rows, "USD")
    tc_tw, tm_tw, tp_tw = section("台股（TWD）", tw_rows, "TWD")

    print(f"\n  {'─'*76}")
    print(f"  美股總損益：${tp_us:>10,.2f} USD  ({tp_us/tc_us*100:>+.1f}%)" if tc_us else "")
    print(f"  台股總損益：${tp_tw:>10,.2f} TWD  ({tp_tw/tc_tw*100:>+.1f}%)" if tc_tw else "")
    print(f"{'='*80}\n")


def fallback_from_snapshot(conn, tickers: list[str]) -> dict[str, float]:
    """從最近一次 snapshot 補齊缺少報價的 ticker。"""
    rows = conn.execute("""
        SELECT ticker, price FROM holdings_snapshot
        WHERE (ticker, snapshot_date) IN (
            SELECT ticker, MAX(snapshot_date)
            FROM holdings_snapshot
            GROUP BY ticker
        )
    """).fetchall()
    return {r[0]: r[1] for r in rows if r[0] in tickers and r[1] is not None}


if __name__ == "__main__":
    conn = sqlite3.connect(DB_PATH)
    add_last_price_columns(conn)

    tickers = [r[0] for r in conn.execute("SELECT ticker FROM holdings").fetchall()]
    print(f"下載最新報價（{', '.join(tickers)}）...")
    prices, price_date = get_prices(tickers)
    print(f"取得 {len(prices)}/{len(tickers)} 檔即時報價（交易日：{price_date or '無'}）")

    # 補齊缺少的（休市、下市等）
    missing = [t for t in tickers if t not in prices]
    if missing:
        fallback = fallback_from_snapshot(conn, missing)
        for t, p in fallback.items():
            prices[t] = p
        print(f"回退至歷史快照補齊：{', '.join(fallback.keys())}")

    update(conn, prices)
    insert_snapshot(conn, prices)
    print_summary(conn)
    conn.close()
