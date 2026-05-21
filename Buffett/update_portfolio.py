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

# 觀察池：追蹤但未持有的股票
WATCHLIST_TICKERS = [
    "AAPL", "AMD", "AMZN", "ARM", "AVGO", "BE", "CLS", "COHR", "CORZ", "CRM",
    "CRWD", "CRWV", "DELL", "GLW", "GOOGL", "IBM", "IFNNY", "INTC", "IREN",
    "LITE", "LOGI", "META", "MRVL", "MSFT", "MSTR", "MU", "NET", "NOK", "NOW",
    "NTAP", "NVDA", "OKTA", "ORCL", "PANW", "PLTR", "QCOM", "SNDK", "SSO",
    "TEAM", "TSLA", "TSM", "VIX", "VRT",
]
WATCHLIST_TICKER_MAP = {"VIX": "^VIX"}
WATCHLIST_BENCHMARKS = ["SPY", "QQQ"]

WATCHLIST_CATEGORIES = {
    "COHR":  "光通訊",
    "LITE":  "光通訊",
    "GLW":   "光通訊",
    "MRVL":  "光通訊",
    "NOK":   "光通訊",
    "NVDA":  "AI 晶片",
    "AMD":   "AI 晶片",
    "ARM":   "AI 晶片",
    "AVGO":  "AI 晶片",
    "INTC":  "AI 晶片",
    "TSM":   "AI 晶片",
    "MU":    "AI 晶片",
    "QCOM":  "AI 晶片",
    "IFNNY": "AI 晶片",
    "SNDK":  "AI 晶片",
    "MSFT":  "AI 雲端 & 軟體",
    "GOOGL": "AI 雲端 & 軟體",
    "AMZN":  "AI 雲端 & 軟體",
    "META":  "AI 雲端 & 軟體",
    "CRWV":  "AI 雲端 & 軟體",
    "ORCL":  "AI 雲端 & 軟體",
    "IBM":   "AI 雲端 & 軟體",
    "NOW":   "AI 雲端 & 軟體",
    "CRM":   "AI 雲端 & 軟體",
    "PLTR":  "AI 雲端 & 軟體",
    "CRWD":  "資安",
    "PANW":  "資安",
    "NET":   "資安",
    "OKTA":  "資安",
    "VRT":   "電力 & 資料中心",
    "BE":    "電力 & 資料中心",
    "CLS":   "電力 & 資料中心",
    "DELL":  "電力 & 資料中心",
    "TSLA":  "電力 & 資料中心",
    "CORZ":  "比特幣 & 加密",
    "IREN":  "比特幣 & 加密",
    "MSTR":  "比特幣 & 加密",
    "AAPL":  "消費 & 其他",
    "LOGI":  "消費 & 其他",
    "TEAM":  "消費 & 其他",
    "NTAP":  "消費 & 其他",
    "SSO":   "消費 & 其他",
    "VIX":   "消費 & 其他",
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


def ensure_watchlist_history_table(conn):
    conn.execute("""CREATE TABLE IF NOT EXISTS watchlist_price_history (
        date   TEXT,
        ticker TEXT,
        close  REAL,
        PRIMARY KEY (date, ticker)
    )""")
    conn.commit()


def update_watchlist_history(conn, tickers: list[str]) -> str:
    all_tickers = sorted(set(tickers + WATCHLIST_BENCHMARKS))
    yf_tickers = [WATCHLIST_TICKER_MAP.get(t, t) for t in all_tickers]

    # backfill 40 天（首次或資料不足），否則只補最近 5 天
    count = conn.execute("SELECT COUNT(DISTINCT date) FROM watchlist_price_history").fetchone()[0]
    period = "40d" if count < 20 else "5d"

    raw = yf.download(yf_tickers, period=period, progress=False, auto_adjust=True)["Close"]
    if isinstance(raw, pd.Series):
        raw = raw.to_frame(name=yf_tickers[0])
    raw = raw.dropna(how="all")

    inserted = 0
    for date_idx in raw.index:
        date_str = str(date_idx.date())
        for ticker in all_tickers:
            yf_key = WATCHLIST_TICKER_MAP.get(ticker, ticker)
            val = raw.loc[date_idx, yf_key] if yf_key in raw.columns else None
            if val is None or (hasattr(val, "__float__") and pd.isna(float(val))):
                val = raw.loc[date_idx, ticker] if ticker in raw.columns else None
            if val is not None and not pd.isna(val):
                conn.execute(
                    "INSERT OR IGNORE INTO watchlist_price_history (date, ticker, close) VALUES (?,?,?)",
                    (date_str, ticker, float(val)),
                )
                inserted += 1
    conn.commit()
    return f"{'Backfill' if period == '40d' else '更新'} {inserted} 筆歷史價格（{period}）"


def ensure_watchlist_tables(conn):
    conn.execute("""CREATE TABLE IF NOT EXISTS watchlist (
        ticker   TEXT PRIMARY KEY,
        name     TEXT,
        category TEXT
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS watchlist_snapshot (
        snapshot_date TEXT,
        ticker        TEXT,
        price         REAL,
        prev_close    REAL,
        change_dollar REAL,
        change_pct    REAL,
        PRIMARY KEY (snapshot_date, ticker)
    )""")
    # add category column if missing (migration)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(watchlist)").fetchall()}
    if "category" not in cols:
        conn.execute("ALTER TABLE watchlist ADD COLUMN category TEXT")
    existing = {r[0] for r in conn.execute("SELECT ticker FROM watchlist").fetchall()}
    for ticker in WATCHLIST_TICKERS:
        cat = WATCHLIST_CATEGORIES.get(ticker, "其他")
        if ticker not in existing:
            conn.execute("INSERT OR IGNORE INTO watchlist (ticker, category) VALUES (?,?)", (ticker, cat))
        else:
            conn.execute("UPDATE watchlist SET category=? WHERE ticker=?", (cat, ticker))
    conn.commit()


def get_watchlist_prices(tickers: list[str]) -> dict[str, dict]:
    yf_tickers = [WATCHLIST_TICKER_MAP.get(t, t) for t in tickers]
    raw = yf.download(yf_tickers, period="5d", progress=False, auto_adjust=True)["Close"]
    if isinstance(raw, pd.Series):
        raw = raw.to_frame(name=yf_tickers[0])
    raw = raw.dropna(how="all")
    if len(raw) < 2:
        return {}
    results = {}
    for ticker in tickers:
        yf_key = WATCHLIST_TICKER_MAP.get(ticker, ticker)
        col = raw.get(yf_key) if yf_key in raw.columns else raw.get(ticker)
        if col is None:
            continue
        col = col.dropna()
        if len(col) < 2:
            continue
        price = float(col.iloc[-1])
        prev_close = float(col.iloc[-2])
        change_dollar = price - prev_close
        change_pct = change_dollar / prev_close * 100 if prev_close else 0.0
        results[ticker] = {
            "price": price,
            "prev_close": prev_close,
            "change_dollar": change_dollar,
            "change_pct": change_pct,
        }
    return results


def insert_watchlist_snapshot(conn, watchlist_prices: dict[str, dict]):
    existing = {r[0] for r in conn.execute(
        "SELECT ticker FROM watchlist_snapshot WHERE snapshot_date=?", (TODAY,)
    ).fetchall()}
    for ticker, data in watchlist_prices.items():
        if ticker in existing:
            conn.execute("""UPDATE watchlist_snapshot
                SET price=?, prev_close=?, change_dollar=?, change_pct=?
                WHERE snapshot_date=? AND ticker=?""",
                (data["price"], data["prev_close"], data["change_dollar"], data["change_pct"], TODAY, ticker))
        else:
            conn.execute("""INSERT INTO watchlist_snapshot
                (snapshot_date, ticker, price, prev_close, change_dollar, change_pct)
                VALUES (?,?,?,?,?,?)""",
                (TODAY, ticker, data["price"], data["prev_close"], data["change_dollar"], data["change_pct"]))
    conn.commit()


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
    ensure_watchlist_tables(conn)
    ensure_watchlist_history_table(conn)

    tickers = [r[0] for r in conn.execute("SELECT ticker FROM holdings").fetchall()]
    print(f"下載最新報價（{', '.join(tickers)}）...")
    prices, price_date = get_prices(tickers)
    print(f"取得 {len(prices)}/{len(tickers)} 檔即時報價（交易日：{price_date or '無'}）")

    missing = [t for t in tickers if t not in prices]
    if missing:
        fallback = fallback_from_snapshot(conn, missing)
        for t, p in fallback.items():
            prices[t] = p
        print(f"回退至歷史快照補齊：{', '.join(fallback.keys())}")

    update(conn, prices)
    insert_snapshot(conn, prices)
    print_summary(conn)

    wl_tickers = [r[0] for r in conn.execute("SELECT ticker FROM watchlist ORDER BY ticker").fetchall()]
    print(f"下載觀察池報價（{len(wl_tickers)} 檔）...")
    watchlist_prices = get_watchlist_prices(wl_tickers)
    print(f"觀察池取得 {len(watchlist_prices)}/{len(wl_tickers)} 檔報價")
    insert_watchlist_snapshot(conn, watchlist_prices)

    print("更新觀察池歷史價格...")
    msg = update_watchlist_history(conn, wl_tickers)
    print(msg)

    conn.close()
