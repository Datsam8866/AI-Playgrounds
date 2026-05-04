"""
init_portfolio.py
建立 portfolio.sqlite，並初始化持倉資料。
執行一次即可；之後用 update_portfolio.py 更新。
"""
import sqlite3
import sys
from datetime import date

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DB_PATH = "portfolio.sqlite"
TODAY = str(date.today())

# --- Schema ---
SCHEMA = """
CREATE TABLE IF NOT EXISTS holdings (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    market      TEXT NOT NULL,        -- 'US' or 'TW'
    ticker      TEXT NOT NULL,
    name        TEXT,
    shares      REAL NOT NULL,
    avg_cost    REAL,                 -- 成本均價（本地幣）
    currency    TEXT NOT NULL,        -- 'USD' or 'TWD'
    last_updated TEXT,
    UNIQUE(market, ticker)
);

CREATE TABLE IF NOT EXISTS holdings_snapshot (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_date TEXT NOT NULL,
    market        TEXT NOT NULL,
    ticker        TEXT NOT NULL,
    name          TEXT,
    shares        REAL,
    price         REAL,               -- 快照當下價格
    market_value  REAL,               -- shares * price
    avg_cost      REAL,
    pnl           REAL,               -- 損益金額
    pnl_pct       REAL,               -- 損益率 %
    currency      TEXT,
    created_at    TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

# --- 初始持倉（2026-05-04 截圖資料）---
US_HOLDINGS = [
    # (ticker, name, shares, price_snapshot, avg_cost)
    # avg_cost 目前未知，先填 None；之後可手動更新
    ("AMD",  "Advanced Micro Devices", 40,  364.57, None),
    ("ARM",  "Arm Holdings",           60,  213.45, None),
    ("AVGO", "Broadcom",               29,  421.92, None),
    ("COHR", "Coherent Corp",          30,  341.60, None),
    ("INTC", "Intel",                 173,  100.85, None),
    ("LITE", "Lumentum",               11,  981.06, None),
    ("NVDA", "NVIDIA",                 58,  199.78, None),
    ("SNDK", "SanDisk",                11, 1230.71, None),
    ("TSLA", "Tesla",                  24,  392.75, None),
    ("TSM",  "TSMC ADR",               27,  406.66, None),
    ("VOO",  "Vanguard S&P 500 ETF",  258,  663.54, None),
]

TW_HOLDINGS = [
    # (ticker, name, shares, price_snapshot, avg_cost)
    # 0050 avg_cost = 92.49（成交均）
    # price ≈ 92.49 + 18841/10000 = 94.37
    ("0050", "元大台灣50", 10000, 94.37, 92.49),
]


def init_db(conn):
    conn.executescript(SCHEMA)
    conn.commit()


def seed_holdings(conn):
    cur = conn.cursor()

    for ticker, name, shares, _, avg_cost in US_HOLDINGS:
        cur.execute("""
            INSERT OR REPLACE INTO holdings
                (market, ticker, name, shares, avg_cost, currency, last_updated)
            VALUES (?, ?, ?, ?, ?, 'USD', ?)
        """, ("US", ticker, name, shares, avg_cost, TODAY))

    for ticker, name, shares, _, avg_cost in TW_HOLDINGS:
        cur.execute("""
            INSERT OR REPLACE INTO holdings
                (market, ticker, name, shares, avg_cost, currency, last_updated)
            VALUES (?, ?, ?, ?, ?, 'TWD', ?)
        """, ("TW", ticker, name, shares, avg_cost, TODAY))

    conn.commit()
    print(f"holdings 初始化完成：{len(US_HOLDINGS)} 檔美股 + {len(TW_HOLDINGS)} 檔台股")


def seed_snapshot(conn):
    cur = conn.cursor()

    for ticker, name, shares, price, avg_cost in US_HOLDINGS:
        mv = shares * price
        pnl = (price - avg_cost) * shares if avg_cost else None
        pnl_pct = (price / avg_cost - 1) * 100 if avg_cost else None
        cur.execute("""
            INSERT INTO holdings_snapshot
                (snapshot_date, market, ticker, name, shares, price,
                 market_value, avg_cost, pnl, pnl_pct, currency)
            VALUES (?, 'US', ?, ?, ?, ?, ?, ?, ?, ?, 'USD')
        """, (TODAY, ticker, name, shares, price, mv, avg_cost, pnl, pnl_pct))

    for ticker, name, shares, price, avg_cost in TW_HOLDINGS:
        mv = shares * price
        pnl = (price - avg_cost) * shares if avg_cost else None
        pnl_pct = (price / avg_cost - 1) * 100 if avg_cost else None
        cur.execute("""
            INSERT INTO holdings_snapshot
                (snapshot_date, market, ticker, name, shares, price,
                 market_value, avg_cost, pnl, pnl_pct, currency)
            VALUES (?, 'TW', ?, ?, ?, ?, ?, ?, ?, ?, 'TWD')
        """, (TODAY, ticker, name, shares, price, mv, avg_cost, pnl, pnl_pct))

    conn.commit()
    print(f"holdings_snapshot 已記錄（{TODAY}）")


def print_summary(conn):
    print("\n=== 目前持倉摘要 ===")
    rows = conn.execute("""
        SELECT market, ticker, name, shares, avg_cost, currency
        FROM holdings ORDER BY market DESC, ticker
    """).fetchall()

    for r in rows:
        cost_str = f"均 {r[4]:.2f}" if r[4] else "均 N/A"
        print(f"  [{r[0]}] {r[1]:6s}  {r[3]:>8g} 股  {cost_str} {r[5]}")


if __name__ == "__main__":
    conn = sqlite3.connect(DB_PATH)
    init_db(conn)
    seed_holdings(conn)
    seed_snapshot(conn)
    print_summary(conn)
    conn.close()
    print(f"\nportfolio.sqlite 已建立於 {DB_PATH}")
