import sqlite3
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

COSTS = {
    "AMD":  244.21,
    "ARM":  167.24,
    "AVGO": 268.84,
    "COHR": 333.67,
    "CRWD": 474.39,
    "INTC":  65.18,
    "LITE": 902.53,
    "MU":   586.45,
    "NET":  231.29,
    "NVDA": 170.52,
    "PLTR": 140.01,
    "SNDK": 1073.72,
    "TSLA": 419.35,
    "TSM":  276.31,
    "VOO":  527.00,
}

conn = sqlite3.connect("portfolio.sqlite")

for ticker, avg_cost in COSTS.items():
    conn.execute(
        "UPDATE holdings SET avg_cost=? WHERE market='US' AND ticker=?",
        (avg_cost, ticker)
    )
conn.commit()
print("成本均價更新完成\n")

print(f"  {'代號':<6} {'股數':>5} {'均價 $':>9} {'總成本':>12} {'現價 $':>9} {'市值':>12} {'損益 $':>10} {'損益%':>7}")
print("  " + "-" * 80)

rows = conn.execute("""
    SELECT h.ticker, h.shares, h.avg_cost, s.price
    FROM holdings h
    LEFT JOIN (
        SELECT ticker, price
        FROM holdings_snapshot
        WHERE snapshot_date = (SELECT MAX(snapshot_date) FROM holdings_snapshot)
          AND market = 'US'
    ) s ON h.ticker = s.ticker
    WHERE h.market = 'US'
    ORDER BY h.ticker
""").fetchall()

total_cost = total_mv = total_pnl = 0.0

for ticker, shares, avg_cost, price in rows:
    cost = shares * avg_cost
    mv   = shares * price if price else 0.0
    pnl  = mv - cost
    pct  = pnl / cost * 100 if cost else 0.0
    total_cost += cost
    total_mv   += mv
    total_pnl  += pnl
    price_str = f"{price:>9.2f}" if price else "      N/A"
    mv_str    = f"{mv:>12,.2f}" if price else "           N/A"
    print(f"  {ticker:<6} {shares:>5g}  {avg_cost:>9.2f}  {cost:>12,.2f}  {price_str}  {mv_str}  {pnl:>10,.2f}  {pct:>+6.1f}%")

print("  " + "-" * 80)
total_pct = total_pnl / total_cost * 100 if total_cost else 0.0
print(f"  {'合計':<6} {'':>5}  {'':>9}  {total_cost:>12,.2f}  {'':>9}  {total_mv:>12,.2f}  {total_pnl:>10,.2f}  {total_pct:>+6.1f}%")

conn.close()
