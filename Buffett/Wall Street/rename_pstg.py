import sqlite3, sys
sys.stdout.reconfigure(encoding="utf-8")

conn = sqlite3.connect("stock_forecast.sqlite")

before = conn.execute("SELECT COUNT(*), MIN(date), MAX(date) FROM price_history WHERE ticker='PSTG'").fetchone()
print(f"PSTG 資料：{before[0]} 筆，{before[1]} ~ {before[2]}")

p_before = conn.execute("SELECT COUNT(*) FROM price_history WHERE ticker='P'").fetchone()[0]
print(f"P 目前筆數：{p_before}")

conn.execute("UPDATE price_history SET ticker='P' WHERE ticker='PSTG'")
conn.commit()

after = conn.execute("SELECT COUNT(*), MIN(date), MAX(date) FROM price_history WHERE ticker='P'").fetchone()
pstg_left = conn.execute("SELECT COUNT(*) FROM price_history WHERE ticker='PSTG'").fetchone()[0]
print(f"P 更新後：{after[0]} 筆，{after[1]} ~ {after[2]}")
print(f"PSTG 殘留：{pstg_left} 筆")
conn.close()
print("完成")
