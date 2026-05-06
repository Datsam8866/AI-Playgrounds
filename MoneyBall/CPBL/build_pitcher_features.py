"""
build_pitcher_features.py

為每場比賽計算先發投手的 pre-game rolling ERA 等指標，
加入 game_features 表作為額外欄位（或另存 pitcher_game_features 表）。

設計原則（無洩漏）：
  每場的先發投手 rolling stats 只用「這場之前」的出賽資料計算。

Rolling window：最近 N 場先發（預設 5 場）。
最少需 MIN_STARTS 場才產生特徵（不足者該欄位為 NULL）。

輸出欄位（加入 game_features）：
  home_sp_era_roll   : 主隊先發投手近 N 場 ERA
  home_sp_whip_roll  : 主隊先發近 N 場 WHIP
  home_sp_k9_roll    : 主隊先發近 N 場 K/9
  home_sp_ip_roll    : 主隊先發近 N 場平均 IP（投球量）
  vis_sp_era_roll    : 客隊先發投手近 N 場 ERA
  vis_sp_whip_roll   : 客隊先發近 N 場 WHIP
  vis_sp_k9_roll     : 客隊先發近 N 場 K/9
  vis_sp_ip_roll     : 客隊先發近 N 場平均 IP

差異特徵（home - vis，ERA/WHIP 用 vis - home 表示越低越好）：
  diff_sp_era        : vis_era - home_era（主隊有利為正）
  diff_sp_whip       : vis_whip - home_whip
  diff_sp_k9         : home_k9 - vis_k9（主隊有利為正）
  diff_sp_ip         : home_ip - vis_ip（主隊先發較能吃局為正）
"""

import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

DB_PATH = Path("cpbl.sqlite")
WINDOW = 10      # 先發投手 rolling window（先發場數）
MIN_STARTS = 5   # 最少需要的先發歷史

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def ip_to_float(ip_int: int, div3: int = 0) -> float:
    """innings pitched: 5 innings + 2 outs = 5.667"""
    return (ip_int or 0) + (div3 or 0) / 3.0


def rolling_sp_stats(history: list[dict], window: int) -> dict | None:
    """
    history: [{"ip": float, "er": int, "k": int, "bb": int, "h": int}, ...]
    """
    subset = history[-window:]
    n = len(subset)
    if n < MIN_STARTS:
        return None
    total_ip = sum(g["ip"] for g in subset)
    total_er = sum(g["er"] for g in subset)
    total_k  = sum(g["k"]  for g in subset)
    total_bb = sum(g["bb"] for g in subset)
    total_h  = sum(g["h"]  for g in subset)
    if total_ip == 0:
        return None
    era  = total_er * 9 / total_ip
    whip = (total_bb + total_h) / total_ip
    k9   = total_k * 9 / total_ip
    return {
        "era":  era,
        "whip": whip,
        "k9":   k9,
        "ip":   total_ip / n,   # avg IP per start
    }


def build_pitcher_features(conn):
    # 讀取先發投手資料 + 對應比賽順序
    rows = conn.execute("""
        SELECT
            gsp.season_year,
            gsp.kind_code,
            gsp.game_sno,
            gsp.home_sp_acnt,
            gsp.vis_sp_acnt,
            gsp.home_sp_ip,
            gsp.vis_sp_ip,
            gsp.home_sp_er,
            gsp.vis_sp_er,
            gsp.home_sp_k,
            gsp.vis_sp_k,
            gsp.home_sp_bb,
            gsp.vis_sp_bb,
            gsp.home_sp_h,
            gsp.vis_sp_h,
            gf.game_date
        FROM game_starting_pitchers gsp
        JOIN game_features gf
          ON gf.season_year = gsp.season_year
         AND gf.kind_code   = gsp.kind_code
         AND gf.game_sno    = gsp.game_sno
        WHERE gsp.scrape_status = 'ok'
          AND gsp.home_sp_acnt IS NOT NULL
        ORDER BY gf.game_date, gf.game_sno
    """).fetchall()

    if not rows:
        print("No data from game_starting_pitchers join game_features.")
        return []

    print(f"Loaded {len(rows)} games with SP data")

    # 每位投手的歷史出賽紀錄（按時間累積）
    sp_history: dict[str, list] = defaultdict(list)
    results = []

    for r in rows:
        year, kind_code, sno, home_acnt, vis_acnt = r[0], r[1], r[2], r[3], r[4]
        home_ip = ip_to_float(r[5] or 0)
        vis_ip  = ip_to_float(r[6] or 0)
        home_er, vis_er = r[7] or 0, r[8] or 0
        home_k,  vis_k  = r[9] or 0, r[10] or 0
        home_bb, vis_bb = r[11] or 0, r[12] or 0
        home_h,  vis_h  = r[13] or 0, r[14] or 0

        # 先計算 rolling（用這場「之前」的資料）
        home_stats = rolling_sp_stats(sp_history[home_acnt], WINDOW) if home_acnt else None
        vis_stats  = rolling_sp_stats(sp_history[vis_acnt],  WINDOW) if vis_acnt  else None

        row = {
            "season_year": year,
            "kind_code": kind_code,
            "game_sno": sno,
            "home_sp_era_roll":  home_stats["era"]  if home_stats else None,
            "home_sp_whip_roll": home_stats["whip"] if home_stats else None,
            "home_sp_k9_roll":   home_stats["k9"]   if home_stats else None,
            "home_sp_ip_roll":   home_stats["ip"]   if home_stats else None,
            "vis_sp_era_roll":   vis_stats["era"]   if vis_stats  else None,
            "vis_sp_whip_roll":  vis_stats["whip"]  if vis_stats  else None,
            "vis_sp_k9_roll":    vis_stats["k9"]    if vis_stats  else None,
            "vis_sp_ip_roll":    vis_stats["ip"]    if vis_stats  else None,
        }
        # 差異特徵（主隊有利 = 正值）
        if home_stats and vis_stats:
            row["diff_sp_era"]  = vis_stats["era"]  - home_stats["era"]   # 越正對主隊越有利
            row["diff_sp_whip"] = vis_stats["whip"] - home_stats["whip"]
            row["diff_sp_k9"]   = home_stats["k9"]  - vis_stats["k9"]
            row["diff_sp_ip"]   = home_stats["ip"]  - vis_stats["ip"]
        else:
            row["diff_sp_era"]  = None
            row["diff_sp_whip"] = None
            row["diff_sp_k9"]   = None
            row["diff_sp_ip"]   = None

        results.append(row)

        # 更新歷史（這場結束後）
        if home_acnt:
            sp_history[home_acnt].append({
                "ip": home_ip, "er": home_er,
                "k": home_k,  "bb": home_bb,
                "h": home_h,
            })
        if vis_acnt:
            sp_history[vis_acnt].append({
                "ip": vis_ip, "er": vis_er,
                "k": vis_k,  "bb": vis_bb,
                "h": vis_h,
            })

    return results


def add_columns_if_missing(conn):
    """若 game_features 還沒有這些欄位，則 ALTER TABLE 加入"""
    existing = {row[1] for row in conn.execute("PRAGMA table_info(game_features)")}
    new_cols = [
        ("home_sp_era_roll",  "REAL"),
        ("home_sp_whip_roll", "REAL"),
        ("home_sp_k9_roll",   "REAL"),
        ("home_sp_ip_roll",   "REAL"),
        ("vis_sp_era_roll",   "REAL"),
        ("vis_sp_whip_roll",  "REAL"),
        ("vis_sp_k9_roll",    "REAL"),
        ("vis_sp_ip_roll",    "REAL"),
        ("diff_sp_era",       "REAL"),
        ("diff_sp_whip",      "REAL"),
        ("diff_sp_k9",        "REAL"),
        ("diff_sp_ip",        "REAL"),
    ]
    for col, dtype in new_cols:
        if col not in existing:
            conn.execute(f"ALTER TABLE game_features ADD COLUMN {col} {dtype}")
    conn.commit()


def write_back(conn, results: list[dict]):
    for r in results:
        conn.execute("""
            UPDATE game_features SET
              home_sp_era_roll  = ?,
              home_sp_whip_roll = ?,
              home_sp_k9_roll   = ?,
              home_sp_ip_roll   = ?,
              vis_sp_era_roll   = ?,
              vis_sp_whip_roll  = ?,
              vis_sp_k9_roll    = ?,
              vis_sp_ip_roll    = ?,
              diff_sp_era       = ?,
              diff_sp_whip      = ?,
              diff_sp_k9        = ?,
              diff_sp_ip        = ?
            WHERE season_year = ? AND kind_code = ? AND game_sno = ?
        """, (
            r["home_sp_era_roll"],  r["home_sp_whip_roll"],
            r["home_sp_k9_roll"],   r["home_sp_ip_roll"],
            r["vis_sp_era_roll"],   r["vis_sp_whip_roll"],
            r["vis_sp_k9_roll"],    r["vis_sp_ip_roll"],
            r["diff_sp_era"],       r["diff_sp_whip"],
            r["diff_sp_k9"],        r["diff_sp_ip"],
            r["season_year"],       r["kind_code"],
            r["game_sno"],
        ))
    conn.commit()
    print(f"Updated {len(results)} rows in game_features")


def main():
    conn = sqlite3.connect(DB_PATH)
    try:
        print("Building pitcher rolling features...")
        add_columns_if_missing(conn)

        results = build_pitcher_features(conn)
        if results:
            write_back(conn, results)

        # 統計有多少筆有先發投手特徵
        cur = conn.cursor()
        cur.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN diff_sp_era IS NOT NULL THEN 1 ELSE 0 END) as with_sp
            FROM game_features
        """)
        total, with_sp = cur.fetchone()
        print(f"game_features: {total} rows, {with_sp} with SP features ({100*with_sp/total:.1f}%)")

        # 逐年統計
        cur.execute("""
            SELECT season_year,
                   COUNT(*) as games,
                   SUM(CASE WHEN diff_sp_era IS NOT NULL THEN 1 ELSE 0 END) as with_sp
            FROM game_features
            WHERE season_year >= 2011
            GROUP BY season_year ORDER BY season_year
        """)
        print(f"\n{'Year':<6} {'Games':>6} {'WithSP':>8} {'Pct':>6}")
        for r in cur.fetchall():
            pct = 100*r[2]/r[1] if r[1] else 0
            print(f"  {r[0]:<6} {r[1]:>6} {r[2]:>8} {pct:>5.1f}%")

    finally:
        conn.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
