"""
build_kbo_pitcher_features.py

計算先發投手 pre-game rolling stats，UPDATE 到 game_features 表。

設計原則（無洩漏）：
  每場的 rolling stats 只用「這場之前」的先發紀錄計算。
  SP history 從完整例行賽先發紀錄建立，再只回寫 game_features target rows。
  這可納入 early rows / 平手場的 prior starts，避免因 game_features burn-in 低估 SP coverage。

Rolling window=10，MIN_STARTS=5。

KBO 注意事項：
  bb 欄位為 4사구（BB+HBP 合併），WHIP = (bb + hits) / ip 直接可用。
  diff 特徵以相同方式計算，偏差在差值中抵消，無需修正。
"""

import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

DB_PATH = Path("kbo.sqlite")
WINDOW     = 10
MIN_STARTS = 5

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def rolling_sp_stats(history: list) -> dict | None:
    subset = history[-WINDOW:]
    n = len(subset)
    if n < MIN_STARTS:
        return None
    total_ip = sum(g["ip"] for g in subset)
    if total_ip == 0:
        return None
    total_er = sum(g["er"] for g in subset)
    total_k  = sum(g["k"]  for g in subset)
    total_bb = sum(g["bb"] for g in subset)
    total_h  = sum(g["h"]  for g in subset)
    return {
        "era":  total_er * 9 / total_ip,
        "whip": (total_bb + total_h) / total_ip,
        "k9":   total_k * 9 / total_ip,
        "ip":   total_ip / n,
    }


def build_pitcher_features(conn) -> list[dict]:
    target_game_ids = {
        r[0] for r in conn.execute("SELECT game_id FROM game_features").fetchall()
    }

    # 讀取完整例行賽 SP 記錄；只對 game_features target rows 輸出 rolling features。
    rows = conn.execute("""
        SELECT
            g.game_id,
            g.season_year,
            g.game_date,
            g.start_time,
            sp_h.player_name  AS home_sp,
            sp_h.team_code    AS home_sp_team,
            sp_a.player_name  AS away_sp,
            sp_a.team_code    AS away_sp_team,
            sp_h.ip_numeric   AS home_ip,
            sp_a.ip_numeric   AS away_ip,
            COALESCE(sp_h.er, 0)         AS home_er,
            COALESCE(sp_a.er, 0)         AS away_er,
            COALESCE(sp_h.strikeouts, 0) AS home_k,
            COALESCE(sp_a.strikeouts, 0) AS away_k,
            COALESCE(sp_h.bb, 0)         AS home_bb,
            COALESCE(sp_a.bb, 0)         AS away_bb,
            COALESCE(sp_h.hits, 0)       AS home_h,
            COALESCE(sp_a.hits, 0)       AS away_h
        FROM team_game_results g
        LEFT JOIN game_starting_pitchers sp_h
            ON sp_h.game_id = g.game_id AND sp_h.side = 'home'
        LEFT JOIN game_starting_pitchers sp_a
            ON sp_a.game_id = g.game_id AND sp_a.side = 'away'
        WHERE g.game_state = 3
          AND g.sr_id = 0
          AND g.away_score IS NOT NULL
          AND g.home_score IS NOT NULL
        ORDER BY g.game_date, COALESCE(g.start_time, ''), g.game_id
    """).fetchall()

    if not rows:
        print("No pitcher rows found — run kbo_boxscore_scraper.py first.")
        return []

    print(f"Loaded {len(rows)} regular-season games for SP history")
    print(f"Target game_features rows: {len(target_game_ids)}")

    sp_history: dict[str, list] = defaultdict(list)
    results = []

    for r in rows:
        (game_id, yr, date, start_time,
         home_sp, home_sp_team, away_sp, away_sp_team,
         home_ip, away_ip,
         home_er, away_er,
         home_k,  away_k,
         home_bb, away_bb,
         home_h,  away_h) = r

        home_ip = home_ip or 0.0
        away_ip = away_ip or 0.0
        home_key = (home_sp, home_sp_team) if home_sp else None
        away_key = (away_sp, away_sp_team) if away_sp else None

        # Rolling stats（這場「之前」的資料）
        h_stats = rolling_sp_stats(sp_history[home_key]) if home_key else None
        a_stats = rolling_sp_stats(sp_history[away_key]) if away_key else None

        if game_id in target_game_ids:
            rec = {
                "game_id":          game_id,
                "home_sp_era_roll":  h_stats["era"]  if h_stats else None,
                "home_sp_whip_roll": h_stats["whip"] if h_stats else None,
                "home_sp_k9_roll":   h_stats["k9"]   if h_stats else None,
                "home_sp_ip_roll":   h_stats["ip"]   if h_stats else None,
                "away_sp_era_roll":  a_stats["era"]  if a_stats else None,
                "away_sp_whip_roll": a_stats["whip"] if a_stats else None,
                "away_sp_k9_roll":   a_stats["k9"]   if a_stats else None,
                "away_sp_ip_roll":   a_stats["ip"]   if a_stats else None,
            }
            if h_stats and a_stats:
                rec["diff_sp_era"]  = a_stats["era"]  - h_stats["era"]   # 客隊−主隊，正=主隊有利
                rec["diff_sp_whip"] = a_stats["whip"] - h_stats["whip"]
                rec["diff_sp_k9"]   = h_stats["k9"]   - a_stats["k9"]    # 主隊−客隊，正=主隊有利
                rec["diff_sp_ip"]   = h_stats["ip"]   - a_stats["ip"]
            else:
                rec["diff_sp_era"] = rec["diff_sp_whip"] = rec["diff_sp_k9"] = rec["diff_sp_ip"] = None

            results.append(rec)

        # 更新歷史（這場結束後）
        if home_key and home_ip > 0:
            sp_history[home_key].append({
                "ip": home_ip, "er": home_er,
                "k":  home_k,  "bb": home_bb, "h": home_h,
            })
        if away_key and away_ip > 0:
            sp_history[away_key].append({
                "ip": away_ip, "er": away_er,
                "k":  away_k,  "bb": away_bb, "h": away_h,
            })

    return results


def write_back(conn, results: list[dict]):
    conn.executemany("""
        UPDATE game_features SET
            home_sp_era_roll  = :home_sp_era_roll,
            home_sp_whip_roll = :home_sp_whip_roll,
            home_sp_k9_roll   = :home_sp_k9_roll,
            home_sp_ip_roll   = :home_sp_ip_roll,
            away_sp_era_roll  = :away_sp_era_roll,
            away_sp_whip_roll = :away_sp_whip_roll,
            away_sp_k9_roll   = :away_sp_k9_roll,
            away_sp_ip_roll   = :away_sp_ip_roll,
            diff_sp_era       = :diff_sp_era,
            diff_sp_whip      = :diff_sp_whip,
            diff_sp_k9        = :diff_sp_k9,
            diff_sp_ip        = :diff_sp_ip
        WHERE game_id = :game_id
    """, results)
    conn.commit()
    print(f"Updated {len(results)} rows in game_features")


def main():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        results = build_pitcher_features(conn)
        if results:
            write_back(conn, results)

        total, with_sp = conn.execute("""
            SELECT COUNT(*),
                   SUM(CASE WHEN diff_sp_era IS NOT NULL THEN 1 ELSE 0 END)
            FROM game_features
        """).fetchone()
        pct = 100 * with_sp / total if total else 0
        print(f"game_features: {total} rows, {with_sp} with SP features ({pct:.1f}%)")

        print(f"\n{'Year':<6} {'Games':>6} {'WithSP':>8} {'Pct':>6}")
        for r in conn.execute("""
            SELECT season_year, COUNT(*),
                   SUM(CASE WHEN diff_sp_era IS NOT NULL THEN 1 ELSE 0 END)
            FROM game_features
            GROUP BY season_year ORDER BY season_year
        """).fetchall():
            p = 100 * r[2] / r[1] if r[1] else 0
            print(f"  {r[0]:<6} {r[1]:>6} {r[2]:>8} {p:>5.1f}%")

    finally:
        conn.close()
    print("\n完成。")


if __name__ == "__main__":
    main()
