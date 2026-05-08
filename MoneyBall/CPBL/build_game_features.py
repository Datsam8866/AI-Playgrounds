"""
build_game_features.py

為每場比賽計算兩隊的 pre-game rolling stats，
寫入 game_features 表，供逐場預測模型使用。

設計原則（無洩漏）：
  每場比賽的 rolling stats 只用「這場之前」的比賽計算。
  排序依據：game_date + kind_code + game_sno（同一天多場用 sno 排序）。

Rolling window：最近 N 場（預設 20 場）。
最少需 MIN_GAMES 場才產生特徵（不足者排除）。

輸出欄位（以 home_/vis_ 為前綴）：
  _win_pct      : 近 N 場勝率
  _rs_per_game  : 近 N 場每場得分
  _ra_per_game  : 近 N 場每場失分
  _rd_per_game  : 近 N 場得失分差
  _pyth_wp      : 近 N 場 Pythagorean Win%

差異特徵（home - visiting）：
  diff_win_pct, diff_rs, diff_ra, diff_rd, diff_pyth_wp

目標：
  home_win  : 1 = 主隊勝，0 = 客隊勝（平局排除）
"""

import sqlite3
import sys
from pathlib import Path
from collections import defaultdict
from math import sqrt

DB_PATH = Path("cpbl.sqlite")
WINDOW = 20      # rolling window 場數
MIN_GAMES = 10   # 最少需要的歷史場次

FRANCHISE_MAP = {
    "ACC011": "ACN011",
    "AEG011": "AEO011",
    "AEM011": "AEO011",
    "AJK011": "AJL011",
}

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def normalize(code: str) -> str:
    return FRANCHISE_MAP.get(code, code)

def pyth_wp(rs: float, ra: float) -> float:
    if rs + ra == 0:
        return 0.5
    return rs ** 2 / (rs ** 2 + ra ** 2)


def load_games(conn) -> list[dict]:
    rows = list(conn.execute("""
        SELECT season_year, kind_code, game_date, game_sno,
               visiting_team_code, home_team_code,
               visiting_score, home_score
        FROM team_game_results
        WHERE game_status = 3
          AND kind_code = 'A'
          AND visiting_score IS NOT NULL
          AND home_score IS NOT NULL
        ORDER BY game_date, kind_code, game_sno
    """))
    games = []
    for year, kind_code, date, sno, vis, home, vs, hs in rows:
        if vs == hs:            # 平局排除
            continue
        games.append({
            "season_year": year,
            "kind_code": kind_code,
            "game_date": date[:10],   # 只取日期部分
            "game_sno": sno,
            "vis_code": normalize(vis),
            "home_code": normalize(home),
            "vis_score": vs,
            "home_score": hs,
            "home_win": 1 if hs > vs else 0,
        })
    return games


def rolling_stats(history: list[dict], window: int) -> dict | None:
    """取最近 window 場的統計，不足 MIN_GAMES 回傳 None。"""
    subset = history[-window:]
    n = len(subset)
    if n < MIN_GAMES:
        return None
    rs = sum(g["rs"] for g in subset)
    ra = sum(g["ra"] for g in subset)
    wins = sum(g["win"] for g in subset)
    return {
        "win_pct":     wins / n,
        "rs_per_game": rs / n,
        "ra_per_game": ra / n,
        "rd_per_game": (rs - ra) / n,
        "pyth_wp":     pyth_wp(rs, ra),
        "n_games":     n,
    }


def build_features(games: list[dict]) -> list[dict]:
    """
    按時間順序處理每場比賽，用截至此場之前的紀錄計算 rolling stats。
    """
    # 每支球隊的歷史紀錄（按時間累積）
    history: dict[str, list] = defaultdict(list)
    features = []

    for g in games:
        vis = g["vis_code"]
        home = g["home_code"]

        vis_stats = rolling_stats(history[vis], WINDOW)
        home_stats = rolling_stats(history[home], WINDOW)

        if vis_stats and home_stats:
            row = {
                "season_year":      g["season_year"],
                "kind_code":        g["kind_code"],
                "game_date":        g["game_date"],
                "game_sno":         g["game_sno"],
                "home_team_code":   home,
                "vis_team_code":    vis,
                "home_win":         g["home_win"],
                # home team stats
                "home_win_pct":     home_stats["win_pct"],
                "home_rs_per_game": home_stats["rs_per_game"],
                "home_ra_per_game": home_stats["ra_per_game"],
                "home_rd_per_game": home_stats["rd_per_game"],
                "home_pyth_wp":     home_stats["pyth_wp"],
                "home_n_games":     home_stats["n_games"],
                # visiting team stats
                "vis_win_pct":      vis_stats["win_pct"],
                "vis_rs_per_game":  vis_stats["rs_per_game"],
                "vis_ra_per_game":  vis_stats["ra_per_game"],
                "vis_rd_per_game":  vis_stats["rd_per_game"],
                "vis_pyth_wp":      vis_stats["pyth_wp"],
                "vis_n_games":      vis_stats["n_games"],
                # difference features (home - visiting)
                "diff_win_pct":     home_stats["win_pct"]     - vis_stats["win_pct"],
                "diff_rs":          home_stats["rs_per_game"] - vis_stats["rs_per_game"],
                "diff_ra":          home_stats["ra_per_game"] - vis_stats["ra_per_game"],
                "diff_rd":          home_stats["rd_per_game"] - vis_stats["rd_per_game"],
                "diff_pyth_wp":     home_stats["pyth_wp"]     - vis_stats["pyth_wp"],
            }
            features.append(row)

        # 比賽結束後更新歷史
        history[vis].append({"rs": g["vis_score"], "ra": g["home_score"], "win": 1 - g["home_win"]})
        history[home].append({"rs": g["home_score"], "ra": g["vis_score"], "win": g["home_win"]})

    return features


def ip_to_float(ip_int: int, div3: int = 0) -> float:
    """innings pitched: 5 innings + 2 outs = 5.667"""
    return (ip_int or 0) + (div3 or 0) / 3.0


def rolling_sp_stats(history: list[dict], window: int) -> dict | None:
    """
    history: [{"ip": float, "er": int, "k": int, "bb": int, "h": int}, ...]
    """
    subset = history[-window:]
    n = len(subset)
    if n < 5:
        return None
    total_ip = sum(g["ip"] for g in subset)
    total_er = sum(g["er"] for g in subset)
    total_k = sum(g["k"] for g in subset)
    total_bb = sum(g["bb"] for g in subset)
    total_h = sum(g["h"] for g in subset)
    if total_ip == 0:
        return None
    return {
        "era": total_er * 9 / total_ip,
        "whip": (total_bb + total_h) / total_ip,
        "k9": total_k * 9 / total_ip,
        "ip": total_ip / n,
    }


def build_pitcher_feature_rows(conn) -> list[dict]:
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
        return []

    sp_history: dict[str, list] = defaultdict(list)
    results = []

    for r in rows:
        year, kind_code, sno, home_acnt, vis_acnt = r[0], r[1], r[2], r[3], r[4]
        home_stats = rolling_sp_stats(sp_history[home_acnt], 10) if home_acnt else None
        vis_stats = rolling_sp_stats(sp_history[vis_acnt], 10) if vis_acnt else None

        row = {
            "season_year": year,
            "kind_code": kind_code,
            "game_sno": sno,
            "home_sp_era_roll": home_stats["era"] if home_stats else None,
            "home_sp_whip_roll": home_stats["whip"] if home_stats else None,
            "home_sp_k9_roll": home_stats["k9"] if home_stats else None,
            "home_sp_ip_roll": home_stats["ip"] if home_stats else None,
            "vis_sp_era_roll": vis_stats["era"] if vis_stats else None,
            "vis_sp_whip_roll": vis_stats["whip"] if vis_stats else None,
            "vis_sp_k9_roll": vis_stats["k9"] if vis_stats else None,
            "vis_sp_ip_roll": vis_stats["ip"] if vis_stats else None,
            "diff_sp_era": (vis_stats["era"] - home_stats["era"]) if home_stats and vis_stats else None,
            "diff_sp_whip": (vis_stats["whip"] - home_stats["whip"]) if home_stats and vis_stats else None,
            "diff_sp_k9": (home_stats["k9"] - vis_stats["k9"]) if home_stats and vis_stats else None,
            "diff_sp_ip": (home_stats["ip"] - vis_stats["ip"]) if home_stats and vis_stats else None,
        }
        results.append(row)

        home_ip = ip_to_float(r[5] or 0)
        vis_ip = ip_to_float(r[6] or 0)
        home_er, vis_er = r[7] or 0, r[8] or 0
        home_k, vis_k = r[9] or 0, r[10] or 0
        home_bb, vis_bb = r[11] or 0, r[12] or 0
        home_h, vis_h = r[13] or 0, r[14] or 0

        if home_acnt:
            sp_history[home_acnt].append({
                "ip": home_ip,
                "er": home_er,
                "k": home_k,
                "bb": home_bb,
                "h": home_h,
            })
        if vis_acnt:
            sp_history[vis_acnt].append({
                "ip": vis_ip,
                "er": vis_er,
                "k": vis_k,
                "bb": vis_bb,
                "h": vis_h,
            })

    return results


def write_pitcher_features(conn, rows: list[dict]):
    if not rows:
        return
    for r in rows:
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
            r["home_sp_era_roll"], r["home_sp_whip_roll"],
            r["home_sp_k9_roll"], r["home_sp_ip_roll"],
            r["vis_sp_era_roll"], r["vis_sp_whip_roll"],
            r["vis_sp_k9_roll"], r["vis_sp_ip_roll"],
            r["diff_sp_era"], r["diff_sp_whip"],
            r["diff_sp_k9"], r["diff_sp_ip"],
            r["season_year"], r["kind_code"], r["game_sno"],
        ))
    conn.commit()


def update_sp_era_z_scores(conn):
    year_rows = conn.execute("""
        SELECT season_year, home_sp_era_roll, vis_sp_era_roll
        FROM game_features
        ORDER BY season_year, game_date, game_sno
    """).fetchall()

    values_by_year: dict[int, list[float]] = defaultdict(list)
    for season_year, home_era, vis_era in year_rows:
        if home_era is not None:
            values_by_year[season_year].append(home_era)
        if vis_era is not None:
            values_by_year[season_year].append(vis_era)

    stats_by_year: dict[int, tuple[float | None, float | None]] = {}
    for season_year, values in values_by_year.items():
        n = len(values)
        if n < 10:
            stats_by_year[season_year] = (None, None)
            continue
        mean = sum(values) / n
        variance = sum((value - mean) ** 2 for value in values) / n
        std = sqrt(variance)
        stats_by_year[season_year] = (mean, std if std > 0 else None)

    for season_year, mean_std in stats_by_year.items():
        mean, std = mean_std
        if mean is None or std is None:
            conn.execute("""
                UPDATE game_features
                SET home_sp_era_z = NULL,
                    vis_sp_era_z = NULL,
                    diff_sp_era_z = NULL
                WHERE season_year = ?
            """, (season_year,))
            continue
        conn.execute("""
            UPDATE game_features
            SET home_sp_era_z = CASE
                    WHEN home_sp_era_roll IS NULL THEN NULL
                    ELSE (home_sp_era_roll - ?) / ?
                END,
                vis_sp_era_z = CASE
                    WHEN vis_sp_era_roll IS NULL THEN NULL
                    ELSE (vis_sp_era_roll - ?) / ?
                END,
                diff_sp_era_z = CASE
                    WHEN home_sp_era_roll IS NULL OR vis_sp_era_roll IS NULL THEN NULL
                    ELSE ((home_sp_era_roll - ?) / ?) - ((vis_sp_era_roll - ?) / ?)
                END
            WHERE season_year = ?
        """, (mean, std, mean, std, mean, std, mean, std, season_year))
    conn.commit()


def create_table(conn):
    conn.execute("DROP TABLE IF EXISTS game_features")
    conn.execute("""
        CREATE TABLE game_features (
            feature_id      INTEGER PRIMARY KEY AUTOINCREMENT,
            season_year     INTEGER NOT NULL,
            kind_code       TEXT NOT NULL,
            game_date       TEXT NOT NULL,
            game_sno        INTEGER NOT NULL,
            home_team_code  TEXT NOT NULL,
            vis_team_code   TEXT NOT NULL,
            home_win        INTEGER NOT NULL,
            home_win_pct    REAL, home_rs_per_game REAL,
            home_ra_per_game REAL, home_rd_per_game REAL,
            home_pyth_wp    REAL, home_n_games INTEGER,
            vis_win_pct     REAL, vis_rs_per_game REAL,
            vis_ra_per_game REAL, vis_rd_per_game REAL,
            vis_pyth_wp     REAL, vis_n_games INTEGER,
            diff_win_pct    REAL, diff_rs REAL,
            diff_ra         REAL, diff_rd REAL,
            diff_pyth_wp    REAL,
            home_sp_era_roll REAL, home_sp_whip_roll REAL,
            home_sp_k9_roll REAL, home_sp_ip_roll REAL,
            vis_sp_era_roll REAL, vis_sp_whip_roll REAL,
            vis_sp_k9_roll REAL, vis_sp_ip_roll REAL,
            diff_sp_era REAL, diff_sp_whip REAL,
            diff_sp_k9 REAL, diff_sp_ip REAL,
            home_sp_era_z REAL, vis_sp_era_z REAL,
            diff_sp_era_z REAL,
            UNIQUE(season_year, kind_code, game_sno, home_team_code)
        )
    """)
    conn.commit()


def insert_features(conn, features: list[dict]):
    cols = list(features[0].keys())
    ph = ",".join("?" * len(cols))
    conn.executemany(
        f"INSERT INTO game_features ({','.join(cols)}) VALUES ({ph})",
        [tuple(f[c] for c in cols) for f in features],
    )
    conn.commit()


def main():
    conn = sqlite3.connect(DB_PATH)
    try:
        print("載入比賽資料...")
        games = load_games(conn)
        print(f"  總場數（排除平局）: {len(games)}")

        print("計算 pre-game rolling features...")
        features = build_features(games)
        print(f"  有效特徵筆數（兩隊各需 >= {MIN_GAMES} 場歷史）: {len(features)}")

        print("寫入 game_features...")
        create_table(conn)
        insert_features(conn, features)

        print("回填先發投手 rolling features...")
        pitcher_features = build_pitcher_feature_rows(conn)
        write_pitcher_features(conn, pitcher_features)
        print(f"  SP rolling features rows: {len(pitcher_features)}")

        print("計算 season-level SP ERA z-score...")
        update_sp_era_z_scores(conn)

        # 摘要
        cur = conn.cursor()
        cur.execute("SELECT MIN(season_year), MAX(season_year), COUNT(*) FROM game_features")
        mn, mx, cnt = cur.fetchone()
        print(f"  game_features: {cnt} 筆，{mn}–{mx} 年")

        cur.execute("""
            SELECT season_year, kind_code, COUNT(*) as games,
                   ROUND(AVG(home_win),3) as home_win_rate
            FROM game_features GROUP BY season_year, kind_code ORDER BY season_year, kind_code
        """)
        print(f"\n{'Year':<6} {'Kind':<5} {'Games':>6} {'HomeWin%':>9}")
        for r in cur.fetchall():
            print(f"  {r[0]:<6} {r[1]:<5} {r[2]:>6} {r[3]:>9}")

        cur.execute("""
            SELECT COUNT(*), COUNT(home_sp_era_z), COUNT(diff_sp_era_z)
            FROM game_features
        """)
        total, home_z_count, diff_z_count = cur.fetchone()
        print(f"\n  SP ERA z-score: home={home_z_count}/{total}, diff={diff_z_count}/{total}")
    finally:
        conn.close()
    print("\n完成。")


if __name__ == "__main__":
    main()
