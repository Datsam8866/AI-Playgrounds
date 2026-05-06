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
    finally:
        conn.close()
    print("\n完成。")


if __name__ == "__main__":
    main()
