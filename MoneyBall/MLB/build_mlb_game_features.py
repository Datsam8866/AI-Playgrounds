"""
build_mlb_game_features.py

為每場 MLB 正規賽計算 pre-game 特徵，寫入 game_features 表：
  - 球隊近 20 場 rolling stats（win_pct, RS/G, RA/G, RD/G, Pythagorean WP）
  - Elo rating（K=20, HOME_ADV=25, 每季開季 regression=0.35）
  - 先發投手近 5 先發 rolling ERA/WHIP/K9/IP
  - MLB 特有 flag：universal_dh_era, coors_field_factor, is_interleague

無前瞻洩漏：所有 rolling stats 僅用「這場之前」資料。
"""

import sqlite3
from collections import defaultdict
from pathlib import Path

DB_PATH = Path("mlb.sqlite")

ELO_K = 8
ELO_HOME_ADV = 25
ELO_REGRESSION = 0.35
ELO_BASE = 1500.0

WINDOW = 20
MIN_GAMES = 10

SP_WINDOW = 8
SP_MIN_STARTS = 5


def ip_mlb_to_float(ip) -> float:
    """MLB inningsPitched: 6.2 means 6 and 2/3 innings (2 outs)."""
    if ip is None:
        return 0.0
    whole = int(ip)
    outs = round((ip - whole) * 10)
    return whole + outs / 3.0


def pyth_wp(rs: float, ra: float) -> float:
    if rs + ra == 0:
        return 0.5
    return rs ** 2 / (rs ** 2 + ra ** 2)


def elo_expected(home_elo: float, vis_elo: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-(home_elo + ELO_HOME_ADV - vis_elo) / 400.0))


def rolling_team_stats(history: list, window: int):
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


def rolling_sp_stats(history: list, window: int):
    subset = history[-window:]
    n = len(subset)
    if n < SP_MIN_STARTS:
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


def load_games(conn):
    return conn.execute("""
        SELECT game_pk, season_year, game_date,
               home_team_id, vis_team_id,
               home_score, vis_score, winner
        FROM team_game_results
        WHERE status = 'Final'
          AND home_score IS NOT NULL
          AND vis_score IS NOT NULL
          AND home_score != vis_score
        ORDER BY game_date, game_pk
    """).fetchall()


def load_sp_data(conn):
    """Return {game_pk: {team_id: {pitcher_id, ip, er, k, bb, h}}}"""
    rows = conn.execute("""
        SELECT game_pk, team_id, pitcher_id, ip, er, k, bb, h
        FROM game_starting_pitchers
        WHERE pitcher_id IS NOT NULL
    """).fetchall()
    sp_data = defaultdict(dict)
    for game_pk, team_id, pitcher_id, ip, er, k, bb, h in rows:
        sp_data[game_pk][team_id] = {
            "pitcher_id": pitcher_id,
            "ip": ip_mlb_to_float(ip),
            "er": er or 0,
            "k":  k  or 0,
            "bb": bb or 0,
            "h":  h  or 0,
        }
    return sp_data


def load_team_leagues(conn):
    """Return {(season_year, team_id): league_name}"""
    rows = conn.execute(
        "SELECT season_year, team_id, league FROM team_season_records"
    ).fetchall()
    return {(yr, tid): lg for yr, tid, lg in rows}


def find_rockies_id(conn):
    row = conn.execute(
        "SELECT team_id FROM team_season_records WHERE team_name LIKE '%Rockies%' LIMIT 1"
    ).fetchone()
    return row[0] if row else None


def create_table(conn):
    conn.execute("DROP TABLE IF EXISTS game_features")
    conn.execute("""
        CREATE TABLE game_features (
            game_pk              INTEGER PRIMARY KEY,
            season_year          INTEGER,
            game_date            TEXT,
            home_team_id         INTEGER,
            vis_team_id          INTEGER,
            home_win             INTEGER,

            home_win_pct         REAL,
            home_rs_per_game     REAL,
            home_ra_per_game     REAL,
            home_rd_per_game     REAL,
            home_pyth_wp         REAL,
            home_n_games         INTEGER,
            vis_win_pct          REAL,
            vis_rs_per_game      REAL,
            vis_ra_per_game      REAL,
            vis_rd_per_game      REAL,
            vis_pyth_wp          REAL,
            vis_n_games          INTEGER,
            diff_win_pct         REAL,
            diff_rs              REAL,
            diff_ra              REAL,
            diff_rd              REAL,
            diff_pyth_wp         REAL,

            home_elo             REAL,
            vis_elo              REAL,
            diff_elo             REAL,
            elo_win_prob         REAL,

            home_sp_era_roll     REAL,
            home_sp_whip_roll    REAL,
            home_sp_k9_roll      REAL,
            home_sp_ip_roll      REAL,
            vis_sp_era_roll      REAL,
            vis_sp_whip_roll     REAL,
            vis_sp_k9_roll       REAL,
            vis_sp_ip_roll       REAL,
            diff_sp_era          REAL,
            diff_sp_whip         REAL,
            diff_sp_k9           REAL,
            diff_sp_ip           REAL,

            universal_dh_era     INTEGER,
            coors_field_factor   INTEGER,
            is_interleague       INTEGER,
            sp_available         INTEGER
        )
    """)
    conn.commit()


def main():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode = WAL")

    try:
        print("Loading data...")
        games = load_games(conn)
        print(f"  Games loaded: {len(games)}")

        sp_data = load_sp_data(conn)
        print(f"  SP records: {sum(len(v) for v in sp_data.values())}")

        league_map = load_team_leagues(conn)
        rockies_id = find_rockies_id(conn)
        print(f"  Rockies team_id: {rockies_id}")

        create_table(conn)

        # Rolling state
        team_history = defaultdict(list)
        elo = defaultdict(lambda: ELO_BASE)
        sp_history = defaultdict(list)
        current_season = None

        features = []

        for game_pk, season_year, game_date, home_id, vis_id, home_score, vis_score, winner in games:
            home_win = 1 if winner == "home" else 0

            # Season transition: Elo regression toward mean
            if season_year != current_season:
                if current_season is not None:
                    for tid in list(elo.keys()):
                        elo[tid] = ELO_BASE * ELO_REGRESSION + elo[tid] * (1 - ELO_REGRESSION)
                current_season = season_year

            # Pre-game features
            home_ts = rolling_team_stats(team_history[home_id], WINDOW)
            vis_ts  = rolling_team_stats(team_history[vis_id],  WINDOW)

            h_elo    = elo[home_id]
            v_elo    = elo[vis_id]
            elo_prob = elo_expected(h_elo, v_elo)

            game_sps = sp_data.get(game_pk, {})
            home_sp  = game_sps.get(home_id)
            vis_sp   = game_sps.get(vis_id)
            home_sp_stats = rolling_sp_stats(sp_history[home_sp["pitcher_id"]], SP_WINDOW) if home_sp else None
            vis_sp_stats  = rolling_sp_stats(sp_history[vis_sp["pitcher_id"]],  SP_WINDOW) if vis_sp  else None

            dh_era     = 1 if season_year >= 2022 else 0
            coors      = 1 if home_id == rockies_id else 0
            h_league   = league_map.get((season_year, home_id))
            v_league   = league_map.get((season_year, vis_id))
            interleague = 1 if (h_league and v_league and h_league != v_league) else 0
            sp_avail   = 1 if (home_sp_stats and vis_sp_stats) else 0

            both_ts = home_ts and vis_ts
            both_sp = home_sp_stats and vis_sp_stats

            features.append((
                game_pk, season_year, game_date, home_id, vis_id, home_win,
                # rolling team
                home_ts["win_pct"]     if home_ts else None,
                home_ts["rs_per_game"] if home_ts else None,
                home_ts["ra_per_game"] if home_ts else None,
                home_ts["rd_per_game"] if home_ts else None,
                home_ts["pyth_wp"]     if home_ts else None,
                home_ts["n_games"]     if home_ts else None,
                vis_ts["win_pct"]      if vis_ts  else None,
                vis_ts["rs_per_game"]  if vis_ts  else None,
                vis_ts["ra_per_game"]  if vis_ts  else None,
                vis_ts["rd_per_game"]  if vis_ts  else None,
                vis_ts["pyth_wp"]      if vis_ts  else None,
                vis_ts["n_games"]      if vis_ts  else None,
                (home_ts["win_pct"]     - vis_ts["win_pct"])     if both_ts else None,
                (home_ts["rs_per_game"] - vis_ts["rs_per_game"]) if both_ts else None,
                (home_ts["ra_per_game"] - vis_ts["ra_per_game"]) if both_ts else None,
                (home_ts["rd_per_game"] - vis_ts["rd_per_game"]) if both_ts else None,
                (home_ts["pyth_wp"]     - vis_ts["pyth_wp"])     if both_ts else None,
                # elo
                h_elo, v_elo, h_elo - v_elo, elo_prob,
                # sp
                home_sp_stats["era"]  if home_sp_stats else None,
                home_sp_stats["whip"] if home_sp_stats else None,
                home_sp_stats["k9"]   if home_sp_stats else None,
                home_sp_stats["ip"]   if home_sp_stats else None,
                vis_sp_stats["era"]   if vis_sp_stats  else None,
                vis_sp_stats["whip"]  if vis_sp_stats  else None,
                vis_sp_stats["k9"]    if vis_sp_stats  else None,
                vis_sp_stats["ip"]    if vis_sp_stats  else None,
                (vis_sp_stats["era"]  - home_sp_stats["era"])  if both_sp else None,
                (vis_sp_stats["whip"] - home_sp_stats["whip"]) if both_sp else None,
                (home_sp_stats["k9"]  - vis_sp_stats["k9"])    if both_sp else None,
                (home_sp_stats["ip"]  - vis_sp_stats["ip"])    if both_sp else None,
                # flags
                dh_era, coors, interleague, sp_avail,
            ))

            # Post-game state updates
            team_history[home_id].append({"rs": home_score, "ra": vis_score, "win": home_win})
            team_history[vis_id].append({"rs": vis_score,   "ra": home_score, "win": 1 - home_win})

            elo[home_id] += ELO_K * (home_win - elo_prob)
            elo[vis_id]  += ELO_K * ((1 - home_win) - (1 - elo_prob))

            if home_sp:
                sp_history[home_sp["pitcher_id"]].append({
                    "ip": home_sp["ip"], "er": home_sp["er"],
                    "k": home_sp["k"],  "bb": home_sp["bb"], "h": home_sp["h"],
                })
            if vis_sp:
                sp_history[vis_sp["pitcher_id"]].append({
                    "ip": vis_sp["ip"], "er": vis_sp["er"],
                    "k": vis_sp["k"],  "bb": vis_sp["bb"], "h": vis_sp["h"],
                })

        print(f"Writing {len(features)} rows...")
        conn.executemany(
            """INSERT INTO game_features VALUES (
                ?,?,?,?,?,?,
                ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,
                ?,?,?,?,
                ?,?,?,?,?,?,?,?,?,?,?,?,
                ?,?,?,?
            )""",
            features,
        )
        conn.commit()

        # Summary
        total, mn, mx = conn.execute(
            "SELECT COUNT(*), MIN(season_year), MAX(season_year) FROM game_features"
        ).fetchone()
        print(f"\ngame_features: {total} rows, {mn}–{mx}\n")

        rows = conn.execute("""
            SELECT season_year, COUNT(*) as games,
                   ROUND(AVG(home_win), 3) as home_win_rate,
                   SUM(CASE WHEN diff_win_pct IS NOT NULL THEN 1 ELSE 0 END) as with_team_stats,
                   SUM(sp_available) as with_sp
            FROM game_features
            GROUP BY season_year ORDER BY season_year
        """).fetchall()
        print(f"{'Year':<6} {'Games':>6} {'HomeWin%':>9} {'TeamStats':>10} {'WithSP':>7}")
        for r in rows:
            print(f"  {r[0]:<6} {r[1]:>6} {r[2]:>9} {r[3]:>10} {r[4]:>7}")

    finally:
        conn.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
