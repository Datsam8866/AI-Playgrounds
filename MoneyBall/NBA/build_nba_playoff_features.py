# -*- coding: utf-8 -*-
"""
build_nba_playoff_features.py

從 nba.sqlite 建立季後賽特徵表 playoff_game_features。
特徵：15 個（正規賽底座4 + 動態Elo3 + 系列賽情境8）
"""

import argparse
import sqlite3
from collections import defaultdict, deque
from datetime import date
from pathlib import Path

from build_nba_game_features import (
    ELO_BASE,
    ELO_HOME_ADV,
    WINDOW,
    pyth_wp,
    elo_win_prob,
    regress_elo,
    compute_lineup_pts,
    update_player_state,
)

DB_PATH = Path(__file__).resolve().parent / "nba.sqlite"

ELO_K_RS = 20.0
ELO_K_PO = 15.0
ELO_REGRESSION = 0.35
MIN_GAMES_NET = 5
FIRST_GAME_REST = 3

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS playoff_game_features (
    feature_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id            TEXT NOT NULL UNIQUE,
    season_year        INTEGER NOT NULL,
    game_date          TEXT NOT NULL,
    home_team_id       INTEGER NOT NULL,
    vis_team_id        INTEGER NOT NULL,
    home_team_abbr     TEXT NOT NULL,
    vis_team_abbr      TEXT NOT NULL,
    home_win           INTEGER NOT NULL,
    playoff_round      INTEGER NOT NULL,
    game_in_series     INTEGER NOT NULL,
    home_series_wins   INTEGER NOT NULL,
    vis_series_wins    INTEGER NOT NULL,
    series_score_diff  INTEGER NOT NULL,
    is_elimination     INTEGER NOT NULL,
    home_has_homecourt INTEGER NOT NULL,
    series_rest_days   INTEGER,
    diff_elo_rs        REAL,
    diff_rs_net_rtg    REAL,
    diff_rs_pyth_wp    REAL,
    diff_rs_lineup_pts REAL,
    diff_elo_po        REAL,
    elo_win_prob_po    REAL,
    diff_elo_change_po REAL
);
CREATE INDEX IF NOT EXISTS idx_pgf_season_year ON playoff_game_features(season_year);
"""

UPSERT_SQL = """
INSERT INTO playoff_game_features (
    game_id, season_year, game_date,
    home_team_id, vis_team_id,
    home_team_abbr, vis_team_abbr,
    home_win, playoff_round,
    game_in_series, home_series_wins, vis_series_wins,
    series_score_diff, is_elimination, home_has_homecourt,
    series_rest_days,
    diff_elo_rs, diff_rs_net_rtg, diff_rs_pyth_wp, diff_rs_lineup_pts,
    diff_elo_po, elo_win_prob_po, diff_elo_change_po
)
VALUES (
    :game_id, :season_year, :game_date,
    :home_team_id, :vis_team_id,
    :home_team_abbr, :vis_team_abbr,
    :home_win, :playoff_round,
    :game_in_series, :home_series_wins, :vis_series_wins,
    :series_score_diff, :is_elimination, :home_has_homecourt,
    :series_rest_days,
    :diff_elo_rs, :diff_rs_net_rtg, :diff_rs_pyth_wp, :diff_rs_lineup_pts,
    :diff_elo_po, :elo_win_prob_po, :diff_elo_change_po
)
ON CONFLICT(game_id) DO UPDATE SET
    season_year = excluded.season_year,
    game_date = excluded.game_date,
    home_team_id = excluded.home_team_id,
    vis_team_id = excluded.vis_team_id,
    home_team_abbr = excluded.home_team_abbr,
    vis_team_abbr = excluded.vis_team_abbr,
    home_win = excluded.home_win,
    playoff_round = excluded.playoff_round,
    game_in_series = excluded.game_in_series,
    home_series_wins = excluded.home_series_wins,
    vis_series_wins = excluded.vis_series_wins,
    series_score_diff = excluded.series_score_diff,
    is_elimination = excluded.is_elimination,
    home_has_homecourt = excluded.home_has_homecourt,
    series_rest_days = excluded.series_rest_days,
    diff_elo_rs = excluded.diff_elo_rs,
    diff_rs_net_rtg = excluded.diff_rs_net_rtg,
    diff_rs_pyth_wp = excluded.diff_rs_pyth_wp,
    diff_rs_lineup_pts = excluded.diff_rs_lineup_pts,
    diff_elo_po = excluded.diff_elo_po,
    elo_win_prob_po = excluded.elo_win_prob_po,
    diff_elo_change_po = excluded.diff_elo_change_po
"""


def connect_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    return conn


def load_rs_games(conn):
    """Load regular season games ordered by season_year, game_date, game_id."""
    rows = conn.execute(
        """
        SELECT game_id, season_year, game_date,
               home_team_id, vis_team_id,
               home_team_abbr, vis_team_abbr,
               home_score, vis_score, home_win
        FROM game_results
        WHERE home_win IS NOT NULL
        ORDER BY season_year, game_date, game_id
        """
    ).fetchall()
    return rows


def load_po_games(conn, target_season=None):
    """Load playoff games ordered by season_year, game_date, game_id."""
    if target_season is not None:
        rows = conn.execute(
            """
            SELECT game_id, season_year, game_date,
                   home_team_id, vis_team_id,
                   home_team_abbr, vis_team_abbr,
                   home_score, vis_score, home_win
            FROM playoff_game_results
            WHERE home_win IS NOT NULL AND season_year = ?
            ORDER BY season_year, game_date, game_id
            """,
            (target_season,)
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT game_id, season_year, game_date,
                   home_team_id, vis_team_id,
                   home_team_abbr, vis_team_abbr,
                   home_score, vis_score, home_win
            FROM playoff_game_results
            WHERE home_win IS NOT NULL
            ORDER BY season_year, game_date, game_id
            """
        ).fetchall()
    return rows


def load_rs_player_game_map(conn) -> dict:
    """Load regular season player stats for lineup_pts computation."""
    try:
        rows = conn.execute(
            "SELECT game_id, team_id, player_id, min_seconds, pts FROM player_game_stats"
        ).fetchall()
    except sqlite3.OperationalError:
        return {}
    result: dict = {}
    for row in rows:
        gid, tid, pid, mins, pts = str(row[0]), int(row[1]), int(row[2]), int(row[3]), float(row[4])
        result.setdefault(gid, {}).setdefault(tid, []).append(
            {"player_id": pid, "min_seconds": mins, "pts": pts}
        )
    return result


def compute_rs_end_state(rs_games, rs_player_game_map) -> dict:
    """
    Run through all regular season games and return end-of-season state per season_year.
    Returns dict: {season_year: {team_id: {elo, net_rtg, pyth_wp, lineup_pts}}}
    """
    elo_by_team = defaultdict(lambda: ELO_BASE)
    team_history = defaultdict(lambda: deque(maxlen=WINDOW))
    team_last_lineup: dict = {}
    player_pts_acc: dict = {}
    current_season = None
    season_end_states: dict = {}

    for game in rs_games:
        season_year = int(game["season_year"])
        home_id = int(game["home_team_id"])
        vis_id = int(game["vis_team_id"])

        if season_year != current_season:
            # Save end state of previous season before regression
            if current_season is not None:
                season_end_states[current_season] = _snapshot_state(
                    current_season, elo_by_team, team_history, team_last_lineup, player_pts_acc
                )
                # Regress elo at season boundary
                regress_elo(elo_by_team)
            current_season = season_year

        home_win = int(game["home_win"])
        vis_win = 1 - home_win
        home_score = int(game["home_score"])
        vis_score = int(game["vis_score"])

        # Update elo (pre-game state already captured via snapshot at season end)
        exp_home = elo_win_prob(elo_by_team[home_id], elo_by_team[vis_id])
        elo_by_team[home_id] += ELO_K_RS * (home_win - exp_home)
        elo_by_team[vis_id] += ELO_K_RS * (vis_win - (1.0 - exp_home))

        # Update history
        team_history[home_id].append({"win": home_win, "pts": home_score, "pts_allowed": vis_score})
        team_history[vis_id].append({"win": vis_win, "pts": vis_score, "pts_allowed": home_score})

        # Update player lineup state
        update_player_state(game, rs_player_game_map, team_last_lineup, player_pts_acc)

    # Save last season
    if current_season is not None:
        season_end_states[current_season] = _snapshot_state(
            current_season, elo_by_team, team_history, team_last_lineup, player_pts_acc
        )

    return season_end_states


def _snapshot_state(season_year, elo_by_team, team_history, team_last_lineup, player_pts_acc) -> dict:
    """Snapshot the end-of-season state for all teams."""
    snapshot = {}
    # Collect all teams seen this season
    all_teams = set(team_history.keys()) | set(elo_by_team.keys())
    for tid in all_teams:
        hist = list(team_history.get(tid, []))
        if len(hist) >= MIN_GAMES_NET:
            recent = hist[-WINDOW:]
            pts = sum(g["pts"] for g in recent)
            pts_allowed = sum(g["pts_allowed"] for g in recent)
            n = len(recent)
            net_rtg = (pts - pts_allowed) / n
            pw = pyth_wp(pts, pts_allowed)
        else:
            net_rtg = None
            pw = None
        # lineup_pts
        lp = compute_lineup_pts(tid, season_year, team_last_lineup, player_pts_acc)
        snapshot[tid] = {
            "elo": float(elo_by_team.get(tid, ELO_BASE)),
            "net_rtg": net_rtg,
            "pyth_wp": pw,
            "lineup_pts": lp,
        }
    return snapshot


def infer_playoff_round(game_id: str) -> int:
    """
    NBA playoff game_id (10 chars): 00 4 YY 00 R SG
      position 7 (0-indexed) = round (1=first round, 2=semis, 3=conf finals, 4=finals)
    """
    try:
        return int(game_id[7])
    except (IndexError, ValueError):
        return 1


def build_playoff_features(po_games, season_end_states) -> list:
    """Build feature rows for all playoff games."""
    # playoff elo: start from RS end elo, K=15, no regression
    elo_po = {}  # {team_id: elo}
    # series tracking: {series_key: {home_id, vis_id, home_wins, vis_wins, game_count, last_date, homecourt_team}}
    series_state: dict = {}
    feature_rows = []

    for game in po_games:
        season_year = int(game["season_year"])
        home_id = int(game["home_team_id"])
        vis_id = int(game["vis_team_id"])
        home_win = int(game["home_win"])
        home_score = int(game["home_score"])
        vis_score = int(game["vis_score"])
        game_date_str = str(game["game_date"])
        game_id = str(game["game_id"])

        # Init elo_po from RS end state if not yet seen
        rs_state = season_end_states.get(season_year, {})
        if home_id not in elo_po:
            elo_po[home_id] = rs_state.get(home_id, {}).get("elo", ELO_BASE)
        if vis_id not in elo_po:
            elo_po[vis_id] = rs_state.get(vis_id, {}).get("elo", ELO_BASE)

        playoff_round = infer_playoff_round(game_id)
        series_key = (season_year, playoff_round, frozenset({home_id, vis_id}))

        if series_key not in series_state:
            series_state[series_key] = {
                "home_id": home_id,  # first-game home team (has homecourt)
                "home_wins": 0,
                "vis_wins": 0,
                "game_count": 0,
                "last_date": None,
                "homecourt_team": home_id,  # G1 home team has homecourt advantage
            }

        ss = series_state[series_key]
        game_in_series = ss["game_count"] + 1

        # Determine homecourt from perspective of current game's home team
        homecourt_team = ss["homecourt_team"]
        home_has_homecourt = 1 if home_id == homecourt_team else 0

        # Series wins BEFORE this game
        # home/vis_series_wins are from the perspective of THIS game's home/vis team
        if home_id == ss["home_id"]:
            home_series_wins = ss["home_wins"]
            vis_series_wins = ss["vis_wins"]
        else:
            # teams swapped perspective from series tracker
            home_series_wins = ss["vis_wins"]
            vis_series_wins = ss["home_wins"]

        series_score_diff = home_series_wins - vis_series_wins
        is_elimination = 1 if (home_series_wins == 3 or vis_series_wins == 3) else 0

        # Rest days
        if ss["last_date"] is None:
            series_rest_days = FIRST_GAME_REST
        else:
            current_dt = date.fromisoformat(game_date_str)
            last_dt = ss["last_date"]
            series_rest_days = max(0, (current_dt - last_dt).days)

        # RS end-state features
        home_rs = rs_state.get(home_id, {})
        vis_rs = rs_state.get(vis_id, {})

        diff_elo_rs = home_rs.get("elo", ELO_BASE) - vis_rs.get("elo", ELO_BASE)

        h_net = home_rs.get("net_rtg")
        v_net = vis_rs.get("net_rtg")
        diff_rs_net_rtg = (h_net - v_net) if (h_net is not None and v_net is not None) else None

        h_pw = home_rs.get("pyth_wp")
        v_pw = vis_rs.get("pyth_wp")
        diff_rs_pyth_wp = (h_pw - v_pw) if (h_pw is not None and v_pw is not None) else None

        h_lp = home_rs.get("lineup_pts")
        v_lp = vis_rs.get("lineup_pts")
        diff_rs_lineup_pts = (h_lp - v_lp) if (h_lp is not None and v_lp is not None) else None

        # Dynamic playoff elo features (pre-game)
        diff_elo_po = elo_po[home_id] - elo_po[vis_id]
        elo_win_prob_po_val = elo_win_prob(elo_po[home_id], elo_po[vis_id])
        diff_elo_change_po = diff_elo_po - diff_elo_rs

        feature_rows.append({
            "game_id": game_id,
            "season_year": season_year,
            "game_date": game_date_str,
            "home_team_id": home_id,
            "vis_team_id": vis_id,
            "home_team_abbr": str(game["home_team_abbr"]),
            "vis_team_abbr": str(game["vis_team_abbr"]),
            "home_win": home_win,
            "playoff_round": playoff_round,
            "game_in_series": game_in_series,
            "home_series_wins": home_series_wins,
            "vis_series_wins": vis_series_wins,
            "series_score_diff": series_score_diff,
            "is_elimination": is_elimination,
            "home_has_homecourt": home_has_homecourt,
            "series_rest_days": series_rest_days,
            "diff_elo_rs": diff_elo_rs,
            "diff_rs_net_rtg": diff_rs_net_rtg,
            "diff_rs_pyth_wp": diff_rs_pyth_wp,
            "diff_rs_lineup_pts": diff_rs_lineup_pts,
            "diff_elo_po": diff_elo_po,
            "elo_win_prob_po": elo_win_prob_po_val,
            "diff_elo_change_po": diff_elo_change_po,
        })

        # Update series state AFTER computing features
        ss["game_count"] += 1
        ss["last_date"] = date.fromisoformat(game_date_str)
        if home_id == ss["home_id"]:
            ss["home_wins"] += home_win
            ss["vis_wins"] += (1 - home_win)
        else:
            ss["vis_wins"] += home_win
            ss["home_wins"] += (1 - home_win)

        # Update playoff elo
        vis_win = 1 - home_win
        exp_home = elo_win_prob(elo_po[home_id], elo_po[vis_id])
        elo_po[home_id] += ELO_K_PO * (home_win - exp_home)
        elo_po[vis_id] += ELO_K_PO * (vis_win - (1.0 - exp_home))

    return feature_rows


def write_features(conn, rows):
    if not rows:
        return 0
    conn.executemany(UPSERT_SQL, rows)
    conn.commit()
    return len(rows)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Build NBA playoff pre-game features into nba.sqlite/playoff_game_features."
    )
    parser.add_argument("--season", type=int, help="Rebuild only one season year (e.g. 2011).")
    return parser


def main():
    args = build_parser().parse_args()
    print("=== NBA Playoff Feature Builder ===")

    conn = connect_db()
    try:
        # Load regular season games and player data
        rs_games = load_rs_games(conn)
        rs_player_map = load_rs_player_game_map(conn)
        print(f"Loaded {len(rs_games):,} regular season games.")

        # Compute end-of-RS state per season
        season_end_states = compute_rs_end_state(rs_games, rs_player_map)
        print(f"Computed RS end states for seasons: {sorted(season_end_states.keys())}")

        # Load playoff games
        po_games = load_po_games(conn, target_season=args.season)
        print(f"Loaded {len(po_games):,} playoff games to process.")

        if not po_games:
            print("No playoff games found. Run nba_playoff_scraper.py first.")
            return

        # Clear existing features for target scope
        if args.season is not None:
            conn.execute("DELETE FROM playoff_game_features WHERE season_year = ?", (args.season,))
        else:
            conn.execute("DELETE FROM playoff_game_features")
        conn.commit()

        # Build features
        feature_rows = build_playoff_features(po_games, season_end_states)
        inserted = write_features(conn, feature_rows)

        # Summary
        from collections import Counter
        season_counts = Counter(row["season_year"] for row in feature_rows)
        for yr, cnt in sorted(season_counts.items()):
            # Check non-null rate for key features
            non_null_elo = sum(1 for r in feature_rows if r["season_year"] == yr and r["diff_elo_rs"] is not None)
            print(f"[{yr}] {cnt} games, diff_elo_rs non-null: {non_null_elo}/{cnt}")

        print(f"Built {inserted:,} playoff features across {len(season_counts)} seasons.")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
