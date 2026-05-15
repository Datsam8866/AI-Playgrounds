import argparse
import sqlite3
from collections import defaultdict, deque
from datetime import date
from pathlib import Path


DB_PATH = Path(__file__).resolve().parent / "nba.sqlite"

ELO_K = 20.0
ELO_HOME_ADV = 100.0
ELO_BASE = 1500.0
ELO_REGRESSION = 0.35

WINDOW = 20
MIN_GAMES = 5
SHORT_WINDOW = 10
FIRST_GAME_REST = 7

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS game_features (
    feature_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id             TEXT NOT NULL UNIQUE,
    season_year         INTEGER NOT NULL,
    game_date           TEXT NOT NULL,
    home_team_id        INTEGER NOT NULL,
    vis_team_id         INTEGER NOT NULL,
    home_team_abbr      TEXT NOT NULL,
    vis_team_abbr       TEXT NOT NULL,
    home_win            INTEGER NOT NULL,

    home_elo            REAL,
    vis_elo             REAL,
    diff_elo            REAL,
    elo_win_prob        REAL,

    home_win_pct_10     REAL,
    vis_win_pct_10      REAL,
    diff_win_pct_10     REAL,
    home_win_pct_20     REAL,
    vis_win_pct_20      REAL,
    diff_win_pct_20     REAL,

    home_pts_pg         REAL,
    vis_pts_pg          REAL,
    home_pts_allowed_pg REAL,
    vis_pts_allowed_pg  REAL,
    home_net_rtg        REAL,
    vis_net_rtg         REAL,
    diff_net_rtg        REAL,

    home_pyth_wp        REAL,
    vis_pyth_wp         REAL,
    diff_pyth_wp        REAL,

    home_rest           INTEGER,
    vis_rest            INTEGER,
    diff_rest           INTEGER,
    home_b2b            INTEGER,
    vis_b2b             INTEGER,

    home_streak         INTEGER,
    vis_streak          INTEGER,
    diff_streak         INTEGER,

    home_games_before   INTEGER,
    vis_games_before    INTEGER,
    is_neutral_site     INTEGER NOT NULL DEFAULT 0,

    home_lineup_pts     REAL,
    vis_lineup_pts      REAL,
    diff_lineup_pts     REAL
);

CREATE INDEX IF NOT EXISTS idx_game_features_season_year
    ON game_features(season_year);
CREATE INDEX IF NOT EXISTS idx_game_features_game_date
    ON game_features(game_date);
"""

UPSERT_SQL = """
INSERT INTO game_features (
    game_id,
    season_year,
    game_date,
    home_team_id,
    vis_team_id,
    home_team_abbr,
    vis_team_abbr,
    home_win,
    home_elo,
    vis_elo,
    diff_elo,
    elo_win_prob,
    home_win_pct_10,
    vis_win_pct_10,
    diff_win_pct_10,
    home_win_pct_20,
    vis_win_pct_20,
    diff_win_pct_20,
    home_pts_pg,
    vis_pts_pg,
    home_pts_allowed_pg,
    vis_pts_allowed_pg,
    home_net_rtg,
    vis_net_rtg,
    diff_net_rtg,
    home_pyth_wp,
    vis_pyth_wp,
    diff_pyth_wp,
    home_rest,
    vis_rest,
    diff_rest,
    home_b2b,
    vis_b2b,
    home_streak,
    vis_streak,
    diff_streak,
    home_games_before,
    vis_games_before,
    is_neutral_site,
    home_lineup_pts,
    vis_lineup_pts,
    diff_lineup_pts
)
VALUES (
    :game_id,
    :season_year,
    :game_date,
    :home_team_id,
    :vis_team_id,
    :home_team_abbr,
    :vis_team_abbr,
    :home_win,
    :home_elo,
    :vis_elo,
    :diff_elo,
    :elo_win_prob,
    :home_win_pct_10,
    :vis_win_pct_10,
    :diff_win_pct_10,
    :home_win_pct_20,
    :vis_win_pct_20,
    :diff_win_pct_20,
    :home_pts_pg,
    :vis_pts_pg,
    :home_pts_allowed_pg,
    :vis_pts_allowed_pg,
    :home_net_rtg,
    :vis_net_rtg,
    :diff_net_rtg,
    :home_pyth_wp,
    :vis_pyth_wp,
    :diff_pyth_wp,
    :home_rest,
    :vis_rest,
    :diff_rest,
    :home_b2b,
    :vis_b2b,
    :home_streak,
    :vis_streak,
    :diff_streak,
    :home_games_before,
    :vis_games_before,
    :is_neutral_site,
    :home_lineup_pts,
    :vis_lineup_pts,
    :diff_lineup_pts
)
ON CONFLICT(game_id) DO UPDATE SET
    season_year = excluded.season_year,
    game_date = excluded.game_date,
    home_team_id = excluded.home_team_id,
    vis_team_id = excluded.vis_team_id,
    home_team_abbr = excluded.home_team_abbr,
    vis_team_abbr = excluded.vis_team_abbr,
    home_win = excluded.home_win,
    home_elo = excluded.home_elo,
    vis_elo = excluded.vis_elo,
    diff_elo = excluded.diff_elo,
    elo_win_prob = excluded.elo_win_prob,
    home_win_pct_10 = excluded.home_win_pct_10,
    vis_win_pct_10 = excluded.vis_win_pct_10,
    diff_win_pct_10 = excluded.diff_win_pct_10,
    home_win_pct_20 = excluded.home_win_pct_20,
    vis_win_pct_20 = excluded.vis_win_pct_20,
    diff_win_pct_20 = excluded.diff_win_pct_20,
    home_pts_pg = excluded.home_pts_pg,
    vis_pts_pg = excluded.vis_pts_pg,
    home_pts_allowed_pg = excluded.home_pts_allowed_pg,
    vis_pts_allowed_pg = excluded.vis_pts_allowed_pg,
    home_net_rtg = excluded.home_net_rtg,
    vis_net_rtg = excluded.vis_net_rtg,
    diff_net_rtg = excluded.diff_net_rtg,
    home_pyth_wp = excluded.home_pyth_wp,
    vis_pyth_wp = excluded.vis_pyth_wp,
    diff_pyth_wp = excluded.diff_pyth_wp,
    home_rest = excluded.home_rest,
    vis_rest = excluded.vis_rest,
    diff_rest = excluded.diff_rest,
    home_b2b = excluded.home_b2b,
    vis_b2b = excluded.vis_b2b,
    home_streak = excluded.home_streak,
    vis_streak = excluded.vis_streak,
    diff_streak = excluded.diff_streak,
    home_games_before = excluded.home_games_before,
    vis_games_before = excluded.vis_games_before,
    is_neutral_site = excluded.is_neutral_site,
    home_lineup_pts = excluded.home_lineup_pts,
    vis_lineup_pts = excluded.vis_lineup_pts,
    diff_lineup_pts = excluded.diff_lineup_pts
"""


def load_player_game_map(conn) -> dict:
    """Returns {game_id: {team_id: [{player_id, min_seconds, pts}]}} from player_game_stats."""
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


def compute_lineup_pts(team_id: int, season_year, team_last_lineup: dict, player_pts_acc: dict):
    lineup = team_last_lineup.get(team_id)
    if not lineup:
        return None
    total = 0.0
    counted = 0
    for pid in lineup:
        acc = player_pts_acc.get((season_year, pid))
        if acc and acc["games"] > 0:
            total += acc["pts"] / acc["games"]
            counted += 1
    return total if counted > 0 else None


def update_player_state(game, player_game_map: dict, team_last_lineup: dict, player_pts_acc: dict) -> None:
    game_id = game["game_id"]
    season_year = game["season_year"]
    team_data = player_game_map.get(game_id, {})
    for team_id_raw, players in team_data.items():
        tid = int(team_id_raw)
        active = [p["player_id"] for p in players if p["min_seconds"] > 0]
        if active:
            team_last_lineup[tid] = active
        for p in players:
            if p["min_seconds"] <= 0:
                continue
            key = (season_year, p["player_id"])
            if key not in player_pts_acc:
                player_pts_acc[key] = {"pts": 0.0, "games": 0}
            player_pts_acc[key]["pts"] += p["pts"]
            player_pts_acc[key]["games"] += 1


def build_parser():
    parser = argparse.ArgumentParser(
        description="Build NBA pre-game features into nba.sqlite/game_features."
    )
    parser.add_argument(
        "--season",
        type=int,
        help="Rebuild only one season year (e.g. 2011 for 2011-12).",
    )
    return parser


def connect_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    return conn


def rebuild_game_features_table(conn):
    conn.execute("DROP TABLE IF EXISTS game_features")
    conn.executescript(SCHEMA_SQL)
    conn.commit()


def load_games(conn):
    rows = conn.execute(
        """
        SELECT game_id,
               season_year,
               game_date,
               home_team_id,
               vis_team_id,
               home_team_abbr,
               vis_team_abbr,
               home_score,
               vis_score,
               home_win
        FROM game_results
        WHERE home_win IS NOT NULL
        ORDER BY season_year, game_date, game_id
        """
    ).fetchall()
    game_ids = [row["game_id"] for row in rows]
    assert len(game_ids) == len(set(game_ids)), "Duplicate game_id found in game_results"
    return rows


def pyth_wp(pts, pts_allowed, exp=13.91):
    if pts + pts_allowed == 0:
        return 0.5
    return (pts**exp) / ((pts**exp) + (pts_allowed**exp))


def is_neutral_site(game) -> int:
    if int(game["season_year"]) == 2019 and game["game_date"] >= "2020-07-30":
        return 1
    return 0


def elo_win_prob(home_elo, vis_elo, neutral=0):
    home_adv = 0.0 if neutral else ELO_HOME_ADV
    return 1.0 / (1.0 + 10.0 ** (-(home_elo + home_adv - vis_elo) / 400.0))


def regress_elo(elo_by_team):
    for team_id in list(elo_by_team.keys()):
        prev_elo = elo_by_team[team_id]
        elo_by_team[team_id] = prev_elo + (ELO_BASE - prev_elo) * ELO_REGRESSION


def summarize_recent(history, window, min_games=MIN_GAMES):
    recent = list(history)[-window:]
    if len(recent) < min_games:
        return None

    games = len(recent)
    wins = sum(item["win"] for item in recent)
    pts = sum(item["pts"] for item in recent)
    pts_allowed = sum(item["pts_allowed"] for item in recent)
    return {
        "games": games,
        "win_pct": wins / games,
        "pts_pg": pts / games,
        "pts_allowed_pg": pts_allowed / games,
        "net_rtg": (pts - pts_allowed) / games,
        "pyth_wp": pyth_wp(pts, pts_allowed),
    }


def current_streak(streak_by_team, team_id):
    return streak_by_team.get(team_id, 0)


def compute_rest(last_game_dates, team_id, game_date_iso, season_games_before):
    if season_games_before == 0:
        return FIRST_GAME_REST
    last_dt = last_game_dates.get(team_id)
    if last_dt is None:
        return FIRST_GAME_REST
    current_dt = date.fromisoformat(game_date_iso)
    return max(0, (current_dt - last_dt).days)


def to_feature_row(game, team_history, elo_by_team, streak_by_team, last_game_dates, season_games, neutral,
                   team_last_lineup=None, player_pts_acc=None):
    home_id = game["home_team_id"]
    vis_id = game["vis_team_id"]
    season_year = game["season_year"]

    home_elo = elo_by_team[home_id]
    vis_elo = elo_by_team[vis_id]
    home_hist = team_history[home_id]
    vis_hist = team_history[vis_id]

    home10 = summarize_recent(home_hist, SHORT_WINDOW)
    vis10 = summarize_recent(vis_hist, SHORT_WINDOW)
    home20 = summarize_recent(home_hist, WINDOW)
    vis20 = summarize_recent(vis_hist, WINDOW)

    home_games_before = season_games[(season_year, home_id)]
    vis_games_before = season_games[(season_year, vis_id)]
    home_rest = compute_rest(last_game_dates, home_id, game["game_date"], home_games_before)
    vis_rest = compute_rest(last_game_dates, vis_id, game["game_date"], vis_games_before)
    home_streak = current_streak(streak_by_team, home_id)
    vis_streak = current_streak(streak_by_team, vis_id)

    home_lineup = compute_lineup_pts(home_id, season_year, team_last_lineup or {}, player_pts_acc or {})
    vis_lineup = compute_lineup_pts(vis_id, season_year, team_last_lineup or {}, player_pts_acc or {})
    diff_lineup = (home_lineup - vis_lineup) if (home_lineup is not None and vis_lineup is not None) else None

    return {
        "game_id": game["game_id"],
        "season_year": season_year,
        "game_date": game["game_date"],
        "home_team_id": home_id,
        "vis_team_id": vis_id,
        "home_team_abbr": game["home_team_abbr"],
        "vis_team_abbr": game["vis_team_abbr"],
        "home_win": game["home_win"],
        "home_elo": home_elo,
        "vis_elo": vis_elo,
        "diff_elo": home_elo - vis_elo,
        "elo_win_prob": elo_win_prob(home_elo, vis_elo, neutral=neutral),
        "home_win_pct_10": home10["win_pct"] if home10 else None,
        "vis_win_pct_10": vis10["win_pct"] if vis10 else None,
        "diff_win_pct_10": (home10["win_pct"] - vis10["win_pct"]) if home10 and vis10 else None,
        "home_win_pct_20": home20["win_pct"] if home20 else None,
        "vis_win_pct_20": vis20["win_pct"] if vis20 else None,
        "diff_win_pct_20": (home20["win_pct"] - vis20["win_pct"]) if home20 and vis20 else None,
        "home_pts_pg": home20["pts_pg"] if home20 else None,
        "vis_pts_pg": vis20["pts_pg"] if vis20 else None,
        "home_pts_allowed_pg": home20["pts_allowed_pg"] if home20 else None,
        "vis_pts_allowed_pg": vis20["pts_allowed_pg"] if vis20 else None,
        "home_net_rtg": home20["net_rtg"] if home20 else None,
        "vis_net_rtg": vis20["net_rtg"] if vis20 else None,
        "diff_net_rtg": (home20["net_rtg"] - vis20["net_rtg"]) if home20 and vis20 else None,
        "home_pyth_wp": home20["pyth_wp"] if home20 else None,
        "vis_pyth_wp": vis20["pyth_wp"] if vis20 else None,
        "diff_pyth_wp": (home20["pyth_wp"] - vis20["pyth_wp"]) if home20 and vis20 else None,
        "home_rest": home_rest,
        "vis_rest": vis_rest,
        "diff_rest": home_rest - vis_rest,
        "home_b2b": 1 if home_rest == 1 else 0,
        "vis_b2b": 1 if vis_rest == 1 else 0,
        "home_streak": home_streak,
        "vis_streak": vis_streak,
        "diff_streak": home_streak - vis_streak,
        "home_games_before": home_games_before,
        "vis_games_before": vis_games_before,
        "is_neutral_site": neutral,
        "home_lineup_pts": home_lineup,
        "vis_lineup_pts": vis_lineup,
        "diff_lineup_pts": diff_lineup,
    }


def update_team_state(game, team_history, elo_by_team, streak_by_team, last_game_dates, season_games, neutral):
    home_id = game["home_team_id"]
    vis_id = game["vis_team_id"]
    season_year = game["season_year"]
    home_win = int(game["home_win"])
    vis_win = 1 - home_win
    home_score = int(game["home_score"])
    vis_score = int(game["vis_score"])

    expected_home = elo_win_prob(elo_by_team[home_id], elo_by_team[vis_id], neutral=neutral)
    elo_by_team[home_id] += ELO_K * (home_win - expected_home)
    elo_by_team[vis_id] += ELO_K * (vis_win - (1.0 - expected_home))

    team_history[home_id].append(
        {"win": home_win, "pts": home_score, "pts_allowed": vis_score}
    )
    team_history[vis_id].append(
        {"win": vis_win, "pts": vis_score, "pts_allowed": home_score}
    )

    streak_by_team[home_id] = next_streak(streak_by_team.get(home_id, 0), home_win)
    streak_by_team[vis_id] = next_streak(streak_by_team.get(vis_id, 0), vis_win)

    game_dt = date.fromisoformat(game["game_date"])
    last_game_dates[home_id] = game_dt
    last_game_dates[vis_id] = game_dt

    season_games[(season_year, home_id)] += 1
    season_games[(season_year, vis_id)] += 1


def next_streak(previous_streak, won_game):
    if won_game:
        return previous_streak + 1 if previous_streak > 0 else 1
    return previous_streak - 1 if previous_streak < 0 else -1


def clear_scope(conn, season_year=None):
    if season_year is None:
        conn.execute("DELETE FROM game_features")
    else:
        conn.execute("DELETE FROM game_features WHERE season_year = ?", (season_year,))
    conn.commit()


def build_features(games, target_season=None, player_game_map=None):
    team_history = defaultdict(lambda: deque(maxlen=WINDOW))
    elo_by_team = defaultdict(lambda: ELO_BASE)
    streak_by_team = {}
    last_game_dates = {}
    season_games = defaultdict(int)
    team_last_lineup: dict = {}
    player_pts_acc: dict = {}

    if player_game_map is None:
        player_game_map = {}

    current_season = None
    feature_rows = []
    season_counts = defaultdict(int)

    for game in games:
        season_year = game["season_year"]
        if season_year != current_season:
            if current_season is not None:
                regress_elo(elo_by_team)
            current_season = season_year
            streak_by_team = {}
            last_game_dates = {}
            team_last_lineup = {}
        neutral = is_neutral_site(game)

        row = to_feature_row(
            game=game,
            team_history=team_history,
            elo_by_team=elo_by_team,
            streak_by_team=streak_by_team,
            last_game_dates=last_game_dates,
            season_games=season_games,
            neutral=neutral,
            team_last_lineup=team_last_lineup,
            player_pts_acc=player_pts_acc,
        )
        if target_season is None or season_year == target_season:
            feature_rows.append(row)
            season_counts[season_year] += 1

        update_team_state(
            game=game,
            team_history=team_history,
            elo_by_team=elo_by_team,
            streak_by_team=streak_by_team,
            last_game_dates=last_game_dates,
            season_games=season_games,
            neutral=neutral,
        )
        update_player_state(game, player_game_map, team_last_lineup, player_pts_acc)

    return feature_rows, dict(sorted(season_counts.items()))


def write_features(conn, rows):
    if not rows:
        return 0
    conn.executemany(UPSERT_SQL, rows)
    conn.commit()
    return len(rows)


def parse_args():
    return build_parser().parse_args()


def main():
    args = parse_args()

    print("=== NBA Feature Builder ===")

    conn = connect_db()
    try:
        games = load_games(conn)
        player_game_map = load_player_game_map(conn)
        if player_game_map:
            print(f"Loaded player data for {len(player_game_map):,} games.")
        else:
            print("No player_game_stats found; lineup features will be NULL.")
        if args.season is None:
            rebuild_game_features_table(conn)
        else:
            clear_scope(conn, args.season)
        feature_rows, season_counts = build_features(games, args.season, player_game_map=player_game_map)
        inserted = write_features(conn, feature_rows)

        for season_year, count in season_counts.items():
            print(f"[{season_year}] {count} games -> {count} features")

        print(f"Built {inserted:,} features across {len(season_counts)} seasons.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
