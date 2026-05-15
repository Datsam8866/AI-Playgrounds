import argparse
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

from nba_api.stats.endpoints import LeagueGameLog


DB_PATH = Path(__file__).resolve().parent / "nba.sqlite"
DEFAULT_START_YEAR = 2011
DEFAULT_END_YEAR = 2025
RATE_LIMIT_SECONDS = 1.5

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS game_results (
    game_id         TEXT PRIMARY KEY,
    season_year     INTEGER NOT NULL,
    game_date       TEXT NOT NULL,
    home_team_id    INTEGER NOT NULL,
    vis_team_id     INTEGER NOT NULL,
    home_team_abbr  TEXT NOT NULL,
    vis_team_abbr   TEXT NOT NULL,
    home_score      INTEGER,
    vis_score       INTEGER,
    home_win        INTEGER
);

CREATE TABLE IF NOT EXISTS seasons_fetched (
    season_year INTEGER PRIMARY KEY,
    fetched_at  TEXT NOT NULL
);
"""


def build_parser():
    parser = argparse.ArgumentParser(
        description="Scrape NBA regular-season game results into SQLite."
    )
    parser.add_argument(
        "--season",
        type=int,
        help="Fetch a single season year (e.g. 2011 for 2011-12).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-fetch selected seasons even if seasons_fetched already contains them.",
    )
    return parser


def validate_season_year(season_year):
    if season_year < DEFAULT_START_YEAR or season_year > DEFAULT_END_YEAR:
        raise SystemExit(
            f"--season must be between {DEFAULT_START_YEAR} and {DEFAULT_END_YEAR}"
        )


def season_label(season_year):
    next_suffix = str((season_year + 1) % 100).zfill(2)
    return f"{season_year}-{next_suffix}"


def utc_now_iso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def connect_db():
    connection = sqlite3.connect(DB_PATH)
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA synchronous = NORMAL")
    connection.executescript(SCHEMA_SQL)
    connection.commit()
    return connection


def fetched_seasons(connection):
    rows = connection.execute("SELECT season_year FROM seasons_fetched").fetchall()
    return {row[0] for row in rows}


def delete_season_rows(connection, season_year):
    connection.execute("DELETE FROM game_results WHERE season_year = ?", (season_year,))
    connection.execute(
        "DELETE FROM seasons_fetched WHERE season_year = ?", (season_year,)
    )
    connection.commit()


def fetch_league_game_log(season_year):
    label = season_label(season_year)
    game_log = LeagueGameLog(
        season=label,
        season_type_all_star="Regular Season",
    )
    try:
        return game_log.get_data_frames()[0]
    finally:
        time.sleep(RATE_LIMIT_SECONDS)


def build_game_rows(df, season_year):
    rows_to_upsert = []

    for game_id, group in df.groupby("GAME_ID", sort=False):
        game_rows = group.to_dict("records")
        home_row = None
        vis_row = None

        for row in game_rows:
            matchup = row.get("MATCHUP") or ""
            if "vs." in matchup:
                home_row = row
            elif "@" in matchup:
                vis_row = row

        if not home_row or not vis_row:
            print(f"warning: skip malformed game_id={game_id}")
            continue

        home_score = to_int(home_row.get("PTS"))
        vis_score = to_int(vis_row.get("PTS"))
        wl_val = home_row.get("WL")
        home_wl = "" if (wl_val != wl_val or wl_val is None) else str(wl_val).strip().upper()

        if home_score is None or vis_score is None:
            home_win = None
        elif home_wl == "W":
            home_win = 1
        elif home_wl == "L":
            home_win = 0
        else:
            home_win = None

        rows_to_upsert.append(
            (
                str(game_id),
                season_year,
                str(home_row["GAME_DATE"]),
                int(home_row["TEAM_ID"]),
                int(vis_row["TEAM_ID"]),
                str(home_row["TEAM_ABBREVIATION"]),
                str(vis_row["TEAM_ABBREVIATION"]),
                home_score,
                vis_score,
                home_win,
            )
        )

    return rows_to_upsert


def to_int(value):
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def upsert_games(connection, game_rows):
    connection.executemany(
        """
        INSERT INTO game_results (
            game_id,
            season_year,
            game_date,
            home_team_id,
            vis_team_id,
            home_team_abbr,
            vis_team_abbr,
            home_score,
            vis_score,
            home_win
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(game_id) DO UPDATE SET
            season_year = excluded.season_year,
            game_date = excluded.game_date,
            home_team_id = excluded.home_team_id,
            vis_team_id = excluded.vis_team_id,
            home_team_abbr = excluded.home_team_abbr,
            vis_team_abbr = excluded.vis_team_abbr,
            home_score = excluded.home_score,
            vis_score = excluded.vis_score,
            home_win = excluded.home_win
        """,
        game_rows,
    )


def mark_season_fetched(connection, season_year):
    connection.execute(
        """
        INSERT INTO seasons_fetched (season_year, fetched_at)
        VALUES (?, ?)
        ON CONFLICT(season_year) DO UPDATE SET
            fetched_at = excluded.fetched_at
        """,
        (season_year, utc_now_iso()),
    )


def season_years_from_args(args):
    if args.season is not None:
        validate_season_year(args.season)
        return [args.season]
    return list(range(DEFAULT_START_YEAR, DEFAULT_END_YEAR + 1))


def main():
    args = build_parser().parse_args()
    seasons = season_years_from_args(args)

    print("=== NBA Scraper ===")

    connection = connect_db()
    total_games = 0

    try:
        fetched = fetched_seasons(connection)

        for season_year in seasons:
            if season_year in fetched and not args.force:
                print(f"[{season_year}] Skipped (already fetched)")
                continue

            if args.force:
                delete_season_rows(connection, season_year)

            label = season_label(season_year)
            print(f"[{season_year}] Fetching {label}...")

            df = fetch_league_game_log(season_year)
            game_rows = build_game_rows(df, season_year)

            upsert_games(connection, game_rows)
            mark_season_fetched(connection, season_year)
            connection.commit()

            fetched_count = len(game_rows)
            total_games += fetched_count
            fetched.add(season_year)
            print(f"[{season_year}] {fetched_count:,} games -> nba.sqlite")
    finally:
        connection.close()

    print(f"Done. Total: {total_games:,} games across {len(seasons)} seasons.")


if __name__ == "__main__":
    main()
