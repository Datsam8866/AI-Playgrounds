# -*- coding: utf-8 -*-
"""
nba_playoff_scraper.py

爬取 NBA 季後賽比賽結果與球員數據，存入 nba.sqlite。
表：playoff_game_results, playoff_player_game_stats, playoff_seasons_fetched
"""

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
CREATE TABLE IF NOT EXISTS playoff_game_results (
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

CREATE TABLE IF NOT EXISTS playoff_seasons_fetched (
    season_year INTEGER PRIMARY KEY,
    fetched_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS playoff_player_game_stats (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id   INTEGER NOT NULL,
    player_name TEXT NOT NULL,
    team_id     INTEGER NOT NULL,
    game_id     TEXT NOT NULL,
    game_date   TEXT NOT NULL,
    season_year INTEGER NOT NULL,
    min_seconds INTEGER NOT NULL DEFAULT 0,
    pts         REAL NOT NULL DEFAULT 0,
    UNIQUE(player_id, game_id)
);
"""


def season_label(season_year: int) -> str:
    next_suffix = str((season_year + 1) % 100).zfill(2)
    return f"{season_year}-{next_suffix}"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_min_to_seconds(min_str) -> int:
    if min_str is None or str(min_str).strip() in ("", "nan"):
        return 0
    s = str(min_str).strip()
    if ":" in s:
        parts = s.split(":")
        try:
            return int(parts[0]) * 60 + int(float(parts[1]))
        except (ValueError, IndexError):
            return 0
    try:
        return int(float(s) * 60)
    except ValueError:
        return 0


def to_int(value):
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def connect_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    return conn


def fetched_seasons(conn):
    rows = conn.execute("SELECT season_year FROM playoff_seasons_fetched").fetchall()
    return {row[0] for row in rows}


def delete_season_rows(conn, season_year: int):
    conn.execute("DELETE FROM playoff_game_results WHERE season_year = ?", (season_year,))
    conn.execute("DELETE FROM playoff_player_game_stats WHERE season_year = ?", (season_year,))
    conn.execute("DELETE FROM playoff_seasons_fetched WHERE season_year = ?", (season_year,))
    conn.commit()


def fetch_team_game_log(season_year: int):
    label = season_label(season_year)
    gl = LeagueGameLog(
        player_or_team_abbreviation="T",
        season=label,
        season_type_all_star="Playoffs",
    )
    try:
        return gl.get_data_frames()[0]
    finally:
        time.sleep(RATE_LIMIT_SECONDS)


def fetch_player_game_log(season_year: int):
    label = season_label(season_year)
    gl = LeagueGameLog(
        player_or_team_abbreviation="P",
        season=label,
        season_type_all_star="Playoffs",
    )
    try:
        return gl.get_data_frames()[0]
    finally:
        time.sleep(RATE_LIMIT_SECONDS)


def build_game_rows(df, season_year: int):
    rows = []
    for game_id, group in df.groupby("GAME_ID", sort=False):
        game_rows_list = group.to_dict("records")
        home_row = None
        vis_row = None
        for row in game_rows_list:
            matchup = str(row.get("MATCHUP") or "")
            if "vs." in matchup:
                home_row = row
            elif "@" in matchup:
                vis_row = row
        if not home_row or not vis_row:
            print(f"  warning: skip malformed game_id={game_id}")
            continue
        # 只保留季後賽 game_id（004...）
        gid = str(game_id)
        if not gid.startswith("004"):
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
        rows.append((
            gid,
            season_year,
            str(home_row["GAME_DATE"]),
            int(home_row["TEAM_ID"]),
            int(vis_row["TEAM_ID"]),
            str(home_row["TEAM_ABBREVIATION"]),
            str(vis_row["TEAM_ABBREVIATION"]),
            home_score,
            vis_score,
            home_win,
        ))
    return rows


def upsert_games(conn, game_rows):
    conn.executemany(
        """
        INSERT INTO playoff_game_results (
            game_id, season_year, game_date,
            home_team_id, vis_team_id,
            home_team_abbr, vis_team_abbr,
            home_score, vis_score, home_win
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


def build_player_rows(df, season_year: int):
    rows = []
    for row in df.to_dict("records"):
        gid = str(row.get("GAME_ID", ""))
        if not gid.startswith("004"):
            continue
        rows.append((
            int(row["PLAYER_ID"]),
            str(row["PLAYER_NAME"]),
            int(row["TEAM_ID"]),
            gid,
            str(row["GAME_DATE"]),
            season_year,
            parse_min_to_seconds(row.get("MIN")),
            float(row.get("PTS") or 0),
        ))
    return rows


def upsert_players(conn, player_rows):
    conn.executemany(
        """
        INSERT INTO playoff_player_game_stats (
            player_id, player_name, team_id,
            game_id, game_date, season_year,
            min_seconds, pts
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(player_id, game_id) DO UPDATE SET
            player_name = excluded.player_name,
            team_id = excluded.team_id,
            game_date = excluded.game_date,
            season_year = excluded.season_year,
            min_seconds = excluded.min_seconds,
            pts = excluded.pts
        """,
        player_rows,
    )


def mark_season_fetched(conn, season_year: int):
    conn.execute(
        """
        INSERT INTO playoff_seasons_fetched (season_year, fetched_at)
        VALUES (?, ?)
        ON CONFLICT(season_year) DO UPDATE SET fetched_at = excluded.fetched_at
        """,
        (season_year, utc_now_iso()),
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description="Scrape NBA playoff game results and player stats into SQLite."
    )
    parser.add_argument("--season", type=int, help="Fetch a single season year (e.g. 2011 for 2011-12).")
    parser.add_argument("--force", action="store_true", help="Re-fetch even if already fetched.")
    return parser


def season_years_from_args(args):
    if args.season is not None:
        if not (DEFAULT_START_YEAR <= args.season <= DEFAULT_END_YEAR):
            raise SystemExit(f"--season must be between {DEFAULT_START_YEAR} and {DEFAULT_END_YEAR}")
        return [args.season]
    return list(range(DEFAULT_START_YEAR, DEFAULT_END_YEAR + 1))


def main():
    args = build_parser().parse_args()
    seasons = season_years_from_args(args)

    print("=== NBA Playoff Scraper ===")
    conn = connect_db()
    total_games = 0

    try:
        fetched = fetched_seasons(conn)
        for season_year in seasons:
            if season_year in fetched and not args.force:
                print(f"[{season_year}] Skipped (already fetched)")
                continue
            if args.force:
                delete_season_rows(conn, season_year)

            label = season_label(season_year)
            print(f"[{season_year}] Fetching {label} (Teams)...")
            team_df = fetch_team_game_log(season_year)
            game_rows = build_game_rows(team_df, season_year)
            upsert_games(conn, game_rows)
            conn.commit()
            print(f"[{season_year}]   {len(game_rows)} playoff games stored.")

            print(f"[{season_year}] Fetching {label} (Players)...")
            player_df = fetch_player_game_log(season_year)
            player_rows = build_player_rows(player_df, season_year)
            upsert_players(conn, player_rows)
            conn.commit()
            print(f"[{season_year}]   {len(player_rows)} player-game records stored.")

            mark_season_fetched(conn, season_year)
            conn.commit()
            total_games += len(game_rows)
            fetched.add(season_year)

    finally:
        conn.close()

    print(f"Done. Total: {total_games} playoff games across {len(seasons)} seasons.")


if __name__ == "__main__":
    main()
