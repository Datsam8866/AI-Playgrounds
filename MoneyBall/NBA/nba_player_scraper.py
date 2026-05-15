import argparse
import sqlite3
import time
from datetime import datetime
from pathlib import Path

from nba_api.stats.endpoints import LeagueGameLog


DB_PATH = Path(__file__).resolve().parent / "nba.sqlite"
DEFAULT_START_YEAR = 2011
DEFAULT_END_YEAR = 2025
RATE_LIMIT_SECONDS = 1.5

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS player_game_stats (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id   INTEGER NOT NULL,
    player_name TEXT NOT NULL,
    team_id     INTEGER NOT NULL,
    game_id     TEXT NOT NULL,
    game_date   TEXT NOT NULL,
    season_year INTEGER NOT NULL,
    min_seconds INTEGER NOT NULL DEFAULT 0,
    pts         REAL NOT NULL DEFAULT 0,
    reb         REAL NOT NULL DEFAULT 0,
    ast         REAL NOT NULL DEFAULT 0,
    UNIQUE(player_id, game_id)
);
CREATE TABLE IF NOT EXISTS player_seasons_fetched (
    season_year INTEGER PRIMARY KEY
);
"""


def season_label(season_year: int) -> str:
    return f"{season_year}-{str((season_year + 1) % 100).zfill(2)}"


def parse_game_date(val) -> str | None:
    if val is None or val != val:
        return None
    s = str(val).strip()
    for fmt in ("%b %d, %Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except ValueError:
            pass
    return None


def parse_minutes(val) -> int:
    if val is None or val != val:
        return 0
    s = str(val).strip()
    if not s or s == "0":
        return 0
    if ":" in s:
        parts = s.split(":")
        try:
            return int(parts[0]) * 60 + (int(parts[1]) if len(parts) > 1 else 0)
        except (ValueError, IndexError):
            return 0
    try:
        return int(float(s)) * 60
    except (ValueError, TypeError):
        return 0


def safe_float(val, default: float = 0.0) -> float:
    if val is None or val != val:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def connect_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    return conn


def load_fetched_seasons(conn: sqlite3.Connection) -> set[int]:
    return {row[0] for row in conn.execute("SELECT season_year FROM player_seasons_fetched").fetchall()}


def load_valid_game_ids(conn: sqlite3.Connection) -> set[str]:
    return {row[0] for row in conn.execute("SELECT DISTINCT game_id FROM game_results").fetchall()}


def delete_season(conn: sqlite3.Connection, season_year: int) -> None:
    conn.execute("DELETE FROM player_game_stats WHERE season_year = ?", (season_year,))
    conn.execute("DELETE FROM player_seasons_fetched WHERE season_year = ?", (season_year,))
    conn.commit()


def fetch_player_log(season_year: int):
    df = LeagueGameLog(
        player_or_team_abbreviation="P",
        season=season_label(season_year),
        season_type_all_star="Regular Season",
    ).get_data_frames()[0]
    time.sleep(RATE_LIMIT_SECONDS)
    return df


def build_rows(df, season_year: int, valid_game_ids: set[str]) -> list[tuple]:
    rows = []
    for _, r in df.iterrows():
        game_id = str(r.get("GAME_ID", "")).strip()
        if game_id not in valid_game_ids:
            continue
        game_date = parse_game_date(r.get("GAME_DATE"))
        if not game_date:
            continue
        player_id = r.get("PLAYER_ID")
        team_id = r.get("TEAM_ID")
        player_name = str(r.get("PLAYER_NAME", ""))
        if not player_id or not team_id:
            continue
        rows.append((
            int(player_id), player_name, int(team_id),
            game_id, game_date, season_year,
            parse_minutes(r.get("MIN")),
            safe_float(r.get("PTS")),
            safe_float(r.get("REB")),
            safe_float(r.get("AST")),
        ))
    return rows


UPSERT_SQL = """
INSERT INTO player_game_stats
    (player_id, player_name, team_id, game_id, game_date, season_year, min_seconds, pts, reb, ast)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(player_id, game_id) DO UPDATE SET
    min_seconds = excluded.min_seconds,
    pts = excluded.pts,
    reb = excluded.reb,
    ast = excluded.ast
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape NBA player game logs into SQLite.")
    parser.add_argument("--season", type=int, help="Single season year to fetch (e.g. 2011).")
    parser.add_argument("--force", action="store_true", help="Re-fetch even if already fetched.")
    args = parser.parse_args()

    seasons = [args.season] if args.season else list(range(DEFAULT_START_YEAR, DEFAULT_END_YEAR + 1))

    print("=== NBA Player Game Log Scraper ===")
    conn = connect_db()
    total = 0
    try:
        fetched = load_fetched_seasons(conn)
        valid_ids = load_valid_game_ids(conn)
        for yr in seasons:
            if yr in fetched and not args.force:
                print(f"[{yr}] Skipped (already fetched)")
                continue
            if args.force:
                delete_season(conn, yr)
            print(f"[{yr}] Fetching {season_label(yr)} ...", flush=True)
            df = fetch_player_log(yr)
            rows = build_rows(df, yr, valid_ids)
            if rows:
                conn.executemany(UPSERT_SQL, rows)
            conn.execute("INSERT OR IGNORE INTO player_seasons_fetched (season_year) VALUES (?)", (yr,))
            conn.commit()
            total += len(rows)
            print(f"[{yr}] {len(rows):,} player-game rows")
    finally:
        conn.close()
    print(f"Done. Total {total:,} rows.")


if __name__ == "__main__":
    main()
