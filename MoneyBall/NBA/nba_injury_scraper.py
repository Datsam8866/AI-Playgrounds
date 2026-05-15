import argparse
import json
import sqlite3
import time
from datetime import date
from pathlib import Path
from urllib.request import urlopen


DB_PATH = Path(__file__).resolve().parent / "nba.sqlite"
TEAMS_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams"
INJURIES_URL_TEMPLATE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{espn_id}/injuries"
RATE_LIMIT_SECONDS = 0.5

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS player_injuries (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    player_name  TEXT NOT NULL,
    team_abbr    TEXT NOT NULL,
    status       TEXT NOT NULL,
    scraped_date TEXT NOT NULL,
    UNIQUE(player_name, team_abbr, scraped_date)
);
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch NBA injuries from ESPN and store them in nba.sqlite.")
    parser.add_argument(
        "--date",
        default=date.today().isoformat(),
        help="Scraped date YYYY-MM-DD (default: today).",
    )
    return parser.parse_args()


def connect_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    return conn


def fetch_json(url: str) -> dict:
    with urlopen(url) as response:
        payload = json.loads(response.read().decode("utf-8"))
    time.sleep(RATE_LIMIT_SECONDS)
    return payload


def fetch_teams() -> list[tuple[str, str]]:
    payload = fetch_json(TEAMS_URL)
    teams = payload["sports"][0]["leagues"][0]["teams"]
    return [
        (str(item["team"]["id"]), str(item["team"]["abbreviation"]))
        for item in teams
    ]


def fetch_and_store_injuries(conn, scraped_date: str) -> dict[str, list[str]]:
    """抓取並存入 DB。回傳 {team_abbr: [player_name, ...]} 僅含 out/doubtful 球員。"""
    conn.executescript(SCHEMA_SQL)
    injury_map: dict[str, list[str]] = {}

    for espn_id, abbr in fetch_teams():
        payload = fetch_json(INJURIES_URL_TEMPLATE.format(espn_id=espn_id))
        injuries = payload.get("injuries")
        if not injuries:
            print(f"No injuries reported for {abbr}")
            continue

        for item in injuries:
            athlete = item.get("athlete") or {}
            injury_type = item.get("type") or {}
            player_name = athlete.get("displayName")
            status = str(injury_type.get("name") or "").lower()
            if not player_name or not status:
                continue
            conn.execute(
                """
                INSERT INTO player_injuries (player_name, team_abbr, status, scraped_date)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(player_name, team_abbr, scraped_date) DO UPDATE SET
                    status = excluded.status
                """,
                (str(player_name), abbr, status, scraped_date),
            )
            if status in {"out", "doubtful"}:
                injury_map.setdefault(abbr, []).append(str(player_name))

    conn.commit()
    return injury_map


def main() -> None:
    args = parse_args()
    conn = connect_db()
    try:
        injury_map = fetch_and_store_injuries(conn, args.date)
    finally:
        conn.close()

    tracked_players = sum(len(players) for players in injury_map.values())
    print(f"Stored injuries for {args.date}")
    print(f"Teams with out/doubtful players: {len(injury_map)}")
    print(f"Total out/doubtful players: {tracked_players}")


if __name__ == "__main__":
    main()
