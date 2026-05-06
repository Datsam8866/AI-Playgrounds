import argparse
import re
import sqlite3
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup


BASE_URL = "https://npb.jp"
DB_PATH = Path(__file__).resolve().parent / "npb.sqlite"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

CREATE_SCHEDULE_TABLE = """
CREATE TABLE IF NOT EXISTS team_game_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    season_year INTEGER,
    game_date TEXT,
    home_code TEXT,
    away_code TEXT,
    home_score INTEGER,
    away_score INTEGER,
    home_win INTEGER,
    league_code TEXT,
    stadium TEXT,
    win_pitcher TEXT,
    lose_pitcher TEXT,
    game_url TEXT UNIQUE,
    status TEXT
);
"""

CREATE_SP_TABLE = """
CREATE TABLE IF NOT EXISTS game_starting_pitchers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    season_year INTEGER,
    game_url TEXT,
    team_code TEXT,
    pitcher_name TEXT,
    ip_outs INTEGER,
    hits INTEGER,
    hr INTEGER,
    bb INTEGER,
    hbp INTEGER,
    strikeouts INTEGER,
    runs INTEGER,
    earned_runs INTEGER,
    pitches INTEGER,
    batters_faced INTEGER,
    result TEXT,
    UNIQUE(game_url, team_code)
);
"""


def clean_text(value):
    if value is None:
        return ""
    return re.sub(r"\s+", " ", value.replace("\u3000", " ")).strip()


def init_db(connection):
    connection.execute(CREATE_SCHEDULE_TABLE)
    connection.execute(CREATE_SP_TABLE)
    connection.commit()


def make_session():
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    return session


def get_pending_games(connection, year=None):
    year_clause = "AND tgr.season_year = ?" if year else ""
    params = (year,) if year else ()
    return connection.execute(
        f"""
        SELECT tgr.season_year, tgr.game_url
        FROM team_game_results tgr
        WHERE tgr.status = 'completed'
          AND tgr.game_url IS NOT NULL
          {year_clause}
          AND NOT EXISTS (
              SELECT 1
              FROM game_starting_pitchers gsp
              WHERE gsp.game_url = tgr.game_url
          )
        ORDER BY tgr.season_year, tgr.game_url
        """,
        params,
    ).fetchall()


def fetch_boxscore(session, game_url, timeout):
    url = f"{BASE_URL}/scores/{game_url}/box.html"
    response = session.get(url, timeout=timeout)
    if response.status_code == 404:
        return None
    response.raise_for_status()
    response.encoding = response.apparent_encoding or "utf-8"
    return response.text


def direct_cells(row):
    return [clean_text(cell.get_text("|", strip=True)) for cell in row.find_all(["th", "td"], recursive=False)]


def is_pitching_table(table):
    header = table.find("thead")
    if not header:
        return False
    first_row = header.find("tr")
    if not first_row:
        return False
    cells = direct_cells(first_row)
    return len(cells) >= 2 and cells[1] == "投手"


def normalize_result(value):
    value = clean_text(value)
    if value == "○":
        return "W"
    if value == "●":
        return "L"
    if value in {"H", "S"}:
        return value
    return None


def parse_int(value):
    value = clean_text(value)
    if value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def ip_to_outs(value):
    value = clean_text(value).replace("|", "").rstrip("+")
    if not value:
        return None

    if value.startswith("."):
        whole = 0
        fraction = value[1:]
    elif "." in value:
        whole_text, fraction = value.split(".", 1)
        whole = int(whole_text or 0)
    else:
        whole = int(value)
        fraction = ""

    outs = whole * 3
    if fraction:
        outs += int(fraction)
    return outs


def parse_pitcher_row(row):
    cells = direct_cells(row)
    if len(cells) < 14:
        return None
    if cells[1] in {"チーム計", "計"}:
        return None

    return {
        "result": normalize_result(cells[0]),
        "pitcher_name": cells[1] or None,
        "pitches": parse_int(cells[2]),
        "batters_faced": parse_int(cells[3]),
        "ip_outs": ip_to_outs(cells[4]),
        "hits": parse_int(cells[5]),
        "hr": parse_int(cells[6]),
        "bb": parse_int(cells[7]),
        "hbp": parse_int(cells[8]),
        "strikeouts": parse_int(cells[9]),
        "runs": parse_int(cells[12]),
        "earned_runs": parse_int(cells[13]),
    }


def parse_starting_pitchers(html):
    soup = BeautifulSoup(html, "html.parser")
    pitching_tables = [table for table in soup.select("table") if is_pitching_table(table)]
    if len(pitching_tables) < 2:
        raise ValueError(f"Expected at least 2 pitching tables, found {len(pitching_tables)}")

    starters = []
    for side, table in zip(("away", "home"), pitching_tables[:2]):
        body = table.find("tbody")
        if not body:
            raise ValueError(f"Missing pitching table body for {side}")

        starter = None
        for row in body.find_all("tr", recursive=False):
            starter = parse_pitcher_row(row)
            if starter:
                break

        if not starter:
            raise ValueError(f"Missing starter row for {side}")

        starter["team_code"] = side
        starters.append(starter)

    return starters


def insert_starter(connection, season_year, game_url, starter):
    connection.execute(
        """
        INSERT OR IGNORE INTO game_starting_pitchers (
            season_year, game_url, team_code, pitcher_name, ip_outs, hits, hr, bb,
            hbp, strikeouts, runs, earned_runs, pitches, batters_faced, result
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            season_year,
            game_url,
            starter["team_code"],
            starter["pitcher_name"],
            starter["ip_outs"],
            starter["hits"],
            starter["hr"],
            starter["bb"],
            starter["hbp"],
            starter["strikeouts"],
            starter["runs"],
            starter["earned_runs"],
            starter["pitches"],
            starter["batters_faced"],
            starter["result"],
        ),
    )


def scrape_game(connection, session, season_year, game_url, sleep_seconds, timeout):
    html = fetch_boxscore(session, game_url, timeout=timeout)
    if sleep_seconds > 0:
        time.sleep(sleep_seconds)

    if html is None:
        print(f"game_url={game_url} status=not_found inserted=0")
        return 0

    try:
        starters = parse_starting_pitchers(html)
    except Exception as exc:
        print(f"warning game_url={game_url} error={type(exc).__name__}: {exc}")
        return 0

    inserted = 0
    for starter in starters:
        before = connection.total_changes
        insert_starter(connection, season_year, game_url, starter)
        inserted += connection.total_changes - before

    connection.commit()
    print(f"game_url={game_url} status=ok inserted={inserted}")
    return inserted


def parse_args():
    parser = argparse.ArgumentParser(description="Scrape NPB box score starting pitchers into SQLite.")
    parser.add_argument("--year", type=int, help="Only process one season year.")
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--sleep-seconds", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=30.0)
    return parser.parse_args()


def main():
    args = parse_args()
    connection = sqlite3.connect(args.db)
    init_db(connection)
    session = make_session()

    total_inserted = 0
    try:
        pending = get_pending_games(connection, year=args.year)
        print(f"pending_games={len(pending)}")
        for season_year, game_url in pending:
            total_inserted += scrape_game(
                connection=connection,
                session=session,
                season_year=season_year,
                game_url=game_url,
                sleep_seconds=args.sleep_seconds,
                timeout=args.timeout,
            )
    finally:
        session.close()
        connection.close()

    print(f"saved_db={args.db.resolve()}")
    print(f"total_sp_inserted={total_inserted}")


if __name__ == "__main__":
    main()
