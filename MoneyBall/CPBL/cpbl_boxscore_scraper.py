import argparse
import json
import re
import sqlite3
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import requests


BASE_URL = "https://www.cpbl.com.tw"
DB_PATH = Path("cpbl.sqlite")
SCHEMA_PATH = Path("cpbl_sqlite_schema.sql")
TOKEN_RE = re.compile(r'name="__RequestVerificationToken" type="hidden" value="([^"]+)"')

SEASON_START_MONTH = 3
SEASON_END_MONTH = 10


def now_text():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def initialize_database(connection):
    connection.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))


def fetch_token(session):
    response = session.get(BASE_URL + "/", timeout=30)
    response.raise_for_status()
    match = TOKEN_RE.search(response.text)
    if not match:
        raise RuntimeError("Unable to find CSRF token on CPBL homepage.")
    return match.group(1)


def fetch_game_list(session, token, game_date_str, kind_code):
    """POST /home/getdetaillist → list of game dicts for that date, or None on failure."""
    response = session.post(
        f"{BASE_URL}/home/getdetaillist",
        data={
            "__RequestVerificationToken": token,
            "GameSno": "",
            "KindCode": kind_code,
            "GameDate": game_date_str,
        },
        timeout=30,
        allow_redirects=False,
    )
    if response.status_code != 200:
        return None
    try:
        payload = response.json()
    except Exception:
        return None
    if not payload.get("Success"):
        return None
    raw = payload.get("GameADetailJson")
    if not raw:
        return []
    try:
        return json.loads(raw)
    except Exception:
        return None


def fetch_game_detail(session, token, year, game_sno, kind_code, game_status):
    """POST /home/gamedetail → CurtGameDetailJson dict, or None on failure."""
    response = session.post(
        f"{BASE_URL}/home/gamedetail",
        data={
            "__RequestVerificationToken": token,
            "GameSno": str(game_sno),
            "Year": str(year),
            "KindCode": kind_code,
            "GameStatus": str(game_status),
        },
        timeout=30,
        allow_redirects=False,
    )
    if response.status_code != 200:
        return None
    try:
        payload = response.json()
    except Exception:
        return None
    if not payload.get("Success"):
        return None
    raw = payload.get("CurtGameDetailJson")
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None


def upsert_game_result(connection, year, kind_code, game_sno, payload):
    connection.execute(
        """
        INSERT INTO team_game_results(
            season_year, kind_code, game_sno, game_date, game_status, game_status_text,
            visiting_team_code, home_team_code, visiting_score, home_score, raw_json, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(season_year, kind_code, game_sno) DO UPDATE SET
            game_date=excluded.game_date,
            game_status=excluded.game_status,
            game_status_text=excluded.game_status_text,
            visiting_team_code=excluded.visiting_team_code,
            home_team_code=excluded.home_team_code,
            visiting_score=excluded.visiting_score,
            home_score=excluded.home_score,
            raw_json=excluded.raw_json,
            created_at=excluded.created_at
        """,
        (
            year,
            kind_code,
            game_sno,
            payload.get("GameDate"),
            payload.get("GameStatus"),
            payload.get("GameStatusChi"),
            payload.get("VisitingTeamCode") or "",
            payload.get("HomeTeamCode") or "",
            payload.get("VisitingTotalScore"),
            payload.get("HomeTotalScore"),
            json.dumps(payload, ensure_ascii=False),
            now_text(),
        ),
    )


def date_already_done(connection, year, kind_code, d):
    """True if d is a past date that already has at least one completed game in DB."""
    if d >= date.today():
        return False
    prefix = d.strftime("%Y-%m-%d")
    row = connection.execute(
        "SELECT COUNT(*) FROM team_game_results "
        "WHERE season_year=? AND kind_code=? AND game_date LIKE ? AND game_status=3",
        (year, kind_code, prefix + "%"),
    ).fetchone()
    return row[0] > 0


def season_dates(year):
    """Yield each calendar date in the CPBL season (March–October), up to today."""
    start = date(year, SEASON_START_MONTH, 1)
    end = min(date(year, SEASON_END_MONTH, 31), date.today())
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def scrape_year(connection, session, year, kind_code, sleep_seconds, refresh):
    token = fetch_token(session)
    hits = 0
    completed = 0

    for d in season_dates(year):
        if not refresh and date_already_done(connection, year, kind_code, d):
            continue

        date_str = d.strftime("%Y/%m/%d")
        games = fetch_game_list(session, token, date_str, kind_code)
        if games is None:
            continue
        if not games:
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
            continue

        for game in games:
            game_sno = game.get("GameSno")
            if not game_sno:
                continue

            game_status = int(game.get("GameStatus") or 0)
            hits += 1

            merged = dict(game)
            if game_status in (2, 3, 8):
                detail = fetch_game_detail(session, token, year, game_sno, kind_code, game_status)
                if detail:
                    merged.update({k: v for k, v in detail.items() if v not in (None, "")})
                completed += 1
                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)

            upsert_game_result(connection, year, kind_code, game_sno, merged)

        connection.commit()
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    return {"year": year, "hits": hits, "completed": completed}


def main():
    parser = argparse.ArgumentParser(description="Scrape CPBL box score totals into SQLite.")
    parser.add_argument("--start-year", type=int, default=2011)
    parser.add_argument("--end-year", type=int, default=datetime.now().year)
    parser.add_argument("--kind-code", default="A")
    parser.add_argument("--sleep-seconds", type=float, default=0.1)
    parser.add_argument(
        "--refresh-range",
        action="store_true",
        help="Re-scrape already-completed dates (upsert, does NOT delete).",
    )
    args = parser.parse_args()

    connection = sqlite3.connect(DB_PATH)
    connection.execute("PRAGMA foreign_keys = ON")
    initialize_database(connection)

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Referer": BASE_URL + "/",
    })

    summaries = []
    try:
        for year in range(args.start_year, args.end_year + 1):
            summary = scrape_year(
                connection=connection,
                session=session,
                year=year,
                kind_code=args.kind_code,
                sleep_seconds=args.sleep_seconds,
                refresh=args.refresh_range,
            )
            summaries.append(summary)
            print(f"year={year} hits={summary['hits']} completed={summary['completed']}")
    finally:
        session.close()
        connection.close()

    total_completed = sum(s["completed"] for s in summaries)
    print(f"saved_db={DB_PATH.resolve()}")
    print(f"years={len(summaries)}")
    print(f"completed_games={total_completed}")


if __name__ == "__main__":
    main()
