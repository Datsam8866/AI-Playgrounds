import argparse
import json
import re
import sqlite3
import time
from datetime import datetime
from pathlib import Path

import requests


BASE_URL = "https://www.cpbl.com.tw"
DB_PATH = Path("cpbl.sqlite")
SCHEMA_PATH = Path("cpbl_sqlite_schema.sql")
TOKEN_RE = re.compile(r'name="__RequestVerificationToken" type="hidden" value="([^"]+)"')


def now_text():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def initialize_database(connection):
    connection.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))


def fetch_token(session, year, kind_code):
    response = session.get(
        f"{BASE_URL}/box",
        params={"GameSno": 1, "KindCode": kind_code, "Year": year},
        timeout=30,
    )
    response.raise_for_status()
    match = TOKEN_RE.search(response.text)
    if not match:
        raise RuntimeError(f"Unable to find CSRF token for year={year}, kind_code={kind_code}.")
    return match.group(1)


def fetch_game_payload(session, token, year, game_sno, kind_code):
    response = session.post(
        f"{BASE_URL}/box/getlive",
        data={
            "__RequestVerificationToken": token,
            "GameSno": str(game_sno),
            "KindCode": kind_code,
            "Year": str(year),
            "PrevOrNext": "",
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
    raw = payload.get("GameDetailJson")
    curt_raw = payload.get("CurtGameDetailJson")
    if not raw and not curt_raw:
        return None

    detail = None
    if raw:
        try:
            rows = json.loads(raw)
            if rows:
                detail = rows[0]
        except Exception:
            detail = None

    curt_detail = None
    if curt_raw:
        try:
            curt_detail = json.loads(curt_raw)
        except Exception:
            curt_detail = None

    if detail and curt_detail:
        merged = dict(detail)
        # CurtGameDetailJson is the live box summary and proved more reliable for
        # current matchup / score fields than GameDetailJson on some 2026 games.
        merged.update({k: v for k, v in curt_detail.items() if v not in (None, "")})
        return merged
    if curt_detail:
        return curt_detail
    return detail


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


def scrape_year(connection, session, year, kind_code, max_game_sno, sleep_seconds, max_consecutive_miss):
    token = fetch_token(session, year, kind_code)
    hits = 0
    completed = 0
    misses = 0
    last_seen_sno = 0

    for game_sno in range(1, max_game_sno + 1):
        payload = fetch_game_payload(session, token, year, game_sno, kind_code)
        if not payload:
            misses += 1
            if hits > 0 and misses >= max_consecutive_miss:
                break
            continue

        misses = 0
        hits += 1
        last_seen_sno = game_sno
        upsert_game_result(connection, year, kind_code, game_sno, payload)
        if int(payload.get("GameStatus") or 0) == 3:
            completed += 1

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    connection.commit()
    return {
        "year": year,
        "hits": hits,
        "completed": completed,
        "last_seen_sno": last_seen_sno,
    }


def main():
    parser = argparse.ArgumentParser(description="Scrape CPBL box score totals into SQLite.")
    parser.add_argument("--start-year", type=int, default=2011)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--kind-code", default="A")
    parser.add_argument("--max-game-sno", type=int, default=450)
    parser.add_argument("--sleep-seconds", type=float, default=0.05)
    parser.add_argument("--max-consecutive-miss", type=int, default=40)
    parser.add_argument(
        "--refresh-range",
        action="store_true",
        help="Delete existing rows for the selected year range before scraping.",
    )
    args = parser.parse_args()

    connection = sqlite3.connect(DB_PATH)
    connection.execute("PRAGMA foreign_keys = ON")
    initialize_database(connection)

    if args.refresh_range:
        connection.execute(
            "DELETE FROM team_game_results WHERE season_year BETWEEN ? AND ? AND kind_code = ?",
            (args.start_year, args.end_year, args.kind_code),
        )
        connection.commit()

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) CPBL-Boxscore-Scraper/1.0",
            "Referer": f"{BASE_URL}/box",
        }
    )

    summaries = []
    try:
        for year in range(args.start_year, args.end_year + 1):
            summary = scrape_year(
                connection=connection,
                session=session,
                year=year,
                kind_code=args.kind_code,
                max_game_sno=args.max_game_sno,
                sleep_seconds=args.sleep_seconds,
                max_consecutive_miss=args.max_consecutive_miss,
            )
            summaries.append(summary)
            print(
                f"year={year} hits={summary['hits']} completed={summary['completed']} "
                f"last_seen_sno={summary['last_seen_sno']}"
            )
    finally:
        session.close()
        connection.close()

    total_completed = sum(item["completed"] for item in summaries)
    print(f"saved_db={DB_PATH.resolve()}")
    print(f"years={len(summaries)}")
    print(f"completed_games={total_completed}")


if __name__ == "__main__":
    main()
