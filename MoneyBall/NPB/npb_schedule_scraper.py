import argparse
import re
import sqlite3
import time
from datetime import date
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup


BASE_URL = "https://npb.jp"
DB_PATH = Path(__file__).resolve().parent / "npb.sqlite"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
MONTHS = range(3, 11)

CL_TEAMS = {"g", "d", "db", "t", "c", "s"}
PL_TEAMS = {"h", "f", "b", "e", "l", "m"}

TEAM_NAME_TO_CODE = {
    "巨人": "g",
    "読売": "g",
    "読売ジャイアンツ": "g",
    "中日": "d",
    "中日ドラゴンズ": "d",
    "DeNA": "db",
    "横浜DeNA": "db",
    "横浜DeNAベイスターズ": "db",
    "阪神": "t",
    "阪神タイガース": "t",
    "広島": "c",
    "広島東洋": "c",
    "広島東洋カープ": "c",
    "ヤクルト": "s",
    "東京ヤクルト": "s",
    "東京ヤクルトスワローズ": "s",
    "ソフトバンク": "h",
    "福岡ソフトバンク": "h",
    "福岡ソフトバンクホークス": "h",
    "日本ハム": "f",
    "北海道日本ハム": "f",
    "北海道日本ハムファイターズ": "f",
    "オリックス": "b",
    "オリックス・バファローズ": "b",
    "楽天": "e",
    "東北楽天": "e",
    "東北楽天ゴールデンイーグルス": "e",
    "西武": "l",
    "埼玉西武": "l",
    "埼玉西武ライオンズ": "l",
    "ロッテ": "m",
    "千葉ロッテ": "m",
    "千葉ロッテマリーンズ": "m",
}

SCORE_URL_RE = re.compile(r"/scores/(\d{4}/\d{4}/([a-z]+)-([a-z]+)-\d+)/?")

CREATE_TABLE = """
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


def clean_text(value):
    if value is None:
        return ""
    return re.sub(r"\s+", " ", value.replace("\u3000", " ")).strip()


def init_db(connection):
    connection.execute(CREATE_TABLE)
    connection.commit()


def make_session():
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    return session


def fetch_page(session, url, timeout):
    response = session.get(url, timeout=timeout)
    if response.status_code == 404:
        return None
    response.raise_for_status()
    response.encoding = response.apparent_encoding or "utf-8"
    return response.text


def extract_game_url_and_codes(row):
    link = row.select_one('a[href*="/scores/"]')
    if not link:
        return None, None, None

    href = link.get("href") or ""
    match = SCORE_URL_RE.search(href)
    if not match:
        return None, None, None
    return match.group(1), match.group(2), match.group(3)


def infer_team_code(team_name):
    normalized = clean_text(team_name)
    return TEAM_NAME_TO_CODE.get(normalized)


def league_code_for(home_code, away_code):
    if home_code in CL_TEAMS and away_code in CL_TEAMS:
        return "CL"
    if home_code in PL_TEAMS and away_code in PL_TEAMS:
        return "PL"
    if (home_code in CL_TEAMS and away_code in PL_TEAMS) or (
        home_code in PL_TEAMS and away_code in CL_TEAMS
    ):
        return "IL"
    return None


def parse_pitcher_credits(row):
    win_pitcher = None
    lose_pitcher = None
    for pit in row.select("td div.pit"):
        text = clean_text(pit.get_text(" ", strip=True))
        if text.startswith("勝："):
            win_pitcher = clean_text(text.split("：", 1)[1])
        elif text.startswith("敗："):
            lose_pitcher = clean_text(text.split("：", 1)[1])
    return win_pitcher or None, lose_pitcher or None


def game_date_from_row(year, row):
    row_id = row.get("id") or ""
    match = re.search(r"date(\d{2})(\d{2})", row_id)
    if not match:
        return None
    month = int(match.group(1))
    day = int(match.group(2))
    return date(year, month, day).isoformat()


def parse_schedule_row(year, row):
    game_date = game_date_from_row(year, row)
    if not game_date:
        return None

    team1 = row.select_one("td div.team1")
    team2 = row.select_one("td div.team2")
    if not team1 or not team2:
        return None

    game_url, link_home_code, link_away_code = extract_game_url_and_codes(row)
    home_code = link_home_code or infer_team_code(team1.get_text(" ", strip=True))
    away_code = link_away_code or infer_team_code(team2.get_text(" ", strip=True))
    if not home_code or not away_code:
        return None

    score1 = clean_text(row.select_one("td div.score1").get_text(" ", strip=True)) if row.select_one("td div.score1") else ""
    score2 = clean_text(row.select_one("td div.score2").get_text(" ", strip=True)) if row.select_one("td div.score2") else ""
    cancelled = row.select_one("td div.cancel") is not None or "中止" in row.get_text(" ", strip=True)

    home_score = int(score1) if score1.isdigit() else None
    away_score = int(score2) if score2.isdigit() else None

    if cancelled:
        status = "cancelled"
    elif home_score is not None and away_score is not None:
        status = "completed"
    else:
        status = "scheduled"

    if status == "completed" and home_score != away_score:
        home_win = 1 if home_score > away_score else 0
    else:
        home_win = None

    stadium_node = row.select_one("td div.place")
    stadium = clean_text(stadium_node.get_text(" ", strip=True)) if stadium_node else None
    win_pitcher, lose_pitcher = parse_pitcher_credits(row)

    return {
        "season_year": year,
        "game_date": game_date,
        "home_code": home_code,
        "away_code": away_code,
        "home_score": home_score,
        "away_score": away_score,
        "home_win": home_win,
        "league_code": league_code_for(home_code, away_code),
        "stadium": stadium,
        "win_pitcher": win_pitcher,
        "lose_pitcher": lose_pitcher,
        "game_url": game_url,
        "status": status,
    }


def existing_game_url(connection, game_url):
    if not game_url:
        return False
    row = connection.execute(
        "SELECT 1 FROM team_game_results WHERE game_url = ? LIMIT 1", (game_url,)
    ).fetchone()
    return row is not None


def existing_scheduled_game(connection, game):
    row = connection.execute(
        """
        SELECT 1
        FROM team_game_results
        WHERE game_url IS NULL
          AND season_year = ?
          AND game_date = ?
          AND home_code = ?
          AND away_code = ?
          AND COALESCE(stadium, '') = COALESCE(?, '')
        LIMIT 1
        """,
        (
            game["season_year"],
            game["game_date"],
            game["home_code"],
            game["away_code"],
            game["stadium"],
        ),
    ).fetchone()
    return row is not None


def find_scheduled_placeholder(connection, game):
    row = connection.execute(
        """
        SELECT id
        FROM team_game_results
        WHERE game_url IS NULL
          AND status = 'scheduled'
          AND season_year = ?
          AND game_date = ?
          AND home_code = ?
          AND away_code = ?
          AND COALESCE(stadium, '') = COALESCE(?, '')
        LIMIT 1
        """,
        (
            game["season_year"],
            game["game_date"],
            game["home_code"],
            game["away_code"],
            game["stadium"],
        ),
    ).fetchone()
    return row[0] if row else None


def insert_game(connection, game):
    connection.execute(
        """
        INSERT INTO team_game_results (
            season_year, game_date, home_code, away_code, home_score, away_score,
            home_win, league_code, stadium, win_pitcher, lose_pitcher, game_url, status
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            game["season_year"],
            game["game_date"],
            game["home_code"],
            game["away_code"],
            game["home_score"],
            game["away_score"],
            game["home_win"],
            game["league_code"],
            game["stadium"],
            game["win_pitcher"],
            game["lose_pitcher"],
            game["game_url"],
            game["status"],
        ),
    )


def update_game(connection, row_id, game):
    connection.execute(
        """
        UPDATE team_game_results
        SET home_score = ?,
            away_score = ?,
            home_win = ?,
            league_code = ?,
            win_pitcher = ?,
            lose_pitcher = ?,
            game_url = ?,
            status = ?
        WHERE id = ?
        """,
        (
            game["home_score"],
            game["away_score"],
            game["home_win"],
            game["league_code"],
            game["win_pitcher"],
            game["lose_pitcher"],
            game["game_url"],
            game["status"],
            row_id,
        ),
    )


def scrape_month(connection, session, year, month, sleep_seconds, timeout):
    url = f"{BASE_URL}/games/{year}/schedule_{month:02d}_detail.html"
    html = fetch_page(session, url, timeout=timeout)
    if sleep_seconds > 0:
        time.sleep(sleep_seconds)

    if html is None:
        print(f"year={year} month={month:02d} status=not_found inserted=0 updated=0 skipped=0")
        return {"inserted": 0, "updated": 0, "skipped": 0, "parsed": 0, "not_found": 1}

    soup = BeautifulSoup(html, "html.parser")
    rows = soup.select("table tr")
    inserted = 0
    updated = 0
    skipped = 0
    parsed = 0

    for row in rows:
        game = parse_schedule_row(year, row)
        if not game:
            continue

        parsed += 1
        if game["game_url"]:
            if existing_game_url(connection, game["game_url"]):
                skipped += 1
                continue
            placeholder_id = find_scheduled_placeholder(connection, game)
            if placeholder_id:
                update_game(connection, placeholder_id, game)
                updated += 1
                continue
        elif existing_scheduled_game(connection, game):
            skipped += 1
            continue

        insert_game(connection, game)
        inserted += 1

    connection.commit()
    print(
        f"year={year} month={month:02d} status=ok parsed={parsed} "
        f"inserted={inserted} updated={updated} skipped={skipped}"
    )
    return {
        "inserted": inserted,
        "updated": updated,
        "skipped": skipped,
        "parsed": parsed,
        "not_found": 0,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Scrape NPB monthly schedules into SQLite.")
    parser.add_argument("--year", type=int, help="Scrape a single season year.")
    parser.add_argument("--start-year", type=int, default=2015)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--sleep-seconds", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=30.0)
    return parser.parse_args()


def main():
    args = parse_args()
    start_year = args.year if args.year else args.start_year
    end_year = args.year if args.year else args.end_year

    connection = sqlite3.connect(args.db)
    init_db(connection)
    session = make_session()

    totals = {"inserted": 0, "updated": 0, "skipped": 0, "parsed": 0, "not_found": 0}
    try:
        for year in range(start_year, end_year + 1):
            for month in MONTHS:
                summary = scrape_month(
                    connection=connection,
                    session=session,
                    year=year,
                    month=month,
                    sleep_seconds=args.sleep_seconds,
                    timeout=args.timeout,
                )
                for key in totals:
                    totals[key] += summary[key]
    finally:
        session.close()
        connection.close()

    print(f"saved_db={args.db.resolve()}")
    print(
        f"total_parsed={totals['parsed']} total_inserted={totals['inserted']} "
        f"total_updated={totals['updated']} total_skipped={totals['skipped']} "
        f"pages_not_found={totals['not_found']}"
    )


if __name__ == "__main__":
    main()
