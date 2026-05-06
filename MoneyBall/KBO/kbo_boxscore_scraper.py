"""
KBO box score scraper — writes to kbo.sqlite.

Strategy:
  1. GetMonthSchedule  → all game IDs for the month
  2. GetScoreBoardScroll → game result, teams, score, crowd
  3. GetBoxScoreScroll   → pitcher stats (SP = first pitcher per team)

Usage:
  python kbo_boxscore_scraper.py --start-year 2011 --end-year 2026
  python kbo_boxscore_scraper.py --start-year 2026 --end-year 2026 --refresh-range
"""

import argparse
import json
import re
import sqlite3
import time
from datetime import datetime
from pathlib import Path

import requests

BASE_URL = "https://www.koreabaseball.com"
DB_PATH = Path("kbo.sqlite")

# Regular season = 0; postseason = 1,3,4,5
SR_IDS_ALL = "0,1,3,4,5,7,9"

GAME_ID_RE = re.compile(r"gameId=(\d{8}[A-Z]{2,6}\d)")

PITCHER_COL_MAP = {
    "선수명": "player_name",
    "이닝":   "ip_raw",       # "6", "2 2/3", "1/3"
    "타자":   "tbf",
    "피안타": "hits",
    "홈런":   "hr",
    "4사구":  "bb",           # BB + HBP combined
    "삼진":   "strikeouts",
    "실점":   "runs",
    "자책":   "er",
    "결과":   "result_code",
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS team_game_results (
    game_id     TEXT PRIMARY KEY,
    season_year INTEGER NOT NULL,
    sr_id       INTEGER NOT NULL,
    game_date   TEXT NOT NULL,
    away_code   TEXT NOT NULL,
    home_code   TEXT NOT NULL,
    away_score  INTEGER,
    home_score  INTEGER,
    game_state  INTEGER,
    stadium     TEXT,
    crowd       TEXT,
    start_time  TEXT,
    end_time    TEXT,
    use_time    TEXT,
    created_at  TEXT
);

CREATE INDEX IF NOT EXISTS idx_tgr_date ON team_game_results(game_date);
CREATE INDEX IF NOT EXISTS idx_tgr_year ON team_game_results(season_year, sr_id);

CREATE TABLE IF NOT EXISTS game_starting_pitchers (
    game_id     TEXT NOT NULL,
    side        TEXT NOT NULL,
    team_code   TEXT NOT NULL,
    player_name TEXT,
    ip_raw      TEXT,
    ip_numeric  REAL,
    tbf         INTEGER,
    hits        INTEGER,
    bb          INTEGER,
    hbp         INTEGER,
    strikeouts  INTEGER,
    hr          INTEGER,
    runs        INTEGER,
    er          INTEGER,
    result_code TEXT,
    season_year INTEGER,
    created_at  TEXT,
    PRIMARY KEY (game_id, side)
);
"""


def now_text():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def init_db(conn):
    conn.executescript(SCHEMA)
    conn.commit()


def ip_to_numeric(ip_str):
    """
    KBO IP format examples: '6', '2 2/3', '1/3', '2/3'
    '6'     → 6.0
    '2 2/3' → 2.667
    '1/3'   → 0.333
    '2/3'   → 0.667
    """
    if not ip_str:
        return None
    s = str(ip_str).strip()
    try:
        if " " in s:
            # e.g. "2 2/3"
            whole_str, frac_str = s.split(" ", 1)
            num, den = frac_str.split("/")
            return int(whole_str) + int(num) / int(den)
        elif "/" in s:
            # e.g. "1/3" or "2/3"
            num, den = s.split("/")
            return int(num) / int(den)
        return float(s)
    except (ValueError, ZeroDivisionError):
        return None


def safe_int(val):
    try:
        v = str(val).strip()
        return int(v) if v not in ("", "-", None) else None
    except (ValueError, TypeError):
        return None


def parse_table(json_str):
    """Parse KBO table JSON → (headers: list[str], rows: list[list[str]])"""
    data = json.loads(json_str)
    headers = []
    for h_row in data.get("headers", []):
        for cell in h_row.get("row", []):
            headers.append(cell.get("Text", "").strip())

    rows = []
    for r in data.get("rows", []):
        cells = [c.get("Text", "").strip() for c in r.get("row", [])]
        rows.append(cells)
    return headers, rows


def extract_sp_row(arr_pitcher_entry):
    """Return (headers, first_pitcher_row) or (None, None) if unavailable."""
    table_str = arr_pitcher_entry.get("table")
    if not table_str:
        return None, None
    try:
        headers, rows = parse_table(table_str)
        if rows:
            return headers, rows[0]
    except Exception:
        pass
    return None, None


def build_pitcher_record(game_id, side, team_code, headers, row, season_year):
    record = {
        "game_id":     game_id,
        "side":        side,
        "team_code":   team_code,
        "player_name": None,
        "ip_raw":      None,
        "ip_numeric":  None,
        "tbf":         None,
        "hits":        None,
        "bb":          None,
        "hbp":         None,
        "strikeouts":  None,
        "hr":          None,
        "runs":        None,
        "er":          None,
        "result_code": None,
        "season_year": season_year,
        "created_at":  now_text(),
    }
    if not headers or not row:
        return record

    for i, col_name in enumerate(headers):
        if i >= len(row):
            break
        val = row[i].strip()
        mapped = PITCHER_COL_MAP.get(col_name)
        if not mapped:
            continue
        if mapped == "player_name":
            record["player_name"] = val or None
        elif mapped == "ip_raw":
            record["ip_raw"] = val or None
            record["ip_numeric"] = ip_to_numeric(val)
        elif mapped == "result_code":
            record["result_code"] = val or None
        else:
            record[mapped] = safe_int(val)
    return record


def get_month_game_ids(session, year, month):
    """Return sorted list of game IDs for the given year/month."""
    r = session.post(
        f"{BASE_URL}/ws/Schedule.asmx/GetMonthSchedule",
        data={
            "leId":      "1",
            "srIdList":  SR_IDS_ALL,
            "seasonId":  str(year),
            "gameMonth": f"{month:02d}",
        },
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()
    game_ids = set()
    for row_obj in data.get("rows", []):
        for cell in row_obj.get("row", []):
            text = cell.get("Text", "")
            for gid in GAME_ID_RE.findall(text):
                game_ids.add(gid)
    return sorted(game_ids)


def scrape_game(session, game_id, year, sr_id=0):
    """
    Fetch ScoreBoard + BoxScore for one completed game.
    Returns (scoreboard_dict, boxscore_dict) or (None, None) on failure.
    """
    params = {
        "leId":     "1",
        "srId":     str(sr_id),
        "seasonId": str(year),
        "gameId":   game_id,
    }

    try:
        r1 = session.post(
            f"{BASE_URL}/ws/Schedule.asmx/GetScoreBoardScroll",
            data=params, timeout=30,
        )
        r1.raise_for_status()
        sb = r1.json()
    except Exception:
        return None, None

    if sb.get("code") != "100":
        # Try other SR_IDs for postseason games
        for alt_sr in [1, 3, 4, 5]:
            if alt_sr == sr_id:
                continue
            try:
                params["srId"] = str(alt_sr)
                r1 = session.post(
                    f"{BASE_URL}/ws/Schedule.asmx/GetScoreBoardScroll",
                    data=params, timeout=30,
                )
                r1.raise_for_status()
                sb = r1.json()
                if sb.get("code") == "100":
                    break
            except Exception:
                continue
        else:
            return None, None

    try:
        r2 = session.post(
            f"{BASE_URL}/ws/Schedule.asmx/GetBoxScoreScroll",
            data=params, timeout=30,
        )
        r2.raise_for_status()
        bs = r2.json()
    except Exception:
        bs = {}

    return sb, bs


def upsert_game(conn, game_id, sb, bs, season_year):
    sr_id = sb.get("SR_ID", 0)
    away_code = sb.get("AWAY_ID", "")
    home_code = sb.get("HOME_ID", "")

    conn.execute(
        """
        INSERT INTO team_game_results
            (game_id, season_year, sr_id, game_date, away_code, home_code,
             away_score, home_score, game_state, stadium, crowd,
             start_time, end_time, use_time, created_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(game_id) DO UPDATE SET
            away_score=excluded.away_score,
            home_score=excluded.home_score,
            game_state=excluded.game_state,
            stadium=excluded.stadium,
            crowd=excluded.crowd,
            start_time=excluded.start_time,
            end_time=excluded.end_time,
            use_time=excluded.use_time,
            created_at=excluded.created_at
        """,
        (
            game_id,
            season_year,
            sr_id,
            sb.get("G_DT", ""),
            away_code,
            home_code,
            safe_int(sb.get("T_SCORE_CN")),
            safe_int(sb.get("B_SCORE_CN")),
            3,  # completed
            sb.get("S_NM"),
            sb.get("CROWD_CN"),
            sb.get("START_TM"),
            sb.get("END_TM"),
            sb.get("USE_TM"),
            now_text(),
        ),
    )

    arr_pitcher = bs.get("arrPitcher", [])
    sides = [("away", away_code), ("home", home_code)]
    for i, (side, team_code) in enumerate(sides):
        if i >= len(arr_pitcher):
            break
        headers, sp_row = extract_sp_row(arr_pitcher[i])
        rec = build_pitcher_record(game_id, side, team_code, headers, sp_row, season_year)
        conn.execute(
            """
            INSERT INTO game_starting_pitchers
                (game_id, side, team_code, player_name, ip_raw, ip_numeric,
                 tbf, hits, bb, hbp, strikeouts, hr, runs, er, result_code,
                 season_year, created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(game_id, side) DO UPDATE SET
                player_name=excluded.player_name,
                ip_raw=excluded.ip_raw,
                ip_numeric=excluded.ip_numeric,
                tbf=excluded.tbf,
                hits=excluded.hits,
                bb=excluded.bb,
                hbp=excluded.hbp,
                strikeouts=excluded.strikeouts,
                hr=excluded.hr,
                runs=excluded.runs,
                er=excluded.er,
                result_code=excluded.result_code,
                created_at=excluded.created_at
            """,
            (
                rec["game_id"], rec["side"], rec["team_code"],
                rec["player_name"], rec["ip_raw"], rec["ip_numeric"],
                rec["tbf"], rec["hits"], rec["bb"], rec["hbp"],
                rec["strikeouts"], rec["hr"], rec["runs"], rec["er"],
                rec["result_code"], rec["season_year"], rec["created_at"],
            ),
        )


def already_scraped(conn, game_id):
    row = conn.execute(
        "SELECT 1 FROM team_game_results WHERE game_id=?", (game_id,)
    ).fetchone()
    return row is not None


def scrape_year(conn, session, year, sleep_seconds, force_refresh):
    total_games = 0
    skipped = 0
    failed = 0

    kbo_months = range(3, 11)  # March to October
    for month in kbo_months:
        game_ids = get_month_game_ids(session, year, month)
        if not game_ids:
            continue

        month_new = 0
        for game_id in game_ids:
            if not force_refresh and already_scraped(conn, game_id):
                skipped += 1
                continue

            sb, bs = scrape_game(session, game_id, year)
            if sb is None:
                failed += 1
                continue

            upsert_game(conn, game_id, sb, bs, year)
            month_new += 1
            total_games += 1

            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

        if month_new:
            conn.commit()
            print(f"  {year}-{month:02d}: +{month_new} games (skipped={skipped}, failed={failed})")
        else:
            print(f"  {year}-{month:02d}: all skipped ({len(game_ids)} already in DB)")

    return total_games


def main():
    parser = argparse.ArgumentParser(description="Scrape KBO box scores into kbo.sqlite.")
    parser.add_argument("--start-year", type=int, default=2011)
    parser.add_argument("--end-year",   type=int, default=2026)
    parser.add_argument("--sleep-seconds", type=float, default=0.2)
    parser.add_argument(
        "--refresh-range",
        action="store_true",
        help="Re-scrape games already in the DB for the given year range.",
    )
    args = parser.parse_args()

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    init_db(conn)

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) KBO-Scraper/1.0",
        "Referer": f"{BASE_URL}/",
        "Origin":  BASE_URL,
    })

    grand_total = 0
    try:
        for year in range(args.start_year, args.end_year + 1):
            print(f"\n=== {year} ===")
            n = scrape_year(conn, session, year, args.sleep_seconds, args.refresh_range)
            grand_total += n
            print(f"  → {year} done: {n} new games")
    finally:
        conn.close()
        session.close()

    print(f"\nFinished. DB={DB_PATH.resolve()}  total_new={grand_total}")


if __name__ == "__main__":
    main()
