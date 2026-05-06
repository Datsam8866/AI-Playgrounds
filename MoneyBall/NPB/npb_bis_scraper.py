"""
Scrape NPB historical game results from npb.jp/bis/ archive (2006-2015).

The /bis/ archive predates the /games/ and /scores/ systems used from 2016+.
Calendar pages enumerate all games per month; individual game pages have box
scores including complete pitcher lines.

Data is written into the same team_game_results and game_starting_pitchers
tables used by the 2016+ scrapers (game_url prefix 'bis/' avoids collisions).

Usage:
    python npb_bis_scraper.py --start-year 2006 --end-year 2015
    python npb_bis_scraper.py --year 2011
    python npb_bis_scraper.py --year 2011 --sleep-seconds 0.5
"""

import argparse
import re
import sqlite3
import sys
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_URL = "https://npb.jp"
DB_PATH = Path(__file__).resolve().parent / "npb.sqlite"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

# Calendar index files: "04" covers March+April; 05-10 are single months; 11 for late Japan Series
CALENDAR_MONTHS = ["04", "05", "06", "07", "08", "09", "10", "11"]

CL_TEAMS = {"g", "d", "db", "t", "c", "s"}
PL_TEAMS = {"h", "f", "b", "e", "l", "m"}

# Pattern for large flag images in the score box: flag2006_f_1l.gif → "f"
FLAG_CODE_RE = re.compile(r"flag\d{4}_([a-z]+)_1l\.gif")

# Some historical flag codes differ from our canonical 2-letter team codes
FLAG_TO_TEAM_CODE = {
    "g": "g", "d": "d", "db": "db", "t": "t", "c": "c", "s": "s",
    "h": "h", "f": "f", "b": "b", "e": "e", "l": "l", "m": "m",
    "bay": "db",   # 横浜ベイスターズ pre-2012
    "bs":  "b",    # オリックス・バファローズ variant seen in some years
    "bw":  "b",    # BlueWave legacy (pre-merger, safety)
}

# Full and short team name → 2-letter code; covers all eras 2006-2015
TEAM_NAME_TO_CODE = {
    "巨人": "g", "読売": "g", "読売ジャイアンツ": "g",
    "中日": "d", "中日ドラゴンズ": "d",
    "DeNA": "db", "横浜DeNA": "db", "横浜DeNAベイスターズ": "db",
    "横浜": "db", "横浜ベイスターズ": "db",
    "阪神": "t", "阪神タイガース": "t",
    "広島": "c", "広島東洋": "c", "広島東洋カープ": "c",
    "ヤクルト": "s", "東京ヤクルト": "s", "東京ヤクルトスワローズ": "s",
    "ソフトバンク": "h", "福岡ソフトバンク": "h", "福岡ソフトバンクホークス": "h",
    "ダイエー": "h", "福岡ダイエー": "h", "福岡ダイエーホークス": "h",
    "日本ハム": "f", "北海道日本ハム": "f", "北海道日本ハムファイターズ": "f",
    "オリックス": "b", "オリックス・バファローズ": "b", "オリックスバファローズ": "b",
    "楽天": "e", "東北楽天": "e", "東北楽天ゴールデンイーグルス": "e",
    "西武": "l", "埼玉西武": "l", "埼玉西武ライオンズ": "l",
    "ロッテ": "m", "千葉ロッテ": "m", "千葉ロッテマリーンズ": "m",
}

# Path pattern inside calendar HTML: /bis/2006/games/s2006032500104.html
BIS_GAME_PATH_RE = re.compile(r"/(bis/\d{4}/games/s\d+)\.html")

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


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def init_db(conn):
    conn.execute(CREATE_SCHEDULE_TABLE)
    conn.execute(CREATE_SP_TABLE)
    conn.commit()


def game_url_exists(conn, game_url):
    row = conn.execute(
        "SELECT 1 FROM team_game_results WHERE game_url = ? LIMIT 1", (game_url,)
    ).fetchone()
    return row is not None


def insert_game(conn, game):
    conn.execute(
        """
        INSERT OR IGNORE INTO team_game_results (
            season_year, game_date, home_code, away_code, home_score, away_score,
            home_win, league_code, stadium, win_pitcher, lose_pitcher, game_url, status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            game["season_year"], game["game_date"],
            game["home_code"], game["away_code"],
            game["home_score"], game["away_score"],
            game["home_win"], game["league_code"],
            game["stadium"], game["win_pitcher"], game["lose_pitcher"],
            game["game_url"], game["status"],
        ),
    )


def insert_starter(conn, season_year, game_url, starter):
    conn.execute(
        """
        INSERT OR IGNORE INTO game_starting_pitchers (
            season_year, game_url, team_code, pitcher_name, ip_outs, hits, hr, bb,
            hbp, strikeouts, runs, earned_runs, pitches, batters_faced, result
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            season_year, game_url,
            starter["team_code"], starter["pitcher_name"],
            starter["ip_outs"], starter["hits"], starter["hr"],
            starter["bb"], starter["hbp"], starter["strikeouts"],
            starter["runs"], starter["earned_runs"],
            starter["pitches"], starter["batters_faced"],
            starter["result"],
        ),
    )


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

def make_session():
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    return session


def fetch_page(session, url, timeout=30):
    r = session.get(url, timeout=timeout)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    r.encoding = r.apparent_encoding or "utf-8"
    return r.text


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def clean(text):
    if not text:
        return ""
    return re.sub(r"\s+", " ", str(text).replace("　", " ")).strip()


def league_code_for(home_code, away_code):
    if home_code in CL_TEAMS and away_code in CL_TEAMS:
        return "CL"
    if home_code in PL_TEAMS and away_code in PL_TEAMS:
        return "PL"
    if (home_code in CL_TEAMS) != (away_code in CL_TEAMS):
        return "IL"
    return None


def extract_game_paths_from_calendar(html):
    """Return ordered-unique BIS game paths from a calendar page HTML."""
    seen = set()
    result = []
    for path in BIS_GAME_PATH_RE.findall(html):
        if path not in seen:
            seen.add(path)
            result.append(path)
    return result


def flag_to_team_code(img_src):
    m = FLAG_CODE_RE.search(img_src or "")
    if not m:
        return None
    raw = m.group(1)
    return FLAG_TO_TEAM_CODE.get(raw, raw if raw in CL_TEAMS | PL_TEAMS else None)


def parse_teams_from_title(soup):
    """
    Extract (home_code, away_code) from page title pattern '（HOME vs AWAY）'.
    Title format: '2006年3月25日 …（北海道日本ハムvs東北楽天）'
    """
    title_tag = soup.find("title")
    if not title_tag:
        return None, None
    title = title_tag.get_text()
    m = re.search(r"[（(]([^）)]+?)[vｖ]s([^）)]+?)[）)]", title)
    if not m:
        return None, None
    home_name = clean(m.group(1))
    away_name = clean(m.group(2))
    return TEAM_NAME_TO_CODE.get(home_name), TEAM_NAME_TO_CODE.get(away_name)


def parse_score_box(soup):
    """
    Find the two score-box rows (large flag images _1l.gif).
    Returns [(code, score), (code, score)] ordered WINNER first (BIS format).
    Do NOT assume positional home/away; look up by team code.
    """
    entries = []
    seen_rows = set()
    for img in soup.find_all("img"):
        src = img.get("src", "")
        if "_1l.gif" not in src:
            continue
        code = flag_to_team_code(src)
        if not code:
            continue
        row = img.find_parent("tr")
        if row is None or id(row) in seen_rows:
            continue
        seen_rows.add(id(row))
        cells = row.find_all(["td", "th"])
        score = None
        for cell in reversed(cells):
            t = clean(cell.get_text())
            if t.isdigit():
                score = int(t)
                break
        entries.append((code, score))
        if len(entries) == 2:
            break
    return entries


def is_pitching_table(table):
    """Identify a BIS pitching table by '投回' (IP) in its header row."""
    first_row = table.find("tr")
    if not first_row:
        return False
    for cell in first_row.find_all(["th", "td"]):
        t = cell.get_text(" ", strip=True)
        if "投" in t and "回" in t:
            return True
    return False


def ip_to_outs(whole_text, frac_text):
    """
    Convert split BIS IP columns to total outs.
    whole='7', frac=''  → 21   (7 full innings)
    whole='4', frac='.2' → 14  (4 innings + 2 outs)
    whole='',  frac='+'  → 0   (faced batter(s) but recorded no out)
    """
    whole_text = clean(whole_text)
    frac_text = clean(frac_text).lstrip("|").strip()
    whole = int(whole_text) if whole_text.isdigit() else 0
    if frac_text in ("+", "＋"):
        frac = 0
    elif frac_text.startswith(".") and len(frac_text) > 1 and frac_text[1].isdigit():
        frac = int(frac_text[1])
    else:
        frac = 0
    return whole * 3 + frac


def parse_int_safe(text):
    text = clean(text)
    try:
        return int(text)
    except (ValueError, TypeError):
        return None


def parse_pitcher_row(row):
    """
    Parse one row from a BIS pitching table.
    Columns: result | name | IP_whole | IP_frac | BF | H | BB | HBP | K | ER
    Returns dict or None if this is a header/total/empty row.
    """
    cells = row.find_all(["td", "th"])
    if len(cells) < 8:
        return None
    texts = [clean(c.get_text(" ")) for c in cells]

    # Skip header rows (contain 投回)
    if any("投" in t and "回" in t for t in texts):
        return None
    # Skip totals
    if texts[1] in ("チーム計", "計", ""):
        return None
    # Need a non-empty pitcher name
    name = texts[1]
    if not name:
        return None

    result_map = {"○": "W", "●": "L", "Ｈ": "H", "H": "H", "Ｓ": "S", "S": "S"}
    result = result_map.get(texts[0])

    ip_outs = ip_to_outs(texts[2], texts[3])

    return {
        "pitcher_name": name,
        "result": result,
        "ip_outs": ip_outs,
        "hits": parse_int_safe(texts[5]) if len(texts) > 5 else None,
        "hr": None,           # not in BIS format
        "bb": parse_int_safe(texts[6]) if len(texts) > 6 else None,
        "hbp": parse_int_safe(texts[7]) if len(texts) > 7 else None,
        "strikeouts": parse_int_safe(texts[8]) if len(texts) > 8 else None,
        "runs": None,         # not in BIS format
        "earned_runs": parse_int_safe(texts[9]) if len(texts) > 9 else None,
        "pitches": None,      # not in BIS format
        "batters_faced": parse_int_safe(texts[4]) if len(texts) > 4 else None,
    }


def parse_bis_game_page(html, game_path):
    """
    Parse a complete BIS game page.
    game_path: e.g. 'bis/2006/games/s2006032500104' (no leading slash, no .html)
    Returns a dict with game info + 'starters' list, or None if unparseable.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Year from path
    ym = re.search(r"bis/(\d{4})/", game_path)
    dm = re.search(r"/s(\d{8})\d+$", game_path)
    if not ym or not dm:
        return None
    season_year = int(ym.group(1))
    ds = dm.group(1)
    game_date = f"{ds[:4]}-{ds[4:6]}-{ds[6:8]}"

    # Teams: title is the only reliable source (score box shows WINNER first, not HOME first)
    home_code, away_code = parse_teams_from_title(soup)
    if not home_code or not away_code:
        return None  # could not determine teams

    # Scores from score box flag images — BIS shows WINNER first, not HOME first,
    # so look up by team code rather than position.
    entries = parse_score_box(soup)
    score_map = {code: score for code, score in entries if code}
    home_score = score_map.get(home_code)
    away_score = score_map.get(away_code)

    if home_score is not None and away_score is not None:
        status = "completed"
        if home_score > away_score:
            home_win = 1
        elif home_score < away_score:
            home_win = 0
        else:
            home_win = None  # draw
    else:
        status = "cancelled"
        home_win = None

    # Win/loss pitchers
    win_pitcher = lose_pitcher = None
    for row in soup.find_all("tr"):
        cells = row.find_all(["td", "th"])
        if len(cells) < 2:
            continue
        label = clean(cells[0].get_text())
        value = clean(cells[1].get_text())
        if "勝投手" in label:
            win_pitcher = re.sub(r"\s*\(.*", "", value).strip() or None
        elif "敗投手" in label:
            lose_pitcher = re.sub(r"\s*\(.*", "", value).strip() or None

    # Stadium: look for the row that contains "試合時間"
    stadium = None
    for row in soup.find_all("tr"):
        for cell in row.find_all(["td", "th"]):
            text = clean(cell.get_text(" "))
            if "試合時間" in text:
                part = text.split("試合時間")[0].strip()
                if 2 < len(part) < 30:
                    stadium = part
                break
        if stadium:
            break

    # Starting pitchers: first data row of each pitching table
    # Away team's table comes before home team's table in BIS layout
    pitching_tables = [t for t in soup.find_all("table") if is_pitching_table(t)]
    starters = []
    for side, table in zip(("away", "home"), pitching_tables[:2]):
        for row in table.find_all("tr"):
            parsed = parse_pitcher_row(row)
            if parsed:
                parsed["team_code"] = side
                starters.append(parsed)
                break  # first non-header row = starter

    return {
        "season_year": season_year,
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
        "game_url": game_path,
        "status": status,
        "starters": starters,
    }


# ---------------------------------------------------------------------------
# Scraping orchestration
# ---------------------------------------------------------------------------

def collect_game_paths_for_year(session, year, sleep_seconds, timeout):
    """Fetch all calendar pages for a year and return deduplicated game paths."""
    all_paths = []
    seen = set()
    for month_code in CALENDAR_MONTHS:
        url = f"{BASE_URL}/bis/{year}/calendar/index_{month_code}.html"
        html = fetch_page(session, url, timeout=timeout)
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
        if html is None:
            continue
        for path in extract_game_paths_from_calendar(html):
            if path not in seen:
                seen.add(path)
                all_paths.append(path)
    return all_paths


def scrape_year(conn, session, year, sleep_seconds, timeout):
    print(f"year={year} collecting game URLs...")
    game_paths = collect_game_paths_for_year(session, year, sleep_seconds, timeout)
    print(f"year={year} found={len(game_paths)} games in calendar")

    inserted = 0
    skipped = 0
    errors = 0

    for game_path in game_paths:
        if game_url_exists(conn, game_path):
            skipped += 1
            continue

        url = f"{BASE_URL}/{game_path}.html"
        html = fetch_page(session, url, timeout=timeout)
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

        if html is None:
            print(f"  not_found: {game_path}")
            errors += 1
            continue

        try:
            game = parse_bis_game_page(html, game_path)
        except Exception as exc:
            print(f"  parse_error: {game_path} — {type(exc).__name__}: {exc}")
            errors += 1
            continue

        if game is None:
            print(f"  unparseable: {game_path}")
            errors += 1
            continue

        insert_game(conn, game)
        for starter in game.get("starters", []):
            insert_starter(conn, game["season_year"], game_path, starter)
        conn.commit()
        inserted += 1

    print(
        f"year={year} inserted={inserted} skipped={skipped} errors={errors}"
    )
    return {"inserted": inserted, "skipped": skipped, "errors": errors}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Scrape NPB /bis/ archive (2006-2015) into SQLite."
    )
    parser.add_argument("--year", type=int, help="Scrape a single year.")
    parser.add_argument("--start-year", type=int, default=2006)
    parser.add_argument("--end-year", type=int, default=2015)
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--sleep-seconds", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=30.0)
    return parser.parse_args()


def main():
    args = parse_args()
    start_year = args.year if args.year else args.start_year
    end_year = args.year if args.year else args.end_year

    conn = sqlite3.connect(args.db)
    init_db(conn)
    session = make_session()

    totals = {"inserted": 0, "skipped": 0, "errors": 0}
    try:
        for year in range(start_year, end_year + 1):
            result = scrape_year(
                conn=conn,
                session=session,
                year=year,
                sleep_seconds=args.sleep_seconds,
                timeout=args.timeout,
            )
            for k in totals:
                totals[k] += result[k]
    finally:
        session.close()
        conn.close()

    print(f"saved_db={args.db.resolve()}")
    print(
        f"total_inserted={totals['inserted']} "
        f"total_skipped={totals['skipped']} "
        f"total_errors={totals['errors']}"
    )


if __name__ == "__main__":
    main()
