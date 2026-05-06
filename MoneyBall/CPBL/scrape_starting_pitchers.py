"""
scrape_starting_pitchers.py

從 CPBL /box/getlive API 爬每場比賽的先發投手資料。

資料來源：
  GET  https://en.cpbl.com.tw/box?KindCode=A&Year={year}&GameSno={sno}
  POST https://en.cpbl.com.tw/box/getlive  (頁面自動呼叫，攔截 response)

抽取欄位（來自 CurtGameDetailJson[0]）：
  VisitingFirstMover / HomeFirstMover  — 先發投手中文名（可能亂碼）
  VisitingFirstAcnt  / HomeFirstAcnt   — 先發投手帳號（唯一 ID）
  WinningPitcherName / WinningPitcherAcnt
  LosePitcherName    / LosePitcherAcnt
  CloserPitcherName  / CloserPitcherAcnt

另外從 PitchingJson 中確認先發（RoleType 判斷）：
  Seq = 1, RoleType 含 '先發' 或 Lineup 最小者

執行前提：
  pip install playwright
  playwright install chromium

用法：
  python scrape_starting_pitchers.py           # 全部
  python scrape_starting_pitchers.py --year 2024   # 只爬某年
"""

import asyncio
import sqlite3
import json
import sys
import random
from pathlib import Path

DB_PATH = Path("cpbl.sqlite")

CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS game_starting_pitchers (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    season_year     INTEGER NOT NULL,
    game_sno        INTEGER NOT NULL,
    -- 先發投手 (帳號 ID，唯一識別)
    home_sp_acnt    TEXT,
    vis_sp_acnt     TEXT,
    -- 英文名（API 提供）
    home_sp_en      TEXT,
    vis_sp_en       TEXT,
    -- W/L/S
    win_acnt        TEXT,
    lose_acnt       TEXT,
    save_acnt       TEXT,
    -- 先發投手本場成績（from PitchingJson）
    home_sp_ip      REAL,    -- innings pitched
    vis_sp_ip       REAL,
    home_sp_er      INTEGER, -- earned runs
    vis_sp_er       INTEGER,
    home_sp_k       INTEGER, -- strikeouts
    vis_sp_k        INTEGER,
    home_sp_bb      INTEGER, -- walks
    vis_sp_bb       INTEGER,
    home_sp_h       INTEGER, -- hits allowed
    vis_sp_h        INTEGER,
    home_sp_pitch   INTEGER, -- pitch count
    vis_sp_pitch    INTEGER,
    -- 爬取狀態
    scrape_status   TEXT DEFAULT 'ok',
    scraped_at      TEXT DEFAULT (datetime('now')),
    UNIQUE(season_year, game_sno)
)
"""


def init_db(conn):
    conn.execute(CREATE_TABLE)
    existing = {row[1] for row in conn.execute("PRAGMA table_info(game_starting_pitchers)")}
    for col in ("home_sp_h", "vis_sp_h"):
        if col not in existing:
            conn.execute(f"ALTER TABLE game_starting_pitchers ADD COLUMN {col} INTEGER")
    conn.commit()


def get_pending_games(conn, year_filter=None, kind_code="A"):
    q = """
        SELECT tgr.season_year, tgr.game_sno
        FROM team_game_results tgr
        WHERE tgr.game_status = 3
          AND tgr.season_year >= 2011
          AND tgr.kind_code = '{kind_code}'
          {year_clause}
          AND (
              NOT EXISTS (
                  SELECT 1 FROM game_starting_pitchers gsp
                  WHERE gsp.season_year = tgr.season_year
                    AND gsp.kind_code   = tgr.kind_code
                    AND gsp.game_sno    = tgr.game_sno
              )
              OR EXISTS (
                  SELECT 1 FROM game_starting_pitchers gsp
                  WHERE gsp.season_year = tgr.season_year
                    AND gsp.kind_code   = tgr.kind_code
                    AND gsp.game_sno    = tgr.game_sno
                    AND (gsp.home_sp_h IS NULL OR gsp.vis_sp_h IS NULL)
              )
          )
        GROUP BY tgr.season_year, tgr.game_sno
        ORDER BY tgr.season_year, tgr.game_sno
    """
    year_clause = f"AND tgr.season_year = {year_filter}" if year_filter else ""
    rows = conn.execute(q.format(kind_code=kind_code, year_clause=year_clause)).fetchall()
    return [(r[0], r[1]) for r in rows]


def parse_api_response(body: bytes, year: int, sno: int, kind_code: str = "A") -> dict:
    """從 /box/getlive JSON body 提取先發投手資料"""
    result = {
        "season_year": year,
        "kind_code": kind_code,
        "game_sno": sno,
        "home_sp_acnt": None, "vis_sp_acnt": None,
        "home_sp_en": None,   "vis_sp_en": None,
        "win_acnt": None,     "lose_acnt": None, "save_acnt": None,
        "home_sp_ip": None,   "vis_sp_ip": None,
        "home_sp_er": None,   "vis_sp_er": None,
        "home_sp_k": None,    "vis_sp_k": None,
        "home_sp_bb": None,   "vis_sp_bb": None,
        "home_sp_h": None,    "vis_sp_h": None,
        "home_sp_pitch": None,"vis_sp_pitch": None,
        "scrape_status": "ok",
    }

    try:
        data = json.loads(body)
    except Exception as e:
        result["scrape_status"] = f"json_error:{str(e)[:80]}"
        return result

    # ── CurtGameDetailJson ──
    curt_raw = data.get("CurtGameDetailJson")
    if curt_raw:
        try:
            curt = json.loads(curt_raw)
            detail = curt[0] if isinstance(curt, list) else curt
            result["home_sp_acnt"]  = detail.get("HomeFirstAcnt") or None
            result["vis_sp_acnt"]   = detail.get("VisitingFirstAcnt") or None
            result["win_acnt"]      = detail.get("WinningPitcherAcnt") or None
            result["lose_acnt"]     = detail.get("LosePitcherAcnt") or None
            result["save_acnt"]     = detail.get("CloserPitcherAcnt") or None
        except Exception:
            pass

    # ── FirstSnoJson：找英文名（DefendStation==1 是投手）──
    first_raw = data.get("FirstSnoJson")
    if first_raw:
        try:
            first = json.loads(first_raw)
            # 先發投手：DefendStation == '1', 且 MainEventNoS 全 0（表示從比賽開始就上場）
            # 或是 Lineup == 0 的投手（通常第一個是先發）
            home_pitchers = [p for p in first
                             if p.get("DefendStation") == "1"
                             and p.get("VisitingHomeType") == "2"]
            vis_pitchers  = [p for p in first
                             if p.get("DefendStation") == "1"
                             and p.get("VisitingHomeType") == "1"]

            # 先發投手 = MainEventNoS 全 0 或 Lineup 最小
            def find_starter(pitchers):
                starters = [p for p in pitchers
                            if p.get("MainEventNoS", "0") == "0000000000"]
                if starters:
                    return starters[0]
                # 退而求其次：Lineup 最小
                if pitchers:
                    return min(pitchers, key=lambda p: p.get("Lineup", 99))
                return None

            home_sp = find_starter(home_pitchers)
            vis_sp  = find_starter(vis_pitchers)

            if home_sp:
                result["home_sp_en"]   = home_sp.get("Engname") or None
                if not result["home_sp_acnt"]:
                    result["home_sp_acnt"] = home_sp.get("Acnt") or None
            if vis_sp:
                result["vis_sp_en"]    = vis_sp.get("Engname") or None
                if not result["vis_sp_acnt"]:
                    result["vis_sp_acnt"] = vis_sp.get("Acnt") or None
        except Exception:
            pass

    # ── PitchingJson：先發投手本場成績 ──
    pitch_raw = data.get("PitchingJson")
    if pitch_raw:
        try:
            pitchings = json.loads(pitch_raw)
            # 先發投手 = 同一隊的第一個投手（Lineup 最小 or Seq 最小）
            home_pitches = [p for p in pitchings if p.get("VisitingHomeType") == "2"]
            vis_pitches  = [p for p in pitchings if p.get("VisitingHomeType") == "1"]

            def get_starter_stats(pitches, sp_acnt):
                """找先發（優先用 acnt 匹配，否則用第一個）"""
                if sp_acnt:
                    matched = [p for p in pitches if p.get("PitcherAcnt") == sp_acnt]
                    if matched:
                        return matched[0]
                # 退而求其次：第一個（最小 Lineup）
                if pitches:
                    return pitches[0]
                return None

            home_stats = get_starter_stats(home_pitches, result["home_sp_acnt"])
            vis_stats  = get_starter_stats(vis_pitches,  result["vis_sp_acnt"])

            if home_stats:
                result["home_sp_ip"]    = home_stats.get("InningPitchedCnt")
                result["home_sp_er"]    = home_stats.get("EarnedRunCnt")
                result["home_sp_k"]     = home_stats.get("StrikeOutCnt")
                result["home_sp_bb"]    = home_stats.get("BasesONBallsCnt")
                result["home_sp_h"]     = home_stats.get("HittingCnt")
                result["home_sp_pitch"] = home_stats.get("PitchCnt")
                # 補英文名
                if not result["home_sp_en"]:
                    result["home_sp_en"] = home_stats.get("PitcherEnName") or None

            if vis_stats:
                result["vis_sp_ip"]    = vis_stats.get("InningPitchedCnt")
                result["vis_sp_er"]    = vis_stats.get("EarnedRunCnt")
                result["vis_sp_k"]     = vis_stats.get("StrikeOutCnt")
                result["vis_sp_bb"]    = vis_stats.get("BasesONBallsCnt")
                result["vis_sp_h"]     = vis_stats.get("HittingCnt")
                result["vis_sp_pitch"] = vis_stats.get("PitchCnt")
                if not result["vis_sp_en"]:
                    result["vis_sp_en"] = vis_stats.get("PitcherEnName") or None

        except Exception:
            pass

    # 如果完全沒找到先發資料
    if not result["home_sp_acnt"] and not result["vis_sp_acnt"]:
        result["scrape_status"] = "no_pitcher_data"

    return result


async def scrape_game(page, year: int, sno: int, kind_code: str = "A") -> dict:
    url = f"https://en.cpbl.com.tw/box?KindCode={kind_code}&Year={year}&GameSno={sno}"
    captured = None

    async def handle_response(response):
        nonlocal captured
        if "getlive" in response.url and response.status == 200:
            try:
                captured = await response.body()
            except Exception:
                pass

    page.on("response", handle_response)
    empty = {
        "season_year": year, "kind_code": kind_code, "game_sno": sno,
        "home_sp_acnt": None, "vis_sp_acnt": None,
        "home_sp_en": None, "vis_sp_en": None,
        "win_acnt": None, "lose_acnt": None, "save_acnt": None,
        "home_sp_ip": None, "vis_sp_ip": None,
        "home_sp_er": None, "vis_sp_er": None,
        "home_sp_k": None, "vis_sp_k": None,
        "home_sp_bb": None, "vis_sp_bb": None,
        "home_sp_h": None, "vis_sp_h": None,
        "home_sp_pitch": None, "vis_sp_pitch": None,
    }
    try:
        await page.goto(url, wait_until="networkidle", timeout=30000)
    except Exception as e:
        page.remove_listener("response", handle_response)
        return {**empty, "scrape_status": f"timeout:{str(e)[:80]}"}

    page.remove_listener("response", handle_response)

    if not captured:
        return {**empty, "scrape_status": "no_api_response"}

    return parse_api_response(captured, year, sno, kind_code)


def upsert_result(conn, result: dict):
    cols = [c for c in result if c not in ("id",)]
    vals = tuple(result[c] for c in cols)
    placeholders = ",".join("?" * len(cols))
    conn.execute(
        f"INSERT OR REPLACE INTO game_starting_pitchers ({','.join(cols)}) VALUES ({placeholders})",
        vals,
    )
    conn.commit()


async def main():
    from playwright.async_api import async_playwright

    year_filter = None
    kind_code = "A"
    for arg in sys.argv[1:]:
        if arg.startswith("--year"):
            part = arg.split("=", 1)
            year_filter = int(part[1]) if len(part) == 2 else int(sys.argv[sys.argv.index(arg)+1])
        if arg.startswith("--kind-code"):
            part = arg.split("=", 1)
            kind_code = part[1] if len(part) == 2 else sys.argv[sys.argv.index(arg)+1]

    conn = sqlite3.connect(DB_PATH)
    init_db(conn)

    pending = get_pending_games(conn, year_filter, kind_code)
    total = len(pending)
    year_str = f" ({year_filter})" if year_filter else ""
    print(f"Pending games: {total}{year_str}")

    if total == 0:
        print("全部已爬完。")
        conn.close()
        return

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36"
        )
        page = await context.new_page()

        ok = not_found = error = 0

        for i, (year, sno) in enumerate(pending):
            result = await scrape_game(page, year, sno, kind_code)
            upsert_result(conn, result)

            s = result["scrape_status"]
            if s == "ok":
                ok += 1
            elif s == "no_pitcher_data":
                not_found += 1
            else:
                error += 1

            if (i + 1) % 10 == 0 or (i + 1) == total:
                home_en = (result.get('home_sp_en') or '?').encode('ascii', 'replace').decode()
                vis_en  = (result.get('vis_sp_en')  or '?').encode('ascii', 'replace').decode()
                print(f"  [{i+1}/{total}] ok={ok} no_data={not_found} err={error}  "
                      f"{year}/{sno}  home={home_en}  vis={vis_en}")

            # 禮貌延遲
            await asyncio.sleep(random.uniform(0.6, 1.2))

        await browser.close()

    # 摘要
    cur = conn.cursor()
    cur.execute("""
        SELECT season_year,
               COUNT(*) as games,
               SUM(CASE WHEN home_sp_acnt IS NOT NULL THEN 1 ELSE 0 END) as with_sp
        FROM game_starting_pitchers
        GROUP BY season_year ORDER BY season_year
    """)
    print(f"\n{'Year':<6} {'Games':>6} {'WithSP':>8}")
    for r in cur.fetchall():
        print(f"  {r[0]:<6} {r[1]:>6} {r[2]:>8}")

    conn.close()
    print(f"\nDone. ok={ok} no_data={not_found} error={error}")


if __name__ == "__main__":
    asyncio.run(main())
