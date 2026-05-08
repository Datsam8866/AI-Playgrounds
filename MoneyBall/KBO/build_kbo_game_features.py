"""
build_kbo_game_features.py

為每場比賽計算 pre-game features，寫入 kbo.sqlite 的 game_features 表。

設計原則（無洩漏）：
  所有 rolling stats 只用「這場之前」的比賽計算。
  Elo 由全賽事（sr_id=0,1,3,4,5）更新，但 game_features 只產生 sr_id=0 例行賽。
  例行賽 rolling / rest / streak / season_games_before 只用 sr_id=0，避免季前賽污染 early regime。

Elo 參數（由 Gemini G1 審查確認）：
  K=48, home_advantage=10, ELO_REGRESSION=0.50, 初始=1500
  新球隊（NC 2013、KT 2015）從 1500 起算（Elo 零和，合理）

Rolling windows：
  主窗口 = 20 場，短窗口 = 3/5/10，場地分割 = 5/10，趨勢 = 5 vs 10
"""

import argparse
import re
import sqlite3
import sys
from collections import defaultdict
from datetime import date as date_cls
from pathlib import Path

import requests

DB_PATH = Path("kbo.sqlite")

WINDOW      = 20
MIN_GAMES   = 10
TEAM_BURN_IN = 10

ELO_K          = 48
ELO_HOME_ADV   = 10
ELO_REGRESSION = 0.50
ELO_INIT       = 1500.0

REST_CAP = 10

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ── helpers ──────────────────────────────────────────────────────────────────

def pyth_wp(rs: float, ra: float) -> float:
    return rs ** 2 / (rs ** 2 + ra ** 2) if (rs + ra) > 0 else 0.5


def elo_win_prob(ra: float, rb: float, home_adv: float = ELO_HOME_ADV) -> float:
    return 1.0 / (1.0 + 10 ** (-(ra - rb + home_adv) / 400.0))


def rolling_stats(history: list, window: int, min_games: int = MIN_GAMES):
    subset = history[-window:]
    n = len(subset)
    if n < min_games:
        return None
    rs = sum(g["rs"] for g in subset)
    ra = sum(g["ra"] for g in subset)
    wins = sum(g["win"] for g in subset)
    return {
        "win_pct":     wins / n,
        "rs_per_game": rs / n,
        "ra_per_game": ra / n,
        "rd_per_game": (rs - ra) / n,
        "pyth_wp":     pyth_wp(rs, ra),
        "n":           n,
    }


def streak_val(history: list) -> int:
    """正數=連勝，負數=連敗，0=無歷史。"""
    if not history:
        return 0
    cur_win = history[-1]["win"]
    count = 0
    for g in reversed(history):
        if g["win"] == cur_win:
            count += 1
        else:
            break
    return count if cur_win else -count


def rest_days(last_date: str | None, today: str) -> int:
    if not last_date:
        return REST_CAP
    from datetime import date
    d0 = date.fromisoformat(last_date)
    d1 = date.fromisoformat(today)
    return min((d1 - d0).days, REST_CAP)


def build_park_context(conn) -> dict[int, dict]:
    """Build prior-season park context keyed by season_year.

    Each target season maps to cumulative regular-season stadium stats using only
    seasons strictly before that year, keeping the feature walk-forward safe.
    """
    rows = conn.execute("""
        SELECT season_year,
               stadium,
               COUNT(*) AS games,
               SUM(home_score + away_score) AS total_runs,
               SUM(CASE WHEN home_score > away_score THEN 1 ELSE 0 END) AS home_wins
        FROM team_game_results
        WHERE game_state = 3
          AND sr_id = 0
          AND away_score IS NOT NULL
          AND home_score IS NOT NULL
          AND stadium IS NOT NULL
          AND stadium != ''
        GROUP BY season_year, stadium
        ORDER BY season_year, stadium
    """).fetchall()

    season_totals: dict[int, dict] = defaultdict(lambda: {
        "league_games": 0,
        "league_runs": 0,
        "league_home_wins": 0,
        "stadiums": defaultdict(lambda: {"games": 0, "runs": 0, "home_wins": 0}),
    })

    for season_year, stadium, games, total_runs, home_wins in rows:
        bucket = season_totals[season_year]
        bucket["league_games"] += games
        bucket["league_runs"] += total_runs
        bucket["league_home_wins"] += home_wins
        bucket["stadiums"][stadium]["games"] += games
        bucket["stadiums"][stadium]["runs"] += total_runs
        bucket["stadiums"][stadium]["home_wins"] += home_wins

    park_context: dict[int, dict] = {}
    cum_league_games = 0
    cum_league_runs = 0
    cum_league_home_wins = 0
    cum_stadiums: dict[str, dict] = defaultdict(lambda: {"games": 0, "runs": 0, "home_wins": 0})

    for season_year in sorted(season_totals):
        park_context[season_year] = {
            "league_games": cum_league_games,
            "league_runs": cum_league_runs,
            "league_home_wins": cum_league_home_wins,
            "stadiums": {stadium: stats.copy() for stadium, stats in cum_stadiums.items()},
        }

        bucket = season_totals[season_year]
        cum_league_games += bucket["league_games"]
        cum_league_runs += bucket["league_runs"]
        cum_league_home_wins += bucket["league_home_wins"]
        for stadium, stats in bucket["stadiums"].items():
            cum_stadiums[stadium]["games"] += stats["games"]
            cum_stadiums[stadium]["runs"] += stats["runs"]
            cum_stadiums[stadium]["home_wins"] += stats["home_wins"]

    return park_context


def get_park_features(season_year: int, stadium: str | None, park_context: dict[int, dict]):
    if not stadium:
        return None, None

    context = park_context.get(season_year)
    if not context or context["league_games"] <= 0:
        return None, None

    stadium_stats = context["stadiums"].get(stadium)
    if not stadium_stats or stadium_stats["games"] < 30:
        return None, None

    league_avg_rpg = context["league_runs"] / context["league_games"]
    league_hw_rate = context["league_home_wins"] / context["league_games"]
    if league_avg_rpg <= 0:
        return None, None

    stadium_avg_rpg = stadium_stats["runs"] / stadium_stats["games"]
    stadium_hw_rate = stadium_stats["home_wins"] / stadium_stats["games"]
    return stadium_avg_rpg / league_avg_rpg, stadium_hw_rate - league_hw_rate


# ── load data ─────────────────────────────────────────────────────────────────

def load_games(conn) -> list[dict]:
    rows = conn.execute("""
        SELECT game_id, season_year, sr_id, game_date,
               away_code, home_code, away_score, home_score, start_time, stadium
        FROM team_game_results
        WHERE game_state = 3
          AND away_score IS NOT NULL AND home_score IS NOT NULL
        ORDER BY game_date, COALESCE(start_time, ''), game_id
    """).fetchall()

    games = []
    for game_id, yr, sr_id, date, away, home, as_, hs, start_time, stadium in rows:
        if as_ == hs:
            continue  # 排除平局
        games.append({
            "game_id":    game_id,
            "season_year": yr,
            "sr_id":      sr_id,
            "game_date":  date[:10],
            "away_code":  away,
            "home_code":  home,
            "away_score": as_,
            "home_score": hs,
            "home_win":   1 if hs > as_ else 0,
            "start_time":  start_time,
            "stadium":    stadium,
        })
    return games


def load_season_records(games: list[dict]) -> dict:
    """計算每隊每季例行賽（sr_id=0）勝率，供 prev_diff 特徵使用。"""
    from collections import defaultdict
    records = defaultdict(lambda: [0, 0, 0, 0])  # [rs, ra, wins, games]
    for g in games:
        if g["sr_id"] != 0:
            continue
        key_h = (g["season_year"], g["home_code"])
        key_a = (g["season_year"], g["away_code"])
        records[key_h][0] += g["home_score"]; records[key_h][1] += g["away_score"]
        records[key_h][2] += g["home_win"];   records[key_h][3] += 1
        records[key_a][0] += g["away_score"]; records[key_a][1] += g["home_score"]
        records[key_a][2] += 1 - g["home_win"]; records[key_a][3] += 1

    season_pct = {}
    for (yr, team), (rs, ra, wins, n) in records.items():
        if n > 0:
            season_pct[(yr, team)] = {
                "win_pct": wins / n,
                "rd_pg":   (rs - ra) / n,
                "pyth":    pyth_wp(rs, ra),
            }
    return season_pct


# ── feature building ──────────────────────────────────────────────────────────

_KBO_BASE_URL = "https://www.koreabaseball.com"
_GAME_ID_RE   = re.compile(r"gameId=(\d{8}[A-Z]{2,6}\d)")

# Korean team name (as shown in schedule) → 2-char DB code
_KO_TEAM_CODE = {
    "삼성": "SS",   # Samsung Lions
    "두산": "OB",   # Doosan Bears
    "키움": "WO",   # Kiwoom Heroes
    "롯데": "LT",   # Lotte Giants
    "KIA":  "HT",   # KIA Tigers
    "NC":   "NC",   # NC Dinos
    "LG":   "LG",   # LG Twins
    "KT":   "KT",   # KT Wiz
    "SSG":  "SK",   # SSG Landers
    "한화": "HH",   # Hanwha Eagles
}

# Pattern: <li>AWAY : HOME [STADIUM]</li>  (scheduled games lack href/gameId)
_SCHEDULED_LI_RE = re.compile(
    r"<li>([^<:]+?)\s*:\s*([^<\[]+?)\s*\[([^\]]+)\]</li>"
)


def _parse_scheduled_cell(cell_text: str, date_prefix: str) -> list[dict]:
    """Extract scheduled games from a GetMonthSchedule HTML cell.

    For future games, the cell contains Korean team names without gameId links.
    We construct game IDs following the established pattern: YYYYMMDDAWAYYHOMEID0.
    """
    results = []
    for m in _SCHEDULED_LI_RE.finditer(cell_text):
        away_name = m.group(1).strip()
        home_name = m.group(2).strip()
        away_code = _KO_TEAM_CODE.get(away_name)
        home_code = _KO_TEAM_CODE.get(home_name)
        if not away_code or not home_code:
            print(f"[WARN] unknown team name: '{away_name}' or '{home_name}'", file=sys.stderr)
            continue
        game_id = f"{date_prefix}{away_code}{home_code}0"
        results.append((game_id, away_code, home_code))
    return results


def load_scheduled_games(target_date: date_cls, known_game_ids: set[str]) -> list[dict]:
    """Fetch today's scheduled KBO regular-season games from the KBO API.

    For completed games, GetMonthSchedule embeds gameId links.
    For future games, it shows Korean team names only — we parse those and
    construct game IDs from the established YYYYMMDDAWAYYHOMEID0 pattern.
    Returns list of dicts for games not already in known_game_ids.
    """
    date_prefix = target_date.strftime("%Y%m%d")  # e.g. "20260429"
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) KBO-Scraper/1.0"})
    try:
        r = session.post(
            f"{_KBO_BASE_URL}/ws/Schedule.asmx/GetMonthSchedule",
            data={
                "leId":      "1",
                "srIdList":  "0,1,3,4,5,7,9",
                "seasonId":  str(target_date.year),
                "gameMonth": target_date.strftime("%m"),
            },
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as exc:
        print(f"[WARN] load_scheduled_games: KBO API error: {exc}", file=sys.stderr)
        return []

    scheduled = []
    seen_ids = set()

    for row_obj in data.get("rows", []):
        for cell in row_obj.get("row", []):
            text = cell.get("Text", "")

            # Completed game IDs (href links)
            for gid in _GAME_ID_RE.findall(text):
                if gid.startswith(date_prefix) and gid not in seen_ids and gid not in known_game_ids:
                    seen_ids.add(gid)
                    scheduled.append({
                        "game_id":    gid,
                        "season_year": target_date.year,
                        "sr_id":      0,
                        "game_date":  target_date.isoformat(),
                        "away_code":  gid[8:10],
                        "home_code":  gid[10:12],
                    })

            # Scheduled games (Korean names, no gameId link) — only in the target day cell
            if f">{target_date.day}<" in text or f">{target_date.day}</li>" in text:
                for gid, away_code, home_code in _parse_scheduled_cell(text, date_prefix):
                    if gid not in seen_ids and gid not in known_game_ids:
                        seen_ids.add(gid)
                        scheduled.append({
                            "game_id":    gid,
                            "season_year": target_date.year,
                            "sr_id":      0,
                            "game_date":  target_date.isoformat(),
                            "away_code":  away_code,
                            "home_code":  home_code,
                        })

    return scheduled


def build_features(
    games: list[dict],
    park_context: dict[int, dict],
    scheduled_games: list[dict] | None = None,
) -> list[dict]:
    # Elo 狀態
    elo: dict[str, float] = defaultdict(lambda: ELO_INIT)
    current_season: dict[str, int] = {}  # 每隊當前賽季
    season_start_elo: dict[tuple, float] = {}  # 記錄賽季初 Elo（for regression）

    # 例行賽滾動歷史。Elo 仍由全賽事更新，但 predictive rolling features 不混用季前賽/季後賽。
    history:      dict[str, list] = defaultdict(list)
    home_history: dict[str, list] = defaultdict(list)
    away_history: dict[str, list] = defaultdict(list)

    last_game_date: dict[str, str] = {}
    season_games:   dict[tuple, int] = defaultdict(int)  # (year, team) → games played this season

    season_records = load_season_records(games)
    features = []

    for g in games:
        away = g["away_code"]
        home = g["home_code"]
        yr   = g["season_year"]
        date = g["game_date"]
        sr   = g["sr_id"]
        stadium = g.get("stadium")

        # 賽季初 Elo regression
        for team in (away, home):
            if current_season.get(team) != yr:
                if team in current_season:
                    old = elo[team]
                    elo[team] = ELO_INIT + (old - ELO_INIT) * (1 - ELO_REGRESSION)
                current_season[team] = yr
                season_start_elo[(yr, team)] = elo[team]

        # ── 計算特徵（只在 sr_id=0 時輸出到 game_features）──
        if sr == 0:
            h_elo = elo[home]
            a_elo = elo[away]
            diff_elo = h_elo - a_elo
            elo_prob = elo_win_prob(h_elo, a_elo)

            h_stats  = rolling_stats(history[home], WINDOW)
            a_stats  = rolling_stats(history[away], WINDOW)

            # 短窗口
            h3  = rolling_stats(history[home], 3,  min_games=1)
            h5  = rolling_stats(history[home], 5,  min_games=1)
            h10 = rolling_stats(history[home], 10, min_games=1)
            a3  = rolling_stats(history[away], 3,  min_games=1)
            a5  = rolling_stats(history[away], 5,  min_games=1)
            a10 = rolling_stats(history[away], 10, min_games=1)

            # 場地分割（主場在主場/客場在客場）
            h_split5  = rolling_stats(home_history[home], 5,  min_games=1)
            h_split10 = rolling_stats(home_history[home], 10, min_games=1)
            a_split5  = rolling_stats(away_history[away], 5,  min_games=1)
            a_split10 = rolling_stats(away_history[away], 10, min_games=1)

            # 休息天數
            h_rest = rest_days(last_game_date.get(home), date)
            a_rest = rest_days(last_game_date.get(away), date)

            # 連勝/連敗
            h_streak = streak_val(history[home])
            a_streak = streak_val(history[away])

            # 本季已出賽場次（這場之前）
            h_sg = season_games.get((yr, home), 0)
            a_sg = season_games.get((yr, away), 0)

            # 前季數據
            prev_h = season_records.get((yr - 1, home))
            prev_a = season_records.get((yr - 1, away))
            park_factor, stadium_hwa = get_park_features(yr, stadium, park_context)

            def diff(a, b, key, default=None):
                if a and b:
                    return a[key] - b[key]
                return default

            row = {
                "game_id":    g["game_id"],
                "season_year": yr,
                "sr_id":       sr,
                "game_date":   date,
                "away_code":   away,
                "home_code":   home,
                "home_win":    g["home_win"],

                # Elo
                "home_elo":       h_elo,
                "away_elo":       a_elo,
                "diff_elo":       diff_elo,
                "elo_home_prob":  elo_prob,

                # Rolling window=20
                "diff_win_pct":  diff(h_stats, a_stats, "win_pct"),
                "diff_rs":       diff(h_stats, a_stats, "rs_per_game"),
                "diff_ra":       diff(h_stats, a_stats, "ra_per_game"),
                "diff_rd":       diff(h_stats, a_stats, "rd_per_game"),
                "diff_pyth_wp":  diff(h_stats, a_stats, "pyth_wp"),

                # 短窗口
                "diff_w3_win_pct":  diff(h3,  a3,  "win_pct"),
                "diff_w5_win_pct":  diff(h5,  a5,  "win_pct"),
                "diff_w10_win_pct": diff(h10, a10, "win_pct"),
                "diff_w3_rd_pg":    diff(h3,  a3,  "rd_per_game"),
                "diff_w5_rd_pg":    diff(h5,  a5,  "rd_per_game"),
                "diff_w10_rd_pg":   diff(h10, a10, "rd_per_game"),

                # 場地分割
                "diff_split5_win_pct":  diff(h_split5,  a_split5,  "win_pct"),
                "diff_split10_win_pct": diff(h_split10, a_split10, "win_pct"),
                "diff_split5_rd_pg":    diff(h_split5,  a_split5,  "rd_per_game"),
                "diff_split10_rd_pg":   diff(h_split10, a_split10, "rd_per_game"),
                "diff_split5_rs_pg":    diff(h_split5,  a_split5,  "rs_per_game"),
                "diff_split5_ra_pg":    diff(h_split5,  a_split5,  "ra_per_game"),
                "diff_split10_rs_pg":   diff(h_split10, a_split10, "rs_per_game"),
                "diff_split10_ra_pg":   diff(h_split10, a_split10, "ra_per_game"),

                # 趨勢（5 場 vs 10 場）
                "diff_trend_win_pct": diff(h5, a5, "win_pct") - diff(h10, a10, "win_pct")
                    if (h5 and a5 and h10 and a10) else None,
                "diff_trend_rd_pg":   diff(h5, a5, "rd_per_game") - diff(h10, a10, "rd_per_game")
                    if (h5 and a5 and h10 and a10) else None,

                # 情境
                "home_rest":    h_rest,
                "away_rest":    a_rest,
                "diff_rest":    h_rest - a_rest,
                "diff_streak":  h_streak - a_streak,
                "home_season_games_before": h_sg,
                "away_season_games_before": a_sg,

                # 前季
                "prev_diff_win_pct": (prev_h["win_pct"] - prev_a["win_pct"])
                    if (prev_h and prev_a) else None,
                "prev_diff_rd_pg":   (prev_h["rd_pg"] - prev_a["rd_pg"])
                    if (prev_h and prev_a) else None,
                "prev_diff_pyth":    (prev_h["pyth"] - prev_a["pyth"])
                    if (prev_h and prev_a) else None,
                "park_factor": park_factor,
                "stadium_hwa": stadium_hwa,
            }
            features.append(row)

        # ── 比賽結束後更新狀態 ──
        h_win = g["home_win"]
        h_actual = h_win
        a_actual = 1 - h_win

        # Elo 更新
        h_exp = elo_win_prob(elo[home], elo[away])
        a_exp = 1 - h_exp
        elo[home] += ELO_K * (h_actual - h_exp)
        elo[away] += ELO_K * (a_actual - a_exp)

        # 只用例行賽更新 rolling / rest / streak / season_games_before。
        if sr == 0:
            history[home].append({"rs": g["home_score"], "ra": g["away_score"], "win": h_win})
            history[away].append({"rs": g["away_score"], "ra": g["home_score"], "win": 1 - h_win})
            home_history[home].append({"rs": g["home_score"], "ra": g["away_score"], "win": h_win})
            away_history[away].append({"rs": g["away_score"], "ra": g["home_score"], "win": 1 - h_win})

            last_game_date[home] = date
            last_game_date[away] = date
            season_games[(yr, home)] = season_games.get((yr, home), 0) + 1
            season_games[(yr, away)] = season_games.get((yr, away), 0) + 1

    # ── 排程賽事特徵（使用當前最終狀態，不更新 Elo/rolling）──
    if scheduled_games:
        for g in scheduled_games:
            away = g["away_code"]
            home = g["home_code"]
            yr   = g["season_year"]
            date = g["game_date"]

            # 賽季初 Elo regression（如尚未進入本季）
            for team in (away, home):
                if current_season.get(team) != yr:
                    if team in current_season:
                        old = elo[team]
                        elo[team] = ELO_INIT + (old - ELO_INIT) * (1 - ELO_REGRESSION)
                    current_season[team] = yr
                    season_start_elo[(yr, team)] = elo[team]

            h_elo = elo[home]
            a_elo = elo[away]
            diff_elo = h_elo - a_elo
            elo_prob = elo_win_prob(h_elo, a_elo)

            h_stats  = rolling_stats(history[home], WINDOW)
            a_stats  = rolling_stats(history[away], WINDOW)
            h3  = rolling_stats(history[home], 3,  min_games=1)
            h5  = rolling_stats(history[home], 5,  min_games=1)
            h10 = rolling_stats(history[home], 10, min_games=1)
            a3  = rolling_stats(history[away], 3,  min_games=1)
            a5  = rolling_stats(history[away], 5,  min_games=1)
            a10 = rolling_stats(history[away], 10, min_games=1)
            h_split5  = rolling_stats(home_history[home], 5,  min_games=1)
            h_split10 = rolling_stats(home_history[home], 10, min_games=1)
            a_split5  = rolling_stats(away_history[away], 5,  min_games=1)
            a_split10 = rolling_stats(away_history[away], 10, min_games=1)

            h_rest = rest_days(last_game_date.get(home), date)
            a_rest = rest_days(last_game_date.get(away), date)
            h_streak = streak_val(history[home])
            a_streak = streak_val(history[away])
            h_sg = season_games.get((yr, home), 0)
            a_sg = season_games.get((yr, away), 0)
            prev_h = season_records.get((yr - 1, home))
            prev_a = season_records.get((yr - 1, away))
            park_factor, stadium_hwa = get_park_features(yr, g.get("stadium"), park_context)

            def diff(a, b, key, default=None):
                if a and b:
                    return a[key] - b[key]
                return default

            row = {
                "game_id":    g["game_id"],
                "season_year": yr,
                "sr_id":       0,
                "game_date":   date,
                "away_code":   away,
                "home_code":   home,
                "home_win":    None,
                "home_elo":       h_elo,
                "away_elo":       a_elo,
                "diff_elo":       diff_elo,
                "elo_home_prob":  elo_prob,
                "diff_win_pct":  diff(h_stats, a_stats, "win_pct"),
                "diff_rs":       diff(h_stats, a_stats, "rs_per_game"),
                "diff_ra":       diff(h_stats, a_stats, "ra_per_game"),
                "diff_rd":       diff(h_stats, a_stats, "rd_per_game"),
                "diff_pyth_wp":  diff(h_stats, a_stats, "pyth_wp"),
                "diff_w3_win_pct":  diff(h3,  a3,  "win_pct"),
                "diff_w5_win_pct":  diff(h5,  a5,  "win_pct"),
                "diff_w10_win_pct": diff(h10, a10, "win_pct"),
                "diff_w3_rd_pg":    diff(h3,  a3,  "rd_per_game"),
                "diff_w5_rd_pg":    diff(h5,  a5,  "rd_per_game"),
                "diff_w10_rd_pg":   diff(h10, a10, "rd_per_game"),
                "diff_split5_win_pct":  diff(h_split5,  a_split5,  "win_pct"),
                "diff_split10_win_pct": diff(h_split10, a_split10, "win_pct"),
                "diff_split5_rd_pg":    diff(h_split5,  a_split5,  "rd_per_game"),
                "diff_split10_rd_pg":   diff(h_split10, a_split10, "rd_per_game"),
                "diff_split5_rs_pg":    diff(h_split5,  a_split5,  "rs_per_game"),
                "diff_split5_ra_pg":    diff(h_split5,  a_split5,  "ra_per_game"),
                "diff_split10_rs_pg":   diff(h_split10, a_split10, "rs_per_game"),
                "diff_split10_ra_pg":   diff(h_split10, a_split10, "ra_per_game"),
                "diff_trend_win_pct": diff(h5, a5, "win_pct") - diff(h10, a10, "win_pct")
                    if (h5 and a5 and h10 and a10) else None,
                "diff_trend_rd_pg":   diff(h5, a5, "rd_per_game") - diff(h10, a10, "rd_per_game")
                    if (h5 and a5 and h10 and a10) else None,
                "home_rest":    h_rest,
                "away_rest":    a_rest,
                "diff_rest":    h_rest - a_rest,
                "diff_streak":  h_streak - a_streak,
                "home_season_games_before": h_sg,
                "away_season_games_before": a_sg,
                "prev_diff_win_pct": (prev_h["win_pct"] - prev_a["win_pct"])
                    if (prev_h and prev_a) else None,
                "prev_diff_rd_pg":   (prev_h["rd_pg"] - prev_a["rd_pg"])
                    if (prev_h and prev_a) else None,
                "prev_diff_pyth":    (prev_h["pyth"] - prev_a["pyth"])
                    if (prev_h and prev_a) else None,
                "park_factor": park_factor,
                "stadium_hwa": stadium_hwa,
            }
            features.append(row)

    return features


# ── DB ────────────────────────────────────────────────────────────────────────

SCHEMA = """
DROP TABLE IF EXISTS game_features;
CREATE TABLE game_features (
    feature_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id      TEXT NOT NULL UNIQUE,
    season_year  INTEGER NOT NULL,
    sr_id        INTEGER NOT NULL,
    game_date    TEXT NOT NULL,
    away_code    TEXT NOT NULL,
    home_code    TEXT NOT NULL,
    home_win     INTEGER,

    home_elo REAL, away_elo REAL, diff_elo REAL, elo_home_prob REAL,

    diff_win_pct REAL, diff_rs REAL, diff_ra REAL, diff_rd REAL, diff_pyth_wp REAL,

    diff_w3_win_pct REAL, diff_w5_win_pct REAL, diff_w10_win_pct REAL,
    diff_w3_rd_pg   REAL, diff_w5_rd_pg   REAL, diff_w10_rd_pg   REAL,

    diff_split5_win_pct REAL, diff_split10_win_pct REAL,
    diff_split5_rd_pg   REAL, diff_split10_rd_pg   REAL,
    diff_split5_rs_pg   REAL, diff_split5_ra_pg    REAL,
    diff_split10_rs_pg  REAL, diff_split10_ra_pg   REAL,

    diff_trend_win_pct REAL, diff_trend_rd_pg REAL,

    home_rest INTEGER, away_rest INTEGER, diff_rest INTEGER,
    diff_streak INTEGER,
    home_season_games_before INTEGER, away_season_games_before INTEGER,

    prev_diff_win_pct REAL, prev_diff_rd_pg REAL, prev_diff_pyth REAL,
    park_factor REAL, stadium_hwa REAL,

    -- SP features (filled by build_kbo_pitcher_features.py)
    home_sp_era_roll  REAL, home_sp_whip_roll REAL,
    home_sp_k9_roll   REAL, home_sp_ip_roll   REAL,
    away_sp_era_roll  REAL, away_sp_whip_roll REAL,
    away_sp_k9_roll   REAL, away_sp_ip_roll   REAL,
    diff_sp_era  REAL, diff_sp_whip REAL, diff_sp_k9 REAL, diff_sp_ip REAL
);
CREATE INDEX IF NOT EXISTS idx_gf_date ON game_features(game_date);
CREATE INDEX IF NOT EXISTS idx_gf_year ON game_features(season_year);
"""


def write_features(conn, features: list[dict]):
    if not features:
        return
    cols = list(features[0].keys())
    ph = ",".join("?" * len(cols))
    conn.executemany(
        f"INSERT OR REPLACE INTO game_features ({','.join(cols)}) VALUES ({ph})",
        [tuple(f[c] for c in cols) for f in features],
    )
    conn.commit()


def main():
    parser = argparse.ArgumentParser(description="Build KBO game features")
    parser.add_argument(
        "--include-scheduled",
        action="store_true",
        help="Also generate features for today's scheduled (not yet played) games",
    )
    parser.add_argument(
        "--date",
        default=date_cls.today().isoformat(),
        help="Target date for scheduled games (default: today, YYYY-MM-DD)",
    )
    args = parser.parse_args()

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        print("載入比賽資料…")
        games = load_games(conn)
        print(f"  總場次（含季後賽，排除平局）: {len(games)}")
        park_context = build_park_context(conn)

        scheduled = None
        if args.include_scheduled:
            target_date = date_cls.fromisoformat(args.date)
            known_ids = {g["game_id"] for g in games}
            print(f"查詢 {target_date} 排程賽事（KBO API）…")
            scheduled = load_scheduled_games(target_date, known_ids)
            print(f"  找到 {len(scheduled)} 場排程賽事")
            for s in scheduled:
                print(f"    {s['game_id']}: {s['away_code']} @ {s['home_code']}")

        print("計算 pre-game features…")
        features = build_features(games, park_context, scheduled_games=scheduled)
        completed_cnt = sum(1 for f in features if f["home_win"] is not None)
        scheduled_cnt = sum(1 for f in features if f["home_win"] is None)
        print(f"  有效 game_features（sr_id=0）: 完成 {completed_cnt}，排程 {scheduled_cnt}")

        print("寫入 game_features…")
        conn.executescript(SCHEMA)
        write_features(conn, features)

        # 摘要
        mn, mx, cnt = conn.execute(
            "SELECT MIN(season_year), MAX(season_year), COUNT(*) FROM game_features"
        ).fetchone()
        print(f"  game_features: {cnt} 筆，{mn}–{mx} 年")

        home_win_rate = conn.execute(
            "SELECT ROUND(AVG(home_win),4) FROM game_features WHERE home_win IS NOT NULL"
        ).fetchone()[0]
        print(f"  整體主場勝率（in features）: {home_win_rate}")

        print(f"\n{'Year':<6} {'Games':>6} {'HomeWin%':>9}")
        for r in conn.execute("""
            SELECT season_year, COUNT(*), ROUND(AVG(home_win),3)
            FROM game_features WHERE home_win IS NOT NULL
            GROUP BY season_year ORDER BY season_year
        """).fetchall():
            print(f"  {r[0]:<6} {r[1]:>6} {r[2]:>9}")

        print(f"\n{'Stadium':<12} {'Games':>6} {'Park':>8} {'HWA':>8}")
        for stadium, games_cnt, park_factor, stadium_hwa in conn.execute("""
            SELECT tgr.stadium,
                   COUNT(gf.park_factor) AS games_cnt,
                   ROUND(AVG(gf.park_factor), 3) AS park_factor,
                   ROUND(AVG(gf.stadium_hwa), 3) AS stadium_hwa
            FROM game_features gf
            JOIN team_game_results tgr ON tgr.game_id = gf.game_id
            WHERE gf.home_win IS NOT NULL
              AND gf.park_factor IS NOT NULL
              AND tgr.stadium IS NOT NULL
              AND tgr.stadium != ''
            GROUP BY tgr.stadium
            ORDER BY park_factor DESC, tgr.stadium
        """).fetchall():
            print(f"  {stadium:<12} {games_cnt:>6} {park_factor:>8} {stadium_hwa:>8}")

    finally:
        conn.close()
    print("\n完成。接著執行 build_kbo_pitcher_features.py")


if __name__ == "__main__":
    main()
