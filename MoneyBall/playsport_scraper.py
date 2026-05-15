"""
Scrape daily baseball schedule, starting pitchers, and moneyline odds
from playsport.cc for the dashboard pipeline.

This source only exposes relative day pages (`yesterday`, `today`, `tomorrow`),
so historical backfill dates outside that window intentionally return no data.
"""

from __future__ import annotations

import json
import re
from datetime import date, datetime
from zoneinfo import ZoneInfo

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.playsport.cc/predict/games"
LIVESCORE_SERVERS = (
    "https://ls6.playsport.cc",
    "https://ls7.playsport.cc",
)
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

ALLIANCE_IDS = {
    "mlb": 1,
    "npb": 2,
    "nba": 3,
    "cpbl": 6,
    "kbo": 9,
}

TEAM_NAME_MAPS: dict[str, dict[str, str]] = {
    "mlb": {
        "響尾蛇": "Arizona Diamondbacks",
        "勇士": "Atlanta Braves",
        "金鶯": "Baltimore Orioles",
        "紅襪": "Boston Red Sox",
        "小熊": "Chicago Cubs",
        "白襪": "Chicago White Sox",
        "紅人": "Cincinnati Reds",
        "守護者": "Cleveland Guardians",
        "落磯": "Colorado Rockies",
        "老虎": "Detroit Tigers",
        "太空人": "Houston Astros",
        "皇家": "Kansas City Royals",
        "天使": "Los Angeles Angels",
        "道奇": "Los Angeles Dodgers",
        "馬林魚": "Miami Marlins",
        "釀酒人": "Milwaukee Brewers",
        "雙城": "Minnesota Twins",
        "大都會": "New York Mets",
        "洋基": "New York Yankees",
        "運動家": "Oakland Athletics",
        "費城人": "Philadelphia Phillies",
        "海盜": "Pittsburgh Pirates",
        "教士": "San Diego Padres",
        "巨人": "San Francisco Giants",
        "水手": "Seattle Mariners",
        "紅雀": "St. Louis Cardinals",
        "光芒": "Tampa Bay Rays",
        "遊騎兵": "Texas Rangers",
        "藍鳥": "Toronto Blue Jays",
        "國民": "Washington Nationals",
    },
    "npb": {
        "西武": "西武",
        "樂天": "楽天",
        "火腿": "日ハム",
        "歐力士": "オリ",
        "歐力士猛牛": "オリ",
        "養樂多": "ヤクルト",
        "中日": "中日",
        "巨人": "巨人",
        "橫濱": "DeNA",
        "DeNA": "DeNA",
        "廣島": "広島",
        "阪神": "阪神",
        "羅德": "ロッテ",
        "軟銀": "SB",
        "日本火腿": "日ハム",
        "日本火腿鬥士": "日ハム",
        "千葉羅德": "ロッテ",
        "福岡軟銀": "SB",
        "福岡軟銀鷹": "SB",
        "東北樂天": "楽天",
        "東京養樂多": "ヤクルト",
        "橫濱DeNA": "DeNA",
    },
    "cpbl": {
        "統一": "統一獅",
        "統一獅": "統一獅",
        "味全": "味全龍",
        "味全龍": "味全龍",
        "台鋼": "台鋼雄鷹",
        "台鋼雄鷹": "台鋼雄鷹",
        "富邦": "富邦悍將",
        "富邦悍將": "富邦悍將",
        "樂天": "樂天桃猿",
        "樂天桃猿": "樂天桃猿",
        "兄弟": "中信兄弟",
        "中信兄弟": "中信兄弟",
    },
    "nba": {
        "塞爾蒂克": "BOS",
        "籃網": "BKN",
        "尼克": "NYK",
        "七六人": "PHI",
        "暴龍": "TOR",
        "公牛": "CHI",
        "騎士": "CLE",
        "活塞": "DET",
        "溜馬": "IND",
        "步行者": "IND",
        "公鹿": "MIL",
        "雄鹿": "MIL",
        "鷹": "ATL",
        "老鷹": "ATL",
        "黃蜂": "CHA",
        "熱火": "MIA",
        "魔術": "ORL",
        "奇才": "WAS",
        "巫師": "WAS",
        "金塊": "DEN",
        "灰狼": "MIN",
        "雷霆": "OKC",
        "開拓者": "POR",
        "拓荒者": "POR",
        "爵士": "UTA",
        "勇士": "GSW",
        "快艇": "LAC",
        "湖人": "LAL",
        "太陽": "PHX",
        "國王": "SAC",
        "獨行俠": "DAL",
        "小牛": "DAL",
        "火箭": "HOU",
        "灰熊": "MEM",
        "鵜鶘": "NOP",
        "馬刺": "SAS",
    },
    "kbo": {
        "三星獅": "삼성 라이온즈",
        "三星": "삼성 라이온즈",
        "斗山熊": "두산 베어스",
        "斗山": "두산 베어스",
        "熊": "두산 베어스",
        "培證英雄": "키움 히어로즈",
        "培證": "키움 히어로즈",
        "樂天巨人": "롯데 자이언츠",
        "樂天": "롯데 자이언츠",
        "起亞老虎": "KIA 타이거즈",
        "起亞虎": "KIA 타이거즈",
        "NC恐龍": "NC 다이노스",
        "恐龍": "NC 다이노스",
        "LG雙子": "LG 트윈스",
        "雙子": "LG 트윈스",
        "KT巫師": "KT 위즈",
        "巫師": "KT 위즈",
        "SSG登陸者": "SSG 랜더스",
        "登陸者": "SSG 랜더스",
        "韓華老鷹": "한화 이글스",
        "華老鷹": "한화 이글스",
    },
}


def _normalize_text(value: str | None) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def _parse_decimal(value: str) -> float | None:
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)", value)
    return float(match.group(1)) if match else None


def _parse_int(value) -> int | None:
    if value in (None, "", "-"):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _resolve_gameday(target_date: date) -> str | None:
    today_tw = datetime.now(ZoneInfo("Asia/Taipei")).date()
    delta = (target_date - today_tw).days
    if delta == -1:
        return "yesterday"
    if delta == 0:
        return "today"
    if delta == 1:
        return "tomorrow"
    return None


def _parse_team_block(node, league: str) -> dict | None:
    name_node = node.select_one("a") or node.select_one("h3") or node
    raw_team = _normalize_text(name_node.get_text(" ", strip=True) if name_node else "")
    if not raw_team:
        return None

    pitcher_node = node.select_one("p")
    pitcher = _normalize_text(pitcher_node.get_text(" ", strip=True) if pitcher_node else "") or "TBD"
    mapped_team = TEAM_NAME_MAPS.get(league, {}).get(raw_team, raw_team)
    return {
        "raw_team": raw_team,
        "team": mapped_team,
        "pitcher": pitcher,
    }


def _extract_team_entries(pair, league: str) -> list[dict]:
    first_cell = pair[0].select_one("td.td-teaminfo")
    if first_cell is not None and (
        first_cell.get("rowspan") == "2"
        or len(first_cell.select('a[href*="teams?"]')) >= 2
    ):
        blocks = first_cell.select("td.winnerteam, td.secondteam")
        if not blocks:
            blocks = first_cell.select("tr td") or [first_cell]

        entries = []
        for block in blocks:
            team = _parse_team_block(block, league)
            if team:
                entries.append(team)
        return entries[:2]

    entries = []
    for tr in pair:
        team_cell = tr.select_one("td.td-teaminfo")
        if team_cell is None:
            continue
        team = _parse_team_block(team_cell, league)
        if team:
            entries.append(team)
    return entries[:2]


def _extract_row_meta(tr) -> dict:
    ml_cell = tr.select_one("td.td-bank-bet03")
    side_node = ml_cell.select_one("strong.team-side") if ml_cell else tr.select_one("td.td-universal-bet01 strong.team-side")
    side = _normalize_text(side_node.get_text(" ", strip=True) if side_node else "")
    odds = _parse_decimal(ml_cell.get_text(" ", strip=True)) if ml_cell is not None else None
    return {"side": side, "odds": odds}


def fetch_playsport_games(league: str, target_date: date, timeout: int = 20) -> dict:
    """
    Return:
      {
        "ok": bool,
        "reason": str | None,
        "games": [
          {
            "source_id": str,
            "home": str,
            "away": str,
            "home_sp": str,
            "away_sp": str,
            "odds_home": float | None,
            "odds_away": float | None,
          }
        ]
      }
    """
    league = league.lower()
    alliance_id = ALLIANCE_IDS.get(league)
    if alliance_id is None:
        return {"ok": False, "reason": "unknown_league", "games": []}

    gameday = _resolve_gameday(target_date)
    if gameday is None:
        return {"ok": False, "reason": "unsupported_gameday", "games": []}

    response = requests.get(
        BASE_URL,
        params={"allianceid": alliance_id, "gameday": gameday},
        headers={"User-Agent": USER_AGENT},
        timeout=timeout,
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    rows = soup.select("table.predictgame-table tr[gameid]")

    games = []
    idx = 0
    while idx < len(rows):
        game_id = rows[idx].get("gameid", "")
        pair = [rows[idx]]
        idx += 1
        while idx < len(rows) and rows[idx].get("gameid", "") == game_id and len(pair) < 2:
            pair.append(rows[idx])
            idx += 1

        teams = _extract_team_entries(pair, league)
        row_meta = [_extract_row_meta(tr) for tr in pair[:2]]
        if len(teams) < 2 or len(row_meta) < 2:
            continue

        team_rows = [
            {**teams[i], **row_meta[i]}
            for i in range(min(len(teams), len(row_meta)))
        ]
        if len(team_rows) < 2:
            continue

        away = next((t for t in team_rows if t["side"] == "客"), team_rows[0])
        home = next((t for t in team_rows if t["side"] == "主"), team_rows[-1])
        games.append({
            "source_id": game_id,
            "home": home["team"],
            "away": away["team"],
            "home_sp": home["pitcher"],
            "away_sp": away["pitcher"],
            "odds_home": home["odds"],
            "odds_away": away["odds"],
        })

    return {"ok": True, "reason": None, "games": games}


def fetch_playsport_game_map(league: str, target_date: date, timeout: int = 20) -> dict:
    result = fetch_playsport_games(league, target_date, timeout=timeout)
    result["game_map"] = {
        (g["home"], g["away"]): g
        for g in result["games"]
    }
    return result


def _parse_jsonp_payload(text: str) -> dict:
    match = re.match(r"^\?\((.*)\)\s*$", text, re.S)
    if not match:
        raise ValueError("unexpected_jsonp")
    return json.loads(match.group(1))


def _is_final_status(status_text: str, away_score: int | None, home_score: int | None) -> bool:
    normalized = _normalize_text(status_text).lower()
    if away_score is None or home_score is None:
        return False
    final_markers = ("final", "結束", "終了", "比賽結束")
    return any(marker.lower() in normalized for marker in final_markers)


def fetch_playsport_results(league: str, target_date: date, timeout: int = 20) -> dict:
    """
    Return standardized result rows from playsport livescore JSON.

    Shape:
      {
        "ok": bool,
        "reason": str | None,
        "games": [
          {
            "official_id": str | None,
            "source_id": str | None,
            "date": "YYYY-MM-DD",
            "home": str,
            "away": str,
            "home_score": int | None,
            "away_score": int | None,
            "status_text": str,
            "is_final": bool,
            "home_sp": str | None,
            "away_sp": str | None,
            "raw": dict,
          }
        ]
      }
    """
    league = league.lower()
    alliance_id = ALLIANCE_IDS.get(league)
    if alliance_id is None:
        return {"ok": False, "reason": "unknown_league", "games": []}

    gamedate = target_date.strftime("%Y%m%d")
    params = {
        "alliance": alliance_id,
        "gamedate": gamedate,
        "pbp": 1,
        "teamStat": 1,
        "oid": "",
        "xcb": "?",
    }
    headers = {
        "User-Agent": USER_AGENT,
        "Referer": f"https://www.playsport.cc/livescore/{alliance_id}?gamedate={gamedate}&mode=1&",
    }

    last_error = None
    for server in LIVESCORE_SERVERS:
        try:
            response = requests.get(
                f"{server}/ls_json.php",
                params=params,
                headers=headers,
                timeout=timeout,
            )
            response.raise_for_status()
            payload = _parse_jsonp_payload(response.text)
            games = []
            for value in payload.values():
                if not isinstance(value, dict) or not value.get("gameid"):
                    continue

                mapped_away = TEAM_NAME_MAPS.get(league, {}).get(value.get("aname"), value.get("aname"))
                mapped_home = TEAM_NAME_MAPS.get(league, {}).get(value.get("hname"), value.get("hname"))
                scores = value.get("r") or []
                away_score = _parse_int(scores[0]) if len(scores) >= 1 else None
                home_score = _parse_int(scores[1]) if len(scores) >= 2 else None
                status_text = _normalize_text(((value.get("gs") or {}).get("ss")) or value.get("ss") or "")

                games.append({
                    "official_id": value.get("official_id"),
                    "source_id": value.get("gameid"),
                    "date": target_date.isoformat(),
                    "home": mapped_home,
                    "away": mapped_away,
                    "home_score": home_score,
                    "away_score": away_score,
                    "status_text": status_text,
                    "is_final": _is_final_status(status_text, away_score, home_score),
                    "home_sp": _normalize_text(value.get("homepitcher") or "") or None,
                    "away_sp": _normalize_text(value.get("visitpitcher") or "") or None,
                    "raw": value,
                })

            return {"ok": True, "reason": None, "games": games}
        except Exception as exc:
            last_error = exc

    return {"ok": False, "reason": str(last_error) if last_error else "fetch_failed", "games": []}
