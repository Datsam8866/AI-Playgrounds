from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import date, datetime
from pathlib import Path

from playsport_scraper import fetch_playsport_results

BASE_DIR = Path(__file__).resolve().parent

CPBL_DB = BASE_DIR / "CPBL" / "cpbl.sqlite"
KBO_DB = BASE_DIR / "KBO" / "kbo.sqlite"
MLB_DB = BASE_DIR / "MLB" / "mlb.sqlite"
NPB_DB = BASE_DIR / "NPB" / "npb.sqlite"

# When adding a CPBL team: also update
#   1. TEAM_NAME_MAPS["cpbl"] in playsport_scraper.py
#   2. TEAM_NAMES in CPBL/predict_today.py
#   3. This dict (CPBL_NAME_TO_CODE)
CPBL_NAME_TO_CODE = {
    "中信兄弟": "ACN011",
    "味全龍": "AAA011",
    "樂天桃猿": "AJL011",
    "富邦悍將": "AEO011",
    "統一獅": "ADD011",
    "台鋼雄鷹": "AKP011",
}

KBO_NAME_TO_CODE = {
    "LG 트윈스": "LG",
    "KT 위즈": "KT",
    "삼성 라이온즈": "SS",
    "NC 다이노스": "NC",
    "두산 베어스": "OB",
    "KIA 타이거즈": "HT",
    "롯데 자이언츠": "LT",
    "SSG 랜더스": "SK",
    "한화 이글스": "HH",
    "키움 히어로즈": "WO",
}

NPB_NAME_TO_CODE = {
    "巨人": "g",
    "中日": "d",
    "DeNA": "db",
    "阪神": "t",
    "広島": "c",
    "ヤクルト": "s",
    "SB": "h",
    "日ハム": "f",
    "オリ": "b",
    "楽天": "e",
    "西武": "l",
    "ロッテ": "m",
}

NPB_CL_TEAMS = {"g", "d", "db", "t", "c", "s"}
NPB_PL_TEAMS = {"h", "f", "b", "e", "l", "m"}


def _now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _calc_npb_league_code(home_code: str, away_code: str) -> str | None:
    if home_code in NPB_CL_TEAMS and away_code in NPB_CL_TEAMS:
        return "CL"
    if home_code in NPB_PL_TEAMS and away_code in NPB_PL_TEAMS:
        return "PL"
    if (home_code in NPB_CL_TEAMS and away_code in NPB_PL_TEAMS) or (
        home_code in NPB_PL_TEAMS and away_code in NPB_CL_TEAMS
    ):
        return "IL"
    return None


def _winner_from_scores(home_score: int | None, away_score: int | None) -> str | None:
    if home_score is None or away_score is None or home_score == away_score:
        return None
    return "home" if home_score > away_score else "vis"


def _home_win_from_scores(home_score: int | None, away_score: int | None) -> int | None:
    if home_score is None or away_score is None or home_score == away_score:
        return None
    return 1 if home_score > away_score else 0


def _summary(result: dict, inserted: int, updated: int, skipped: int) -> dict:
    return {
        "ok": result.get("ok", False),
        "reason": result.get("reason"),
        "fetched": len(result.get("games", [])),
        "inserted": inserted,
        "updated": updated,
        "skipped": skipped,
        "changed": bool(inserted or updated),
    }


def sync_cpbl_results(target_date: date) -> dict:
    result = fetch_playsport_results("cpbl", target_date)
    if not result["ok"]:
        return _summary(result, 0, 0, 0)

    conn = sqlite3.connect(CPBL_DB)
    inserted = updated = skipped = 0
    try:
        existing = {}
        for row in conn.execute(
            """
            SELECT game_result_id, game_sno, home_team_code, visiting_team_code,
                   home_score, visiting_score, game_status, game_status_text
            FROM team_game_results
            WHERE DATE(game_date) = ? AND kind_code = 'A'
            """,
            (target_date.isoformat(),),
        ):
            existing[(row[2], row[3])] = {
                "game_result_id": row[0],
                "game_sno": row[1],
                "home_score": row[4],
                "away_score": row[5],
                "game_status": row[6],
                "game_status_text": row[7],
            }

        next_sno = conn.execute(
            "SELECT COALESCE(MAX(game_sno), 0) FROM team_game_results WHERE season_year = ? AND kind_code = 'A'",
            (target_date.year,),
        ).fetchone()[0] or 0

        for game in result["games"]:
            home_code = CPBL_NAME_TO_CODE.get(game["home"])
            away_code = CPBL_NAME_TO_CODE.get(game["away"])
            if not home_code or not away_code:
                skipped += 1
                unknown = [n for n in (game["home"], game["away"]) if n not in CPBL_NAME_TO_CODE]
                print(f"[CPBL sync] WARNING: unknown team name(s) {unknown} — update CPBL_NAME_TO_CODE, TEAM_NAME_MAPS, and TEAM_NAMES", file=__import__('sys').stderr)
                continue

            new_status = 3 if game["is_final"] else 1
            new_status_text = "比賽結束" if game["is_final"] else (game["status_text"] or "未開賽")
            new_home_score = game["home_score"]
            new_away_score = game["away_score"]
            raw_json = json.dumps(game["raw"], ensure_ascii=False)
            key = (home_code, away_code)
            current = existing.get(key)

            if current is not None:
                if (
                    current["home_score"] == new_home_score
                    and current["away_score"] == new_away_score
                    and current["game_status"] == new_status
                    and (current["game_status_text"] or "") == new_status_text
                ):
                    skipped += 1
                    continue

                conn.execute(
                    """
                    UPDATE team_game_results
                    SET game_date = ?,
                        game_status = ?,
                        game_status_text = ?,
                        visiting_score = ?,
                        home_score = ?,
                        raw_json = ?,
                        created_at = ?
                    WHERE game_result_id = ?
                    """,
                    (
                        f"{target_date.isoformat()}T00:00:00",
                        new_status,
                        new_status_text,
                        new_away_score,
                        new_home_score,
                        raw_json,
                        _now_text(),
                        current["game_result_id"],
                    ),
                )
                updated += 1
                continue

            next_sno += 1
            conn.execute(
                """
                INSERT INTO team_game_results(
                    season_year, kind_code, game_sno, game_date, game_status, game_status_text,
                    visiting_team_code, home_team_code, visiting_score, home_score, raw_json, created_at
                )
                VALUES (?, 'A', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    target_date.year,
                    next_sno,
                    f"{target_date.isoformat()}T00:00:00",
                    new_status,
                    new_status_text,
                    away_code,
                    home_code,
                    new_away_score,
                    new_home_score,
                    raw_json,
                    _now_text(),
                ),
            )
            inserted += 1

        conn.commit()
    finally:
        conn.close()

    return _summary(result, inserted, updated, skipped)


def sync_kbo_results(target_date: date) -> dict:
    result = fetch_playsport_results("kbo", target_date)
    if not result["ok"]:
        return _summary(result, 0, 0, 0)

    conn = sqlite3.connect(KBO_DB)
    inserted = updated = skipped = 0
    try:
        existing = {}
        for row in conn.execute(
            """
            SELECT game_id, home_code, away_code, home_score, away_score, game_state, start_time
            FROM team_game_results
            WHERE game_date = ?
            """,
            (target_date.isoformat(),),
        ):
            existing[(row[1], row[2])] = {
                "game_id": row[0],
                "home_score": row[3],
                "away_score": row[4],
                "game_state": row[5],
                "start_time": row[6],
            }

        for game in result["games"]:
            home_code = KBO_NAME_TO_CODE.get(game["home"])
            away_code = KBO_NAME_TO_CODE.get(game["away"])
            if not home_code or not away_code:
                skipped += 1
                continue

            key = (home_code, away_code)
            current = existing.get(key)
            new_state = 3 if game["is_final"] else 1
            new_home_score = game["home_score"]
            new_away_score = game["away_score"]
            start_time = None
            raw_dateon = (game["raw"] or {}).get("dateon") or ""
            if len(raw_dateon) >= 16:
                start_time = raw_dateon[11:16]

            if current is not None:
                if (
                    current["home_score"] == new_home_score
                    and current["away_score"] == new_away_score
                    and current["game_state"] == new_state
                    and (current["start_time"] or "") == (start_time or "")
                ):
                    skipped += 1
                    continue

                conn.execute(
                    """
                    UPDATE team_game_results
                    SET away_score = ?,
                        home_score = ?,
                        game_state = ?,
                        start_time = COALESCE(?, start_time),
                        created_at = ?
                    WHERE game_id = ?
                    """,
                    (
                        new_away_score,
                        new_home_score,
                        new_state,
                        start_time,
                        _now_text(),
                        current["game_id"],
                    ),
                )
                updated += 1
                continue

            synthetic_game_id = f"{target_date.strftime('%Y%m%d')}{away_code}{home_code}0"
            conn.execute(
                """
                INSERT INTO team_game_results(
                    game_id, season_year, sr_id, game_date, away_code, home_code,
                    away_score, home_score, game_state, start_time, created_at
                )
                VALUES (?, ?, 0, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    synthetic_game_id,
                    target_date.year,
                    target_date.isoformat(),
                    away_code,
                    home_code,
                    new_away_score,
                    new_home_score,
                    new_state,
                    start_time,
                    _now_text(),
                ),
            )
            inserted += 1

        conn.commit()
    finally:
        conn.close()

    return _summary(result, inserted, updated, skipped)


def sync_npb_results(target_date: date) -> dict:
    result = fetch_playsport_results("npb", target_date)
    if not result["ok"]:
        return _summary(result, 0, 0, 0)

    conn = sqlite3.connect(NPB_DB)
    inserted = updated = skipped = 0
    try:
        existing = {}
        for row in conn.execute(
            """
            SELECT id, home_code, away_code, home_score, away_score, home_win, status, game_url
            FROM team_game_results
            WHERE game_date = ?
            """,
            (target_date.isoformat(),),
        ):
            existing[(row[1], row[2])] = {
                "id": row[0],
                "home_score": row[3],
                "away_score": row[4],
                "home_win": row[5],
                "status": row[6],
                "game_url": row[7],
            }

        for idx, game in enumerate(result["games"], start=1):
            home_code = NPB_NAME_TO_CODE.get(game["home"])
            away_code = NPB_NAME_TO_CODE.get(game["away"])
            if not home_code or not away_code:
                skipped += 1
                continue

            key = (home_code, away_code)
            current = existing.get(key)
            new_home_score = game["home_score"]
            new_away_score = game["away_score"]
            new_home_win = _home_win_from_scores(new_home_score, new_away_score) if game["is_final"] else None
            new_status = "completed" if game["is_final"] else "scheduled"

            if current is not None:
                if (
                    current["home_score"] == new_home_score
                    and current["away_score"] == new_away_score
                    and current["home_win"] == new_home_win
                    and (current["status"] or "") == new_status
                ):
                    skipped += 1
                    continue

                conn.execute(
                    """
                    UPDATE team_game_results
                    SET home_score = ?,
                        away_score = ?,
                        home_win = ?,
                        status = ?
                    WHERE id = ?
                    """,
                    (
                        new_home_score,
                        new_away_score,
                        new_home_win,
                        new_status,
                        current["id"],
                    ),
                )
                updated += 1
                continue

            synthetic_url = f"playsport/{target_date.strftime('%Y%m%d')}/{away_code}-{home_code}-{idx:02d}"
            conn.execute(
                """
                INSERT INTO team_game_results(
                    season_year, game_date, home_code, away_code, home_score, away_score,
                    home_win, league_code, stadium, win_pitcher, lose_pitcher, game_url, status
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, ?, ?)
                """,
                (
                    target_date.year,
                    target_date.isoformat(),
                    home_code,
                    away_code,
                    new_home_score,
                    new_away_score,
                    new_home_win,
                    _calc_npb_league_code(home_code, away_code),
                    synthetic_url,
                    new_status,
                ),
            )
            inserted += 1

        conn.commit()
    finally:
        conn.close()

    return _summary(result, inserted, updated, skipped)


def sync_mlb_results(target_date: date) -> dict:
    result = fetch_playsport_results("mlb", target_date)
    if not result["ok"]:
        return _summary(result, 0, 0, 0)

    conn = sqlite3.connect(MLB_DB)
    inserted = updated = skipped = 0
    try:
        team_id_by_name = {}
        for team_id, team_name in conn.execute(
            """
            SELECT home_team_id, home_team_name FROM team_game_results
            UNION
            SELECT vis_team_id, vis_team_name FROM team_game_results
            """
        ):
            if team_id is not None and team_name:
                team_id_by_name[team_name] = team_id

        existing = {}
        for row in conn.execute(
            """
            SELECT game_pk, home_team_name, vis_team_name,
                   home_score, vis_score, status, winner
            FROM team_game_results
            WHERE game_date = ?
            """,
            (target_date.isoformat(),),
        ):
            existing[(row[1], row[2])] = {
                "game_pk": row[0],
                "home_score": row[3],
                "away_score": row[4],
                "status": row[5],
                "winner": row[6],
            }

        for game in result["games"]:
            home_name = game["home"]
            away_name = game["away"]
            key = (home_name, away_name)
            current = existing.get(key)
            new_home_score = game["home_score"]
            new_away_score = game["away_score"]
            new_status = "Final" if game["is_final"] else (game["status_text"] or "Scheduled")
            new_winner = _winner_from_scores(new_home_score, new_away_score) if game["is_final"] else None
            game_pk = current["game_pk"] if current is not None else int(game["source_id"])

            if current is not None:
                if (
                    current["home_score"] == new_home_score
                    and current["away_score"] == new_away_score
                    and (current["status"] or "") == new_status
                    and current["winner"] == new_winner
                ):
                    skipped += 1
                    continue

                conn.execute(
                    """
                    UPDATE team_game_results
                    SET home_score = ?,
                        vis_score = ?,
                        status = ?,
                        winner = ?
                    WHERE game_pk = ?
                    """,
                    (
                        new_home_score,
                        new_away_score,
                        new_status,
                        new_winner,
                        game_pk,
                    ),
                )
                updated += 1
                continue

            conn.execute(
                """
                INSERT INTO team_game_results(
                    season_year, game_pk, game_date, home_team_id, home_team_name,
                    vis_team_id, vis_team_name, home_score, vis_score, game_type, status, winner
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'R', ?, ?)
                """,
                (
                    target_date.year,
                    game_pk,
                    target_date.isoformat(),
                    team_id_by_name.get(home_name),
                    home_name,
                    team_id_by_name.get(away_name),
                    away_name,
                    new_home_score,
                    new_away_score,
                    new_status,
                    new_winner,
                ),
            )
            inserted += 1

        conn.commit()
    finally:
        conn.close()

    return _summary(result, inserted, updated, skipped)


def sync_results_for_league(league: str, target_date: date) -> dict:
    league = league.lower()
    if league == "cpbl":
        return sync_cpbl_results(target_date)
    if league == "kbo":
        return sync_kbo_results(target_date)
    if league == "mlb":
        return sync_mlb_results(target_date)
    if league == "npb":
        return sync_npb_results(target_date)
    return {"ok": False, "reason": "unknown_league", "fetched": 0, "inserted": 0, "updated": 0, "skipped": 0, "changed": False}


def _parse_args():
    parser = argparse.ArgumentParser(description="Sync daily baseball results from playsport livescore into league sqlite DBs.")
    parser.add_argument("--league", required=True, help="cpbl, kbo, mlb, npb, or comma-separated list")
    parser.add_argument("--date", default=date.today().isoformat(), help="Target date YYYY-MM-DD")
    return parser.parse_args()


def main():
    args = _parse_args()
    target_date = date.fromisoformat(args.date)
    leagues = [part.strip().lower() for part in args.league.split(",") if part.strip()]

    for league in leagues:
        summary = sync_results_for_league(league, target_date)
        print(
            f"{league} ok={summary['ok']} fetched={summary['fetched']} "
            f"inserted={summary['inserted']} updated={summary['updated']} skipped={summary['skipped']} "
            f"changed={summary['changed']} reason={summary['reason']}"
        )


if __name__ == "__main__":
    main()
