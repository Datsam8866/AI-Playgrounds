"""
Update NPB starting-pitcher rolling features in game_features_npb.

Run this after build_game_features_npb.py. Pitcher features are computed from
same-season prior starts only, using the latest 10 starts before each game.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path


DB_PATH = Path(__file__).resolve().parent / "npb.sqlite"
WINDOW = 10
MIN_STARTS = 5


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


@dataclass(frozen=True)
class StarterLine:
    season_year: int
    game_date: date
    game_url: str
    team_code: str
    pitcher_name: str | None
    ip_outs: int
    hits: int
    bb: int
    strikeouts: int
    earned_runs: int


def parse_date(value: str) -> date:
    return date.fromisoformat(value[:10])


def safe_int(value: int | None) -> int:
    return 0 if value is None else int(value)


def rolling_sp_stats(history: list[dict]) -> dict | None:
    subset = history[-WINDOW:]
    if len(subset) < MIN_STARTS:
        return None
    total_outs = sum(g["ip_outs"] for g in subset)
    if total_outs <= 0:
        return None
    innings = total_outs / 3.0
    earned_runs = sum(g["earned_runs"] for g in subset)
    hits = sum(g["hits"] for g in subset)
    walks = sum(g["bb"] for g in subset)
    strikeouts = sum(g["strikeouts"] for g in subset)
    return {
        "era": earned_runs / innings * 9.0,
        "whip": (hits + walks) / innings,
        "k9": strikeouts / innings * 9.0,
        "starts": len(subset),
    }


def load_starters(conn: sqlite3.Connection) -> list[StarterLine]:
    rows = conn.execute(
        """
        SELECT
            gsp.season_year,
            tgr.game_date,
            gsp.game_url,
            gsp.team_code,
            gsp.pitcher_name,
            gsp.ip_outs,
            gsp.hits,
            gsp.bb,
            gsp.strikeouts,
            gsp.earned_runs
        FROM game_starting_pitchers gsp
        JOIN team_game_results tgr
          ON tgr.game_url = gsp.game_url
        WHERE tgr.status = 'completed'
          AND tgr.home_score IS NOT NULL
          AND tgr.away_score IS NOT NULL
        ORDER BY gsp.season_year ASC, tgr.game_date ASC, gsp.game_url ASC, gsp.team_code ASC
        """,
    ).fetchall()
    return [
        StarterLine(
            season_year=row["season_year"],
            game_date=parse_date(row["game_date"]),
            game_url=row["game_url"],
            team_code=row["team_code"],
            pitcher_name=row["pitcher_name"],
            ip_outs=safe_int(row["ip_outs"]),
            hits=safe_int(row["hits"]),
            bb=safe_int(row["bb"]),
            strikeouts=safe_int(row["strikeouts"]),
            earned_runs=safe_int(row["earned_runs"]),
        )
        for row in rows
    ]


def reset_target_rows(conn: sqlite3.Connection, target_year: int | None) -> None:
    if target_year is None:
        conn.execute(
            """
            UPDATE game_features_npb
            SET diff_sp_era = NULL,
                diff_sp_whip = NULL,
                diff_sp_k9 = NULL,
                sp_available = 0
            """
        )
    else:
        conn.execute(
            """
            UPDATE game_features_npb
            SET diff_sp_era = NULL,
                diff_sp_whip = NULL,
                diff_sp_k9 = NULL,
                sp_available = 0
            WHERE season_year = ?
            """,
            (target_year,),
        )
    conn.commit()


def update_row(
    conn: sqlite3.Connection,
    game_url: str,
    home_stats: dict | None,
    away_stats: dict | None,
) -> bool:
    if home_stats and away_stats:
        values = (
            away_stats["era"] - home_stats["era"],
            away_stats["whip"] - home_stats["whip"],
            home_stats["k9"] - away_stats["k9"],
            1,
            game_url,
        )
    else:
        values = (None, None, None, 0, game_url)
    cur = conn.execute(
        """
        UPDATE game_features_npb
        SET diff_sp_era = ?,
            diff_sp_whip = ?,
            diff_sp_k9 = ?,
            sp_available = ?
        WHERE game_url = ?
        """,
        values,
    )
    return cur.rowcount > 0


def build_pitcher_features(conn: sqlite3.Connection, target_year: int | None) -> tuple[int, int]:
    starters = load_starters(conn)
    by_game: dict[str, list[StarterLine]] = defaultdict(list)
    game_order: dict[str, tuple[int, date]] = {}
    for starter in starters:
        by_game[starter.game_url].append(starter)
        game_order[starter.game_url] = (starter.season_year, starter.game_date)

    # Key by pitcher_name only (cross-season) so career history carries forward
    pitcher_history: dict[str, list[dict]] = defaultdict(list)
    updated_rows = 0
    available_rows = 0

    for game_url in sorted(by_game, key=lambda url: (game_order[url][0], game_order[url][1], url)):
        game_year = game_order[game_url][0]
        lines = by_game[game_url]
        by_side = {line.team_code: line for line in lines}
        home_line = by_side.get("home")
        away_line = by_side.get("away")

        home_stats = (
            rolling_sp_stats(pitcher_history[home_line.pitcher_name])
            if home_line and home_line.pitcher_name
            else None
        )
        away_stats = (
            rolling_sp_stats(pitcher_history[away_line.pitcher_name])
            if away_line and away_line.pitcher_name
            else None
        )

        if target_year is None or game_year == target_year:
            if update_row(conn, game_url, home_stats, away_stats):
                updated_rows += 1
                if home_stats and away_stats:
                    available_rows += 1

        for line in lines:
            if not line.pitcher_name:
                continue
            pitcher_history[line.pitcher_name].append(
                {
                    "ip_outs": line.ip_outs,
                    "hits": line.hits,
                    "bb": line.bb,
                    "strikeouts": line.strikeouts,
                    "earned_runs": line.earned_runs,
                }
            )

    conn.commit()
    return updated_rows, available_rows


def print_summary(
    conn: sqlite3.Connection,
    target_year: int | None,
    updated_rows: int,
    available_rows: int,
) -> None:
    if target_year is None:
        total, available = conn.execute(
            """
            SELECT COUNT(*), SUM(CASE WHEN sp_available = 1 THEN 1 ELSE 0 END)
            FROM game_features_npb
            """
        ).fetchone()
    else:
        total, available = conn.execute(
            """
            SELECT COUNT(*), SUM(CASE WHEN sp_available = 1 THEN 1 ELSE 0 END)
            FROM game_features_npb
            WHERE season_year = ?
            """,
            (target_year,),
        ).fetchone()
    available = available or 0
    pct = 100.0 * available / total if total else 0.0
    print(f"SP game rows matched in game_features_npb: {updated_rows}")
    print(f"SP rows available during replay: {available_rows}")
    print(f"game_features_npb rows in scope: {total}")
    print(f"sp_available=1: {available} ({pct:.1f}%)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update NPB pitcher rolling features.")
    parser.add_argument("--year", type=int, help="Update only this season year.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        reset_target_rows(conn, args.year)
        updated_rows, available_rows = build_pitcher_features(conn, args.year)
        print_summary(conn, args.year, updated_rows, available_rows)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
