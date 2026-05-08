"""
Build NPB pre-game team features into game_features_npb.

The script replays games chronologically and captures feature snapshots before
each completed game is used to update team state. With --year, prior seasons are
still replayed so Elo, rest, and prior history are correct for the target year.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable


DB_PATH = Path(__file__).resolve().parent / "npb.sqlite"

CL_TEAMS = {"g", "d", "db", "t", "c", "s"}
PL_TEAMS = {"h", "f", "b", "e", "l", "m"}
TEAM_CODES = CL_TEAMS | PL_TEAMS
TEAM_CODE_MAP = {
    "bs": "b",
}

TEAM_WINDOW = 20
TEAM_MIN_GAMES = 10
SP_WINDOW = 10
SP_MIN_STARTS = 3
ELO_K = 52.0
ELO_HOME_ADVANTAGE = 10.0
ELO_REGRESSION = 0.45
TRAIN_START_YEAR = 2011
VALID_LEAGUES = {"CL", "PL", "IL"}

FEATURE_COLUMNS = [
    "game_url",
    "season_year",
    "game_date",
    "home_code",
    "away_code",
    "league_code",
    "is_interleague",
    "home_elo",
    "vis_elo",
    "diff_elo",
    "home_win_pct",
    "vis_win_pct",
    "diff_win_pct",
    "home_rs_pg",
    "vis_rs_pg",
    "home_ra_pg",
    "vis_ra_pg",
    "diff_rd_pg",
    "diff_pyth_wp",
    "diff_w5_win_pct",
    "diff_w10_win_pct",
    "diff_w5_rd_pg",
    "diff_w10_rd_pg",
    "home_streak",
    "vis_streak",
    "diff_streak",
    "home_rest",
    "vis_rest",
    "diff_rest",
    "home_season_games_before",
    "vis_season_games_before",
    "home_sp_fip_roll",
    "vis_sp_fip_roll",
    "diff_sp_fip",
    "diff_sp_era",
    "diff_sp_whip",
    "diff_sp_k9",
    "sp_available",
    "home_win",
]


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


@dataclass(frozen=True)
class Game:
    season_year: int
    game_date: date
    home_code: str
    away_code: str
    home_score: int | None
    away_score: int | None
    home_win: int | None
    league_code: str | None
    game_url: str
    status: str


@dataclass(frozen=True)
class StartingPitcherStart:
    season_year: int
    game_date: date
    game_url: str
    team_code: str
    pitcher_name: str
    ip_outs: int
    hits: int
    hr: int
    bb: int
    hbp: int
    strikeouts: int
    earned_runs: int


def normalize_team(code: str | None) -> str | None:
    if code is None:
        return None
    return TEAM_CODE_MAP.get(code.lower(), code.lower())


def league_for(home_code: str | None, away_code: str | None, raw_league: str | None) -> str | None:
    if raw_league in VALID_LEAGUES:
        return raw_league
    if home_code in CL_TEAMS and away_code in CL_TEAMS:
        return "CL"
    if home_code in PL_TEAMS and away_code in PL_TEAMS:
        return "PL"
    if home_code in TEAM_CODES and away_code in TEAM_CODES:
        return "IL"
    return None


def parse_date(value: str) -> date:
    return date.fromisoformat(value[:10])


def pythagorean_wp(rs: float, ra: float) -> float:
    if rs <= 0 and ra <= 0:
        return 0.5
    exp = 1.83
    rs_exp = rs**exp
    ra_exp = ra**exp
    denom = rs_exp + ra_exp
    return 0.5 if denom == 0 else rs_exp / denom


def summarize(history: list[dict], window: int, min_games: int) -> dict | None:
    subset = history[-window:]
    n = len(subset)
    if n < min_games:
        return None
    rs = sum(g["rs"] for g in subset)
    ra = sum(g["ra"] for g in subset)
    wins = sum(g["win"] for g in subset)
    return {
        "n": n,
        "win_pct": wins / n,
        "rs_pg": rs / n,
        "ra_pg": ra / n,
        "rd_pg": (rs - ra) / n,
        "pyth_wp": pythagorean_wp(rs, ra),
    }


def streak_value(history: list[dict]) -> int:
    if not history:
        return 0
    last = history[-1]["win"]
    if last == 0.5:
        return 0
    sign = 1 if last == 1.0 else -1
    streak = 0
    for game in reversed(history):
        if game["win"] == 0.5:
            break
        current_sign = 1 if game["win"] == 1.0 else -1
        if current_sign != sign:
            break
        streak += current_sign
    return streak


def rest_days(last_game_date: date | None, game_date: date) -> int:
    if last_game_date is None:
        return 10
    return max(0, min(10, (game_date - last_game_date).days))


def safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def calc_era(ip_outs: int, earned_runs: int) -> float | None:
    return safe_ratio(27.0 * earned_runs, ip_outs)


def calc_whip(ip_outs: int, hits: int, bb: int) -> float | None:
    return safe_ratio(3.0 * (hits + bb), ip_outs)


def calc_k9(ip_outs: int, strikeouts: int) -> float | None:
    return safe_ratio(27.0 * strikeouts, ip_outs)


def calc_fip(ip_outs: int, hr: int, bb: int, hbp: int, strikeouts: int, fip_constant: float | None) -> float | None:
    if fip_constant is None:
        return None
    innings = ip_outs / 3.0
    if innings <= 0:
        return None
    return ((13.0 * hr) + (3.0 * (bb + hbp)) - (2.0 * strikeouts)) / innings + fip_constant


def summarize_pitcher_history(
    history: list[StartingPitcherStart],
    fip_constant: float | None,
    window: int = SP_WINDOW,
    min_starts: int = SP_MIN_STARTS,
) -> dict | None:
    subset = history[-window:]
    if len(subset) < min_starts:
        return None
    ip_outs = sum(start.ip_outs for start in subset)
    if ip_outs <= 0:
        return None
    hits = sum(start.hits for start in subset)
    hr = sum(start.hr for start in subset)
    bb = sum(start.bb for start in subset)
    hbp = sum(start.hbp for start in subset)
    strikeouts = sum(start.strikeouts for start in subset)
    earned_runs = sum(start.earned_runs for start in subset)
    return {
        "starts": len(subset),
        "era": calc_era(ip_outs, earned_runs),
        "whip": calc_whip(ip_outs, hits, bb),
        "k9": calc_k9(ip_outs, strikeouts),
        "fip": calc_fip(ip_outs, hr, bb, hbp, strikeouts, fip_constant),
    }


def starter_team_code(team_side: str, home_code: str, away_code: str) -> str | None:
    if team_side == "home":
        return home_code
    if team_side == "away":
        return away_code
    return None


def is_valid_pitcher_name(name: str | None) -> bool:
    if name is None:
        return False
    normalized = name.strip()
    return bool(normalized) and not normalized.isdigit()


def compute_fip_constants_by_season(starts: list[StartingPitcherStart]) -> dict[int, float | None]:
    season_totals: dict[int, dict[str, int]] = defaultdict(
        lambda: {"ip_outs": 0, "earned_runs": 0, "hr": 0, "bb": 0, "hbp": 0, "strikeouts": 0}
    )
    for start in starts:
        if start.ip_outs <= 0:
            continue
        totals = season_totals[start.season_year]
        totals["ip_outs"] += start.ip_outs
        totals["earned_runs"] += start.earned_runs
        totals["hr"] += start.hr
        totals["bb"] += start.bb
        totals["hbp"] += start.hbp
        totals["strikeouts"] += start.strikeouts

    cumulative = {"ip_outs": 0, "earned_runs": 0, "hr": 0, "bb": 0, "hbp": 0, "strikeouts": 0}
    constants: dict[int, float | None] = {}
    for season_year in sorted(season_totals):
        ip_outs = cumulative["ip_outs"]
        if ip_outs <= 0:
            constants[season_year] = None
        else:
            innings = ip_outs / 3.0
            league_era = (27.0 * cumulative["earned_runs"]) / ip_outs
            fip_core = (
                (13.0 * cumulative["hr"])
                + (3.0 * (cumulative["bb"] + cumulative["hbp"]))
                - (2.0 * cumulative["strikeouts"])
            ) / innings
            constants[season_year] = league_era - fip_core
        for key, value in season_totals[season_year].items():
            cumulative[key] += value
    return constants


def load_starting_pitchers(
    conn: sqlite3.Connection,
) -> tuple[dict[str, dict[str, StartingPitcherStart]], dict[int, float | None]]:
    rows = conn.execute(
        """
        SELECT gsp.season_year,
               tgr.game_date,
               gsp.game_url,
               gsp.team_code AS team_side,
               tgr.home_code,
               tgr.away_code,
               tgr.status,
               tgr.league_code,
               gsp.pitcher_name,
               gsp.ip_outs,
               gsp.hits,
               gsp.hr,
               gsp.bb,
               gsp.hbp,
               gsp.strikeouts,
               gsp.earned_runs
        FROM game_starting_pitchers gsp
        JOIN team_game_results tgr
          ON tgr.game_url = gsp.game_url
        ORDER BY tgr.game_date ASC, gsp.game_url ASC, gsp.team_code ASC
        """
    ).fetchall()

    starters_by_game: dict[str, dict[str, StartingPitcherStart]] = defaultdict(dict)
    completed_starts: list[StartingPitcherStart] = []
    for row in rows:
        home_code = normalize_team(row["home_code"])
        away_code = normalize_team(row["away_code"])
        team_code = starter_team_code(row["team_side"], home_code or "", away_code or "")
        if team_code not in TEAM_CODES:
            continue
        league_code = league_for(home_code, away_code, row["league_code"])
        if league_code not in VALID_LEAGUES:
            continue
        if not is_valid_pitcher_name(row["pitcher_name"]):
            continue
        start = StartingPitcherStart(
            season_year=row["season_year"],
            game_date=parse_date(row["game_date"]),
            game_url=row["game_url"],
            team_code=team_code,
            pitcher_name=row["pitcher_name"],
            ip_outs=row["ip_outs"] or 0,
            hits=row["hits"] or 0,
            hr=row["hr"] or 0,
            bb=row["bb"] or 0,
            hbp=row["hbp"] or 0,
            strikeouts=row["strikeouts"] or 0,
            earned_runs=row["earned_runs"] or 0,
        )
        starters_by_game[start.game_url][row["team_side"]] = start
        if row["status"] == "completed":
            completed_starts.append(start)
    return starters_by_game, compute_fip_constants_by_season(completed_starts)


class GameState:
    def __init__(
        self,
        starters_by_game: dict[str, dict[str, StartingPitcherStart]],
        fip_constants_by_season: dict[int, float | None],
    ) -> None:
        self.elo = {team: 1500.0 for team in TEAM_CODES}
        self.histories: dict[tuple[int, str], list[dict]] = defaultdict(list)
        self.pitcher_histories: dict[tuple[str, str], list[StartingPitcherStart]] = defaultdict(list)
        self.last_dates: dict[str, date] = {}
        self.current_season: int | None = None
        self.processed_completed_games = 0
        self.starters_by_game = starters_by_game
        self.fip_constants_by_season = fip_constants_by_season

    def maybe_advance_season(self, season_year: int) -> None:
        if self.current_season is None:
            self.current_season = season_year
            return
        if season_year == self.current_season:
            return
        self.regress_elo_by_league()
        self.current_season = season_year

    def regress_elo_by_league(self) -> None:
        for pool in (CL_TEAMS, PL_TEAMS):
            mean = sum(self.elo[team] for team in pool) / len(pool)
            for team in pool:
                self.elo[team] = self.elo[team] * (1.0 - ELO_REGRESSION) + mean * ELO_REGRESSION

    def snapshot(self, game: Game) -> dict:
        home_history = self.histories[(game.season_year, game.home_code)]
        away_history = self.histories[(game.season_year, game.away_code)]
        home_long = summarize(home_history, TEAM_WINDOW, TEAM_MIN_GAMES)
        away_long = summarize(away_history, TEAM_WINDOW, TEAM_MIN_GAMES)
        home_w5 = summarize(home_history, 5, 5)
        away_w5 = summarize(away_history, 5, 5)
        home_w10 = summarize(home_history, 10, 10)
        away_w10 = summarize(away_history, 10, 10)

        home_elo = self.elo[game.home_code]
        away_elo = self.elo[game.away_code]
        home_streak = streak_value(home_history)
        away_streak = streak_value(away_history)
        home_rest = rest_days(self.last_dates.get(game.home_code), game.game_date)
        away_rest = rest_days(self.last_dates.get(game.away_code), game.game_date)
        current_fip_constant = self.fip_constants_by_season.get(game.season_year)
        game_starters = self.starters_by_game.get(game.game_url, {})
        home_start = game_starters.get("home")
        away_start = game_starters.get("away")
        home_sp = (
            summarize_pitcher_history(
                self.pitcher_histories[(home_start.team_code, home_start.pitcher_name)],
                current_fip_constant,
            )
            if home_start
            else None
        )
        away_sp = (
            summarize_pitcher_history(
                self.pitcher_histories[(away_start.team_code, away_start.pitcher_name)],
                current_fip_constant,
            )
            if away_start
            else None
        )

        row = {
            "game_url": game.game_url,
            "season_year": game.season_year,
            "game_date": game.game_date.isoformat(),
            "home_code": game.home_code,
            "away_code": game.away_code,
            "league_code": game.league_code,
            "is_interleague": 1 if game.league_code == "IL" else 0,
            "home_elo": home_elo,
            "vis_elo": away_elo,
            "diff_elo": home_elo - away_elo,
            "home_win_pct": home_long["win_pct"] if home_long else None,
            "vis_win_pct": away_long["win_pct"] if away_long else None,
            "diff_win_pct": (
                home_long["win_pct"] - away_long["win_pct"] if home_long and away_long else None
            ),
            "home_rs_pg": home_long["rs_pg"] if home_long else None,
            "vis_rs_pg": away_long["rs_pg"] if away_long else None,
            "home_ra_pg": home_long["ra_pg"] if home_long else None,
            "vis_ra_pg": away_long["ra_pg"] if away_long else None,
            "diff_rd_pg": (
                home_long["rd_pg"] - away_long["rd_pg"] if home_long and away_long else None
            ),
            "diff_pyth_wp": (
                home_long["pyth_wp"] - away_long["pyth_wp"] if home_long and away_long else None
            ),
            "diff_w5_win_pct": (
                home_w5["win_pct"] - away_w5["win_pct"] if home_w5 and away_w5 else None
            ),
            "diff_w10_win_pct": (
                home_w10["win_pct"] - away_w10["win_pct"] if home_w10 and away_w10 else None
            ),
            "diff_w5_rd_pg": home_w5["rd_pg"] - away_w5["rd_pg"] if home_w5 and away_w5 else None,
            "diff_w10_rd_pg": (
                home_w10["rd_pg"] - away_w10["rd_pg"] if home_w10 and away_w10 else None
            ),
            "home_streak": home_streak,
            "vis_streak": away_streak,
            "diff_streak": home_streak - away_streak,
            "home_rest": home_rest,
            "vis_rest": away_rest,
            "diff_rest": home_rest - away_rest,
            "home_season_games_before": len(home_history),
            "vis_season_games_before": len(away_history),
            "home_sp_fip_roll": home_sp["fip"] if home_sp else None,
            "vis_sp_fip_roll": away_sp["fip"] if away_sp else None,
            "diff_sp_fip": (
                home_sp["fip"] - away_sp["fip"]
                if home_sp and away_sp and home_sp["fip"] is not None and away_sp["fip"] is not None
                else None
            ),
            "diff_sp_era": (
                home_sp["era"] - away_sp["era"]
                if home_sp and away_sp and home_sp["era"] is not None and away_sp["era"] is not None
                else None
            ),
            "diff_sp_whip": (
                home_sp["whip"] - away_sp["whip"]
                if home_sp and away_sp and home_sp["whip"] is not None and away_sp["whip"] is not None
                else None
            ),
            "diff_sp_k9": (
                home_sp["k9"] - away_sp["k9"]
                if home_sp and away_sp and home_sp["k9"] is not None and away_sp["k9"] is not None
                else None
            ),
            "sp_available": int(home_sp is not None and away_sp is not None),
            "home_win": game.home_win,
        }
        return row

    def update(self, game: Game) -> None:
        if not game_is_completed_with_score(game):
            return
        actual_home = 0.5 if game.home_score == game.away_score else float(game.home_score > game.away_score)
        actual_away = 1.0 - actual_home

        self.histories[(game.season_year, game.home_code)].append(
            {"rs": game.home_score, "ra": game.away_score, "win": actual_home}
        )
        self.histories[(game.season_year, game.away_code)].append(
            {"rs": game.away_score, "ra": game.home_score, "win": actual_away}
        )

        expected_home = 1.0 / (
            1.0 + 10.0 ** ((self.elo[game.away_code] - (self.elo[game.home_code] + ELO_HOME_ADVANTAGE)) / 400.0)
        )
        self.elo[game.home_code] += ELO_K * (actual_home - expected_home)
        self.elo[game.away_code] += ELO_K * (actual_away - (1.0 - expected_home))
        self.last_dates[game.home_code] = game.game_date
        self.last_dates[game.away_code] = game.game_date
        for start in self.starters_by_game.get(game.game_url, {}).values():
            if start.ip_outs >= 3:
                self.pitcher_histories[(start.team_code, start.pitcher_name)].append(start)
        self.processed_completed_games += 1


def game_is_usable(game: Game) -> bool:
    return game.home_code in TEAM_CODES and game.away_code in TEAM_CODES and game.league_code in VALID_LEAGUES


def game_is_completed_with_score(game: Game) -> bool:
    return (
        game.status == "completed"
        and game.home_score is not None
        and game.away_score is not None
        and game_is_usable(game)
    )


def should_insert_game(game: Game, target_year: int | None, include_scheduled: bool) -> bool:
    if target_year is not None and game.season_year != target_year:
        return False
    if not game_is_usable(game):
        return False
    if game.status == "completed":
        return game.season_year >= TRAIN_START_YEAR and game.league_code in {"CL", "PL"}
    if include_scheduled and game.status == "scheduled":
        return game.season_year >= TRAIN_START_YEAR and game.league_code in VALID_LEAGUES
    return False


def load_games(conn: sqlite3.Connection) -> list[Game]:
    rows = conn.execute(
        """
        SELECT season_year, game_date, home_code, away_code,
               home_score, away_score, home_win, league_code, game_url, status
        FROM team_game_results
        ORDER BY game_date ASC, game_url ASC
        """
    ).fetchall()
    games = []
    for row in rows:
        home_code = normalize_team(row["home_code"])
        away_code = normalize_team(row["away_code"])
        league_code = league_for(home_code, away_code, row["league_code"])
        games.append(
            Game(
                season_year=row["season_year"],
                game_date=parse_date(row["game_date"]),
                home_code=home_code or "",
                away_code=away_code or "",
                home_score=row["home_score"],
                away_score=row["away_score"],
                home_win=row["home_win"],
                league_code=league_code,
                game_url=row["game_url"],
                status=row["status"],
            )
        )
    return games


def create_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS game_features_npb (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_url TEXT UNIQUE,
            season_year INTEGER,
            game_date TEXT,
            home_code TEXT,
            away_code TEXT,
            league_code TEXT,
            is_interleague INTEGER,
            home_elo REAL,
            vis_elo REAL,
            diff_elo REAL,
            home_win_pct REAL,
            vis_win_pct REAL,
            diff_win_pct REAL,
            home_rs_pg REAL,
            vis_rs_pg REAL,
            home_ra_pg REAL,
            vis_ra_pg REAL,
            diff_rd_pg REAL,
            diff_pyth_wp REAL,
            diff_w5_win_pct REAL,
            diff_w10_win_pct REAL,
            diff_w5_rd_pg REAL,
            diff_w10_rd_pg REAL,
            home_streak INTEGER,
            vis_streak INTEGER,
            diff_streak INTEGER,
            home_rest INTEGER,
            vis_rest INTEGER,
            diff_rest INTEGER,
            home_season_games_before INTEGER,
            vis_season_games_before INTEGER,
            home_sp_fip_roll REAL,
            vis_sp_fip_roll REAL,
            diff_sp_fip REAL,
            diff_sp_era REAL,
            diff_sp_whip REAL,
            diff_sp_k9 REAL,
            sp_available INTEGER,
            home_win INTEGER
        )
        """
    )
    existing_columns = {row["name"] for row in conn.execute("PRAGMA table_info(game_features_npb)").fetchall()}
    required_columns = {
        "home_sp_fip_roll": "REAL",
        "vis_sp_fip_roll": "REAL",
        "diff_sp_fip": "REAL",
    }
    for column_name, column_type in required_columns.items():
        if column_name not in existing_columns:
            conn.execute(f"ALTER TABLE game_features_npb ADD COLUMN {column_name} {column_type}")
    conn.commit()


def clear_existing(conn: sqlite3.Connection, target_year: int | None) -> None:
    if target_year is None:
        conn.execute("DELETE FROM game_features_npb")
    else:
        conn.execute("DELETE FROM game_features_npb WHERE season_year = ?", (target_year,))
    conn.commit()


def insert_rows(conn: sqlite3.Connection, rows: Iterable[dict]) -> int:
    rows = list(rows)
    if not rows:
        return 0
    placeholders = ",".join("?" for _ in FEATURE_COLUMNS)
    conn.executemany(
        f"""
        INSERT OR REPLACE INTO game_features_npb ({",".join(FEATURE_COLUMNS)})
        VALUES ({placeholders})
        """,
        [tuple(row[col] for col in FEATURE_COLUMNS) for row in rows],
    )
    conn.commit()
    return len(rows)


def build_features(games: list[Game], target_year: int | None, include_scheduled: bool) -> tuple[list[dict], GameState]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        starters_by_game, fip_constants_by_season = load_starting_pitchers(conn)
    finally:
        conn.close()
    state = GameState(starters_by_game, fip_constants_by_season)
    rows = []
    for game in games:
        state.maybe_advance_season(game.season_year)
        if should_insert_game(game, target_year, include_scheduled):
            rows.append(state.snapshot(game))
        if game.status == "completed":
            state.update(game)
    return rows, state


def print_summary(conn: sqlite3.Connection, inserted: int, state: GameState, target_year: int | None) -> None:
    year_filter = "WHERE season_year = ?" if target_year is not None else ""
    params = (target_year,) if target_year is not None else ()
    total = conn.execute(f"SELECT COUNT(*) FROM game_features_npb {year_filter}", params).fetchone()[0]
    completed = conn.execute(
        f"SELECT COUNT(*) FROM game_features_npb {year_filter} AND home_win IS NOT NULL"
        if target_year is not None
        else "SELECT COUNT(*) FROM game_features_npb WHERE home_win IS NOT NULL",
        params,
    ).fetchone()[0]

    print(f"Rows inserted/replaced: {inserted}")
    print(f"game_features_npb rows in scope: {total}")
    print(f"Rows with binary home_win label: {completed}")
    print(f"Completed games replayed into state: {state.processed_completed_games}")
    print("\nElo sample (initial 1500.0 for every team):")
    for team in ["g", "d", "db", "t", "c", "s", "h", "f", "b", "e", "l", "m"]:
        print(f"  {team:>2}: {state.elo[team]:.1f}")
    for name, pool in [("CL", CL_TEAMS), ("PL", PL_TEAMS)]:
        mean = sum(state.elo[team] for team in pool) / len(pool)
        print(f"  {name} mean: {mean:.1f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build NPB game-level pre-game features.")
    parser.add_argument("--year", type=int, help="Rebuild only rows for this season year.")
    parser.add_argument(
        "--include-scheduled",
        action="store_true",
        help="Also insert scheduled games in scope with home_win=NULL.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        create_table(conn)
        clear_existing(conn, args.year)
        games = load_games(conn)
        rows, state = build_features(games, args.year, args.include_scheduled)
        inserted = insert_rows(conn, rows)
        print_summary(conn, inserted, state, args.year)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
