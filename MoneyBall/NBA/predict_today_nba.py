from __future__ import annotations

import argparse
import sqlite3
import sys
import time
import warnings
from collections import defaultdict, deque
from datetime import date, datetime, timezone
from pathlib import Path

from nba_api.stats.endpoints import ScoreboardV2
from nba_api.stats.static import teams as nba_teams

from build_nba_game_features import (
    ELO_BASE,
    WINDOW,
    current_streak,
    compute_rest,
    elo_win_prob,
    is_neutral_site,
    load_games,
    next_streak,
    regress_elo,
    summarize_recent,
    to_feature_row,
    update_team_state,
)
from train_nba_model import (
    FEATURES,
    MIN_CALIB_ROWS,
    apply_calibrator,
    fit_calibrator,
    fit_xgb,
    predict_probs,
    rolling_train_years,
    target_vector,
)


BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "nba.sqlite"
RATE_LIMIT_SECONDS = 1.5
FIRST_GAME_REST = 7
HIGH_CONF_THRESHOLD = 0.65

PREDICTION_TRACKING_SCHEMA = """
CREATE TABLE IF NOT EXISTS prediction_tracking (
    prediction_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    season_year         INTEGER NOT NULL,
    game_id             TEXT NOT NULL UNIQUE,
    game_date           TEXT NOT NULL,
    home_team_id        INTEGER NOT NULL,
    vis_team_id         INTEGER NOT NULL,
    home_team_abbr      TEXT NOT NULL,
    vis_team_abbr       TEXT NOT NULL,
    prob_home_win       REAL NOT NULL,
    prob_home_win_cal   REAL NOT NULL,
    predicted_home_win  INTEGER NOT NULL,
    confidence          REAL NOT NULL,
    confidence_level    TEXT NOT NULL,
    is_high_confidence  INTEGER NOT NULL,
    threshold           REAL NOT NULL,
    actual_home_win     INTEGER,
    is_correct          INTEGER,
    verified_at         TEXT,
    created_at          TEXT NOT NULL,
    updated_at          TEXT NOT NULL
);
"""

UPSERT_PREDICTION_SQL = """
INSERT INTO prediction_tracking (
    season_year,
    game_id,
    game_date,
    home_team_id,
    vis_team_id,
    home_team_abbr,
    vis_team_abbr,
    prob_home_win,
    prob_home_win_cal,
    predicted_home_win,
    confidence,
    confidence_level,
    is_high_confidence,
    threshold,
    actual_home_win,
    is_correct,
    verified_at,
    created_at,
    updated_at
)
VALUES (
    :season_year,
    :game_id,
    :game_date,
    :home_team_id,
    :vis_team_id,
    :home_team_abbr,
    :vis_team_abbr,
    :prob_home_win,
    :prob_home_win_cal,
    :predicted_home_win,
    :confidence,
    :confidence_level,
    :is_high_confidence,
    :threshold,
    :actual_home_win,
    :is_correct,
    :verified_at,
    :created_at,
    :updated_at
)
ON CONFLICT(game_id) DO UPDATE SET
    season_year = excluded.season_year,
    game_date = excluded.game_date,
    home_team_id = excluded.home_team_id,
    vis_team_id = excluded.vis_team_id,
    home_team_abbr = excluded.home_team_abbr,
    vis_team_abbr = excluded.vis_team_abbr,
    prob_home_win = excluded.prob_home_win,
    prob_home_win_cal = excluded.prob_home_win_cal,
    predicted_home_win = excluded.predicted_home_win,
    confidence = excluded.confidence,
    confidence_level = excluded.confidence_level,
    is_high_confidence = excluded.is_high_confidence,
    threshold = excluded.threshold,
    actual_home_win = COALESCE(prediction_tracking.actual_home_win, excluded.actual_home_win),
    is_correct = COALESCE(prediction_tracking.is_correct, excluded.is_correct),
    verified_at = COALESCE(prediction_tracking.verified_at, excluded.verified_at),
    updated_at = excluded.updated_at
"""


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings(
    "ignore",
    message="ScoreboardV2 has known issues with line score data for 2025-26 season games*",
    category=DeprecationWarning,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict NBA games for a target date.")
    parser.add_argument(
        "--date",
        default=date.today().isoformat(),
        help="Target date YYYY-MM-DD for prediction mode (default: today).",
    )
    parser.add_argument(
        "--verify",
        metavar="YYYY-MM-DD",
        help="Verify completed results for the specified date.",
    )
    return parser.parse_args()


def connect_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.executescript(PREDICTION_TRACKING_SCHEMA)
    conn.commit()
    return conn


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def infer_season_year(target_date: date) -> int:
    return target_date.year if target_date.month >= 10 else target_date.year - 1


def build_team_lookup() -> dict[int, str]:
    mapping = {}
    for team in nba_teams.get_teams():
        mapping[int(team["id"])] = str(team["abbreviation"])
    return mapping


def fetch_scoreboard_games(target_date: date, team_lookup: dict[int, str]) -> list[dict]:
    scoreboard = ScoreboardV2(
        game_date=target_date.strftime("%Y-%m-%d"),
        league_id="00",
    )
    try:
        frames = scoreboard.get_data_frames()
    finally:
        time.sleep(RATE_LIMIT_SECONDS)

    if not frames:
        return []
    game_header = frames[0]
    if game_header.empty:
        return []

    games = []
    season_year = infer_season_year(target_date)
    for row in game_header.to_dict("records"):
        home_team_id = int(row["HOME_TEAM_ID"])
        vis_team_id = int(row["VISITOR_TEAM_ID"])
        games.append(
            {
                "game_id": str(row["GAME_ID"]),
                "season_year": season_year,
                "game_date": target_date.isoformat(),
                "home_team_id": home_team_id,
                "vis_team_id": vis_team_id,
                "home_team_abbr": team_lookup.get(home_team_id, str(home_team_id)),
                "vis_team_abbr": team_lookup.get(vis_team_id, str(vis_team_id)),
                "home_win": None,
                "game_status_text": str(row.get("GAME_STATUS_TEXT") or ""),
            }
        )
    return games


def build_state_until(conn: sqlite3.Connection, target_date: date):
    all_completed = load_games(conn)
    completed = [row for row in all_completed if row["game_date"] < target_date.isoformat()]

    team_history = defaultdict(lambda: deque(maxlen=WINDOW))
    elo_by_team = defaultdict(lambda: ELO_BASE)
    streak_by_team: dict[int, int] = {}
    last_game_dates: dict[int, date] = {}
    season_games = defaultdict(int)

    current_season = None
    for game in completed:
        season_year = int(game["season_year"])
        if season_year != current_season:
            if current_season is not None:
                regress_elo(elo_by_team)
            current_season = season_year
            streak_by_team = {}
            last_game_dates = {}
        update_team_state(
            game=game,
            team_history=team_history,
            elo_by_team=elo_by_team,
            streak_by_team=streak_by_team,
            last_game_dates=last_game_dates,
            season_games=season_games,
            neutral=is_neutral_site(game),
        )

    return team_history, elo_by_team, streak_by_team, last_game_dates, season_games


def build_today_feature_rows(conn: sqlite3.Connection, target_date: date, scheduled_games: list[dict]) -> list[dict]:
    team_history, elo_by_team, streak_by_team, last_game_dates, season_games = build_state_until(conn, target_date)
    feature_rows = []
    for game in scheduled_games:
        neutral = is_neutral_site(game)
        feature_rows.append(
            to_feature_row(
                game=game,
                team_history=team_history,
                elo_by_team=elo_by_team,
                streak_by_team=streak_by_team,
                last_game_dates=last_game_dates,
                season_games=season_games,
                neutral=neutral,
            )
        )
    return feature_rows


def load_training_rows(conn: sqlite3.Connection, target_date: date, target_season: int) -> list[dict]:
    train_years = rolling_train_years(target_season + 1)
    placeholders = ",".join("?" for _ in train_years)
    query = f"""
        SELECT season_year, game_id, game_date, home_team_abbr, vis_team_abbr, home_win,
               {", ".join(FEATURES)}
        FROM game_features
        WHERE season_year IN ({placeholders})
          AND game_date < ?
          AND home_win IS NOT NULL
        ORDER BY season_year, game_date, game_id
    """
    params = (*train_years, target_date.isoformat())
    rows = conn.execute(query, params).fetchall()
    return [dict(row) for row in rows]


def fit_models(train_rows: list[dict], target_season: int):
    if not train_rows:
        raise RuntimeError("No training rows found in game_features for the recent 5-season window.")

    model, medians = fit_xgb(train_rows, FEATURES)

    latest_season = max(int(row["season_year"]) for row in train_rows)
    calib_rows = [row for row in train_rows if int(row["season_year"]) == latest_season]
    pretrain_rows = [row for row in train_rows if int(row["season_year"]) < latest_season]

    calibrator = None
    if latest_season == target_season and len(calib_rows) >= MIN_CALIB_ROWS and pretrain_rows:
        calib_model, calib_medians = fit_xgb(pretrain_rows, FEATURES)
        calib_probs = predict_probs(calib_model, calib_medians, calib_rows, FEATURES)
        calibrator = fit_calibrator(calib_probs.tolist(), target_vector(calib_rows).tolist())
    elif latest_season < target_season and len(calib_rows) >= MIN_CALIB_ROWS:
        full_probs = predict_probs(model, medians, calib_rows, FEATURES)
        calibrator = fit_calibrator(full_probs.tolist(), target_vector(calib_rows).tolist())

    return model, medians, calibrator


def confidence_label(confidence: float) -> str:
    if confidence > HIGH_CONF_THRESHOLD:
        return "HIGH"
    if confidence > 0.55:
        return "MEDIUM"
    return "LOW"


def make_predictions(feature_rows: list[dict], model, medians, calibrator) -> list[dict]:
    raw_probs = predict_probs(model, medians, feature_rows, FEATURES)
    predictions = []
    for row, raw_prob in zip(feature_rows, raw_probs):
        cal_prob = apply_calibrator(calibrator, raw_prob)
        predicted_home_win = int(cal_prob >= 0.5)
        confidence = cal_prob if cal_prob >= 0.5 else 1.0 - cal_prob
        predictions.append(
            {
                **row,
                "prob_home_win": float(raw_prob),
                "prob_home_win_cal": float(cal_prob),
                "predicted_home_win": predicted_home_win,
                "confidence": float(confidence),
                "confidence_level": confidence_label(confidence),
                "is_high_confidence": int(confidence > HIGH_CONF_THRESHOLD),
                "threshold": HIGH_CONF_THRESHOLD,
                "actual_home_win": None,
                "is_correct": None,
                "verified_at": None,
            }
        )
    return predictions


def upsert_predictions(conn: sqlite3.Connection, predictions: list[dict]) -> int:
    now_iso = utc_now_iso()
    rows = []
    for prediction in predictions:
        rows.append(
            {
                **prediction,
                "created_at": now_iso,
                "updated_at": now_iso,
            }
        )
    if rows:
        conn.executemany(UPSERT_PREDICTION_SQL, rows)
        conn.commit()
    return len(rows)


def print_predictions(target_date: date, predictions: list[dict]) -> None:
    print(f"=== NBA Predictions: {target_date.isoformat()} ===\n")
    high_games = 0
    for idx, row in enumerate(predictions, start=1):
        home_team = row["home_team_abbr"]
        vis_team = row["vis_team_abbr"]
        pred_team = home_team if int(row["predicted_home_win"]) else vis_team
        label = row["confidence_level"]
        if int(row["is_high_confidence"]):
            high_games += 1
        print(f"Game {idx}: {home_team} vs {vis_team}")
        print(f"  Prob({home_team} wins): {row['prob_home_win_cal']:.3f}  [{label}]")
        print(f"  Predicted: {pred_team}")
        print()
    print(f"High-confidence games (p > {HIGH_CONF_THRESHOLD:.2f}): {high_games}/{len(predictions)}")


def verify_predictions(conn: sqlite3.Connection, target_date: date) -> int:
    rows = conn.execute(
        """
        SELECT game_id, home_win
        FROM game_results
        WHERE game_date = ?
          AND home_win IS NOT NULL
        ORDER BY game_id
        """,
        (target_date.isoformat(),),
    ).fetchall()
    if not rows:
        print(f"Warning: game_results has no completed rows for {target_date.isoformat()}.")
        return 0

    result_map = {str(row["game_id"]): int(row["home_win"]) for row in rows}
    existing = conn.execute(
        """
        SELECT game_id, predicted_home_win
        FROM prediction_tracking
        WHERE game_date = ?
        ORDER BY game_id
        """,
        (target_date.isoformat(),),
    ).fetchall()
    if not existing:
        print(f"Warning: prediction_tracking has no rows for {target_date.isoformat()}.")
        return 0

    now_iso = utc_now_iso()
    updated = 0
    correct = 0
    total = 0
    for row in existing:
        game_id = str(row["game_id"])
        if game_id not in result_map:
            continue
        actual_home_win = result_map[game_id]
        is_correct = int(int(row["predicted_home_win"]) == actual_home_win)
        conn.execute(
            """
            UPDATE prediction_tracking
            SET actual_home_win = ?,
                is_correct = ?,
                verified_at = ?,
                updated_at = ?
            WHERE game_id = ?
            """,
            (actual_home_win, is_correct, now_iso, now_iso, game_id),
        )
        updated += 1
        total += 1
        correct += is_correct
    conn.commit()

    print(f"=== NBA Verification: {target_date.isoformat()} ===")
    print(f"Verified rows: {updated}")
    if total:
        print(f"Accuracy: {correct}/{total} = {correct / total:.1%}")
    return updated


def run_predict_mode(target_date: date) -> None:
    team_lookup = build_team_lookup()
    conn = connect_db()
    try:
        scheduled_games = fetch_scoreboard_games(target_date, team_lookup)
        if not scheduled_games:
            print(f"No games scheduled for {target_date.isoformat()}.")
            return

        target_season = infer_season_year(target_date)
        train_rows = load_training_rows(conn, target_date, target_season)
        model, medians, calibrator = fit_models(train_rows, target_season)
        feature_rows = build_today_feature_rows(conn, target_date, scheduled_games)
        predictions = make_predictions(feature_rows, model, medians, calibrator)
        upsert_predictions(conn, predictions)
        print_predictions(target_date, predictions)
    finally:
        conn.close()


def run_verify_mode(target_date: date) -> None:
    conn = connect_db()
    try:
        verify_predictions(conn, target_date)
    finally:
        conn.close()


def main() -> None:
    args = parse_args()
    if args.verify:
        run_verify_mode(date.fromisoformat(args.verify))
        return
    run_predict_mode(date.fromisoformat(args.date))


if __name__ == "__main__":
    main()
