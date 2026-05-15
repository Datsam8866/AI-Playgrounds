# -*- coding: utf-8 -*-
"""
predict_today_nba_playoffs.py

預測今日 NBA 季後賽比賽。
使用歷史 playoff_game_features 訓練，動態重建季後賽 Elo 進行今日預測。
"""

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
    elo_win_prob,
    is_neutral_site,
    regress_elo,
    update_player_state,
)
from build_nba_playoff_features import (
    ELO_K_RS,
    ELO_K_PO,
    ELO_HOME_ADV,
    FIRST_GAME_REST,
    load_rs_games,
    load_rs_player_game_map,
    compute_rs_end_state,
    infer_playoff_round,
)
from train_nba_playoff_model import (
    FEATURES,
    MIN_CALIB_ROWS,
    MIN_TRAIN_ROWS,
    apply_calibrator,
    fit_calibrator,
    fit_xgb,
    predict_probs,
    rolling_train_years,
    target_vector,
    can_fit,
)

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "nba.sqlite"
RATE_LIMIT_SECONDS = 1.5
HIGH_CONF_THRESHOLD = 0.65

PLAYOFF_PREDICTION_TRACKING_SCHEMA = """
CREATE TABLE IF NOT EXISTS playoff_prediction_tracking (
    prediction_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    season_year         INTEGER NOT NULL,
    game_id             TEXT NOT NULL UNIQUE,
    game_date           TEXT NOT NULL,
    home_team_id        INTEGER NOT NULL,
    vis_team_id         INTEGER NOT NULL,
    home_team_abbr      TEXT NOT NULL,
    vis_team_abbr       TEXT NOT NULL,
    playoff_round       INTEGER NOT NULL,
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
INSERT INTO playoff_prediction_tracking (
    season_year, game_id, game_date,
    home_team_id, vis_team_id,
    home_team_abbr, vis_team_abbr,
    playoff_round,
    prob_home_win, prob_home_win_cal,
    predicted_home_win, confidence, confidence_level,
    is_high_confidence, threshold,
    actual_home_win, is_correct, verified_at,
    created_at, updated_at
)
VALUES (
    :season_year, :game_id, :game_date,
    :home_team_id, :vis_team_id,
    :home_team_abbr, :vis_team_abbr,
    :playoff_round,
    :prob_home_win, :prob_home_win_cal,
    :predicted_home_win, :confidence, :confidence_level,
    :is_high_confidence, :threshold,
    :actual_home_win, :is_correct, :verified_at,
    :created_at, :updated_at
)
ON CONFLICT(game_id) DO UPDATE SET
    season_year = excluded.season_year,
    game_date = excluded.game_date,
    home_team_id = excluded.home_team_id,
    vis_team_id = excluded.vis_team_id,
    home_team_abbr = excluded.home_team_abbr,
    vis_team_abbr = excluded.vis_team_abbr,
    playoff_round = excluded.playoff_round,
    prob_home_win = excluded.prob_home_win,
    prob_home_win_cal = excluded.prob_home_win_cal,
    predicted_home_win = excluded.predicted_home_win,
    confidence = excluded.confidence,
    confidence_level = excluded.confidence_level,
    is_high_confidence = excluded.is_high_confidence,
    threshold = excluded.threshold,
    actual_home_win = COALESCE(playoff_prediction_tracking.actual_home_win, excluded.actual_home_win),
    is_correct = COALESCE(playoff_prediction_tracking.is_correct, excluded.is_correct),
    verified_at = COALESCE(playoff_prediction_tracking.verified_at, excluded.verified_at),
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
    parser = argparse.ArgumentParser(description="Predict NBA playoff games for a target date.")
    parser.add_argument("--date", default=date.today().isoformat(), help="Target date YYYY-MM-DD.")
    parser.add_argument("--verify", metavar="YYYY-MM-DD", help="Verify completed results for the specified date.")
    return parser.parse_args()


def connect_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.executescript(PLAYOFF_PREDICTION_TRACKING_SCHEMA)
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

    season_year = infer_season_year(target_date)
    games = []
    for row in game_header.to_dict("records"):
        gid = str(row["GAME_ID"])
        if not gid.startswith("004"):
            continue  # skip non-playoff games
        home_team_id = int(row["HOME_TEAM_ID"])
        vis_team_id = int(row["VISITOR_TEAM_ID"])
        games.append({
            "game_id": gid,
            "season_year": season_year,
            "game_date": target_date.isoformat(),
            "home_team_id": home_team_id,
            "vis_team_id": vis_team_id,
            "home_team_abbr": team_lookup.get(home_team_id, str(home_team_id)),
            "vis_team_abbr": team_lookup.get(vis_team_id, str(vis_team_id)),
            "home_win": None,
            "game_status_text": str(row.get("GAME_STATUS_TEXT") or ""),
        })
    return games


def load_po_games_for_season(conn, season_year: int) -> list[dict]:
    rows = conn.execute(
        """
        SELECT game_id, season_year, game_date,
               home_team_id, vis_team_id,
               home_team_abbr, vis_team_abbr,
               home_score, vis_score, home_win
        FROM playoff_game_results
        WHERE season_year = ? AND home_win IS NOT NULL
        ORDER BY game_date, game_id
        """,
        (season_year,)
    ).fetchall()
    return [dict(row) for row in rows]


def build_current_season_state(conn, target_date: date, season_year: int, season_end_states: dict) -> dict:
    """
    Build current playoff state for season_year up to (but not including) target_date.
    Returns: {team_id: {elo_po, series_info}}
    Also returns series_state dict for determining game_in_series, wins, etc.
    """
    rs_state = season_end_states.get(season_year, {})
    elo_po: dict[int, float] = {}
    series_state: dict = {}

    po_games = load_po_games_for_season(conn, season_year)
    # Filter games strictly before target_date
    po_games = [g for g in po_games if str(g["game_date"]) < target_date.isoformat()]

    for game in po_games:
        home_id = int(game["home_team_id"])
        vis_id = int(game["vis_team_id"])
        home_win = int(game["home_win"])
        game_id = str(game["game_id"])
        game_date_str = str(game["game_date"])

        if home_id not in elo_po:
            elo_po[home_id] = rs_state.get(home_id, {}).get("elo", ELO_BASE)
        if vis_id not in elo_po:
            elo_po[vis_id] = rs_state.get(vis_id, {}).get("elo", ELO_BASE)

        playoff_round = infer_playoff_round(game_id)
        series_key = (season_year, playoff_round, frozenset({home_id, vis_id}))

        if series_key not in series_state:
            series_state[series_key] = {
                "home_id": home_id,
                "home_wins": 0,
                "vis_wins": 0,
                "game_count": 0,
                "last_date": None,
                "homecourt_team": home_id,
            }

        ss = series_state[series_key]
        ss["game_count"] += 1
        ss["last_date"] = date.fromisoformat(game_date_str)
        if home_id == ss["home_id"]:
            ss["home_wins"] += home_win
            ss["vis_wins"] += (1 - home_win)
        else:
            ss["vis_wins"] += home_win
            ss["home_wins"] += (1 - home_win)

        # Update playoff elo
        vis_win = 1 - home_win
        exp_home = elo_win_prob(elo_po[home_id], elo_po[vis_id])
        elo_po[home_id] += ELO_K_PO * (home_win - exp_home)
        elo_po[vis_id] += ELO_K_PO * (vis_win - (1.0 - exp_home))

    return elo_po, series_state


def build_today_playoff_features(
    scheduled_games: list[dict],
    season_end_states: dict,
    elo_po: dict,
    series_state: dict,
    target_date: date,
) -> list[dict]:
    """Build feature rows for today's playoff games."""
    feature_rows = []
    season_year = infer_season_year(target_date)
    rs_state = season_end_states.get(season_year, {})

    for game in scheduled_games:
        home_id = int(game["home_team_id"])
        vis_id = int(game["vis_team_id"])
        game_id = str(game["game_id"])

        # Init elo_po from RS end state if not seen yet
        if home_id not in elo_po:
            elo_po[home_id] = rs_state.get(home_id, {}).get("elo", ELO_BASE)
        if vis_id not in elo_po:
            elo_po[vis_id] = rs_state.get(vis_id, {}).get("elo", ELO_BASE)

        playoff_round = infer_playoff_round(game_id)
        series_key = (season_year, playoff_round, frozenset({home_id, vis_id}))

        if series_key not in series_state:
            series_state[series_key] = {
                "home_id": home_id,
                "home_wins": 0,
                "vis_wins": 0,
                "game_count": 0,
                "last_date": None,
                "homecourt_team": home_id,
            }

        ss = series_state[series_key]
        game_in_series = ss["game_count"] + 1

        homecourt_team = ss["homecourt_team"]
        home_has_homecourt = 1 if home_id == homecourt_team else 0

        if home_id == ss["home_id"]:
            home_series_wins = ss["home_wins"]
            vis_series_wins = ss["vis_wins"]
        else:
            home_series_wins = ss["vis_wins"]
            vis_series_wins = ss["home_wins"]

        series_score_diff = home_series_wins - vis_series_wins
        is_elimination = 1 if (home_series_wins == 3 or vis_series_wins == 3) else 0

        if ss["last_date"] is None:
            series_rest_days = FIRST_GAME_REST
        else:
            series_rest_days = max(0, (target_date - ss["last_date"]).days)

        home_rs = rs_state.get(home_id, {})
        vis_rs = rs_state.get(vis_id, {})

        diff_elo_rs = home_rs.get("elo", ELO_BASE) - vis_rs.get("elo", ELO_BASE)
        h_net = home_rs.get("net_rtg")
        v_net = vis_rs.get("net_rtg")
        diff_rs_net_rtg = (h_net - v_net) if (h_net is not None and v_net is not None) else None
        h_pw = home_rs.get("pyth_wp")
        v_pw = vis_rs.get("pyth_wp")
        diff_rs_pyth_wp = (h_pw - v_pw) if (h_pw is not None and v_pw is not None) else None
        h_lp = home_rs.get("lineup_pts")
        v_lp = vis_rs.get("lineup_pts")
        diff_rs_lineup_pts = (h_lp - v_lp) if (h_lp is not None and v_lp is not None) else None

        diff_elo_po = elo_po[home_id] - elo_po[vis_id]
        elo_win_prob_po_val = elo_win_prob(elo_po[home_id], elo_po[vis_id])
        diff_elo_change_po = diff_elo_po - diff_elo_rs

        feature_rows.append({
            **game,
            "playoff_round": playoff_round,
            "game_in_series": game_in_series,
            "home_series_wins": home_series_wins,
            "vis_series_wins": vis_series_wins,
            "series_score_diff": series_score_diff,
            "is_elimination": is_elimination,
            "home_has_homecourt": home_has_homecourt,
            "series_rest_days": series_rest_days,
            "diff_elo_rs": diff_elo_rs,
            "diff_rs_net_rtg": diff_rs_net_rtg,
            "diff_rs_pyth_wp": diff_rs_pyth_wp,
            "diff_rs_lineup_pts": diff_rs_lineup_pts,
            "diff_elo_po": diff_elo_po,
            "elo_win_prob_po": elo_win_prob_po_val,
            "diff_elo_change_po": diff_elo_change_po,
        })

    return feature_rows


def load_training_rows(conn, target_season: int) -> list[dict]:
    train_years = rolling_train_years(target_season + 1)
    placeholders = ",".join("?" for _ in train_years)
    query = f"""
        SELECT season_year, game_id, game_date, home_team_abbr, vis_team_abbr, home_win,
               {", ".join(FEATURES)}
        FROM playoff_game_features
        WHERE season_year IN ({placeholders})
          AND home_win IS NOT NULL
        ORDER BY season_year, game_date, game_id
    """
    rows = conn.execute(query, train_years).fetchall()
    return [dict(row) for row in rows]


def fit_models(train_rows: list[dict], target_season: int):
    if not train_rows or not can_fit(train_rows):
        raise RuntimeError(f"Insufficient training rows: n={len(train_rows)} (need {MIN_TRAIN_ROWS})")

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
        predictions.append({
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
        })
    return predictions


def upsert_predictions(conn, predictions: list[dict]) -> int:
    now_iso = utc_now_iso()
    rows = [{**p, "created_at": now_iso, "updated_at": now_iso} for p in predictions]
    if rows:
        conn.executemany(UPSERT_PREDICTION_SQL, rows)
        conn.commit()
    return len(rows)


def print_predictions(target_date: date, predictions: list[dict]) -> None:
    print(f"=== NBA Playoff Predictions: {target_date.isoformat()} ===\n")
    high_games = 0
    for idx, row in enumerate(predictions, start=1):
        home_team = row["home_team_abbr"]
        vis_team = row["vis_team_abbr"]
        pred_team = home_team if int(row["predicted_home_win"]) else vis_team
        label = row["confidence_level"]
        rnd = row.get("playoff_round", "?")
        series_info = f"G{row.get('game_in_series','?')} ({row.get('home_series_wins',0)}-{row.get('vis_series_wins',0)})"
        if int(row["is_high_confidence"]):
            high_games += 1
        print(f"Game {idx}: {home_team} vs {vis_team} | Round {rnd} {series_info}")
        print(f"  Prob({home_team} wins): {row['prob_home_win_cal']:.3f}  [{label}]")
        print(f"  Predicted: {pred_team}")
        print()
    print(f"High-confidence games (p > {HIGH_CONF_THRESHOLD:.2f}): {high_games}/{len(predictions)}")


def verify_predictions(conn, target_date: date) -> int:
    rows = conn.execute(
        """
        SELECT game_id, home_win
        FROM playoff_game_results
        WHERE game_date = ? AND home_win IS NOT NULL
        ORDER BY game_id
        """,
        (target_date.isoformat(),),
    ).fetchall()
    if not rows:
        print(f"Warning: playoff_game_results has no completed rows for {target_date.isoformat()}.")
        return 0

    result_map = {str(row["game_id"]): int(row["home_win"]) for row in rows}
    existing = conn.execute(
        """
        SELECT game_id, predicted_home_win
        FROM playoff_prediction_tracking
        WHERE game_date = ?
        ORDER BY game_id
        """,
        (target_date.isoformat(),),
    ).fetchall()
    if not existing:
        print(f"Warning: playoff_prediction_tracking has no rows for {target_date.isoformat()}.")
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
            UPDATE playoff_prediction_tracking
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

    print(f"=== NBA Playoff Verification: {target_date.isoformat()} ===")
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
            print(f"No playoff games scheduled for {target_date.isoformat()}.")
            return

        season_year = infer_season_year(target_date)
        print(f"Found {len(scheduled_games)} playoff game(s) for {target_date.isoformat()} (season {season_year}).")

        # Build RS end state
        rs_games = load_rs_games(conn)
        rs_player_map = load_rs_player_game_map(conn)
        season_end_states = compute_rs_end_state(rs_games, rs_player_map)

        # Build current playoff state
        elo_po, series_state = build_current_season_state(conn, target_date, season_year, season_end_states)

        # Build features for today
        feature_rows = build_today_playoff_features(
            scheduled_games, season_end_states, elo_po, series_state, target_date
        )

        # Load training data and fit model
        train_rows = load_training_rows(conn, season_year)
        print(f"Training on {len(train_rows):,} historical playoff games.")
        model, medians, calibrator = fit_models(train_rows, season_year)

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
