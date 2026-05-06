"""
evaluate_game_predictions_regime.py

Regime-based game prediction:
  - early-season model: prior-heavy (Elo + previous-season strength + light context)
  - in-season model: current advanced ensemble (XGBoost)

Routing rule:
  If either team has fewer than TEAM_BURN_IN games before the matchup,
  or either starter has fewer than STARTER_BURN_IN prior starts,
  route to the early-season model.
"""

import argparse
import csv
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import xgboost as xgb

from evaluate_game_predictions_advanced import (
    ADVANCED_FALLBACK_FEATURES,
    ADVANCED_PRIMARY_FEATURES,
    BACKTEST_END_YEAR,
    BACKTEST_START_YEAR,
    DB_PATH,
    ELO_HOME_ADVANTAGE,
    ELO_K,
    ELO_REGRESSION,
    FRANCHISE_MAP,
    XGB_PARAMS,
    build_advanced_rows,
)


REPORT_PATH = Path("team_record_predictions_regime.md")
PREDICTION_FIELDNAMES = [
    "game_date",
    "season_year",
    "game_sno",
    "prob_home",
    "actual_home_win",
    "model_label",
]

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TEAM_BURN_IN = 10
STARTER_BURN_IN = 5
MIN_EARLY_TRAIN_ROWS = 50
EARLY_PROB_SHRINK = 0.50

EARLY_FEATURES = [
    "diff_elo",
    "home_elo",
    "vis_elo",
    "elo_home_prob",
    "prev_diff_win_pct",
    "prev_diff_rd_pg",
    "prev_diff_pyth",
    "home_rest",
    "vis_rest",
    "diff_rest",
    "diff_streak",
    "home_season_games_before",
    "vis_season_games_before",
]


def normalize(code: str) -> str:
    return FRANCHISE_MAP.get(code, code)


def pyth_wp(rs: float, ra: float) -> float:
    if rs + ra == 0:
        return 0.5
    return rs ** 2 / (rs ** 2 + ra ** 2)


def summarize_games(games: list[dict]) -> dict | None:
    if not games:
        return None
    n = len(games)
    rs = sum(g["rs"] for g in games)
    ra = sum(g["ra"] for g in games)
    wins = sum(g["win"] for g in games)
    return {
        "win_pct": wins / n,
        "rd_pg": (rs - ra) / n,
        "pyth": pyth_wp(rs, ra),
        "games": n,
    }


def load_starting_pitcher_map(conn) -> dict[tuple[int, str, int], dict]:
    rows = conn.execute("""
        SELECT season_year, kind_code, game_sno, home_sp_acnt, vis_sp_acnt
        FROM game_starting_pitchers
        WHERE kind_code = 'A'
    """).fetchall()
    return {(row["season_year"], row["kind_code"], row["game_sno"]): dict(row) for row in rows}


def load_completed_games(conn) -> list[sqlite3.Row]:
    return conn.execute("""
        SELECT
            season_year,
            kind_code,
            game_date,
            game_sno,
            visiting_team_code,
            home_team_code,
            visiting_score,
            home_score
        FROM team_game_results
        WHERE game_status = 3
          AND kind_code = 'A'
          AND visiting_score IS NOT NULL
          AND home_score IS NOT NULL
        ORDER BY game_date, game_sno
    """).fetchall()


def build_regime_rows(conn) -> list[dict]:
    base_rows = build_advanced_rows(conn)
    base_by_key = {
        (row["season_year"], row.get("kind_code", "A"), row["game_sno"]): dict(row)
        for row in base_rows
    }
    sp_map = load_starting_pitcher_map(conn)
    raw_games = load_completed_games(conn)

    histories_all: dict[str, list] = defaultdict(list)
    last_dates: dict[str, datetime.date] = {}
    elo: dict[str, float] = defaultdict(lambda: 1500.0)
    sp_starts: dict[str, int] = defaultdict(int)

    current_season = None
    metadata = {}
    for raw in raw_games:
        season_year = raw["season_year"]
        kind_code = raw["kind_code"]
        if current_season is None:
            current_season = season_year
        elif season_year != current_season:
            for team_code in list(elo):
                elo[team_code] = 1500 + (elo[team_code] - 1500) * (1 - ELO_REGRESSION)
            current_season = season_year

        game_date = datetime.fromisoformat(raw["game_date"][:10]).date()
        game_sno = raw["game_sno"]
        vis_code = normalize(raw["visiting_team_code"])
        home_code = normalize(raw["home_team_code"])
        vis_score = float(raw["visiting_score"])
        home_score = float(raw["home_score"])
        actual_home = 0.5 if home_score == vis_score else (1.0 if home_score > vis_score else 0.0)

        sp_row = sp_map.get((season_year, kind_code, game_sno), {})
        home_sp = sp_row.get("home_sp_acnt")
        vis_sp = sp_row.get("vis_sp_acnt")

        if home_score != vis_score and (season_year, kind_code, game_sno) in base_by_key:
            prev_home = summarize_games([g for g in histories_all[home_code] if g["season_year"] == season_year - 1])
            prev_vis = summarize_games([g for g in histories_all[vis_code] if g["season_year"] == season_year - 1])
            current_home = summarize_games([g for g in histories_all[home_code] if g["season_year"] == season_year])
            current_vis = summarize_games([g for g in histories_all[vis_code] if g["season_year"] == season_year])

            metadata[(season_year, kind_code, game_sno)] = {
                "home_season_games_before": current_home["games"] if current_home else 0,
                "vis_season_games_before": current_vis["games"] if current_vis else 0,
                "home_sp_starts_before": sp_starts.get(home_sp, 0) if home_sp else 0,
                "vis_sp_starts_before": sp_starts.get(vis_sp, 0) if vis_sp else 0,
                "prev_diff_win_pct": (
                    (prev_home["win_pct"] if prev_home else 0.0)
                    - (prev_vis["win_pct"] if prev_vis else 0.0)
                ),
                "prev_diff_rd_pg": (
                    (prev_home["rd_pg"] if prev_home else 0.0)
                    - (prev_vis["rd_pg"] if prev_vis else 0.0)
                ),
                "prev_diff_pyth": (
                    (prev_home["pyth"] if prev_home else 0.0)
                    - (prev_vis["pyth"] if prev_vis else 0.0)
                ),
            }

        histories_all[home_code].append({
            "season_year": season_year,
            "rs": home_score,
            "ra": vis_score,
            "win": actual_home,
        })
        histories_all[vis_code].append({
            "season_year": season_year,
            "rs": vis_score,
            "ra": home_score,
            "win": 1 - actual_home,
        })

        expected_home = 1 / (
            1 + 10 ** ((elo[vis_code] - (elo[home_code] + ELO_HOME_ADVANTAGE)) / 400)
        )
        elo[home_code] += ELO_K * (actual_home - expected_home)
        elo[vis_code] += ELO_K * ((1 - actual_home) - (1 - expected_home))
        last_dates[home_code] = game_date
        last_dates[vis_code] = game_date

        if home_sp:
            sp_starts[home_sp] += 1
        if vis_sp:
            sp_starts[vis_sp] += 1

    rows = []
    for row in base_rows:
        merged = dict(row)
        meta = metadata[(row["season_year"], row.get("kind_code", "A"), row["game_sno"])]
        merged.update(meta)
        merged["early_flag"] = 1.0 if (
            merged["home_season_games_before"] < TEAM_BURN_IN
            or merged["vis_season_games_before"] < TEAM_BURN_IN
            or merged["home_sp_starts_before"] < STARTER_BURN_IN
            or merged["vis_sp_starts_before"] < STARTER_BURN_IN
        ) else 0.0
        rows.append(merged)
    return rows


def fit_xgb(train_rows, feature_names):
    X = np.array([[row[name] for name in feature_names] for row in train_rows], dtype=float)
    y = np.array([row["home_win"] for row in train_rows], dtype=float)
    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(X, y)
    return model


def predict_xgb(model, row, feature_names):
    X = np.array([[row[name] for name in feature_names]], dtype=float)
    return float(model.predict_proba(X)[0, 1])


def shrink_early_probability(prob: float) -> float:
    return 0.5 + (prob - 0.5) * EARLY_PROB_SHRINK


def team_readiness(row: dict) -> float:
    home_games = max(0.0, min(float(row.get("home_season_games_before", 0)), TEAM_BURN_IN))
    vis_games = max(0.0, min(float(row.get("vis_season_games_before", 0)), TEAM_BURN_IN))
    return min(home_games, vis_games) / TEAM_BURN_IN


def starter_readiness(row: dict) -> float:
    home_starts = max(0.0, min(float(row.get("home_sp_starts_before", 0)), STARTER_BURN_IN))
    vis_starts = max(0.0, min(float(row.get("vis_sp_starts_before", 0)), STARTER_BURN_IN))
    return min(home_starts, vis_starts) / STARTER_BURN_IN


def soft_regime_weights(row: dict, has_primary_model: bool) -> dict:
    team_ready = team_readiness(row)
    sp_ready = starter_readiness(row) if row.get("sp_available", 0) > 0.5 else 0.0

    early_weight = 1.0 - team_ready
    primary_weight = (1.0 - early_weight) * sp_ready if has_primary_model else 0.0
    fallback_weight = max(0.0, 1.0 - early_weight - primary_weight)

    total = early_weight + fallback_weight + primary_weight
    if total <= 0:
        return {"early": 0.0, "fallback": 1.0, "primary": 0.0}
    return {
        "early": early_weight / total,
        "fallback": fallback_weight / total,
        "primary": primary_weight / total,
    }


def model_label(weights: dict) -> str:
    if weights["early"] >= 0.999:
        return "early_shrunk"
    if weights["primary"] >= 0.999:
        return "advanced+SP"
    if weights["fallback"] >= 0.999:
        return "advanced(no_SP)"
    return "soft_blend"


def fit_regime_models(train_rows: list[dict]) -> dict:
    primary_train = [row for row in train_rows if row["sp_available"] > 0.5]
    early_train = [row for row in train_rows if row["early_flag"] > 0.5]
    if len(early_train) < MIN_EARLY_TRAIN_ROWS:
        early_train = train_rows
    return {
        "fallback": fit_xgb(train_rows, ADVANCED_FALLBACK_FEATURES),
        "primary": fit_xgb(primary_train, ADVANCED_PRIMARY_FEATURES) if primary_train else None,
        "early": fit_xgb(early_train, EARLY_FEATURES),
    }


def predict_regime_from_models(models: dict, test_row: dict) -> dict:
    early_raw = predict_xgb(models["early"], test_row, EARLY_FEATURES)
    early_prob = shrink_early_probability(early_raw)
    fallback_prob = predict_xgb(models["fallback"], test_row, ADVANCED_FALLBACK_FEATURES)

    primary_prob = None
    has_primary = models["primary"] is not None and test_row.get("sp_available", 0) > 0.5
    if has_primary:
        primary_prob = predict_xgb(models["primary"], test_row, ADVANCED_PRIMARY_FEATURES)

    weights = soft_regime_weights(test_row, has_primary)
    prob = weights["early"] * early_prob + weights["fallback"] * fallback_prob
    if primary_prob is not None:
        prob += weights["primary"] * primary_prob

    return {
        "prob_home_win": float(prob),
        "model_used": model_label(weights),
        "early_prob_raw": early_raw,
        "early_prob_shrunk": early_prob,
        "fallback_prob": fallback_prob,
        "primary_prob": primary_prob,
        "early_weight": weights["early"],
        "fallback_weight": weights["fallback"],
        "primary_weight": weights["primary"],
        "team_readiness": team_readiness(test_row),
        "starter_readiness": starter_readiness(test_row),
    }


def predict_advanced(train_rows, test_row):
    primary_train = [row for row in train_rows if row["sp_available"] > 0.5]
    fallback_model = fit_xgb(train_rows, ADVANCED_FALLBACK_FEATURES)

    if primary_train and test_row["sp_available"] > 0.5:
        primary_model = fit_xgb(primary_train, ADVANCED_PRIMARY_FEATURES)
        X = np.array([[test_row[name] for name in ADVANCED_PRIMARY_FEATURES]], dtype=float)
        return float(primary_model.predict_proba(X)[0, 1]), "advanced_primary"

    X = np.array([[test_row[name] for name in ADVANCED_FALLBACK_FEATURES]], dtype=float)
    return float(fallback_model.predict_proba(X)[0, 1]), "advanced_fallback"


def predict_regime(train_rows, test_row):
    details = predict_regime_details(train_rows, test_row)
    return details["prob_home_win"], details["model_used"]


def predict_regime_details(train_rows, test_row):
    models = fit_regime_models(train_rows)
    return predict_regime_from_models(models, test_row)


def append_prediction_row(output_path: Path, row: dict):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.exists() or output_path.stat().st_size == 0
    with output_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PREDICTION_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def evaluate_period(rows, start_date: str, end_date: str) -> dict:
    target_rows = [
        row for row in rows
        if start_date <= row["game_date"] <= end_date
    ]
    target_rows.sort(key=lambda row: (row["game_date"], row["game_sno"]))

    advanced_correct = 0
    regime_correct = 0
    early_games = 0
    details = []

    for row in target_rows:
        train_rows = [
            train_row for train_row in rows
            if (train_row["game_date"] < row["game_date"])
            or (train_row["game_date"] == row["game_date"] and train_row["game_sno"] < row["game_sno"])
        ]
        if not train_rows:
            continue

        advanced_prob, advanced_mode = predict_advanced(train_rows, row)
        regime_details = predict_regime_details(train_rows, row)
        regime_prob = regime_details["prob_home_win"]
        regime_mode = regime_details["model_used"]
        actual = row["home_win"]
        advanced_pick = 1 if advanced_prob >= 0.5 else 0
        regime_pick = 1 if regime_prob >= 0.5 else 0

        advanced_correct += int(advanced_pick == actual)
        regime_correct += int(regime_pick == actual)
        early_games += int(regime_details["early_weight"] > 0)
        details.append({
            "game_date": row["game_date"],
            "game_sno": row["game_sno"],
            "actual_home_win": actual,
            "advanced_prob": advanced_prob,
            "advanced_mode": advanced_mode,
            "advanced_ok": advanced_pick == actual,
            "regime_prob": regime_prob,
            "regime_mode": regime_mode,
            "regime_ok": regime_pick == actual,
            "early_flag": row["early_flag"],
            "early_weight": regime_details["early_weight"],
            "fallback_weight": regime_details["fallback_weight"],
            "primary_weight": regime_details["primary_weight"],
        })

    games = len(details)
    return {
        "games": games,
        "advanced_correct": advanced_correct,
        "advanced_accuracy": advanced_correct / games if games else float("nan"),
        "regime_correct": regime_correct,
        "regime_accuracy": regime_correct / games if games else float("nan"),
        "early_games": early_games,
        "details": details,
    }


def evaluate_benchmark(rows, save_predictions_path: Path | None = None) -> dict:
    target_rows = [row for row in rows if BACKTEST_START_YEAR <= row["season_year"] <= BACKTEST_END_YEAR]
    target_rows.sort(key=lambda row: (row["game_date"], row["game_sno"]))
    correct = 0
    early_games = 0

    for row in target_rows:
        train_rows = [
            train_row for train_row in rows
            if (train_row["game_date"] < row["game_date"])
            or (train_row["game_date"] == row["game_date"] and train_row["game_sno"] < row["game_sno"])
        ]
        if not train_rows:
            continue
        regime_details = predict_regime_details(train_rows, row)
        prob = regime_details["prob_home_win"]
        early_games += int(regime_details["early_weight"] > 0)
        correct += int((1 if prob >= 0.5 else 0) == row["home_win"])
        if save_predictions_path is not None:
            append_prediction_row(
                save_predictions_path,
                {
                    "game_date": row["game_date"],
                    "season_year": row["season_year"],
                    "game_sno": row["game_sno"],
                    "prob_home": prob,
                    "actual_home_win": int(row["home_win"]),
                    "model_label": regime_details["model_used"],
                },
            )

    games = len(target_rows)
    return {
        "games": games,
        "correct": correct,
        "accuracy": correct / games if games else float("nan"),
        "early_games": early_games,
    }


def build_report(opening_result, cumulative_result, recent_result, benchmark_result, latest_completed: str):
    lines = [
        "# Team Record Predictions (Regime Model)",
        "",
        f"_As of latest completed game date: `{latest_completed}`_",
        "",
        "## Soft Routing Rule",
        "",
        f"- Early model weight fades out as both teams reach `{TEAM_BURN_IN}` prior games",
        f"- SP model weight fades in as both starters reach `{STARTER_BURN_IN}` prior starts",
        f"- Early model probability is shrunk toward `0.500` with multiplier `{EARLY_PROB_SHRINK:.2f}` before blending",
        "- Fallback advanced model absorbs weight when SP data is incomplete",
        "",
        "## Early-Season Features",
        "",
        "- Elo state: `diff_elo`, `home_elo`, `vis_elo`, `elo_home_prob`",
        "- Previous-season priors: `prev_diff_win_pct`, `prev_diff_rd_pg`, `prev_diff_pyth`",
        "- Light context only: `home_rest`, `vis_rest`, `diff_rest`, `diff_streak`",
        "- Burn-in counters: `home_season_games_before`, `vis_season_games_before`",
        "",
        "## 2026 Opening Stretch",
        "",
        "| Model | Games | Correct | Accuracy |",
        "| --- | ---: | ---: | ---: |",
        f"| current advanced ensemble | {opening_result['games']} | {opening_result['advanced_correct']} | {opening_result['advanced_accuracy']:.2%} |",
        f"| regime model | {opening_result['games']} | {opening_result['regime_correct']} | {opening_result['regime_accuracy']:.2%} |",
        "",
        f"- Games with non-zero early weight in this window: `{opening_result['early_games']} / {opening_result['games']}`",
        "",
        "## 2026 Through Latest Completed Date",
        "",
        f"Window: `2026-04-01` to `{latest_completed}`",
        "",
        "| Model | Games | Correct | Accuracy |",
        "| --- | ---: | ---: | ---: |",
        f"| current advanced ensemble | {cumulative_result['games']} | {cumulative_result['advanced_correct']} | {cumulative_result['advanced_accuracy']:.2%} |",
        f"| regime model | {cumulative_result['games']} | {cumulative_result['regime_correct']} | {cumulative_result['regime_accuracy']:.2%} |",
        "",
        f"- Games with non-zero early weight in this window: `{cumulative_result['early_games']} / {cumulative_result['games']}`",
        "",
        "## Recent Window",
        "",
        f"Window: `2026-04-08` to `{latest_completed}`",
        "",
        "| Model | Games | Correct | Accuracy |",
        "| --- | ---: | ---: | ---: |",
        f"| current advanced ensemble | {recent_result['games']} | {recent_result['advanced_correct']} | {recent_result['advanced_accuracy']:.2%} |",
        f"| regime model | {recent_result['games']} | {recent_result['regime_correct']} | {recent_result['regime_accuracy']:.2%} |",
        "",
        f"- Games with non-zero early weight in this window: `{recent_result['early_games']} / {recent_result['games']}`",
        "",
        "## 2016–2025 Benchmark",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Games | {benchmark_result['games']} |",
        f"| Accuracy | {benchmark_result['accuracy']:.2%} |",
        f"| Games with non-zero early weight | {benchmark_result['early_games']} |",
        "",
    ]

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Evaluate regime model against advanced ensemble")
    parser.add_argument(
        "--recent-start",
        default="2026-04-08",
        help="Recent validation window start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--save-predictions",
        type=Path,
        help="Optional CSV path to append walk-forward benchmark predictions",
    )
    args = parser.parse_args()

    if args.save_predictions:
        args.save_predictions.unlink(missing_ok=True)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        rows = build_regime_rows(conn)
        opening_result = evaluate_period(rows, "2026-03-28", "2026-04-07")
        completed_2026 = [row["game_date"] for row in rows if row["season_year"] == 2026]
        latest_completed = max(completed_2026) if completed_2026 else "N/A"
        cumulative_result = evaluate_period(rows, "2026-04-01", latest_completed) if completed_2026 else {
            "games": 0,
            "advanced_correct": 0,
            "advanced_accuracy": float("nan"),
            "regime_correct": 0,
            "regime_accuracy": float("nan"),
            "early_games": 0,
            "details": [],
        }
        recent_result = evaluate_period(rows, args.recent_start, latest_completed) if completed_2026 else {
            "games": 0,
            "advanced_correct": 0,
            "advanced_accuracy": float("nan"),
            "regime_correct": 0,
            "regime_accuracy": float("nan"),
            "early_games": 0,
            "details": [],
        }
        benchmark_result = evaluate_benchmark(rows, args.save_predictions)
        build_report(opening_result, cumulative_result, recent_result, benchmark_result, latest_completed)

        print(f"report={REPORT_PATH.resolve()}")
        print(f"latest_completed={latest_completed}")
        print(f"opening_games={opening_result['games']}")
        print(f"opening_advanced_accuracy={opening_result['advanced_accuracy']:.4f}")
        print(f"opening_regime_accuracy={opening_result['regime_accuracy']:.4f}")
        print(f"recent_games={recent_result['games']}")
        print(f"recent_advanced_accuracy={recent_result['advanced_accuracy']:.4f}")
        print(f"recent_regime_accuracy={recent_result['regime_accuracy']:.4f}")
        print(f"benchmark_accuracy={benchmark_result['accuracy']:.4f}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
