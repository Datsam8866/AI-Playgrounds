"""
evaluate_game_predictions_npb_regime.py

NPB game win/loss prediction with CPBL-style regime routing:
  early season -> fixed home-win baseline
  in season -> primary model with SP-feature median imputation

Backtest protocol:
  expanding walk-forward by season, train on all rows before each test date,
  evaluate 2016+ games, and fit imputers only on each training split.
"""

from __future__ import annotations

import argparse
import math
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression


DB_PATH = Path(__file__).resolve().parent / "npb.sqlite"
TABLE_NAME = "game_features_npb"

DEFAULT_TRAIN_LEAGUES = ("CL", "PL", "IL")
COMPARE_CONFIGS = (
    ("baseline_cl_pl", ("CL", "PL")),
    ("train_with_il", ("CL", "PL", "IL")),
)
EVAL_LEAGUES = ("CL", "PL", "IL")
TEST_START_YEAR = 2016
EARLY_BASELINE_PROBABILITY = 0.535
BASELINE_ACCURACY = EARLY_BASELINE_PROBABILITY
EARLY_MIN_TRAIN_ROWS = 30
EARLY_MODEL_FEATURES = ["diff_elo", "home_elo", "vis_elo"]

BASE_FEATURES = [
    "diff_elo",
    "home_elo",
    "vis_elo",
    "diff_win_pct",
    "diff_rd_pg",
    "diff_pyth_wp",
    "diff_w5_win_pct",
    "diff_w10_win_pct",
    "diff_w5_rd_pg",
    "diff_w10_rd_pg",
    "diff_streak",
    "home_rest",
    "vis_rest",
    "diff_rest",
    "home_season_games_before",
    "vis_season_games_before",
    "home_win_pct",
    "vis_win_pct",
    "home_rs_pg",
    "home_ra_pg",
    "vis_rs_pg",
    "vis_ra_pg",
]

FALLBACK_FEATURES = BASE_FEATURES
PRIMARY_FEATURES = BASE_FEATURES + [
    "diff_sp_era",
    "diff_sp_whip",
    "diff_sp_k9",
    "diff_sp_fip",
    "sp_available",
]

XGB_PARAMS = dict(
    n_estimators=50,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=15,
    reg_lambda=3.0,
    eval_metric="logloss",
    use_label_encoder=False,
    random_state=42,
    verbosity=0,
)

REGIME_FEATURES = {
    "fallback": FALLBACK_FEATURES,
    "primary": PRIMARY_FEATURES,
}

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Walk-forward evaluate NPB regime models.")
    parser.add_argument(
        "--train-leagues",
        help="Comma-separated training leagues, e.g. CL,PL or CL,PL,IL. Default follows current production setting.",
    )
    parser.add_argument(
        "--compare-il",
        action="store_true",
        help="Run both CL/PL baseline and CL/PL/IL training for comparison.",
    )
    return parser.parse_args()


def parse_leagues(value: str | None) -> tuple[str, ...]:
    if not value:
        return DEFAULT_TRAIN_LEAGUES
    leagues = tuple(part.strip().upper() for part in value.split(",") if part.strip())
    if not leagues:
        raise RuntimeError("At least one training league is required.")
    invalid = [league for league in leagues if league not in EVAL_LEAGUES]
    if invalid:
        raise RuntimeError(f"Invalid league codes: {', '.join(invalid)}")
    return leagues


def load_rows() -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        placeholders = ",".join("?" for _ in EVAL_LEAGUES)
        query = f"""
            SELECT *
            FROM {TABLE_NAME}
            WHERE home_win IS NOT NULL
              AND league_code IN ({placeholders})
            ORDER BY season_year, game_date, game_url
        """
        return [dict(row) for row in conn.execute(query, EVAL_LEAGUES).fetchall()]
    finally:
        conn.close()


def validate_features(rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError("No training rows found in game_features_npb.")
    required = sorted({feature for features in REGIME_FEATURES.values() for feature in features})
    required += ["game_date", "game_url", "home_win", "league_code", "season_year", "sp_available"]
    missing = [name for name in required if name not in rows[0]]
    if missing:
        raise RuntimeError(f"Missing required columns: {', '.join(missing)}")


def is_early_season(row: dict) -> bool:
    return (
        int(row["home_season_games_before"] or 0) < 10
        or int(row["vis_season_games_before"] or 0) < 10
    )


def route_regime(row: dict) -> str:
    if is_early_season(row):
        return "early_baseline"
    return "primary" if has_primary_pitcher_features(row) else "fallback"


def as_float(value) -> float:
    if value is None:
        return math.nan
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def fit_medians(rows: list[dict], features: list[str]) -> dict[str, float]:
    medians = {}
    for feature in features:
        values = [as_float(row.get(feature)) for row in rows]
        clean = [value for value in values if not math.isnan(value)]
        medians[feature] = float(np.median(clean)) if clean else 0.0
    return medians


def transform(rows: list[dict], features: list[str], medians: dict[str, float]) -> np.ndarray:
    matrix = []
    for row in rows:
        matrix.append([
            medians[feature] if math.isnan(as_float(row.get(feature))) else as_float(row.get(feature))
            for feature in features
        ])
    return np.array(matrix, dtype=float)


def fit_model(rows: list[dict], features: list[str]) -> tuple[xgb.XGBClassifier, dict[str, float]]:
    medians = fit_medians(rows, features)
    x_train = transform(rows, features, medians)
    y_train = np.array([int(row["home_win"]) for row in rows], dtype=int)
    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(x_train, y_train)
    return model, medians


def predicted_side_probability(home_probability: float) -> float:
    return home_probability if home_probability >= 0.5 else 1.0 - home_probability


def has_primary_pitcher_features(row: dict) -> bool:
    return as_float(row.get("sp_available")) >= 0.5


def format_accuracy(value: float) -> str:
    return f"{value * 100:.2f}%"


def empty_stats() -> dict:
    return {"total": 0, "correct": 0}


def record_result(stats: dict, key: str, correct: int) -> None:
    bucket = stats[key]
    bucket["total"] += 1
    bucket["correct"] += correct


def print_bucket(label: str, stats: dict) -> None:
    total = stats["total"]
    accuracy = stats["correct"] / total if total else math.nan
    accuracy_text = format_accuracy(accuracy) if total else "nan%"
    print(f"{label}: N={total} accuracy={accuracy_text}")


def evaluate_configuration(rows: list[dict], train_leagues: tuple[str, ...], label: str) -> dict:
    test_rows = [row for row in rows if int(row["season_year"]) >= TEST_START_YEAR]
    test_rows.sort(key=lambda row: (row["game_date"], row.get("game_url") or ""))

    year_stats = defaultdict(lambda: {
        "overall": empty_stats(),
        "cl_pl": empty_stats(),
        "il_only": empty_stats(),
        "early_baseline": empty_stats(),
        "fallback": empty_stats(),
        "primary": empty_stats(),
    })
    conf_buckets = {0.60: [], 0.70: [], 0.80: []}

    cached_date = None
    cached_models: dict = {}

    for row in test_rows:
        game_date = row["game_date"]
        if game_date != cached_date:
            cached_models = {}
            train = [
                train_row
                for train_row in rows
                if train_row["game_date"] < game_date and train_row["league_code"] in train_leagues
            ]
            if not train:
                continue

            fallback_train = [train_row for train_row in train if not is_early_season(train_row)]
            if not fallback_train:
                continue
            cached_models["fallback"] = fit_model(fallback_train, FALLBACK_FEATURES)

            primary_train = [train_row for train_row in train if route_regime(train_row) == "primary"]
            primary_labels = {int(train_row["home_win"]) for train_row in primary_train}
            cached_models["primary"] = (
                fit_model(primary_train, PRIMARY_FEATURES)
                if primary_train and len(primary_labels) > 1
                else None
            )

            early_train = [train_row for train_row in train if route_regime(train_row) == "early_baseline"]
            early_labels = {int(train_row["home_win"]) for train_row in early_train}
            if len(early_train) >= EARLY_MIN_TRAIN_ROWS and len(early_labels) > 1:
                medians = fit_medians(early_train, EARLY_MODEL_FEATURES)
                x_early = transform(early_train, EARLY_MODEL_FEATURES, medians)
                y_early = np.array([int(train_row["home_win"]) for train_row in early_train], dtype=int)
                lr = LogisticRegression(C=1.0, max_iter=500)
                lr.fit(x_early, y_early)
                cached_models["early"] = (lr, medians)
            else:
                cached_models["early"] = None

            cached_date = game_date

        regime = route_regime(row)
        if regime == "early_baseline":
            early_bundle = cached_models.get("early")
            if early_bundle is not None:
                model, medians = early_bundle
                x_early = transform([row], EARLY_MODEL_FEATURES, medians)
                home_probability = float(model.predict_proba(x_early)[0, 1])
            else:
                home_probability = EARLY_BASELINE_PROBABILITY
        elif regime == "primary" and cached_models.get("primary") is not None:
            model, medians = cached_models["primary"]
            x_test = transform([row], PRIMARY_FEATURES, medians)
            home_probability = float(model.predict_proba(x_test)[0, 1])
        else:
            regime = "fallback"
            model, medians = cached_models["fallback"]
            x_test = transform([row], FALLBACK_FEATURES, medians)
            home_probability = float(model.predict_proba(x_test)[0, 1])

        predicted_home_win = int(home_probability >= 0.5)
        correct = int(predicted_home_win == int(row["home_win"]))
        year = int(row["season_year"])

        record_result(year_stats[year], "overall", correct)
        if row["league_code"] == "IL":
            record_result(year_stats[year], "il_only", correct)
        else:
            record_result(year_stats[year], "cl_pl", correct)
        record_result(year_stats[year], regime, correct)

        side_probability = predicted_side_probability(home_probability)
        for threshold in conf_buckets:
            if side_probability >= threshold:
                conf_buckets[threshold].append(correct)

    overall = empty_stats()
    cl_pl = empty_stats()
    il_only = empty_stats()
    regime_totals = {
        "early_baseline": empty_stats(),
        "fallback": empty_stats(),
        "primary": empty_stats(),
    }

    print(f"=== {label} train_leagues={','.join(train_leagues)} ===")
    for year in sorted(year_stats):
        stats = year_stats[year]
        year_overall = stats["overall"]
        accuracy = year_overall["correct"] / year_overall["total"] if year_overall["total"] else 0.0
        print(
            f"year={year} total={year_overall['total']} correct={year_overall['correct']} "
            f"accuracy={accuracy:.2%} (baseline={BASELINE_ACCURACY:.1%})"
        )
        print_bucket("  cl_pl", stats["cl_pl"])
        print_bucket("  il_only", stats["il_only"])
        if stats["early_baseline"]["total"]:
            print_bucket("  early_baseline", stats["early_baseline"])
        if stats["fallback"]["total"]:
            print_bucket("  fallback", stats["fallback"])
        if stats["primary"]["total"]:
            print_bucket("  primary", stats["primary"])

        for key in ("overall", "cl_pl", "il_only"):
            overall_key = {"overall": overall, "cl_pl": cl_pl, "il_only": il_only}[key]
            overall_key["total"] += stats[key]["total"]
            overall_key["correct"] += stats[key]["correct"]
        for regime_key, bucket in regime_totals.items():
            bucket["total"] += stats[regime_key]["total"]
            bucket["correct"] += stats[regime_key]["correct"]

    print()
    print(f"Walk-forward window: {TEST_START_YEAR}-2026")
    print_bucket("overall_all_leagues", overall)
    print_bucket("overall_cl_pl_only", cl_pl)
    print_bucket("overall_il_only", il_only)
    for regime_key in ("early_baseline", "fallback", "primary"):
        if regime_totals[regime_key]["total"]:
            print_bucket(regime_key, regime_totals[regime_key])
    for threshold, results in sorted(conf_buckets.items()):
        n = len(results)
        accuracy = sum(results) / n if n else math.nan
        accuracy_text = format_accuracy(accuracy) if n else "nan%"
        print(f"high-conf (p>={threshold:.2f}): N={n} accuracy={accuracy_text}")
    print()

    return {
        "label": label,
        "train_leagues": train_leagues,
        "overall_all_leagues": overall,
        "overall_cl_pl_only": cl_pl,
        "overall_il_only": il_only,
    }


def main() -> None:
    args = parse_args()
    rows = load_rows()
    validate_features(rows)

    min_year = min(int(row["season_year"]) for row in rows)
    if min_year > 2011:
        raise RuntimeError("Training data before 2016 is unavailable.")

    if args.compare_il:
        for label, train_leagues in COMPARE_CONFIGS:
            evaluate_configuration(rows, train_leagues, label)
        return

    train_leagues = parse_leagues(args.train_leagues)
    evaluate_configuration(rows, train_leagues, "default")


if __name__ == "__main__":
    main()
