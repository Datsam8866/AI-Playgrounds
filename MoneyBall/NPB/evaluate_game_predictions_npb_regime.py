"""
evaluate_game_predictions_npb_regime.py

NPB game win/loss prediction with CPBL-style regime routing:
  early season -> fixed home-win baseline
  in season -> primary model with SP-feature median imputation

Backtest protocol:
  expanding walk-forward by season, train on all seasons before the test year,
  evaluate 2021-2025, and fit imputers only on each training split.
"""

from __future__ import annotations

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

TRAIN_LEAGUES = ("CL", "PL")
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


def load_rows() -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        placeholders = ",".join("?" for _ in TRAIN_LEAGUES)
        query = f"""
            SELECT *
            FROM {TABLE_NAME}
            WHERE home_win IS NOT NULL
              AND league_code IN ({placeholders})
            ORDER BY season_year, game_date, game_url
        """
        return [dict(row) for row in conn.execute(query, TRAIN_LEAGUES).fetchall()]
    finally:
        conn.close()


def validate_features(rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError("No training rows found in game_features_npb.")
    required = sorted({feature for features in REGIME_FEATURES.values() for feature in features})
    required += ["game_date", "game_url", "home_win", "season_year", "sp_available"]
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


def fit_logistic_model(rows: list[dict], features: list[str]) -> tuple[LogisticRegression, dict[str, float]]:
    medians = fit_medians(rows, features)
    x_train = transform(rows, features, medians)
    y_train = np.array([int(row["home_win"]) for row in rows], dtype=int)
    model = LogisticRegression(C=1.0, max_iter=500)
    model.fit(x_train, y_train)
    return model, medians


def predicted_side_probability(home_probability: float) -> float:
    return home_probability if home_probability >= 0.5 else 1.0 - home_probability


def has_primary_pitcher_features(row: dict) -> bool:
    return as_float(row.get("sp_available")) >= 0.5


def format_accuracy(value: float) -> str:
    return f"{value * 100:.2f}%"


def evaluate_all(rows: list[dict]) -> None:
    all_rows = rows

    test_rows = [row for row in rows if int(row["season_year"]) >= TEST_START_YEAR]
    test_rows.sort(key=lambda row: (row["game_date"], row.get("game_url") or ""))

    year_stats = defaultdict(lambda: {
        "total": 0,
        "correct": 0,
        "early_baseline": {"total": 0, "correct": 0},
        "fallback": {"total": 0, "correct": 0},
        "primary": {"total": 0, "correct": 0},
    })
    conf_buckets = {0.60: [], 0.70: [], 0.80: []}

    cached_date = None
    cached_models = {}

    for row in test_rows:
        game_date = row["game_date"]

        if game_date != cached_date:
            cached_models = {}
            train = [train_row for train_row in all_rows if train_row["game_date"] < game_date]
            if not train:
                continue

            fallback_train = [
                train_row
                for train_row in train
                if not is_early_season(train_row)
            ]
            if not fallback_train:
                continue
            cached_models["fallback"] = fit_model(fallback_train, FALLBACK_FEATURES)

            primary_train = [
                train_row
                for train_row in train
                if route_regime(train_row) == "primary"
            ]
            cached_models["primary"] = fit_model(primary_train, PRIMARY_FEATURES) if primary_train else None

            early_train = [train_row for train_row in train if route_regime(train_row) == "early_baseline"]
            if len(early_train) >= EARLY_MIN_TRAIN_ROWS:
                x_early = np.array(
                    [[as_float(train_row.get(feature)) for feature in EARLY_MODEL_FEATURES] for train_row in early_train],
                    dtype=float,
                )
                y_early = np.array([int(train_row["home_win"]) for train_row in early_train], dtype=int)
                lr = LogisticRegression(C=1.0, max_iter=500)
                lr.fit(x_early, y_early)
                cached_models["early"] = lr
            else:
                cached_models["early"] = None

            cached_date = game_date

        regime = route_regime(row)
        year = int(row["season_year"])

        if regime == "early_baseline":
            lr = cached_models.get("early")
            if lr is not None:
                x_early = np.array([[as_float(row.get(feature)) for feature in EARLY_MODEL_FEATURES]], dtype=float)
                home_probability = float(lr.predict_proba(x_early)[0, 1])
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

        year_stats[year]["total"] += 1
        year_stats[year]["correct"] += correct
        year_stats[year][regime]["total"] += 1
        year_stats[year][regime]["correct"] += correct

        side_probability = predicted_side_probability(home_probability)
        for threshold in conf_buckets:
            if side_probability >= threshold:
                conf_buckets[threshold].append(correct)

    total_correct = 0
    total_games = 0
    for year in sorted(year_stats):
        stats = year_stats[year]
        accuracy = stats["correct"] / stats["total"] if stats["total"] else 0.0
        total_correct += stats["correct"]
        total_games += stats["total"]
        early = stats["early_baseline"]
        fallback = stats["fallback"]
        primary = stats["primary"]
        print(
            f"year={year} total={stats['total']} correct={stats['correct']} "
            f"accuracy={accuracy:.2%} (baseline={BASELINE_ACCURACY:.1%})"
        )
        if early["total"]:
            print(f"  early_baseline: N={early['total']}   acc={early['correct'] / early['total']:.2%}")
        if fallback["total"]:
            print(f"  fallback:       N={fallback['total']}  acc={fallback['correct'] / fallback['total']:.2%}")
        if primary["total"]:
            print(f"  primary:        N={primary['total']}  acc={primary['correct'] / primary['total']:.2%}")

    overall = total_correct / total_games if total_games else 0.0
    print()
    print(f"=== Walk-Forward {TEST_START_YEAR}-2026 ===")
    print(f"total={total_games} correct={total_correct} accuracy={overall:.2%}")
    for threshold, results in sorted(conf_buckets.items()):
        n = len(results)
        accuracy = sum(results) / n if n else math.nan
        if n:
            print(f"high-conf (p>={threshold:.2f}): N={n} accuracy={accuracy:.2%}")
        else:
            print(f"high-conf (p>={threshold:.2f}): N=0 accuracy=nan%")


def main() -> None:
    rows = load_rows()
    validate_features(rows)

    min_year = min(int(row["season_year"]) for row in rows)
    if min_year > 2011:
        raise RuntimeError("Training data before 2016 is unavailable.")

    evaluate_all(rows)


if __name__ == "__main__":
    main()
