# -*- coding: utf-8 -*-
"""
train_nba_playoff_model.py

NBA 季後賽 walk-forward XGBoost backtest。
訓練窗口：5 年歷史季後賽（ROLLING_WINDOW_YEARS=5）
測試年：2016-2025
輸出：nba_playoff_walkforward.csv, nba_playoff_model_report.md
"""

import argparse
import csv
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression

DB_PATH = Path(__file__).resolve().parent / "nba.sqlite"
CSV_PATH = Path(__file__).resolve().parent / "nba_playoff_walkforward.csv"
REPORT_PATH = Path(__file__).resolve().parent / "nba_playoff_model_report.md"

TRAIN_START_YEAR = 2011
BACKTEST_START_YEAR = 2016
BACKTEST_END_YEAR = 2025
ROLLING_WINDOW_YEARS = 5
MIN_CALIB_ROWS = 30
MIN_TRAIN_ROWS = 50

FEATURES = [
    "diff_elo_rs",
    "diff_rs_net_rtg",
    "diff_rs_pyth_wp",
    "diff_rs_lineup_pts",
    "diff_elo_po",
    "elo_win_prob_po",
    "diff_elo_change_po",
    "game_in_series",
    "home_series_wins",
    "vis_series_wins",
    "series_score_diff",
    "is_elimination",
    "playoff_round",
    "series_rest_days",
    "home_has_homecourt",
]

XGB_PARAMS = dict(
    max_depth=3,
    min_child_weight=5,
    n_estimators=200,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.7,
    reg_lambda=2.0,
    reg_alpha=0.5,
    random_state=42,
    eval_metric="logloss",
    verbosity=0,
)


def parse_args():
    parser = argparse.ArgumentParser(description="NBA playoff walk-forward XGBoost backtest.")
    parser.add_argument("--year", type=int, help="Run only one test year (e.g. 2025).")
    return parser.parse_args()


def rolling_train_years(test_year):
    start_year = max(TRAIN_START_YEAR, test_year - ROLLING_WINDOW_YEARS)
    return list(range(start_year, test_year))


def load_rows(conn):
    query = f"""
        SELECT
            season_year,
            game_id,
            game_date,
            home_team_abbr,
            vis_team_abbr,
            home_win,
            {", ".join(FEATURES)}
        FROM playoff_game_features
        WHERE home_win IS NOT NULL
        ORDER BY season_year, game_date, game_id
    """
    rows = conn.execute(query).fetchall()
    columns = [item[0] for item in conn.execute(query + " LIMIT 0").description]
    return [dict(zip(columns, row)) for row in rows]


def compute_feature_medians(rows, features):
    medians = []
    for feature in features:
        values = [float(row[feature]) for row in rows if row.get(feature) is not None]
        medians.append(float(np.median(values)) if values else 0.0)
    return np.array(medians, dtype=float)


def matrix_from_rows(rows, features, medians):
    data = np.empty((len(rows), len(features)), dtype=float)
    for row_idx, row in enumerate(rows):
        for feature_idx, feature in enumerate(features):
            value = row.get(feature)
            data[row_idx, feature_idx] = medians[feature_idx] if value is None else float(value)
    return data


def target_vector(rows):
    return np.array([int(row["home_win"]) for row in rows], dtype=int)


def can_fit(rows):
    if len(rows) < MIN_TRAIN_ROWS:
        return False
    labels = {int(row["home_win"]) for row in rows}
    return len(labels) >= 2


def fit_xgb(rows, features):
    if not can_fit(rows):
        raise ValueError(f"insufficient training rows: n={len(rows)}")
    medians = compute_feature_medians(rows, features)
    X = matrix_from_rows(rows, features, medians)
    y = target_vector(rows)
    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(X, y)
    return model, medians


def predict_probs(model, medians, rows, features):
    X = matrix_from_rows(rows, features, medians)
    return model.predict_proba(X)[:, 1].astype(float)


def fit_calibrator(raw_probs, actuals):
    if len(raw_probs) < MIN_CALIB_ROWS:
        return None
    labels = np.array(actuals, dtype=float)
    if len(set(labels.tolist())) < 2:
        return None
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(np.array(raw_probs, dtype=float), labels)
    return calibrator


def apply_calibrator(calibrator, prob):
    if calibrator is None:
        return float(prob)
    return float(calibrator.predict(np.array([prob], dtype=float))[0])


def brier_score(probs, actuals):
    return float(np.mean((np.array(probs, dtype=float) - np.array(actuals, dtype=float)) ** 2))


def expected_calibration_error(probs, actuals, n_bins=10):
    probs = np.array(probs, dtype=float)
    actuals = np.array(actuals, dtype=float)
    n = len(probs)
    if n == 0:
        return 0.0
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for idx in range(n_bins):
        if idx == n_bins - 1:
            mask = (probs >= bin_edges[idx]) & (probs <= bin_edges[idx + 1])
        else:
            mask = (probs >= bin_edges[idx]) & (probs < bin_edges[idx + 1])
        if mask.sum() == 0:
            continue
        bin_acc = actuals[mask].mean()
        bin_conf = probs[mask].mean()
        ece += mask.sum() / n * abs(bin_conf - bin_acc)
    return float(ece)


def summarize_confidence(results, threshold, prob_key="prob_home_win_cal"):
    selected = []
    for row in results:
        prob = float(row[prob_key])
        confidence = prob if prob >= 0.5 else 1 - prob
        if confidence > threshold:
            selected.append(row)
    n = len(selected)
    if n == 0:
        return {"accuracy": 0.0, "games": 0, "coverage": 0.0}
    accuracy = sum(int(row["correct"]) for row in selected) / n
    coverage = n / len(results) if results else 0.0
    return {"accuracy": accuracy, "games": n, "coverage": coverage}


def collect_feature_importance(importance_accumulator, model, features):
    for feature, importance in zip(features, model.feature_importances_):
        importance_accumulator[feature].append(float(importance))


def build_year_results(test_rows, raw_probs, cal_probs):
    year_results = []
    for row, raw_prob, cal_prob in zip(test_rows, raw_probs, cal_probs):
        predicted_home_win = int(cal_prob >= 0.5)
        year_results.append({
            "season_year": row["season_year"],
            "game_id": row["game_id"],
            "game_date": row["game_date"],
            "home_team_abbr": row["home_team_abbr"],
            "vis_team_abbr": row["vis_team_abbr"],
            "home_win": int(row["home_win"]),
            "prob_home_win": round(float(raw_prob), 6),
            "prob_home_win_cal": round(float(cal_prob), 6),
            "predicted_home_win": predicted_home_win,
            "correct": int(predicted_home_win == int(row["home_win"])),
            "correct_raw": int((float(raw_prob) >= 0.5) == int(row["home_win"])),
        })
    return year_results


def run_fold(all_rows, test_year, importance_accumulator):
    train_years = rolling_train_years(test_year)
    train_rows = [row for row in all_rows if row["season_year"] in train_years]
    test_rows = [row for row in all_rows if row["season_year"] == test_year]
    if not train_rows or not test_rows:
        return None
    if not can_fit(train_rows):
        print(f"[{test_year}] Skipped: insufficient training rows (n={len(train_rows)})")
        return None

    model, medians = fit_xgb(train_rows, FEATURES)
    collect_feature_importance(importance_accumulator, model, FEATURES)

    calib_year = test_year - 1
    pretrain_years = [year for year in train_years if year < calib_year]
    calib_rows = [row for row in train_rows if row["season_year"] == calib_year]
    calibrator = None
    if len(calib_rows) >= MIN_CALIB_ROWS and pretrain_years:
        pretrain_rows = [row for row in train_rows if row["season_year"] in pretrain_years]
        if can_fit(pretrain_rows):
            calib_model, calib_medians = fit_xgb(pretrain_rows, FEATURES)
            calib_probs = predict_probs(calib_model, calib_medians, calib_rows, FEATURES)
            calibrator = fit_calibrator(calib_probs.tolist(), target_vector(calib_rows).tolist())

    raw_probs = predict_probs(model, medians, test_rows, FEATURES)
    cal_probs = np.array([apply_calibrator(calibrator, prob) for prob in raw_probs], dtype=float)
    year_results = build_year_results(test_rows, raw_probs, cal_probs)

    raw_acc = sum(row["correct_raw"] for row in year_results) / len(year_results)
    cal_acc = sum(row["correct"] for row in year_results) / len(year_results)
    high_conf = summarize_confidence(year_results, 0.65)

    return {
        "test_year": test_year,
        "train_years": train_years,
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "raw_accuracy": raw_acc,
        "cal_accuracy": cal_acc,
        "high_conf_065": high_conf,
        "results": year_results,
    }


def walkforward(all_rows, target_year=None):
    years = [target_year] if target_year is not None else list(range(BACKTEST_START_YEAR, BACKTEST_END_YEAR + 1))
    all_results = []
    year_stats = []
    importance_accumulator = defaultdict(list)

    print("=== NBA Playoff Walk-Forward XGBoost ===")
    for test_year in years:
        fold = run_fold(all_rows, test_year, importance_accumulator)
        if fold is None:
            continue
        year_stats.append(fold)
        all_results.extend(fold["results"])

        train_label = f"{fold['train_years'][0]}-{fold['train_years'][-1]}" if fold['train_years'] else "N/A"
        hc = fold["high_conf_065"]
        print(
            f"[{test_year}] train={train_label} (N={fold['train_rows']:,}) | "
            f"test={test_year} (N={fold['test_rows']:,})"
        )
        print(
            f"       raw={fold['raw_accuracy']:.1%} | cal={fold['cal_accuracy']:.1%} | "
            f"p>0.65: {hc['accuracy']:.1%} (N={hc['games']:,}, cov={hc['coverage']:.1%})"
        )

    if all_results:
        overall_cal = sum(row["correct"] for row in all_results) / len(all_results)
        overall_high_conf = summarize_confidence(all_results, 0.65)
        print(
            f"Overall: {overall_cal:.1%}  "
            f"High-conf p>0.65: {overall_high_conf['accuracy']:.1%} "
            f"(N={overall_high_conf['games']:,}, cov={overall_high_conf['coverage']:.1%})"
        )
    else:
        print("No results generated (insufficient data).")

    return all_results, year_stats, importance_accumulator


def write_csv(results):
    fieldnames = [
        "season_year", "game_id", "game_date",
        "home_team_abbr", "vis_team_abbr", "home_win",
        "prob_home_win", "prob_home_win_cal", "predicted_home_win", "correct",
    ]
    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({field: row[field] for field in fieldnames})


def top_feature_importance(importance_accumulator, top_n=10):
    ranked = []
    for feature, values in importance_accumulator.items():
        if not values:
            continue
        ranked.append((feature, float(np.mean(values))))
    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked[:top_n]


def write_report(year_stats, results, importance_accumulator):
    if not results:
        REPORT_PATH.write_text("# NBA Playoff Walk-Forward XGBoost\n\nNo results (insufficient data).\n", encoding="utf-8")
        return

    overall_raw = sum(row["correct_raw"] for row in results) / len(results)
    overall_cal = sum(row["correct"] for row in results) / len(results)
    high_conf_060 = summarize_confidence(results, 0.60)
    high_conf_065 = summarize_confidence(results, 0.65)
    calibrated_probs = [row["prob_home_win_cal"] for row in results]
    actuals = [row["home_win"] for row in results]
    overall_brier = brier_score(calibrated_probs, actuals)
    overall_ece = expected_calibration_error(calibrated_probs, actuals, n_bins=10)
    top10 = top_feature_importance(importance_accumulator)

    lines = [
        "# NBA Playoff Walk-Forward XGBoost",
        "",
        f"Backtest years: {year_stats[0]['test_year']}–{year_stats[-1]['test_year']}" if year_stats else "Backtest years: none",
        f"Rolling window: {ROLLING_WINDOW_YEARS} seasons",
        f"Features: {len(FEATURES)}",
        "",
        "> Note: Each playoff season has ~70-100 games. Single-season accuracy has high variance (±10pp CI). Evaluate cumulative multi-season trends.",
        "",
        "## Per-Season Accuracy",
        "",
        "| Year | Train Window | Train N | Test N | Raw Accuracy | Cal Accuracy |",
        "| ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for stat in year_stats:
        train_label = f"{stat['train_years'][0]}-{stat['train_years'][-1]}" if stat['train_years'] else "N/A"
        lines.append(
            f"| {stat['test_year']} | {train_label} | {stat['train_rows']:,} | {stat['test_rows']:,} "
            f"| {stat['raw_accuracy']:.2%} | {stat['cal_accuracy']:.2%} |"
        )

    lines += [
        "",
        "## Overall Accuracy",
        "",
        f"- Raw: {overall_raw:.2%}",
        f"- Calibrated: {overall_cal:.2%}",
        "",
        "## Calibration Metrics (Isotonic, calibrated probs)",
        "",
        f"- Brier Score: {overall_brier:.4f}  (lower is better; random = 0.25)",
        f"- ECE (10 bins): {overall_ece:.4f}  (lower is better; perfect = 0.00)",
        "",
        "## High-Confidence Subsets (calibrated confidence)",
        "",
        "| Threshold | Games | Coverage | Accuracy |",
        "| ---: | ---: | ---: | ---: |",
        f"| p_cal > 0.60 | {high_conf_060['games']:,} | {high_conf_060['coverage']:.1%} | {high_conf_060['accuracy']:.2%} |",
        f"| p_cal > 0.65 | {high_conf_065['games']:,} | {high_conf_065['coverage']:.1%} | {high_conf_065['accuracy']:.2%} |",
        "",
        "## Feature Importance (top 10, mean across folds)",
        "",
        "| Feature | Importance |",
        "| --- | ---: |",
    ]
    for feature, importance in top10:
        lines.append(f"| {feature} | {importance:.6f} |")

    lines += [
        "",
        "## Model Params",
        "",
        f"- max_depth={XGB_PARAMS['max_depth']}",
        f"- min_child_weight={XGB_PARAMS['min_child_weight']}",
        f"- n_estimators={XGB_PARAMS['n_estimators']}",
        f"- learning_rate={XGB_PARAMS['learning_rate']}",
        f"- subsample={XGB_PARAMS['subsample']}",
        f"- colsample_bytree={XGB_PARAMS['colsample_bytree']}",
        f"- reg_lambda={XGB_PARAMS['reg_lambda']}",
        f"- reg_alpha={XGB_PARAMS['reg_alpha']}",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    if args.year is not None and not (BACKTEST_START_YEAR <= args.year <= BACKTEST_END_YEAR):
        raise SystemExit(f"--year must be between {BACKTEST_START_YEAR} and {BACKTEST_END_YEAR}")

    conn = sqlite3.connect(DB_PATH)
    try:
        rows = load_rows(conn)
    finally:
        conn.close()

    print(f"Loaded {len(rows):,} playoff feature rows.")
    results, year_stats, importance_accumulator = walkforward(rows, target_year=args.year)
    if results:
        write_csv(results)
        write_report(year_stats, results, importance_accumulator)
        print(f"Saved CSV -> {CSV_PATH}")
        print(f"Saved report -> {REPORT_PATH}")
    else:
        print("No results to save. Run nba_playoff_scraper.py and build_nba_playoff_features.py first.")


if __name__ == "__main__":
    main()
