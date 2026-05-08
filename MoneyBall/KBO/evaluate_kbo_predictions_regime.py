"""
evaluate_kbo_predictions_regime.py

Regime-routing XGBoost walk-forward evaluation for KBO.

Routing:
  early_flag → either team < TEAM_BURN_IN games this season
  sp_available → diff_sp_era IS NOT NULL (both SPs have >= 5 prior starts)

  early_flag = True  → early_model   (13 features, prior-heavy)
  early_flag = False, sp_available   → primary_model  (29 features)
  early_flag = False, !sp_available  → fallback_model (24 features)

Walk-forward: season-level expanding window
  Train on TRAIN_START .. Y-1, test on season Y.

Elo params (Gemini-reviewed): K=48, home_adv=10, regression=0.50
"""

import sqlite3
import sys
from pathlib import Path

import numpy as np
import xgboost as xgb

DB_PATH            = Path("kbo.sqlite")
REPORT_PATH        = Path("kbo_regime_benchmark.md")
TRAIN_START_YEAR   = 2013   # 2 years of burn-in from DATA_START=2011
BACKTEST_START_YEAR = 2016  # First full 10-team era year with prior-season data
BACKTEST_END_YEAR  = 2025
TEAM_BURN_IN       = 10
STARTER_BURN_IN    = 5      # used via sp_available flag from game_features
MIN_EARLY_TRAIN    = 50

XGB_PARAMS = {
    "n_estimators":      30,
    "max_depth":         3,
    "learning_rate":     0.05,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "min_child_weight":  30,
    "reg_lambda":        3.0,
    "objective":         "binary:logistic",
    "eval_metric":       "logloss",
    "verbosity":         0,
    "use_label_encoder": False,
    "random_state":      42,
    "n_jobs":            1,
}

EARLY_FEATURES = [
    "diff_elo", "home_elo", "away_elo", "elo_home_prob",
    "prev_diff_win_pct", "prev_diff_rd_pg", "prev_diff_pyth",
    "home_rest", "away_rest", "diff_rest",
    "diff_streak",
    "home_season_games_before", "away_season_games_before",
]

FALLBACK_FEATURES = [
    "diff_elo", "home_elo", "away_elo",
    "diff_win_pct", "diff_rs", "diff_ra", "diff_rd", "diff_pyth_wp",
    "diff_w3_win_pct", "diff_w5_win_pct", "diff_w10_win_pct",
    "diff_w3_rd_pg",   "diff_w5_rd_pg",   "diff_w10_rd_pg",
    "diff_split5_win_pct", "diff_split10_win_pct",
    "diff_split5_rd_pg",   "diff_split10_rd_pg",
    "diff_trend_win_pct",  "diff_trend_rd_pg",
    "home_rest", "away_rest", "diff_rest",
    "diff_streak",
]

PRIMARY_FEATURES = FALLBACK_FEATURES + [
    "diff_sp_era", "diff_sp_whip", "diff_sp_k9", "diff_sp_ip",
    "sp_available",
    "park_factor", "stadium_hwa",
    "diff_split5_rs_pg", "diff_split5_ra_pg",
    "diff_split10_rs_pg", "diff_split10_ra_pg",
]

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ── data loading ──────────────────────────────────────────────────────────────

def load_rows(conn) -> list[dict]:
    rows = conn.execute("""
        SELECT *
        FROM game_features
        WHERE sr_id = 0
        ORDER BY game_date, game_id
    """).fetchall()

    cols = [d[1] for d in conn.execute("PRAGMA table_info(game_features)").fetchall()]
    result = []
    for r in rows:
        d = dict(zip(cols, r))
        # derived flags
        d["early_flag"]    = int(
            d.get("home_season_games_before", 0) < TEAM_BURN_IN
            or d.get("away_season_games_before", 0) < TEAM_BURN_IN
        )
        d["sp_available"]  = int(d.get("diff_sp_era") is not None)
        result.append(d)
    return result


# ── XGBoost helpers ───────────────────────────────────────────────────────────

def to_matrix(rows: list[dict], features: list[str]) -> np.ndarray:
    X = []
    for r in rows:
        X.append([r.get(f) if r.get(f) is not None else np.nan for f in features])
    return np.array(X, dtype=float)


def fit_model(rows: list[dict], features: list[str]) -> xgb.XGBClassifier:
    X = to_matrix(rows, features)
    y = np.array([r["home_win"] for r in rows], dtype=float)
    m = xgb.XGBClassifier(**XGB_PARAMS)
    m.fit(X, y)
    return m


def predict_one(models: dict, row: dict) -> tuple[float, str]:
    """Return (prob_home_win, model_used)."""
    if row["early_flag"]:
        m = models.get("early")
        if m is None:
            return 0.5, "early_missing"
        X = to_matrix([row], EARLY_FEATURES)
        return float(m.predict_proba(X)[0, 1]), "early"

    if row["sp_available"]:
        m = models.get("primary")
        if m is None:
            # fall back to fallback
            m = models.get("fallback")
            if m is None:
                return 0.5, "fallback_missing"
            X = to_matrix([row], FALLBACK_FEATURES)
            return float(m.predict_proba(X)[0, 1]), "fallback"
        X = to_matrix([row], PRIMARY_FEATURES)
        return float(m.predict_proba(X)[0, 1]), "primary"
    else:
        m = models.get("fallback")
        if m is None:
            return 0.5, "fallback_missing"
        X = to_matrix([row], FALLBACK_FEATURES)
        return float(m.predict_proba(X)[0, 1]), "fallback"


def train_models(train_rows: list[dict]) -> dict:
    models = {}
    early  = [r for r in train_rows if r["early_flag"]]
    nosp   = [r for r in train_rows if not r["early_flag"]]
    prim   = [r for r in nosp if r["sp_available"]]

    if len(early) >= MIN_EARLY_TRAIN:
        models["early"] = fit_model(early, EARLY_FEATURES)
    elif train_rows:
        models["early"] = fit_model(train_rows, EARLY_FEATURES)

    if len(nosp) >= 20:
        models["fallback"] = fit_model(nosp, FALLBACK_FEATURES)
    elif train_rows:
        models["fallback"] = fit_model(train_rows, FALLBACK_FEATURES)

    if len(prim) >= 20:
        models["primary"] = fit_model(prim, PRIMARY_FEATURES)

    return models


# ── walk-forward evaluation ───────────────────────────────────────────────────

def home_baseline(rows: list[dict]) -> float:
    wins = sum(r["home_win"] for r in rows)
    return wins / len(rows) if rows else 0.0


def evaluate_walkforward(all_rows: list[dict]) -> dict:
    """Season-level expanding walk-forward for BACKTEST_START..BACKTEST_END."""
    test_rows_all = [
        r for r in all_rows
        if BACKTEST_START_YEAR <= r["season_year"] <= BACKTEST_END_YEAR
    ]

    results_by_year = {}
    correct_total = 0
    games_total   = 0

    # threshold buckets for high-confidence analysis
    buckets = {0.55: [0, 0], 0.60: [0, 0], 0.70: [0, 0], 0.80: [0, 0]}

    for yr in range(BACKTEST_START_YEAR, BACKTEST_END_YEAR + 1):
        train = [r for r in all_rows if TRAIN_START_YEAR <= r["season_year"] < yr]
        test  = [r for r in all_rows if r["season_year"] == yr]
        if not train or not test:
            continue

        models = train_models(train)
        yr_correct = 0
        yr_model_counts = {"early": 0, "primary": 0, "fallback": 0}

        for row in test:
            prob, model_used = predict_one(models, row)
            pred = 1 if prob >= 0.5 else 0
            ok   = int(pred == row["home_win"])
            yr_correct += ok
            correct_total += ok
            games_total   += 1
            key = model_used.split("_")[0]  # 'early'/'primary'/'fallback'
            yr_model_counts[key] = yr_model_counts.get(key, 0) + 1

            # high-confidence (predicted side's confidence)
            conf = prob if prob >= 0.5 else (1 - prob)
            for thr in buckets:
                if conf >= thr:
                    buckets[thr][0] += ok
                    buckets[thr][1] += 1

        results_by_year[yr] = {
            "games":    len(test),
            "correct":  yr_correct,
            "accuracy": yr_correct / len(test) if test else 0,
            "model_counts": yr_model_counts,
        }
        print(f"  {yr}: {len(test)} games, {yr_correct/len(test):.2%} "
              f"  (early={yr_model_counts.get('early',0)} "
              f"primary={yr_model_counts.get('primary',0)} "
              f"fallback={yr_model_counts.get('fallback',0)})")

    baseline = home_baseline(test_rows_all)

    return {
        "by_year":       results_by_year,
        "total_games":   games_total,
        "total_correct": correct_total,
        "accuracy":      correct_total / games_total if games_total else 0,
        "baseline":      baseline,
        "buckets":       {thr: (c/n if n else 0, n) for thr, (c, n) in buckets.items()},
    }


def evaluate_season(all_rows: list[dict], season: int) -> dict:
    """Single-season evaluation (e.g. 2026 YTD)."""
    train = [r for r in all_rows if TRAIN_START_YEAR <= r["season_year"] < season]
    test  = [r for r in all_rows if r["season_year"] == season]
    if not train or not test:
        return {"games": 0, "correct": 0, "accuracy": 0.0}

    models = train_models(train)
    correct = 0
    model_counts = {}
    for row in test:
        prob, model_used = predict_one(models, row)
        pred = 1 if prob >= 0.5 else 0
        correct += int(pred == row["home_win"])
        key = model_used.split("_")[0]
        model_counts[key] = model_counts.get(key, 0) + 1

    return {
        "games":        len(test),
        "correct":      correct,
        "accuracy":     correct / len(test) if test else 0,
        "model_counts": model_counts,
    }


# ── report ────────────────────────────────────────────────────────────────────

def write_report(wf: dict, season_2026: dict):
    lines = [
        "# KBO Regime Model — Walk-Forward Benchmark",
        "",
        f"Train start: {TRAIN_START_YEAR} | Backtest: {BACKTEST_START_YEAR}–{BACKTEST_END_YEAR}",
        f"Elo: K=48, home_adv=10, regression=0.50",
        f"XGBoost: max_depth={XGB_PARAMS['max_depth']}, min_child_weight={XGB_PARAMS['min_child_weight']}, n_estimators={XGB_PARAMS['n_estimators']}",
        "",
        "## Walk-Forward Results (2016–2025)",
        "",
        f"| Metric | Value |",
        f"| --- | ---: |",
        f"| Total games | {wf['total_games']} |",
        f"| Correct | {wf['total_correct']} |",
        f"| **Accuracy** | **{wf['accuracy']:.2%}** |",
        f"| Home baseline | {wf['baseline']:.2%} |",
        "",
        "## Per-Year Breakdown",
        "",
        "| Year | Games | Accuracy | early | primary | fallback |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for yr, r in sorted(wf["by_year"].items()):
        mc = r["model_counts"]
        lines.append(
            f"| {yr} | {r['games']} | {r['accuracy']:.2%} "
            f"| {mc.get('early',0)} | {mc.get('primary',0)} | {mc.get('fallback',0)} |"
        )

    lines += [
        "",
        "## High-Confidence Subset (2016–2025)",
        "",
        "| Threshold | Games | Coverage | Accuracy |",
        "| ---: | ---: | ---: | ---: |",
    ]
    total_test = wf["total_games"]
    for thr in sorted(wf["buckets"]):
        acc, n = wf["buckets"][thr]
        cov = n / total_test if total_test else 0
        lines.append(f"| p >= {thr:.2f} | {n} | {cov:.1%} | {acc:.2%} |")

    lines += [
        "",
        "## 2026 YTD",
        "",
        f"| Games | Correct | Accuracy |",
        f"| ---: | ---: | ---: |",
        f"| {season_2026['games']} | {season_2026['correct']} | {season_2026['accuracy']:.2%} |",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nReport written → {REPORT_PATH}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    conn = sqlite3.connect(DB_PATH)
    try:
        print("Loading game_features…")
        all_rows = load_rows(conn)
        print(f"  {len(all_rows)} rows loaded (sr_id=0)")

        baseline = home_baseline([r for r in all_rows if BACKTEST_START_YEAR <= r["season_year"] <= BACKTEST_END_YEAR])
        print(f"  Home baseline ({BACKTEST_START_YEAR}–{BACKTEST_END_YEAR}): {baseline:.2%}")

        print(f"\nWalk-forward {BACKTEST_START_YEAR}–{BACKTEST_END_YEAR}:")
        wf = evaluate_walkforward(all_rows)

        print(f"\nOverall accuracy: {wf['accuracy']:.4f} ({wf['total_correct']}/{wf['total_games']})")
        print(f"Home baseline:    {wf['baseline']:.4f}")
        print(f"Delta vs baseline: +{wf['accuracy']-wf['baseline']:.4f}")

        print("\nHigh-confidence (2016–2025):")
        for thr in sorted(wf["buckets"]):
            acc, n = wf["buckets"][thr]
            cov = n / wf["total_games"] if wf["total_games"] else 0
            print(f"  p>={thr:.2f}: {n} games ({cov:.1%} coverage) → {acc:.2%}")

        print("\n2026 YTD:")
        s26 = evaluate_season(all_rows, 2026)
        mc = s26.get("model_counts", {})
        print(f"  {s26['games']} games, {s26['correct']} correct, {s26['accuracy']:.2%}")
        print(f"  models: early={mc.get('early',0)} primary={mc.get('primary',0)} fallback={mc.get('fallback',0)}")

        write_report(wf, s26)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
