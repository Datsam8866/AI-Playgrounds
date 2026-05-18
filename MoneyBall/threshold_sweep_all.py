# -*- coding: utf-8 -*-
"""
Multi-league threshold sweep: accuracy & coverage for 60-65% (2020+).
Run from MoneyBall/ root directory.
"""
import sqlite3, sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent
THRESHOLDS = [0.60, 0.61, 0.62, 0.63, 0.64, 0.65]
SINCE_YEAR = 2020


def sweep(preds, league_name):
    """preds: list of {prob_home, actual, season_year}"""
    subset = [p for p in preds if int(p["season_year"]) >= SINCE_YEAR]
    years = sorted(set(int(p["season_year"]) for p in subset))

    print(f"\n{'='*68}")
    print(f"  {league_name}  (2020+, n_total={len(subset)})")
    print(f"{'='*68}")
    print(f"{'Thresh':>7}  {'Year':>6}  {'n_all':>6}  {'n_high':>7}  {'cov':>7}  {'acc_all':>8}  {'acc_high':>9}")
    print("  " + "-"*62)

    summary = {t: {"n_all": 0, "n_high": 0, "c_all": 0, "c_high": 0} for t in THRESHOLDS}

    for thresh in THRESHOLDS:
        for yr in years:
            yr_rows = [p for p in subset if int(p["season_year"]) == yr]
            n_all = len(yr_rows)
            c_all = sum(p["correct"] for p in yr_rows)
            high = [p for p in yr_rows if max(p["prob_home"], 1 - p["prob_home"]) >= thresh]
            n_high = len(high)
            c_high = sum(p["correct"] for p in high)
            cov = n_high / n_all if n_all else 0
            acc_all = c_all / n_all if n_all else 0
            acc_high = c_high / n_high if n_high else 0
            acc_h_str = f"{acc_high:.1%}" if n_high else "—"
            print(f"  {thresh:.0%}  {yr:>6}  {n_all:>6}  {n_high:>7}  {cov:>7.1%}  {acc_all:>8.1%}  {acc_h_str:>9}")
            summary[thresh]["n_all"]  += n_all
            summary[thresh]["n_high"] += n_high
            summary[thresh]["c_all"]  += c_all
            summary[thresh]["c_high"] += c_high
        print()

    print(f"\n{'':=<64}")
    print(f"  AGGREGATE {SINCE_YEAR}–2026  —  {league_name}")
    print(f"{'Thresh':>7}  {'n_all':>6}  {'n_high':>7}  {'coverage':>9}  {'acc_all':>8}  {'acc_high':>9}")
    print("  " + "-"*55)
    for thresh in THRESHOLDS:
        s = summary[thresh]
        cov = s["n_high"] / s["n_all"] if s["n_all"] else 0
        acc_all = s["c_all"] / s["n_all"] if s["n_all"] else 0
        acc_high = s["c_high"] / s["n_high"] if s["n_high"] else 0
        print(f"  {thresh:.0%}  {s['n_all']:>6}  {s['n_high']:>7}  {cov:>9.1%}  {acc_all:>8.1%}  {acc_high:>9.1%}")


# ── KBO ──────────────────────────────────────────────────────────────────────
def run_kbo():
    sys.path.insert(0, str(BASE_DIR / "KBO"))
    from evaluate_kbo_predictions_regime import (
        load_rows, train_models, predict_one, fit_platt, apply_platt,
        BACKTEST_START_YEAR, TRAIN_START_YEAR,
    )
    conn = sqlite3.connect(str(BASE_DIR / "KBO" / "kbo.sqlite"))
    conn.row_factory = sqlite3.Row
    all_rows = load_rows(conn)
    conn.close()
    print(f"KBO: {len(all_rows)} rows loaded", flush=True)

    preds = []
    for yr in range(SINCE_YEAR, 2027):
        train = [r for r in all_rows if r["season_year"] < yr]
        test  = [r for r in all_rows if r["season_year"] == yr]
        if not train or not test:
            continue
        calib_rows   = [r for r in train if r["season_year"] == yr - 1]
        pretrain_rows = [r for r in train if r["season_year"] < yr - 1]
        platt_scaler = None
        if pretrain_rows and calib_rows:
            pretrain_models = train_models(pretrain_rows)
            calib_probs = [predict_one(pretrain_models, r)[0] for r in calib_rows]
            platt_scaler = fit_platt(calib_probs, [r["home_win"] for r in calib_rows])
        models = train_models(train)
        for row in test:
            prob, _ = predict_one(models, row)
            prob = apply_platt(platt_scaler, prob)
            pred = 1 if prob >= 0.5 else 0
            preds.append({"prob_home": prob, "actual": row["home_win"],
                          "season_year": yr, "correct": int(pred == row["home_win"])})
        print(f"  KBO {yr}: {len(test)} games done", flush=True)

    sweep(preds, "KBO")


# ── NPB ──────────────────────────────────────────────────────────────────────
def run_npb():
    sys.path.insert(0, str(BASE_DIR / "NPB"))
    import evaluate_game_predictions_npb_regime as npb_ev

    # Override DB path to work from root
    import types
    orig_load = npb_ev.load_rows

    def load_rows_fixed():
        conn = sqlite3.connect(str(BASE_DIR / "NPB" / "npb.sqlite"))
        conn.row_factory = sqlite3.Row
        rows_raw = conn.execute(
            f"SELECT * FROM {npb_ev.TABLE_NAME} WHERE home_win IS NOT NULL"
            " ORDER BY season_year, game_date, game_url"
        ).fetchall()
        conn.close()
        rows = [dict(r) for r in rows_raw]
        for r in rows:
            r["season_year"] = int(r["season_year"])
        return rows

    all_rows = load_rows_fixed()
    print(f"NPB: {len(all_rows)} rows loaded", flush=True)

    preds = []
    for yr in range(SINCE_YEAR, 2027):
        train = [r for r in all_rows if r["season_year"] < yr]
        test  = [r for r in all_rows if r["season_year"] == yr]
        if not train or not test:
            continue
        # Build regime models for this year
        models = {}
        fallback_train = [r for r in train if npb_ev.route_regime(r) in ("fallback", "primary")]
        if fallback_train:
            models["fallback"], models["fallback_medians"] = npb_ev.fit_model(fallback_train, npb_ev.FALLBACK_FEATURES)
        primary_train = [r for r in train if npb_ev.route_regime(r) == "primary"]
        if primary_train:
            models["primary"], models["primary_medians"] = npb_ev.fit_model(primary_train, npb_ev.PRIMARY_FEATURES)
        early_train = [r for r in train if npb_ev.is_early_season(r)]
        if early_train:
            models["early"], models["early_medians"] = npb_ev.fit_model(early_train, npb_ev.EARLY_MODEL_FEATURES)

        for row in test:
            route = "early" if npb_ev.is_early_season(row) else npb_ev.route_regime(row)
            feat_key = route if route in models else "fallback"
            if feat_key not in models:
                continue
            m, med = models[feat_key], models[feat_key + "_medians"]
            feat_name = (npb_ev.EARLY_MODEL_FEATURES if feat_key == "early" else
                         (npb_ev.PRIMARY_FEATURES if feat_key == "primary" else npb_ev.FALLBACK_FEATURES))
            X = npb_ev.transform([row], feat_name, med)
            prob = float(m.predict_proba(X)[0][1])
            pred = 1 if prob >= 0.5 else 0
            preds.append({"prob_home": prob, "actual": row["home_win"],
                          "season_year": yr, "correct": int(pred == int(row["home_win"]))})
        print(f"  NPB {yr}: {len(test)} games done", flush=True)

    sweep(preds, "NPB")


# ── CPBL ─────────────────────────────────────────────────────────────────────
def run_cpbl():
    sys.path.insert(0, str(BASE_DIR / "CPBL"))
    import importlib.util, types

    # CPBL uses a complex regime model; we replicate with XGBoost on game_features
    import numpy as np
    import xgboost as xgb

    FEATURES = [
        "diff_win_pct", "diff_rs", "diff_ra", "diff_rd", "diff_pyth_wp",
        "diff_sp_era", "diff_sp_whip", "diff_sp_k9",
    ]

    conn = sqlite3.connect(str(BASE_DIR / "CPBL" / "cpbl.sqlite"))
    conn.row_factory = sqlite3.Row
    rows_raw = conn.execute(
        "SELECT season_year, game_date, home_team_code, vis_team_code, home_win,"
        " diff_win_pct, diff_rs, diff_ra, diff_rd, diff_pyth_wp,"
        " diff_sp_era, diff_sp_whip, diff_sp_k9, diff_sp_ip, home_n_games"
        " FROM game_features WHERE home_win IS NOT NULL"
        " ORDER BY season_year, game_date, game_sno"
    ).fetchall()
    conn.close()
    all_rows = []
    for r in rows_raw:
        d = dict(r)
        d["season_year"] = int(d["season_year"])
        d["home_win"] = int(d["home_win"])
        # fill None with 0
        for f in FEATURES:
            if d.get(f) is None:
                d[f] = 0.0
        all_rows.append(d)
    print(f"CPBL: {len(all_rows)} rows loaded", flush=True)

    def fit_xgb_cpbl(rows):
        X = np.array([[r.get(f, 0.0) or 0.0 for f in FEATURES] for r in rows], dtype=float)
        y = np.array([r["home_win"] for r in rows], dtype=float)
        m = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05,
                               subsample=0.8, use_label_encoder=False,
                               eval_metric="logloss", verbosity=0)
        m.fit(X, y)
        return m

    preds = []
    for yr in range(SINCE_YEAR, 2027):
        train = [r for r in all_rows if r["season_year"] < yr]
        test  = [r for r in all_rows if r["season_year"] == yr]
        if len(train) < 50 or not test:
            continue
        model = fit_xgb_cpbl(train)
        X_test = np.array([[r.get(f, 0.0) or 0.0 for f in FEATURES] for r in test], dtype=float)
        probs = model.predict_proba(X_test)[:, 1]
        for row, prob in zip(test, probs):
            prob = float(prob)
            pred = 1 if prob >= 0.5 else 0
            preds.append({"prob_home": prob, "actual": row["home_win"],
                          "season_year": yr, "correct": int(pred == row["home_win"])})
        print(f"  CPBL {yr}: {len(test)} games done", flush=True)

    sweep(preds, "CPBL")


# ── NBA ───────────────────────────────────────────────────────────────────────
def run_nba():
    sys.path.insert(0, str(BASE_DIR / "NBA"))
    from train_nba_model import (
        FEATURES as RS_FEATURES, fit_xgb, predict_probs,
        apply_calibrator, fit_calibrator, target_vector,
        rolling_train_years, MIN_CALIB_ROWS,
    )
    from train_nba_playoff_model import (
        FEATURES as PO_FEATURES, MIN_CALIB_ROWS as PO_MIN_CALIB,
        fit_xgb as po_fit_xgb, predict_probs as po_predict_probs,
        apply_calibrator as po_apply_calibrator,
        fit_calibrator as po_fit_calibrator,
        target_vector as po_target_vector,
        rolling_train_years as po_rolling_train_years,
        can_fit,
    )

    conn = sqlite3.connect(str(BASE_DIR / "NBA" / "nba.sqlite"))
    conn.row_factory = sqlite3.Row

    # Regular season
    rs_rows = [dict(r) for r in conn.execute(
        f"SELECT season_year, game_id, game_date, home_team_abbr, vis_team_abbr, home_win,"
        f" {', '.join(RS_FEATURES)}"
        f" FROM game_features WHERE home_win IS NOT NULL ORDER BY season_year, game_date, game_id"
    ).fetchall()]
    print(f"NBA RS: {len(rs_rows)} rows loaded", flush=True)

    # Playoff
    po_rows = [dict(r) for r in conn.execute(
        f"SELECT season_year, game_id, game_date, home_team_abbr, vis_team_abbr, home_win,"
        f" {', '.join(PO_FEATURES)}"
        f" FROM playoff_game_features WHERE home_win IS NOT NULL ORDER BY season_year, game_date, game_id"
    ).fetchall()]
    print(f"NBA PO: {len(po_rows)} rows loaded", flush=True)
    conn.close()

    def do_nba_sweep(all_rows, features, fit_fn, pred_fn, calib_fn, tcal_fn, tc_fn, apc_fn, roll_fn, min_cal, label):
        preds = []
        for yr in range(SINCE_YEAR, 2027):
            train_years = roll_fn(yr + 1)
            train = [r for r in all_rows if int(r["season_year"]) in train_years
                     and str(r["game_date"]) < f"{yr}-01-01"]
            test  = [r for r in all_rows if int(r["season_year"]) == yr]
            if not train or not test:
                continue
            if callable(getattr(__builtins__ if isinstance(__builtins__, dict) else type(__builtins__), 'get', None)):
                pass
            # check can_fit for PO
            try:
                if not can_fit(train):
                    continue
            except Exception:
                pass
            model, medians = fit_fn(train, features)
            latest = max(int(r["season_year"]) for r in train)
            calib = [r for r in train if int(r["season_year"]) == latest]
            pretrain = [r for r in train if int(r["season_year"]) < latest]
            calibrator = None
            if len(calib) >= min_cal and pretrain:
                cm, cmd = fit_fn(pretrain, features)
                cp = pred_fn(cm, cmd, calib, features)
                calibrator = calib_fn(cp.tolist(), tcal_fn(calib).tolist())
            elif len(calib) >= min_cal:
                fp = pred_fn(model, medians, calib, features)
                calibrator = calib_fn(fp.tolist(), tcal_fn(calib).tolist())
            raw_probs = pred_fn(model, medians, test, features)
            for row, raw in zip(test, raw_probs):
                cal = apc_fn(calibrator, float(raw))
                pred = 1 if cal >= 0.5 else 0
                preds.append({"prob_home": cal, "actual": int(row["home_win"]),
                              "season_year": yr, "correct": int(pred == int(row["home_win"]))})
            print(f"  NBA {label} {yr}: {len(test)} games done", flush=True)
        return preds

    rs_preds = do_nba_sweep(rs_rows, RS_FEATURES, fit_xgb, predict_probs,
                             fit_calibrator, target_vector, None, apply_calibrator,
                             rolling_train_years, MIN_CALIB_ROWS, "RS")
    sweep(rs_preds, "NBA Regular Season")

    po_preds = do_nba_sweep(po_rows, PO_FEATURES, po_fit_xgb, po_predict_probs,
                             po_fit_calibrator, po_target_vector, None, po_apply_calibrator,
                             po_rolling_train_years, PO_MIN_CALIB, "PO")
    sweep(po_preds, "NBA Playoffs")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("league", nargs="?", default="all",
                        choices=["kbo", "npb", "cpbl", "nba", "all"])
    args = parser.parse_args()

    lg = args.league.lower()
    if lg in ("kbo", "all"):
        run_kbo()
    if lg in ("npb", "all"):
        run_npb()
    if lg in ("cpbl", "all"):
        run_cpbl()
    if lg in ("nba", "all"):
        run_nba()
