"""
train_mlb_model.py

MLB Soft-regime XGBoost — year-by-year expanding walk-forward backtest.

Three models per fold:
  early    : Elo + prev-season priors + context  (burn-in games)
  fallback : rolling stats + Elo + rest/streak   (no SP required)
  primary  : fallback + SP rolling ERA/WHIP/K9   (when both SPs ≥ STARTER_BURN_IN starts)

Blending: soft weights fade early→fallback→primary based on team/SP readiness.
Walk-forward: train on 2011..Y-1, test on Y, for Y in 2014..2025.

Outputs: mlb_walkforward.csv, mlb_model_report.md
"""

import csv
import sqlite3
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

DB_PATH = Path("mlb.sqlite")
CSV_PATH = Path("mlb_walkforward.csv")
REPORT_PATH = Path("mlb_model_report.md")

BACKTEST_START_YEAR = 2014
BACKTEST_END_YEAR   = 2026

TEAM_BURN_IN    = 10
STARTER_BURN_IN = 4
EARLY_PROB_SHRINK = 0.55
MIN_EARLY_TRAIN   = 100

ELO_BASE = 1500.0

XGB_PARAMS = dict(
    max_depth=3, min_child_weight=15, n_estimators=200,
    learning_rate=0.04, subsample=0.8, colsample_bytree=0.7,
    reg_lambda=3.0, reg_alpha=0.5, random_state=42, eval_metric="logloss", verbosity=0,
)

FALLBACK_FEATURES = [
    "diff_win_pct", "diff_pyth_wp", "diff_rd",
    "diff_elo", "home_elo", "vis_elo",
    "diff_w5_win_pct", "diff_w5_rd_pg",
    "diff_w10_win_pct", "diff_w10_rd_pg",
    "diff_w30_win_pct", "diff_w30_rd_pg",
    "diff_split5_win_pct", "diff_split5_rd_pg",
    "diff_split10_win_pct", "diff_split10_rd_pg",
    "diff_trend_win_pct", "diff_trend_rd_pg",
    "home_rest", "vis_rest", "diff_rest", "diff_streak",
    "universal_dh_era", "coors_field_factor", "is_interleague",
    "home_park_factor", "series_game_no",
    "diff_pyth_residual", "diff_elo_momentum",
]
PRIMARY_FEATURES = FALLBACK_FEATURES + [
    "diff_sp_era", "diff_sp_whip", "diff_sp_k9", "diff_sp_kbb", "sp_available",
]
EARLY_FEATURES = [
    "diff_elo", "home_elo", "vis_elo", "elo_win_prob",
    "prev_diff_win_pct", "prev_diff_rd_pg", "prev_diff_pyth",
    "home_rest", "vis_rest", "diff_rest", "diff_streak",
    "home_season_games_before", "vis_season_games_before",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ip_mlb_to_float(ip):
    if ip is None:
        return 0.0
    whole = int(ip)
    outs = round((ip - whole) * 10)
    return whole + outs / 3.0


def pyth_wp(rs, ra):
    if rs + ra == 0:
        return 0.5
    return rs ** 2 / (rs ** 2 + ra ** 2)


def rolling_kbb(history, window=5, min_starts=4):
    subset = history[-window:]
    if len(subset) < min_starts:
        return None
    k  = sum(g["k"]  for g in subset)
    bb = sum(g["bb"] for g in subset)
    return k / max(bb, 1)


def summarize(games):
    if not games:
        return None
    n = len(games)
    rs = sum(g["rs"] for g in games)
    ra = sum(g["ra"] for g in games)
    wins = sum(g["win"] for g in games)
    return {"win_pct": wins / n, "rd_pg": (rs - ra) / n, "pyth": pyth_wp(rs, ra)}


def streak_val(games):
    if not games:
        return 0
    last = games[-1]["win"]
    s = 0
    for g in reversed(games):
        if g["win"] != last:
            break
        s += 1 if last > 0.5 else -1
    return s


def rest_days(last_d, cur_d):
    if last_d is None:
        return 0
    return max(0, min((cur_d - last_d).days - 1, 10))


def nn(v, default=0.0):
    """None → default, else v."""
    return default if v is None else v


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------

def build_rows(conn, cutoff_date=None):
    # ---- slow features from game_features -----------------------------------
    col_names = [d[0] for d in conn.execute("SELECT * FROM game_features LIMIT 0").description]
    gf_map = {
        row[col_names.index("game_pk")]: dict(zip(col_names, row))
        for row in conn.execute("SELECT * FROM game_features").fetchall()
    }

    # ---- SP pitcher IDs + stats for burn-in tracking and K/BB ---------------
    sp_pid = defaultdict(dict)   # game_pk -> {team_id: {pitcher_id, k, bb}}
    for game_pk, team_id, pitcher_id, k, bb in conn.execute(
        "SELECT game_pk, team_id, pitcher_id, k, bb FROM game_starting_pitchers WHERE pitcher_id IS NOT NULL"
    ).fetchall():
        sp_pid[game_pk][team_id] = {"pitcher_id": pitcher_id, "k": k or 0, "bb": bb or 0}

    # ---- Raw games in order --------------------------------------------------
    if cutoff_date is not None:
        raw = conn.execute("""
            SELECT game_pk, season_year, game_date,
                   home_team_id, vis_team_id,
                   home_score, vis_score, winner
            FROM team_game_results
            WHERE status = 'Final'
              AND home_score IS NOT NULL AND vis_score IS NOT NULL
              AND home_score != vis_score
              AND date(game_date) < ?
            ORDER BY game_date, game_pk
        """, (cutoff_date.isoformat(),)).fetchall()
    else:
        raw = conn.execute("""
            SELECT game_pk, season_year, game_date,
                   home_team_id, vis_team_id,
                   home_score, vis_score, winner
            FROM team_game_results
            WHERE status = 'Final'
              AND home_score IS NOT NULL AND vis_score IS NOT NULL
              AND home_score != vis_score
            ORDER BY game_date, game_pk
        """).fetchall()

    # ---- Pre-compute prev-season team summaries (for early model priors) ----
    season_team_games = defaultdict(list)
    for game_pk, yr, gd, hid, vid, hs, vs, winner in raw:
        hw = 1 if winner == "home" else 0
        season_team_games[(yr, hid)].append({"rs": hs, "ra": vs, "win": hw})
        season_team_games[(yr, vid)].append({"rs": vs, "ra": hs, "win": 1 - hw})
    prev_stats = {}
    for (yr, tid), games in season_team_games.items():
        s = summarize(games)
        if s:
            prev_stats[(yr + 1, tid)] = s

    # ---- Pre-compute park factors (rolling 2-year home runs/game / league avg) ----
    # For each (season, home_team): avg total runs at home vs league avg
    season_home_runs = defaultdict(list)  # season_year -> [(home_team_id, total_runs)]
    for game_pk, yr, gd, hid, vid, hs, vs, winner in raw:
        season_home_runs[yr].append((hid, hs + vs))
    park_factors = {}
    seasons = sorted(season_home_runs.keys())
    for i, yr in enumerate(seasons):
        window_data = []
        for j in range(max(0, i - 1), i + 1):  # 2-year rolling
            window_data.extend(season_home_runs[seasons[j]])
        if not window_data:
            continue
        league_avg = sum(r for _, r in window_data) / len(window_data)
        team_data = defaultdict(list)
        for tid, r in window_data:
            team_data[tid].append(r)
        for tid, runs_list in team_data.items():
            park_factors[(yr, tid)] = (sum(runs_list) / len(runs_list)) / max(league_avg, 1)

    # ---- Single forward pass ------------------------------------------------
    hist = defaultdict(list)          # team_id → [{rs,ra,win}]
    home_hist = defaultdict(list)     # team_id → home-only history
    away_hist = defaultdict(list)     # team_id → away-only history
    last_date = {}                    # team_id → datetime.date
    sp_starts = defaultdict(int)      # pitcher_id → starts before this game
    sp_kbb_hist = defaultdict(list)   # pitcher_id → [{k, bb}]
    elo_hist = defaultdict(list)      # team_id → [pre-game elo, ...]
    team_season_count = defaultdict(int)
    matchup_last = {}                 # (min_id, max_id) → (date, series_game_no)
    current_season = None
    result_rows = []

    for game_pk, season_year, game_date_str, home_id, vis_id, home_score, vis_score, winner in raw:
        game_date = datetime.fromisoformat(game_date_str[:10]).date()
        home_win  = 1 if winner == "home" else 0

        if season_year != current_season:
            if current_season is not None:
                team_season_count = defaultdict(int)
            current_season = season_year

        gf = gf_map.get(game_pk)

        if gf is not None:
            # Short-window rolling
            w5_h  = summarize(hist[home_id][-5:])
            w5_v  = summarize(hist[vis_id][-5:])
            w10_h = summarize(hist[home_id][-10:])
            w10_v = summarize(hist[vis_id][-10:])
            w20_h = summarize(hist[home_id][-20:])
            w20_v = summarize(hist[vis_id][-20:])
            s5_h  = summarize(home_hist[home_id][-5:])
            s5_v  = summarize(away_hist[vis_id][-5:])
            s10_h = summarize(home_hist[home_id][-10:])
            s10_v = summarize(away_hist[vis_id][-10:])

            w30_h = summarize(hist[home_id][-30:])
            w30_v = summarize(hist[vis_id][-30:])

            dw5_wp  = (w5_h["win_pct"] - w5_v["win_pct"])   if (w5_h and w5_v) else 0.0
            dw5_rd  = (w5_h["rd_pg"]   - w5_v["rd_pg"])     if (w5_h and w5_v) else 0.0
            dw10_wp = (w10_h["win_pct"] - w10_v["win_pct"]) if (w10_h and w10_v) else 0.0
            dw10_rd = (w10_h["rd_pg"]   - w10_v["rd_pg"])   if (w10_h and w10_v) else 0.0
            dw20_wp = (w20_h["win_pct"] - w20_v["win_pct"]) if (w20_h and w20_v) else 0.0
            dw20_rd = (w20_h["rd_pg"]   - w20_v["rd_pg"])   if (w20_h and w20_v) else 0.0
            dw30_wp = (w30_h["win_pct"] - w30_v["win_pct"]) if (w30_h and w30_v) else 0.0
            dw30_rd = (w30_h["rd_pg"]   - w30_v["rd_pg"])   if (w30_h and w30_v) else 0.0

            # Pythagorean residual: actual win% - Pythagorean win% (luck indicator)
            h_pyth_res = (w20_h["win_pct"] - w20_h["pyth"]) if w20_h else 0.0
            v_pyth_res = (w20_v["win_pct"] - w20_v["pyth"]) if w20_v else 0.0
            diff_pyth_res = h_pyth_res - v_pyth_res

            # Elo momentum: Elo change over last 10 games
            h_elo_now  = nn(gf.get("home_elo"), ELO_BASE)
            v_elo_now  = nn(gf.get("vis_elo"),  ELO_BASE)
            h_elo_10ago = elo_hist[home_id][-10] if len(elo_hist[home_id]) >= 10 else ELO_BASE
            v_elo_10ago = elo_hist[vis_id][-10]  if len(elo_hist[vis_id])  >= 10 else ELO_BASE
            diff_elo_mom = (h_elo_now - h_elo_10ago) - (v_elo_now - v_elo_10ago)
            ds5_wp  = (s5_h["win_pct"] - s5_v["win_pct"])   if (s5_h and s5_v) else 0.0
            ds5_rd  = (s5_h["rd_pg"]   - s5_v["rd_pg"])     if (s5_h and s5_v) else 0.0
            ds10_wp = (s10_h["win_pct"] - s10_v["win_pct"]) if (s10_h and s10_v) else 0.0
            ds10_rd = (s10_h["rd_pg"]   - s10_v["rd_pg"])   if (s10_h and s10_v) else 0.0

            h_rest = rest_days(last_date.get(home_id), game_date)
            v_rest = rest_days(last_date.get(vis_id),  game_date)

            home_sg = team_season_count[home_id]
            vis_sg  = team_season_count[vis_id]

            game_sps  = sp_pid.get(game_pk, {})
            home_sp_d = game_sps.get(home_id)
            vis_sp_d  = game_sps.get(vis_id)
            home_pid  = home_sp_d["pitcher_id"] if home_sp_d else None
            vis_pid   = vis_sp_d["pitcher_id"]  if vis_sp_d  else None
            home_sp_s = sp_starts[home_pid] if home_pid else 0
            vis_sp_s  = sp_starts[vis_pid]  if vis_pid  else 0

            # Rolling K/BB for SP command signal
            home_kbb = rolling_kbb(sp_kbb_hist[home_pid]) if home_pid else None
            vis_kbb  = rolling_kbb(sp_kbb_hist[vis_pid])  if vis_pid  else None
            diff_sp_kbb = (home_kbb - vis_kbb) if (home_kbb is not None and vis_kbb is not None) else 0.0

            # Park factor (2-year rolling home run environment vs league avg)
            pf = park_factors.get((season_year, home_id), 1.0)

            # Series position (0=first game of series, 1=second, ...)
            mk = (min(home_id, vis_id), max(home_id, vis_id))
            last_m = matchup_last.get(mk)
            if last_m is None or (game_date - last_m[0]).days > 1:
                series_no = 0
            else:
                series_no = min(last_m[1] + 1, 3)

            ph = prev_stats.get((season_year, home_id))
            pv = prev_stats.get((season_year, vis_id))

            # Soft-regime weights
            team_ready = max(0.0, min(1.0, min(home_sg, vis_sg) / TEAM_BURN_IN))
            sp_avail   = float(gf.get("sp_available") or 0)
            sp_ready   = max(0.0, min(1.0, min(home_sp_s, vis_sp_s) / STARTER_BURN_IN)) if sp_avail else 0.0
            early_w    = 1.0 - team_ready
            primary_w  = (1.0 - early_w) * sp_ready
            fallback_w = (1.0 - early_w) * (1.0 - sp_ready)

            result_rows.append({
                "game_pk":      game_pk,
                "season_year":  season_year,
                "game_date":    game_date_str[:10],
                "home_win":     home_win,
                # Slow features (game_features)
                "diff_win_pct":       nn(gf.get("diff_win_pct")),
                "diff_pyth_wp":       nn(gf.get("diff_pyth_wp")),
                "diff_rd":            nn(gf.get("diff_rd")),
                "diff_elo":           nn(gf.get("diff_elo")),
                "home_elo":           nn(gf.get("home_elo"), ELO_BASE),
                "vis_elo":            nn(gf.get("vis_elo"),  ELO_BASE),
                "elo_win_prob":       nn(gf.get("elo_win_prob"), 0.5),
                "universal_dh_era":   nn(gf.get("universal_dh_era")),
                "coors_field_factor": nn(gf.get("coors_field_factor")),
                "is_interleague":     nn(gf.get("is_interleague")),
                "sp_available":       sp_avail,
                "diff_sp_era":        nn(gf.get("diff_sp_era")),
                "diff_sp_whip":       nn(gf.get("diff_sp_whip")),
                "diff_sp_k9":         nn(gf.get("diff_sp_k9")),
                "diff_sp_kbb":        diff_sp_kbb,
                "home_park_factor":   pf,
                "series_game_no":     float(series_no),
                # Dynamic features
                "diff_w5_win_pct":    dw5_wp,
                "diff_w5_rd_pg":      dw5_rd,
                "diff_w10_win_pct":   dw10_wp,
                "diff_w10_rd_pg":     dw10_rd,
                "diff_w30_win_pct":   dw30_wp,
                "diff_w30_rd_pg":     dw30_rd,
                "diff_split5_win_pct":  ds5_wp,
                "diff_split5_rd_pg":    ds5_rd,
                "diff_split10_win_pct": ds10_wp,
                "diff_split10_rd_pg":   ds10_rd,
                "diff_trend_win_pct": dw5_wp - dw20_wp,
                "diff_trend_rd_pg":   dw5_rd - dw20_rd,
                "diff_pyth_residual": diff_pyth_res,
                "diff_elo_momentum":  diff_elo_mom,
                "home_rest":          h_rest,
                "vis_rest":           v_rest,
                "diff_rest":          h_rest - v_rest,
                "diff_streak":        streak_val(hist[home_id]) - streak_val(hist[vis_id]),
                "home_season_games_before": home_sg,
                "vis_season_games_before":  vis_sg,
                "home_sp_starts_before":    home_sp_s,
                "vis_sp_starts_before":     vis_sp_s,
                # Prev-season priors
                "prev_diff_win_pct": nn(ph["win_pct"] if ph else None) - nn(pv["win_pct"] if pv else None),
                "prev_diff_rd_pg":   nn(ph["rd_pg"]   if ph else None) - nn(pv["rd_pg"]   if pv else None),
                "prev_diff_pyth":    nn(ph["pyth"]    if ph else None) - nn(pv["pyth"]    if pv else None),
                # Regime weights
                "early_weight":    early_w,
                "primary_weight":  primary_w,
                "fallback_weight": fallback_w,
            })

        # Post-game state updates
        hist[home_id].append({"rs": home_score, "ra": vis_score, "win": home_win})
        hist[vis_id].append( {"rs": vis_score,  "ra": home_score, "win": 1 - home_win})
        home_hist[home_id].append({"rs": home_score, "ra": vis_score, "win": home_win})
        away_hist[vis_id].append({"rs": vis_score, "ra": home_score, "win": 1 - home_win})
        last_date[home_id] = game_date
        last_date[vis_id]  = game_date
        elo_hist[home_id].append(nn(gf.get("home_elo"), ELO_BASE) if gf else ELO_BASE)
        elo_hist[vis_id].append( nn(gf.get("vis_elo"),  ELO_BASE) if gf else ELO_BASE)
        team_season_count[home_id] += 1
        team_season_count[vis_id]  += 1
        matchup_last[(min(home_id, vis_id), max(home_id, vis_id))] = (game_date, series_no if gf is not None else 0)
        game_sps = sp_pid.get(game_pk, {})
        if game_sps.get(home_id):
            pid = game_sps[home_id]["pitcher_id"]
            sp_starts[pid] += 1
            sp_kbb_hist[pid].append({"k": game_sps[home_id]["k"], "bb": game_sps[home_id]["bb"]})
        if game_sps.get(vis_id):
            pid = game_sps[vis_id]["pitcher_id"]
            sp_starts[pid] += 1
            sp_kbb_hist[pid].append({"k": game_sps[vis_id]["k"], "bb": game_sps[vis_id]["bb"]})

    return result_rows


# ---------------------------------------------------------------------------
# Model fitting / prediction
# ---------------------------------------------------------------------------

def fit_xgb(rows, features):
    X = np.array([[r[f] for f in features] for r in rows], dtype=float)
    y = np.array([r["home_win"] for r in rows], dtype=float)
    m = xgb.XGBClassifier(**XGB_PARAMS)
    m.fit(X, y)
    return m


def predict_xgb(model, row, features):
    X = np.array([[row[f] for f in features]], dtype=float)
    return float(model.predict_proba(X)[0, 1])


def fit_models(train_rows):
    primary_train = [r for r in train_rows if r["sp_available"] > 0.5]
    early_train   = [r for r in train_rows if r["early_weight"] > 0.0]
    if len(early_train) < MIN_EARLY_TRAIN:
        early_train = train_rows
    return {
        "fallback": fit_xgb(train_rows,   FALLBACK_FEATURES),
        "primary":  fit_xgb(primary_train, PRIMARY_FEATURES) if primary_train else None,
        "early":    fit_xgb(early_train,   EARLY_FEATURES),
    }


def _prob_to_logodds(p):
    p = max(1e-6, min(1 - 1e-6, p))
    return np.log(p / (1 - p))


def fit_platt_scaler(models, calib_rows):
    """Fit A, B such that P_cal = sigmoid(A * logodds(raw) + B).
    Uses NO regularization (C=1e9) so A is free to spread probabilities.
    Input: log-odds of soft-blend prob, not raw probability.
    """
    if not calib_rows:
        return None
    logodds = np.array([_prob_to_logodds(soft_predict(models, r)[0]) for r in calib_rows]).reshape(-1, 1)
    y = np.array([r["home_win"] for r in calib_rows], dtype=int)
    if len(set(y)) < 2:
        return None
    scaler = LogisticRegression(C=1e9, solver="lbfgs", max_iter=1000, random_state=42)
    scaler.fit(logodds, y)
    return scaler


def apply_platt_scaler(scaler, prob):
    if scaler is None:
        return float(prob)
    lo = np.array([[_prob_to_logodds(prob)]])
    return float(scaler.predict_proba(lo)[0, 1])


def soft_predict(models, row):
    early_raw  = predict_xgb(models["early"],    row, EARLY_FEATURES)
    early_prob = 0.5 + (early_raw - 0.5) * EARLY_PROB_SHRINK
    fallback_prob = predict_xgb(models["fallback"], row, FALLBACK_FEATURES)

    ew = row["early_weight"]
    fw = row["fallback_weight"]
    pw = row["primary_weight"]

    if models["primary"] and row["sp_available"] > 0.5 and pw > 0:
        primary_prob = predict_xgb(models["primary"], row, PRIMARY_FEATURES)
    else:
        primary_prob = 0.0
        fw += pw
        pw  = 0.0

    prob = ew * early_prob + fw * fallback_prob + pw * primary_prob

    if ew >= 0.999:
        label = "early"
    elif pw >= 0.999:
        label = "primary"
    elif fw >= 0.999:
        label = "fallback"
    else:
        label = "soft_blend"

    return prob, label


# ---------------------------------------------------------------------------
# Walk-forward
# ---------------------------------------------------------------------------

def walkforward(all_rows):
    all_results = []
    year_stats  = []

    print(f"\n{'Year':<6} {'Games':>6} {'Accuracy':>9} {'EarlyPct':>9} {'SPPct':>7}")
    print("  " + "-" * 42)

    for test_year in range(BACKTEST_START_YEAR, BACKTEST_END_YEAR + 1):
        train_rows = [r for r in all_rows if r["season_year"] <  test_year]
        test_rows  = [r for r in all_rows if r["season_year"] == test_year]
        if not train_rows or not test_rows:
            continue

        models = fit_models(train_rows)
        year_preds = []

        for row in test_rows:
            prob, label = soft_predict(models, row)
            year_preds.append({
                "game_pk":     row["game_pk"],
                "season_year": row["season_year"],
                "game_date":   row["game_date"],
                "prob_home":   round(prob, 4),
                "actual":      row["home_win"],
                "correct":     int((1 if prob >= 0.5 else 0) == row["home_win"]),
                "model_label": label,
            })


        n       = len(year_preds)
        acc     = sum(r["correct"] for r in year_preds) / n
        early_n = sum(1 for r in test_rows if r["early_weight"] > 0)
        sp_n    = sum(1 for r in test_rows if r["sp_available"] > 0.5)
        print(f"  {test_year:<6} {n:>6} {acc:>9.3f} {early_n/n:>9.1%} {sp_n/n:>7.1%}")

        year_stats.append({"year": test_year, "games": n, "accuracy": acc})
        all_results.extend(year_preds)

    total = len(all_results)
    if total:
        overall = sum(r["correct"] for r in all_results) / total
        print(f"  {'ALL':<6} {total:>6} {overall:>9.3f}")

    return all_results, year_stats


# ---------------------------------------------------------------------------
# Report / output
# ---------------------------------------------------------------------------

def write_csv(results):
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["game_pk", "season_year", "game_date",
                                                "prob_home", "actual", "correct", "model_label"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved {len(results)} rows → {CSV_PATH}")


def write_report(year_stats, results):
    total  = len(results)
    overall = sum(r["correct"] for r in results) / total if total else 0.0
    probs = [r["prob_home"] for r in results]
    cover_65 = sum(1 for p in probs if p >= 0.65) / total if total else 0.0

    lines = [
        "# MLB Soft-regime XGBoost Walk-forward",
        "",
        f"Walk-forward: {BACKTEST_START_YEAR}–{BACKTEST_END_YEAR}  |  "
        f"Train start: 2011  |  Three models: early / fallback / primary+SP",
        "",
        "## Per-Year Accuracy",
        "",
        "| Year | Games | Accuracy |",
        "| ---: | ---: | ---: |",
    ]
    for s in year_stats:
        lines.append(f"| {s['year']} | {s['games']} | {s['accuracy']:.3f} |")
    lines += [
        f"| **ALL** | **{total}** | **{overall:.3f}** |",
        "",
        "## Model Parameters",
        "",
        f"- ELO_K=20, HOME_ADV=25, REGRESSION=0.35",
        f"- TEAM_BURN_IN={TEAM_BURN_IN}, STARTER_BURN_IN={STARTER_BURN_IN}",
        f"- EARLY_PROB_SHRINK={EARLY_PROB_SHRINK}",
        (
            f"- XGBoost: max_depth={XGB_PARAMS['max_depth']}, "
            f"n_estimators={XGB_PARAMS['n_estimators']}, "
            f"lr={XGB_PARAMS['learning_rate']}, reg_lambda={XGB_PARAMS['reg_lambda']}, "
            f"min_child_weight={XGB_PARAMS['min_child_weight']}"
        ),
        f"- Post-processing: Platt scaling via train-only OOF probabilities",
        f"- P>=0.65 coverage: {cover_65:.1%}",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved report → {REPORT_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    conn = sqlite3.connect(DB_PATH)
    try:
        print("Building feature rows...")
        all_rows = build_rows(conn)
        print(f"  Total rows: {len(all_rows)}")
        yrs = sorted({r["season_year"] for r in all_rows})
        print(f"  Years: {yrs[0]}–{yrs[-1]}")

        results, year_stats = walkforward(all_rows)
        write_csv(results)
        write_report(year_stats, results)
    finally:
        conn.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
