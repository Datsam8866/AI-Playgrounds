"""
predict_today.py

Predict CPBL game outcomes for a given date (default: today).
All features are computed on-the-fly from completed games BEFORE the target date.

Usage:
    python predict_today.py
    python predict_today.py --date 2026-04-10
    python predict_today.py --date 2026-04-10 --verify

Regime routing:
    - early-season model: prior-heavy (Elo + previous-season strength + light context)
    - in-season model: advanced ensemble (XGBoost)
"""

import argparse
import io
import sqlite3
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta
from math import exp, log
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

DB_PATH = Path("cpbl.sqlite")

FRANCHISE_MAP = {
    "ACC011": "ACN011", "AEG011": "AEO011",
    "AEM011": "AEO011", "AJK011": "AJL011",
}
TEAM_NAMES = {
    "ACN011": "中信兄弟", "AAA011": "味全龍", "AJL011": "樂天桃猿",
    "AEO011": "富邦悍將", "ADD011": "統一獅", "AKP011": "台鋼雄鷹",
}

# ── Model params (same as evaluate_game_predictions_advanced.py) ─────────────
ELO_K = 52
ELO_REGRESSION = 0.35  # 6-team league: high roster continuity, less regression needed
ELO_HOME_ADV = 20  # A/B experiment 2026-04-28: adv=20 beats adv=10 on ECE (0.0562 vs 0.0608)
LOGISTIC_L2 = 3.0
TRAIN_START_YEAR = 2013

XGB_PARAMS = {
    "max_depth": 3,
    "min_child_weight": 20,
    "n_estimators": 50,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 3.0,
    "random_state": 42,
    "eval_metric": "logloss",
    "verbosity": 0,
}

TEAM_BURN_IN = 10
STARTER_BURN_IN = 5  # aligned with SP_MIN_STARTS to eliminate semantic mismatch
DISABLE_SP_MODEL = False
MIN_EARLY_TRAIN_ROWS = 50
EARLY_PROB_SHRINK = 0.50
PLATT_A = 1.450091
PLATT_B = -0.056832
CONFIDENCE_CAP = 0.85      # 建議1: 阻止 Platt 產生假極端信心（歷史前60場從未出現0.90+）
SP_FULL_STARTS = 15        # 建議2: SP需要此出賽數才給完整路由權重
SP_FULL_SHRINK = 0.70      # 建議2: SP active但出賽<SP_FULL_STARTS時，信心收縮至70%
PLATT_REFIT_THRESHOLD = 150 # n≥150 才有足夠樣本做可靠 Platt re-fit（DeepSeek 統計驗證）

# Rolling windows for team stats
TEAM_WINDOW = 20      # long-form baseline
SP_WINDOW = 10        # pitcher rolling window
SP_MIN_STARTS = 5

# League-average prior for SP blending (2013–2025 CPBL regular season, kind_code='A')
LEAGUE_SP_PRIOR = {"era": 5.50, "whip": 1.64, "k9": 6.82, "ip": 5.43, "bb9": 3.50}

ADVANCED_FALLBACK_FEATURES = [
    "diff_win_pct", "diff_rs", "diff_ra", "diff_rd", "diff_pyth_wp",
    "diff_elo", "home_elo", "vis_elo",
    # elo_home_prob removed: r=0.997 with diff_elo, redundant
    "diff_w3_win_pct", "diff_w3_rd_pg",
    "diff_w5_win_pct", "diff_w5_rd_pg",
    "diff_w10_win_pct", "diff_w10_rd_pg",
    "diff_split5_win_pct", "diff_split5_rd_pg",
    "diff_split10_win_pct", "diff_split10_rd_pg",
    "diff_trend_win_pct", "diff_trend_rd_pg",
    "home_rest", "vis_rest", "diff_rest", "diff_streak",
]
ADVANCED_PRIMARY_FEATURES = ADVANCED_FALLBACK_FEATURES + [
    "diff_sp_era", "diff_sp_whip", "diff_sp_k9",
    "diff_sp_ip", "diff_sp_bb9", "sp_available",
    # sp_ip_available removed: r=1.000 with sp_available, completely redundant
    # diff_sp_era_z deferred: requires running z-score in GameState (look-ahead-safe)
]

EARLY_FEATURES = [
    "diff_elo", "home_elo", "vis_elo", "elo_home_prob",
    "prev_diff_win_pct", "prev_diff_rd_pg", "prev_diff_pyth",
    "home_rest", "vis_rest", "diff_rest", "diff_streak",
    "home_season_games_before", "vis_season_games_before",
]


def shrink_early_probability(prob: float) -> float:
    return 0.5 + (prob - 0.5) * EARLY_PROB_SHRINK


def platt_calibrate(p: float, a: float = PLATT_A, b: float = PLATT_B) -> float:
    p = max(0.001, min(0.999, p))
    logit_p = log(p / (1.0 - p))
    return 1.0 / (1.0 + exp(-(a * logit_p + b)))


def apply_confidence_guards(prob: float, weights: dict, row: dict) -> float:
    """建議1+2: SP出賽不足時收縮信心，並對所有預測套用信心上限。"""
    # 建議2: advanced+SP dominant 但 SP 歷史不足 → 額外收縮
    if weights.get("primary", 0) > 0.5:
        min_sp = min(
            row.get("home_sp_starts_before", 0),
            row.get("vis_sp_starts_before", 0),
        )
        if min_sp < SP_FULL_STARTS:
            prob = 0.5 + (prob - 0.5) * SP_FULL_SHRINK
    # 建議1: 信心上限 cap
    return max(1.0 - CONFIDENCE_CAP, min(CONFIDENCE_CAP, prob))


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
    primary_weight = (
        0.0 if DISABLE_SP_MODEL else ((1.0 - early_weight) * sp_ready if has_primary_model else 0.0)
    )
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
    suffix = " [SP-OFF]" if DISABLE_SP_MODEL else ""
    if weights["early"] >= 0.999:
        return f"early_shrunk{suffix}"
    if weights["primary"] >= 0.999:
        return f"advanced+SP{suffix}"
    if weights["fallback"] >= 0.999:
        return f"advanced(no_SP){suffix}"
    return f"soft_blend{suffix}"


# ── Helpers ───────────────────────────────────────────────────────────────────

def norm(code: str) -> str:
    return FRANCHISE_MAP.get(code, code)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

def fit_lr(X, y, lr=0.05, epochs=600, l2=1.0):
    n, d = X.shape
    w = np.zeros(d); b = 0.0
    for _ in range(epochs):
        p = sigmoid(X @ w + b); err = p - y
        w -= lr * ((X.T @ err) / n + l2 * w / n)
        b -= lr * err.mean()
    return w, b

def standardize_fit(X_tr, X_te):
    m = X_tr.mean(0); s = X_tr.std(0); s[s == 0] = 1.0
    return (X_tr - m) / s, (X_te - m) / s, m, s

def pyth(rs, ra):
    return rs**2 / (rs**2 + ra**2) if rs + ra else 0.5

def summarize(games):
    if not games: return None
    n = len(games)
    rs = sum(g["rs"] for g in games); ra = sum(g["ra"] for g in games)
    w  = sum(g["win"] for g in games)
    return {"win_pct": w/n, "rd_pg": (rs-ra)/n, "rs": rs/n, "ra": ra/n}

def streak_val(games):
    if not games: return 0
    last = games[-1]["win"]; s = 0
    for g in reversed(games):
        if g["win"] != last: break
        s += 1 if last == 1 else -1
    return s

def rest_days(last_date, cur_date):
    if last_date is None: return 0
    return max(0, min((cur_date - last_date).days - 1, 10))

def ip_to_float(ip_int, div3=0):
    return (ip_int or 0) + (div3 or 0) / 3.0

def sp_rolling(history, window=SP_WINDOW, min_starts=SP_MIN_STARTS, prior=None):
    subset = history[-window:]
    n = len(subset)
    if n == 0:
        return prior
    tip = sum(g["ip"] for g in subset)
    if tip == 0:
        return prior
    ter = sum(g["er"] for g in subset)
    tk  = sum(g["k"]  for g in subset)
    tbb = sum(g["bb"] for g in subset)
    th  = sum(g["h"]  for g in subset)
    observed = {
        "era":  ter * 9 / tip,
        "whip": (tbb + th) / tip,
        "k9":   tk  * 9 / tip,
        "ip":   tip / n,
        "bb9":  tbb * 9 / tip,
    }
    if n >= min_starts:
        return observed
    if prior is None:
        return None
    # Bayesian blend: weight toward prior when starts < min_starts
    w = n / min_starts
    return {k: w * observed[k] + (1 - w) * prior[k] for k in observed}


# ── State builder (processes all completed games chronologically) ─────────────

class GameState:
    """Maintains rolling team and pitcher state up to a given date."""

    def __init__(self):
        self.hist_all:  dict[str, list] = defaultdict(list)
        self.hist_home: dict[str, list] = defaultdict(list)
        self.hist_away: dict[str, list] = defaultdict(list)
        self.last_date: dict[str, date] = {}
        self.elo:       dict[str, float] = defaultdict(lambda: 1500.0)
        self.sp_hist:   dict[str, list]  = defaultdict(list)   # acnt -> starts
        self.season_games: dict[tuple[str,int], int] = defaultdict(int)  # (team, year)
        self._cur_season: int | None = None

    def process_game(self, yr, gd: date, hc, vc, hs, vs):
        """Update state with a completed game."""
        if self._cur_season is None:
            self._cur_season = yr
        elif yr != self._cur_season:
            for t in list(self.elo):
                self.elo[t] = 1500 + (self.elo[t] - 1500) * (1 - ELO_REGRESSION)
            self._cur_season = yr

        actual = 0.5 if hs == vs else (1.0 if hs > vs else 0.0)

        # Update histories FIRST (pre-game state already captured for feature rows)
        self.hist_all[hc].append({"season_year": yr, "rs": hs, "ra": vs, "win": actual})
        self.hist_all[vc].append({"season_year": yr, "rs": vs, "ra": hs, "win": 1 - actual})
        self.hist_home[hc].append({"season_year": yr, "rs": hs, "ra": vs, "win": actual})
        self.hist_away[vc].append({"season_year": yr, "rs": vs, "ra": hs, "win": 1 - actual})

        exp = 1 / (1 + 10 ** ((self.elo[vc] - (self.elo[hc] + ELO_HOME_ADV)) / 400))
        self.elo[hc] += ELO_K * (actual - exp)
        self.elo[vc] += ELO_K * ((1 - actual) - (1 - exp))
        self.last_date[hc] = gd
        self.last_date[vc] = gd
        self.season_games[(hc, yr)] += 1
        self.season_games[(vc, yr)] += 1

    def process_sp(self, acnt, ip, er, k, bb, h):
        if not acnt: return
        self.sp_hist[acnt].append({"ip": ip, "er": er, "k": k, "bb": bb, "h": h})

    def snapshot(self, gd: date, hc: str, vc: str, yr: int) -> dict:
        """
        Return pre-game feature snapshot for a (home, vis) matchup.
        Call BEFORE process_game() for prediction games.
        """
        row = {}

        # Long-form team stats (window=20)
        hg = self.hist_all[hc][-TEAM_WINDOW:]
        vg = self.hist_all[vc][-TEAM_WINDOW:]
        hs = summarize(hg); vs = summarize(vg)
        if hs and vs:
            row["diff_win_pct"]  = hs["win_pct"] - vs["win_pct"]
            row["diff_rs"]       = hs["rs"]       - vs["rs"]
            row["diff_ra"]       = hs["ra"]       - vs["ra"]
            row["diff_rd"]       = hs["rd_pg"]    - vs["rd_pg"]
            row["diff_pyth_wp"]  = pyth(hs["rs"]*TEAM_WINDOW, hs["ra"]*TEAM_WINDOW) - \
                                   pyth(vs["rs"]*TEAM_WINDOW, vs["ra"]*TEAM_WINDOW)
        else:
            for k2 in ["diff_win_pct","diff_rs","diff_ra","diff_rd","diff_pyth_wp"]:
                row[k2] = 0.0

        # Elo
        he = self.elo[hc]; ve = self.elo[vc]
        row["diff_elo"]       = he - ve
        row["home_elo"]       = he
        row["vis_elo"]        = ve
        row["elo_home_prob"]  = 1 / (1 + 10 ** ((ve - (he + ELO_HOME_ADV)) / 400))

        # Short-form windows
        for w, p in [(3,"w3"),(5,"w5"),(10,"w10"),(20,"w20")]:
            hs2 = summarize(self.hist_all[hc][-w:])
            vs2 = summarize(self.hist_all[vc][-w:])
            for m in ["win_pct","rd_pg"]:
                row[f"diff_{p}_{m}"] = hs2[m] - vs2[m] if hs2 and vs2 else 0.0

        # Venue split
        for w, p in [(5,"split5"),(10,"split10")]:
            hs2 = summarize(self.hist_home[hc][-w:])
            vs2 = summarize(self.hist_away[vc][-w:])
            for m in ["win_pct","rd_pg"]:
                row[f"diff_{p}_{m}"] = hs2[m] - vs2[m] if hs2 and vs2 else 0.0

        # Trend
        row["diff_trend_win_pct"] = row["diff_w5_win_pct"] - row["diff_w20_win_pct"]
        row["diff_trend_rd_pg"]   = row["diff_w5_rd_pg"]   - row["diff_w20_rd_pg"]

        # Rest/streak
        hr = rest_days(self.last_date.get(hc), gd)
        vr = rest_days(self.last_date.get(vc), gd)
        row["home_rest"]   = hr
        row["vis_rest"]    = vr
        row["diff_rest"]   = hr - vr
        row["diff_streak"] = streak_val(self.hist_all[hc]) - streak_val(self.hist_all[vc])

        # Burn-in counters + prior-season priors
        h_games = self.season_games.get((hc, yr), 0)
        v_games = self.season_games.get((vc, yr), 0)
        row["home_season_games_before"] = h_games
        row["vis_season_games_before"] = v_games
        row["_h_season_games"] = h_games
        row["_v_season_games"] = v_games

        prev_h = summarize([g for g in self.hist_all[hc] if g["season_year"] == yr - 1])
        prev_v = summarize([g for g in self.hist_all[vc] if g["season_year"] == yr - 1])
        row["prev_diff_win_pct"] = (
            (prev_h["win_pct"] if prev_h else 0.0) -
            (prev_v["win_pct"] if prev_v else 0.0)
        )
        row["prev_diff_rd_pg"] = (
            (prev_h["rd_pg"] if prev_h else 0.0) -
            (prev_v["rd_pg"] if prev_v else 0.0)
        )
        row["prev_diff_pyth"] = (
            pyth(prev_h["rs"] * len([g for g in self.hist_all[hc] if g["season_year"] == yr - 1]),
                 prev_h["ra"] * len([g for g in self.hist_all[hc] if g["season_year"] == yr - 1])) if prev_h else 0.0
        ) - (
            pyth(prev_v["rs"] * len([g for g in self.hist_all[vc] if g["season_year"] == yr - 1]),
                 prev_v["ra"] * len([g for g in self.hist_all[vc] if g["season_year"] == yr - 1])) if prev_v else 0.0
        )

        return row

    def add_sp_features(self, row: dict, home_acnt: str | None, vis_acnt: str | None):
        """Attach SP rolling stats to a feature row (in-place)."""
        row["home_sp_starts_before"] = len(self.sp_hist.get(home_acnt, [])) if home_acnt else 0
        row["vis_sp_starts_before"] = len(self.sp_hist.get(vis_acnt, [])) if vis_acnt else 0
        row["_early_season"] = (
            row.get("home_season_games_before", 0) < TEAM_BURN_IN or
            row.get("vis_season_games_before", 0) < TEAM_BURN_IN or
            row["home_sp_starts_before"] < STARTER_BURN_IN or
            row["vis_sp_starts_before"] < STARTER_BURN_IN
        )
        h_sp = sp_rolling(self.sp_hist.get(home_acnt, []), prior=LEAGUE_SP_PRIOR) if home_acnt else None
        v_sp = sp_rolling(self.sp_hist.get(vis_acnt,  []), prior=LEAGUE_SP_PRIOR) if vis_acnt  else None

        if h_sp and v_sp:
            row["diff_sp_era"]      = v_sp["era"]  - h_sp["era"]
            row["diff_sp_whip"]     = v_sp["whip"] - h_sp["whip"]
            row["diff_sp_k9"]       = h_sp["k9"]   - v_sp["k9"]
            row["diff_sp_ip"]       = h_sp["ip"]   - v_sp["ip"]
            row["diff_sp_bb9"]      = v_sp["bb9"]  - h_sp["bb9"]  # positive = home better control
            row["sp_available"]     = 1.0
            row["sp_ip_available"]  = 1.0
        else:
            row["diff_sp_era"]  = 0.0
            row["diff_sp_whip"] = 0.0
            row["diff_sp_k9"]   = 0.0
            row["diff_sp_ip"]   = 0.0
            row["diff_sp_bb9"]  = 0.0
            row["sp_available"] = 0.0
            row["sp_ip_available"] = 0.0


# ── Load data ─────────────────────────────────────────────────────────────────

def load_data(conn, target: date):
    """Return (train_rows, predict_games, state_at_target)."""

    completed = conn.execute("""
        SELECT season_year, kind_code, game_date, game_sno,
               visiting_team_code, home_team_code,
               visiting_score, home_score
        FROM team_game_results
        WHERE game_status = 3
          AND visiting_score IS NOT NULL AND home_score IS NOT NULL
          AND DATE(game_date) < ?
        ORDER BY game_date, game_sno
    """, (target.isoformat(),)).fetchall()

    scheduled = conn.execute("""
        SELECT season_year, kind_code, game_date, game_sno,
               visiting_team_code, home_team_code
        FROM team_game_results
        WHERE DATE(game_date) = ?
          AND game_status IN (1, 3, 4, 6)
        ORDER BY game_sno
    """, (target.isoformat(),)).fetchall()

    sp_data = conn.execute("""
        SELECT season_year, kind_code, game_sno,
               home_sp_acnt, vis_sp_acnt,
               home_sp_ip,   vis_sp_ip,
               home_sp_er,   vis_sp_er,
               home_sp_k,    vis_sp_k,
               home_sp_bb,   vis_sp_bb,
               home_sp_h,    vis_sp_h,
               home_sp_en,   vis_sp_en
        FROM game_starting_pitchers
        WHERE scrape_status = 'ok'
        ORDER BY season_year, kind_code, game_sno
    """).fetchall()

    # Build SP lookup by (year, sno)
    sp_lookup = {}
    sp_history_data = {}   # acnt -> list of starts (for building rolling)
    for r in sp_data:
        sp_lookup[(r["season_year"], r["game_sno"])] = r
        sp_history_data[(r["season_year"], r["game_sno"])] = r

    # ── Replay all completed games ────────────────────────────────────────────
    state = GameState()
    train_rows: list[dict] = []

    # Index SP data by (season_year, kind_code, game_sno) to avoid postseason collisions
    sp_idx = {(r["season_year"], r["kind_code"], r["game_sno"]): r for r in sp_data}

    for raw in completed:
        yr        = raw["season_year"]
        kind_code = raw["kind_code"]
        gd  = datetime.fromisoformat(raw["game_date"][:10]).date()
        sno = raw["game_sno"]
        hc  = norm(raw["home_team_code"])
        vc  = norm(raw["visiting_team_code"])
        hs  = float(raw["home_score"])
        vs  = float(raw["visiting_score"])

        if hs == vs:
            state.process_game(yr, gd, hc, vc, hs, vs)
            continue  # ties: update state but no training row

        # Build pre-game feature snapshot (BEFORE updating state)
        if yr >= TRAIN_START_YEAR:
            snap = state.snapshot(gd, hc, vc, yr)

            sp_row = sp_idx.get((yr, kind_code, sno))
            if sp_row:
                state.add_sp_features(
                    snap,
                    sp_row["home_sp_acnt"],
                    sp_row["vis_sp_acnt"],
                )
            else:
                state.add_sp_features(snap, None, None)

            snap["home_win"]    = int(hs > vs)
            snap["season_year"] = yr
            snap["game_date"]   = gd.isoformat()
            snap["game_sno"]    = sno
            # Only regular season games form the training set
            if kind_code == "A":
                train_rows.append(snap)

        # Update SP rolling history AFTER snapshot
        sp_row = sp_idx.get((yr, kind_code, sno))
        if sp_row:
            state.process_sp(
                sp_row["home_sp_acnt"],
                ip_to_float(sp_row["home_sp_ip"] or 0),
                sp_row["home_sp_er"] or 0, sp_row["home_sp_k"] or 0,
                sp_row["home_sp_bb"] or 0, sp_row["home_sp_h"] or 0,
            )
            state.process_sp(
                sp_row["vis_sp_acnt"],
                ip_to_float(sp_row["vis_sp_ip"] or 0),
                sp_row["vis_sp_er"] or 0, sp_row["vis_sp_k"] or 0,
                sp_row["vis_sp_bb"] or 0, sp_row["vis_sp_h"] or 0,
            )

        state.process_game(yr, gd, hc, vc, hs, vs)

    # ── Build prediction rows for target date ─────────────────────────────────
    pred_games = []
    for raw in scheduled:
        yr        = raw["season_year"]
        kind_code = raw["kind_code"]
        gd  = datetime.fromisoformat(raw["game_date"][:10]).date()
        sno = raw["game_sno"]
        hc  = norm(raw["home_team_code"])
        vc  = norm(raw["visiting_team_code"])

        snap = state.snapshot(gd, hc, vc, yr)

        # Check if SP announced for this game
        sp_row = sp_idx.get((yr, kind_code, sno))
        if sp_row:
            state.add_sp_features(snap, sp_row["home_sp_acnt"], sp_row["vis_sp_acnt"])
            snap["home_sp_name"] = sp_row["home_sp_en"] or "?"
            snap["vis_sp_name"]  = sp_row["vis_sp_en"]  or "?"
            snap["home_sp_acnt"] = sp_row["home_sp_acnt"]
            snap["vis_sp_acnt"]  = sp_row["vis_sp_acnt"]
        else:
            state.add_sp_features(snap, None, None)
            snap["home_sp_name"] = "TBD"
            snap["vis_sp_name"]  = "TBD"
            snap["home_sp_acnt"] = None
            snap["vis_sp_acnt"]  = None

        snap["season_year"]   = yr
        snap["game_date"]     = gd.isoformat()
        snap["game_sno"]      = sno
        snap["home_team"]     = hc
        snap["vis_team"]      = vc
        pred_games.append(snap)

    return train_rows, pred_games


# ── Train ensemble and predict ────────────────────────────────────────────────

def fit_xgb(train_rows: list[dict], feature_names: list[str]):
    X = np.array([[r[n] for n in feature_names] for r in train_rows], dtype=float)
    y = np.array([r["home_win"] for r in train_rows], dtype=float)
    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(X, y)
    return model

def predict_xgb(model, row: dict, feature_names: list[str]) -> float:
    X = np.array([[row[n] for n in feature_names]], dtype=float)
    return float(model.predict_proba(X)[0, 1])

def train_and_predict(train_rows: list[dict], pred_rows: list[dict]) -> list[dict]:
    if not train_rows or not pred_rows:
        return []

    pri_filter = lambda r: r.get("sp_available", 0) > 0.5
    pri_train = [r for r in train_rows if pri_filter(r)]
    early_train = [r for r in train_rows if r.get("_early_season")]
    if len(early_train) < MIN_EARLY_TRAIN_ROWS:
        early_train = train_rows

    fb_model = fit_xgb(train_rows, ADVANCED_FALLBACK_FEATURES)
    pri_model = fit_xgb(pri_train, ADVANCED_PRIMARY_FEATURES) if pri_train else None
    early_model = fit_xgb(early_train, EARLY_FEATURES)

    # ── Platt refit (if current season has enough completed games) ────────────
    platt_a = PLATT_A
    platt_b = PLATT_B
    current_year = pred_rows[0].get("season_year") if pred_rows else None
    season_rows = [r for r in train_rows if r.get("season_year") == current_year] if current_year else []
    if len(season_rows) >= PLATT_REFIT_THRESHOLD:
        logit_X = []
        y_labels = []
        for r in season_rows:
            ep_raw = predict_xgb(early_model, r, EARLY_FEATURES)
            ep = shrink_early_probability(ep_raw)
            fp = predict_xgb(fb_model, r, ADVANCED_FALLBACK_FEATURES)
            has_pri = pri_model is not None and pri_filter(r)
            pp = predict_xgb(pri_model, r, ADVANCED_PRIMARY_FEATURES) if has_pri else None
            w = soft_regime_weights(r, has_pri)
            rp = w["early"] * ep + w["fallback"] * fp
            if pp is not None:
                rp += w["primary"] * pp
            rp = max(0.001, min(0.999, rp))
            logit_val = log(rp / (1.0 - rp))
            logit_X.append([logit_val])
            y_labels.append(int(r["home_win"]))
        # Skip refit if only one class present
        if len(set(y_labels)) == 2:
            lr = LogisticRegression(C=1e9, solver="lbfgs", max_iter=1000, fit_intercept=True)
            lr.fit(logit_X, y_labels)
            platt_a = float(lr.coef_[0][0])
            platt_b = float(lr.intercept_[0])
            print(f"  [Platt refit] n={len(season_rows)}, A={platt_a:.4f}, B={platt_b:.4f}")
        else:
            print(f"  [Platt refit] skipped — single class in season_rows (n={len(season_rows)})")

    results = []
    for row in pred_rows:
        early_prob_raw = predict_xgb(early_model, row, EARLY_FEATURES)
        early_prob = shrink_early_probability(early_prob_raw)
        fallback_prob = predict_xgb(fb_model, row, ADVANCED_FALLBACK_FEATURES)
        primary_prob = None
        has_primary = pri_model is not None and pri_filter(row)
        if has_primary:
            primary_prob = predict_xgb(pri_model, row, ADVANCED_PRIMARY_FEATURES)

        weights = soft_regime_weights(row, has_primary)
        raw_prob = weights["early"] * early_prob + weights["fallback"] * fallback_prob
        if primary_prob is not None:
            raw_prob += weights["primary"] * primary_prob
        calibrated_prob = apply_confidence_guards(platt_calibrate(raw_prob, platt_a, platt_b), weights, row)

        results.append({
            **row,
            "raw_prob_home_win": float(raw_prob),
            "prob_home_win": float(calibrated_prob),
            "model_used": model_label(weights),
            "early_prob_raw": early_prob_raw,
            "early_prob_shrunk": early_prob,
            "fallback_prob": fallback_prob,
            "primary_prob": primary_prob,
            "early_weight": weights["early"],
            "fallback_weight": weights["fallback"],
            "primary_weight": weights["primary"],
            "team_readiness": team_readiness(row),
            "starter_readiness": starter_readiness(row),
        })

    return results


# ── Output ────────────────────────────────────────────────────────────────────

def format_report(target: date, predictions: list[dict], verify: bool, conn) -> str:
    date_str = target.strftime("%Y-%m-%d")
    lines = [f"# CPBL Predictions — {date_str}", ""]

    if not predictions:
        lines += [f"No scheduled games found for {date_str}.", ""]
        return "\n".join(lines)

    # Lookup actual results if verify mode
    actuals = {}
    if verify:
        rows = conn.execute("""
            SELECT game_sno, home_score, visiting_score, game_status
            FROM team_game_results
            WHERE DATE(game_date) = ? AND season_year = ?
        """, (date_str, target.year)).fetchall()
        for r in rows:
            if r["game_status"] == 3 and r["home_score"] is not None:
                actuals[r["game_sno"]] = int(float(r["home_score"]) > float(r["visiting_score"]))

    lines += [
        f"- Training games: all completed through {date_str} (exclusive)",
        f"- Soft routing: early weight fades out by `{TEAM_BURN_IN}` team games; SP weight fades in by `{STARTER_BURN_IN}` starter starts",
        f"- Early-model probability shrinkage: `0.500 + (p - 0.500) * {EARLY_PROB_SHRINK:.2f}`",
        "",
        "| SNO | Home | Vis | Prob(Home) | Pred | Conf | Model | SP(H) | SP(V) |" +
        (" Result | Hit |" if verify else ""),
        "| --- | --- | --- | ---: | --- | --- | --- | --- | --- |" +
        (" --- | --- |" if verify else ""),
    ]

    correct = total = 0
    for p in predictions:
        prob = p["prob_home_win"]
        pred = "Home" if prob >= 0.5 else "Vis"
        conf = "HIGH" if prob >= 0.65 or prob <= 0.35 else ("MED" if prob >= 0.55 or prob <= 0.45 else "LOW")
        hn = TEAM_NAMES.get(p["home_team"], p["home_team"])
        vn = TEAM_NAMES.get(p["vis_team"],  p["vis_team"])
        sp_h = (p.get("home_sp_name") or "TBD")[:20]
        sp_v = (p.get("vis_sp_name")  or "TBD")[:20]

        row_line = (f"| {p['game_sno']} | {hn} | {vn} | {prob:.3f} | {pred} | {conf} "
                    f"| {p['model_used']} | {sp_h} | {sp_v} |")

        if verify and p["game_sno"] in actuals:
            actual_home_win = actuals[p["game_sno"]]
            result_str = "Home" if actual_home_win else "Vis"
            hit = (pred == result_str)
            row_line += f" {result_str} | {'✓' if hit else '✗'} |"
            correct += int(hit); total += 1
        elif verify:
            row_line += " ? | ? |"

        lines.append(row_line)

    if verify and total > 0:
        lines += ["", f"**Result: {correct}/{total} = {correct/total:.1%}**"]

    lines += [
        "",
        "## Model Blend",
        "",
        "| SNO | Team Ready | SP Ready | Early Wt | Fallback Wt | Primary Wt | Early Raw | Early Shrunk | Fallback | Primary |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for p in predictions:
        primary_prob = p.get("primary_prob")
        primary_cell = f"{primary_prob:.3f}" if primary_prob is not None else "N/A"
        lines.append(
            f"| {p['game_sno']} "
            f"| {p.get('team_readiness', 0):.2f} "
            f"| {p.get('starter_readiness', 0):.2f} "
            f"| {p.get('early_weight', 0):.2f} "
            f"| {p.get('fallback_weight', 0):.2f} "
            f"| {p.get('primary_weight', 0):.2f} "
            f"| {p.get('early_prob_raw', 0):.3f} "
            f"| {p.get('early_prob_shrunk', 0):.3f} "
            f"| {p.get('fallback_prob', 0):.3f} "
            f"| {primary_cell} |"
        )

    lines += [
        "",
        "## Feature Snapshot",
        "",
        "| SNO | diff_elo | diff_win_pct | diff_pyth | w5_winpct | split5_win | rest | streak | SP? | Early? |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for p in predictions:
        early = "Y" if p.get("_early_season") else "N"
        sp = "Y" if p.get("sp_available", 0) > 0.5 else "N"
        lines.append(
            f"| {p['game_sno']} "
            f"| {p.get('diff_elo',0):+.1f} "
            f"| {p.get('diff_win_pct',0):+.3f} "
            f"| {p.get('diff_pyth_wp',0):+.3f} "
            f"| {p.get('diff_w5_win_pct',0):+.3f} "
            f"| {p.get('diff_split5_win_pct',0):+.3f} "
            f"| {p.get('diff_rest',0):+d} "
            f"| {p.get('diff_streak',0):+d} "
            f"| {sp} | {early} |"
        )

    lines += ["", f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*", ""]
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Predict CPBL game outcomes for a given date")
    parser.add_argument("--date", default=date.today().isoformat(),
                        help="Target date YYYY-MM-DD (default: today)")
    parser.add_argument("--verify", action="store_true",
                        help="Show actual results if games completed")
    args = parser.parse_args()

    target = date.fromisoformat(args.date)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    try:
        print(f"Loading data and building state up to {target}...")
        train_rows, pred_games = load_data(conn, target)
        print(f"Training rows: {len(train_rows)}  |  Games to predict: {len(pred_games)}")

        if not pred_games:
            print(f"No scheduled games on {target}. Check team_game_results.")
            return

        print("Training model and predicting...")
        predictions = train_and_predict(train_rows, pred_games)

        # Console output
        print(f"\n{'='*65}")
        print(f"PREDICTIONS FOR {target}")
        print(f"{'='*65}")
        for p in predictions:
            prob = p["prob_home_win"]
            pred = "Home" if prob >= 0.5 else "Vis"
            conf = "HIGH" if abs(prob - 0.5) >= 0.15 else ("MED" if abs(prob - 0.5) >= 0.05 else "LOW")
            hn = TEAM_NAMES.get(p["home_team"], p["home_team"])
            vn = TEAM_NAMES.get(p["vis_team"],  p["vis_team"])
            if p.get("early_weight", 0) > 0:
                data_warn = " [EARLY-BLEND]"
            elif p.get("_early_season"):
                data_warn = " [SP-LIMITED]"
            else:
                data_warn = ""
            sp_str = f"SP: {p.get('home_sp_name','TBD')} vs {p.get('vis_sp_name','TBD')}"
            print(f"\nGame {p['game_sno']}: {hn}(Home) vs {vn}(Vis)")
            print(f"  {sp_str}")
            print(f"  Pred: {pred} win  |  Prob(Home): {prob:.3f}  |  Confidence: {conf}{data_warn}")
            print(f"  Model: {p['model_used']}  |  "
                  f"diff_elo={p.get('diff_elo',0):+.1f}  "
                  f"w5={p.get('diff_w5_win_pct',0):+.3f}  "
                  f"h_games={p.get('_h_season_games',0)}  v_games={p.get('_v_season_games',0)}  "
                  f"h_sp_starts={p.get('home_sp_starts_before',0)}  v_sp_starts={p.get('vis_sp_starts_before',0)}")
            print(f"  Weights: early={p.get('early_weight',0):.2f}  "
                  f"fallback={p.get('fallback_weight',0):.2f}  primary={p.get('primary_weight',0):.2f}")

        # Verify if requested
        if args.verify:
            actuals = {}
            rows = conn.execute("""
                SELECT game_sno, home_score, visiting_score, game_status
                FROM team_game_results
                WHERE DATE(game_date) = ? AND season_year = ?
            """, (target.isoformat(), target.year)).fetchall()
            correct = total = 0
            for r in rows:
                if r["game_status"] == 3 and r["home_score"] is not None:
                    actual_hw = int(float(r["home_score"]) > float(r["visiting_score"]))
                    pred_hw = next((int(p["prob_home_win"] >= 0.5)
                                    for p in predictions if p["game_sno"] == r["game_sno"]), None)
                    if pred_hw is not None:
                        hit = pred_hw == actual_hw
                        correct += int(hit); total += 1
            if total > 0:
                print(f"\nVerification: {correct}/{total} = {correct/total:.1%}")

        # Write markdown report
        report = format_report(target, predictions, args.verify, conn)
        report_path = Path(f"predictions_{target.strftime('%Y%m%d')}.md")
        report_path.write_text(report, encoding="utf-8")
        print(f"\nReport: {report_path.resolve()}")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
