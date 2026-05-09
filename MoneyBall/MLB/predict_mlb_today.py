"""
predict_mlb_today.py

Train on all historical data, build rolling state, fetch today's schedule
(with probable pitchers), and predict home-win probability.

Usage:
    python predict_mlb_today.py                    # today
    python predict_mlb_today.py --date 2026-05-01
    python predict_mlb_today.py --date 2026-04-27 --save
"""

import argparse
import sqlite3
import sys
import time
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

import requests

# Force UTF-8 output on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from train_mlb_model import (
    FALLBACK_FEATURES, PRIMARY_FEATURES, EARLY_FEATURES,
    TEAM_BURN_IN, STARTER_BURN_IN, EARLY_PROB_SHRINK, ELO_BASE,
    MIN_EARLY_TRAIN, fit_models, soft_predict, build_rows,
    summarize, streak_val, rest_days, nn, rolling_kbb,
    ip_mlb_to_float,
)

DB_PATH      = Path("mlb.sqlite")
SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"

# Must match build_mlb_game_features.py
ELO_K          = 8
ELO_HOME_ADV   = 25
ELO_REGRESSION = 0.35
SP_WINDOW      = 8
SP_MIN_STARTS  = 5


# ---------------------------------------------------------------------------
# Elo / SP helpers
# ---------------------------------------------------------------------------

def elo_expected(home_elo: float, vis_elo: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-(home_elo + ELO_HOME_ADV - vis_elo) / 400.0))


def rolling_sp_stats(history):
    subset = history[-SP_WINDOW:]
    if len(subset) < SP_MIN_STARTS:
        return None
    total_ip = sum(g["ip"] for g in subset)
    if total_ip == 0:
        return None
    return {
        "era":  sum(g["er"] for g in subset) * 9 / total_ip,
        "whip": (sum(g["bb"] for g in subset) + sum(g["h"] for g in subset)) / total_ip,
        "k9":   sum(g["k"]  for g in subset) * 9 / total_ip,
        "ip":   total_ip / len(subset),
    }


# ---------------------------------------------------------------------------
# Build rolling state from all DB games
# ---------------------------------------------------------------------------

def build_live_state(conn, cutoff_date=None):
    """Replay every game in the DB to build current rolling state."""
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

    # SP game-level data
    sp_data = defaultdict(dict)
    for game_pk, team_id, pitcher_id, ip, er, k, bb, h in conn.execute(
        "SELECT game_pk, team_id, pitcher_id, ip, er, k, bb, h "
        "FROM game_starting_pitchers WHERE pitcher_id IS NOT NULL"
    ).fetchall():
        sp_data[game_pk][team_id] = {
            "pitcher_id": pitcher_id,
            "ip": ip_mlb_to_float(ip),
            "er": er or 0, "k": k or 0, "bb": bb or 0, "h": h or 0,
        }

    # League map (for interleague)
    league_map = {}
    for yr, tid, lg in conn.execute(
        "SELECT season_year, team_id, league FROM team_season_records"
    ).fetchall():
        league_map[(yr, tid)] = lg

    # Rockies team_id (Coors field)
    row = conn.execute(
        "SELECT team_id FROM team_season_records WHERE team_name LIKE '%Rockies%' LIMIT 1"
    ).fetchone()
    rockies_id = row[0] if row else None

    # Prev-season summaries (for early model priors)
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

    # Park factors (2-year rolling runs/game at home vs league avg)
    season_home_runs = defaultdict(list)
    for game_pk, yr, gd, hid, vid, hs, vs, _ in raw:
        season_home_runs[yr].append((hid, hs + vs))
    park_factors = {}
    seasons = sorted(season_home_runs.keys())
    for i, yr in enumerate(seasons):
        window_data = []
        for j in range(max(0, i - 1), i + 1):
            window_data.extend(season_home_runs[seasons[j]])
        if not window_data:
            continue
        league_avg = sum(r for _, r in window_data) / len(window_data)
        team_data = defaultdict(list)
        for tid, r in window_data:
            team_data[tid].append(r)
        for tid, runs_list in team_data.items():
            park_factors[(yr, tid)] = (sum(runs_list) / len(runs_list)) / max(league_avg, 1)

    # Rolling state
    hist             = defaultdict(list)
    home_hist        = defaultdict(list)
    away_hist        = defaultdict(list)
    last_date        = {}
    sp_starts        = defaultdict(int)
    sp_hist_raw      = defaultdict(list)
    sp_kbb_hist      = defaultdict(list)
    elo_hist         = defaultdict(list)
    current_elo      = defaultdict(lambda: ELO_BASE)
    team_season_cnt  = defaultdict(int)
    matchup_last     = {}
    current_season   = None

    for game_pk, season_year, game_date_str, home_id, vis_id, home_score, vis_score, winner in raw:
        home_win  = 1 if winner == "home" else 0
        game_date = datetime.fromisoformat(game_date_str[:10]).date()

        if season_year != current_season:
            if current_season is not None:
                for tid in list(current_elo.keys()):
                    current_elo[tid] = ELO_BASE * ELO_REGRESSION + current_elo[tid] * (1 - ELO_REGRESSION)
                team_season_cnt = defaultdict(int)
            current_season = season_year

        # Record pre-game Elo for momentum tracking
        elo_hist[home_id].append(current_elo[home_id])
        elo_hist[vis_id].append(current_elo[vis_id])

        # Matchup series position
        mk = (min(home_id, vis_id), max(home_id, vis_id))
        last_m = matchup_last.get(mk)
        series_no = 0 if (last_m is None or (game_date - last_m[0]).days > 1) else min(last_m[1] + 1, 3)
        matchup_last[mk] = (game_date, series_no)

        # Elo update
        elo_prob = elo_expected(current_elo[home_id], current_elo[vis_id])
        current_elo[home_id] += ELO_K * (home_win - elo_prob)
        current_elo[vis_id]  += ELO_K * ((1 - home_win) - (1 - elo_prob))

        # Team history
        hist[home_id].append({"rs": home_score, "ra": vis_score, "win": home_win})
        hist[vis_id].append({"rs":  vis_score,  "ra": home_score, "win": 1 - home_win})
        home_hist[home_id].append({"rs": home_score, "ra": vis_score, "win": home_win})
        away_hist[vis_id].append({"rs":  vis_score,  "ra": home_score, "win": 1 - home_win})
        last_date[home_id] = game_date
        last_date[vis_id]  = game_date
        team_season_cnt[home_id] += 1
        team_season_cnt[vis_id]  += 1

        # SP history
        for tid in (home_id, vis_id):
            sp = sp_data.get(game_pk, {}).get(tid)
            if sp:
                pid = sp["pitcher_id"]
                sp_starts[pid] += 1
                sp_hist_raw[pid].append({"ip": sp["ip"], "er": sp["er"],
                                         "k": sp["k"],   "bb": sp["bb"], "h": sp["h"]})
                sp_kbb_hist[pid].append({"k": sp["k"], "bb": sp["bb"]})

    return {
        "hist":            hist,
        "home_hist":       home_hist,
        "away_hist":       away_hist,
        "last_date":       last_date,
        "sp_starts":       sp_starts,
        "sp_hist_raw":     sp_hist_raw,
        "sp_kbb_hist":     sp_kbb_hist,
        "elo_hist":        elo_hist,
        "current_elo":     current_elo,
        "team_season_cnt": team_season_cnt,
        "matchup_last":    matchup_last,
        "current_season":  current_season,
        "prev_stats":      prev_stats,
        "park_factors":    park_factors,
        "league_map":      league_map,
        "rockies_id":      rockies_id,
    }


# ---------------------------------------------------------------------------
# Compute features for a single game
# ---------------------------------------------------------------------------

def compute_game_features(state, home_id, vis_id, game_date, season_year,
                           home_pid=None, vis_pid=None):
    hist            = state["hist"]
    home_hist       = state["home_hist"]
    away_hist       = state["away_hist"]
    last_date       = state["last_date"]
    sp_starts       = state["sp_starts"]
    sp_hist_raw     = state["sp_hist_raw"]
    sp_kbb_hist     = state["sp_kbb_hist"]
    elo_hist        = state["elo_hist"]
    current_elo     = state["current_elo"]
    team_season_cnt = state["team_season_cnt"]
    matchup_last    = state["matchup_last"]
    prev_stats      = state["prev_stats"]
    park_factors    = state["park_factors"]
    league_map      = state["league_map"]
    rockies_id      = state["rockies_id"]

    # Rolling team stats
    w5_h  = summarize(hist[home_id][-5:])
    w5_v  = summarize(hist[vis_id][-5:])
    w10_h = summarize(hist[home_id][-10:])
    w10_v = summarize(hist[vis_id][-10:])
    w20_h = summarize(hist[home_id][-20:])
    w20_v = summarize(hist[vis_id][-20:])
    w30_h = summarize(hist[home_id][-30:])
    w30_v = summarize(hist[vis_id][-30:])
    s5_h  = summarize(home_hist[home_id][-5:])
    s5_v  = summarize(away_hist[vis_id][-5:])
    s10_h = summarize(home_hist[home_id][-10:])
    s10_v = summarize(away_hist[vis_id][-10:])

    dw5_wp  = (w5_h["win_pct"]  - w5_v["win_pct"])  if (w5_h  and w5_v)  else 0.0
    dw5_rd  = (w5_h["rd_pg"]    - w5_v["rd_pg"])    if (w5_h  and w5_v)  else 0.0
    dw10_wp = (w10_h["win_pct"] - w10_v["win_pct"]) if (w10_h and w10_v) else 0.0
    dw10_rd = (w10_h["rd_pg"]   - w10_v["rd_pg"])   if (w10_h and w10_v) else 0.0
    dw20_wp = (w20_h["win_pct"] - w20_v["win_pct"]) if (w20_h and w20_v) else 0.0
    dw20_rd = (w20_h["rd_pg"]   - w20_v["rd_pg"])   if (w20_h and w20_v) else 0.0
    dw30_wp = (w30_h["win_pct"] - w30_v["win_pct"]) if (w30_h and w30_v) else 0.0
    dw30_rd = (w30_h["rd_pg"]   - w30_v["rd_pg"])   if (w30_h and w30_v) else 0.0
    ds5_wp  = (s5_h["win_pct"]  - s5_v["win_pct"])  if (s5_h  and s5_v)  else 0.0
    ds5_rd  = (s5_h["rd_pg"]    - s5_v["rd_pg"])    if (s5_h  and s5_v)  else 0.0
    ds10_wp = (s10_h["win_pct"] - s10_v["win_pct"]) if (s10_h and s10_v) else 0.0
    ds10_rd = (s10_h["rd_pg"]   - s10_v["rd_pg"])   if (s10_h and s10_v) else 0.0

    # Pythagorean residual (luck correction)
    h_pyth_res = (w20_h["win_pct"] - w20_h["pyth"]) if w20_h else 0.0
    v_pyth_res = (w20_v["win_pct"] - w20_v["pyth"]) if w20_v else 0.0
    diff_pyth_res = h_pyth_res - v_pyth_res

    # Elo
    h_elo = current_elo[home_id]
    v_elo = current_elo[vis_id]
    elo_prob = elo_expected(h_elo, v_elo)
    h_elo_10ago = elo_hist[home_id][-10] if len(elo_hist[home_id]) >= 10 else ELO_BASE
    v_elo_10ago = elo_hist[vis_id][-10]  if len(elo_hist[vis_id])  >= 10 else ELO_BASE
    diff_elo_mom = (h_elo - h_elo_10ago) - (v_elo - v_elo_10ago)

    # Rest / streak
    h_rest   = rest_days(last_date.get(home_id), game_date)
    v_rest   = rest_days(last_date.get(vis_id),  game_date)
    h_streak = streak_val(hist[home_id])
    v_streak = streak_val(hist[vis_id])

    home_sg = team_season_cnt[home_id]
    vis_sg  = team_season_cnt[vis_id]

    # SP stats
    home_sp_s    = sp_starts[home_pid] if home_pid else 0
    vis_sp_s     = sp_starts[vis_pid]  if vis_pid  else 0
    home_sp_roll = rolling_sp_stats(sp_hist_raw[home_pid]) if home_pid else None
    vis_sp_roll  = rolling_sp_stats(sp_hist_raw[vis_pid])  if vis_pid  else None
    sp_avail     = 1.0 if (home_sp_roll and vis_sp_roll) else 0.0

    diff_sp_era  = (vis_sp_roll["era"]  - home_sp_roll["era"])  if sp_avail else 0.0
    diff_sp_whip = (vis_sp_roll["whip"] - home_sp_roll["whip"]) if sp_avail else 0.0
    diff_sp_k9   = (home_sp_roll["k9"]  - vis_sp_roll["k9"])    if sp_avail else 0.0
    home_kbb     = rolling_kbb(sp_kbb_hist[home_pid]) if home_pid else None
    vis_kbb      = rolling_kbb(sp_kbb_hist[vis_pid])  if vis_pid  else None
    diff_sp_kbb  = (home_kbb - vis_kbb) if (home_kbb is not None and vis_kbb is not None) else 0.0

    # Park factor (fall back to previous season if current not available)
    pf = park_factors.get((season_year, home_id)) or park_factors.get((season_year - 1, home_id), 1.0)

    # Series position
    mk     = (min(home_id, vis_id), max(home_id, vis_id))
    last_m = matchup_last.get(mk)
    series_no = 0 if (last_m is None or (game_date - last_m[0]).days > 1) else min(last_m[1] + 1, 3)

    # MLB flags
    dh_era = 1 if season_year >= 2022 else 0
    coors  = 1 if home_id == rockies_id else 0
    h_lg   = league_map.get((season_year, home_id)) or league_map.get((season_year - 1, home_id))
    v_lg   = league_map.get((season_year, vis_id))  or league_map.get((season_year - 1, vis_id))
    interleague = 1 if (h_lg and v_lg and h_lg != v_lg) else 0

    # Prev-season priors
    ph = prev_stats.get((season_year, home_id))
    pv = prev_stats.get((season_year, vis_id))

    # Soft-regime weights
    team_ready = max(0.0, min(1.0, min(home_sg, vis_sg) / TEAM_BURN_IN))
    sp_ready   = max(0.0, min(1.0, min(home_sp_s, vis_sp_s) / STARTER_BURN_IN)) if sp_avail else 0.0
    early_w    = 1.0 - team_ready
    primary_w  = (1.0 - early_w) * sp_ready
    fallback_w = (1.0 - early_w) * (1.0 - sp_ready)

    return {
        "game_pk":     0,
        "season_year": season_year,
        "game_date":   game_date.isoformat(),
        "home_win":    -1,
        # Game features (mirrors build_rows output)
        "diff_win_pct":       dw20_wp,
        "diff_pyth_wp":       (w20_h["pyth"] - w20_v["pyth"]) if (w20_h and w20_v) else 0.0,
        "diff_rd":            dw20_rd,
        "diff_elo":           h_elo - v_elo,
        "home_elo":           h_elo,
        "vis_elo":            v_elo,
        "elo_win_prob":       elo_prob,
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
        "diff_streak":        h_streak - v_streak,
        "sp_available":       sp_avail,
        "diff_sp_era":        diff_sp_era,
        "diff_sp_whip":       diff_sp_whip,
        "diff_sp_k9":         diff_sp_k9,
        "diff_sp_kbb":        diff_sp_kbb,
        "universal_dh_era":   dh_era,
        "coors_field_factor": coors,
        "is_interleague":     interleague,
        "is_pitch_clock_era": 1.0 if season_year >= 2023 else 0.0,
        "home_park_factor":   pf,
        "series_game_no":     float(series_no),
        "prev_diff_win_pct":  nn(ph["win_pct"] if ph else None) - nn(pv["win_pct"] if pv else None),
        "prev_diff_rd_pg":    nn(ph["rd_pg"]   if ph else None) - nn(pv["rd_pg"]   if pv else None),
        "prev_diff_pyth":     nn(ph["pyth"]    if ph else None) - nn(pv["pyth"]    if pv else None),
        "home_season_games_before": home_sg,
        "vis_season_games_before":  vis_sg,
        "home_sp_starts_before":    home_sp_s,
        "vis_sp_starts_before":     vis_sp_s,
        "early_weight":    early_w,
        "primary_weight":  primary_w,
        "fallback_weight": fallback_w,
    }


# ---------------------------------------------------------------------------
# Fetch today's schedule from MLB Stats API
# ---------------------------------------------------------------------------

def fetch_today_games(session, date_str):
    resp = session.get(
        SCHEDULE_URL,
        params={
            "sportId":   1,
            "startDate": date_str,
            "endDate":   date_str,
            "gameType":  "R",
            "hydrate":   "probablePitcher,linescore",
        },
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()

    games = []
    for date_block in data.get("dates") or []:
        for game in date_block.get("games") or []:
            gp = game.get("gamePk")
            if not gp:
                continue
            teams     = game.get("teams") or {}
            home      = teams.get("home") or {}
            away      = teams.get("away") or {}
            home_team = home.get("team") or {}
            away_team = away.get("team") or {}
            home_pp   = home.get("probablePitcher") or {}
            away_pp   = away.get("probablePitcher") or {}
            status    = ((game.get("status") or {}).get("detailedState") or "Scheduled")

            games.append({
                "game_pk":   gp,
                "home_id":   home_team.get("id"),
                "home_name": home_team.get("name", "?"),
                "vis_id":    away_team.get("id"),
                "vis_name":  away_team.get("name", "?"),
                "home_pid":  home_pp.get("id"),
                "home_pname":home_pp.get("fullName", "TBD"),
                "vis_pid":   away_pp.get("id"),
                "vis_pname": away_pp.get("fullName", "TBD"),
                "status":    status,
            })
    return games


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=date.today().isoformat(),
                        help="Prediction date (YYYY-MM-DD). Default: today.")
    parser.add_argument("--save", action="store_true",
                        help="Save predictions to predictions_YYYY-MM-DD.md")
    args = parser.parse_args()

    pred_date   = datetime.fromisoformat(args.date).date()
    season_year = pred_date.year

    print(f"=== MLB Predictions  {args.date} ===\n")

    conn    = sqlite3.connect(DB_PATH)
    session = requests.Session()
    session.headers.update({"User-Agent": "MLB-Predictor/1.0", "Accept": "application/json"})

    try:
        # 1. Train on all historical data
        print("Training model on historical data...")
        all_rows = build_rows(conn)
        models   = fit_models(all_rows)
        print(f"  {len(all_rows)} games  ({min(r['season_year'] for r in all_rows)}"
              f"–{max(r['season_year'] for r in all_rows)})\n")

        # 2. Build live rolling state
        print("Building rolling state...")
        state = build_live_state(conn)
        last_season = state["current_season"]
        print(f"  State current through season {last_season}.\n")

        # Apply season-start Elo regression if we're in a new season
        if season_year > last_season:
            for tid in list(state["current_elo"].keys()):
                state["current_elo"][tid] = (
                    ELO_BASE * ELO_REGRESSION + state["current_elo"][tid] * (1 - ELO_REGRESSION)
                )
            state["team_season_cnt"] = defaultdict(int)

        # 3. Fetch schedule
        print(f"Fetching schedule for {args.date}...")
        games = fetch_today_games(session, args.date)
        if not games:
            print("No regular-season games found.")
            return
        # For past dates show all games; for today show only unstarted ones
        if pred_date < date.today():
            pending = games
            print(f"  {len(games)} games (past date — predicting all)\n")
        else:
            pending = [g for g in games if "Final" not in g["status"]]
            if not pending:
                pending = games
            print(f"  {len(pending)} upcoming  ({len(games)} total today)\n")

        # 4. Predict
        header = f"{'Away':22} {'Home':22} {'Away SP':18} {'Home SP':18} {'Prob':>6} {'Fav':>4} {'Note'}"
        divider = "─" * len(header)
        print(header)
        print(divider)

        rows_md = []
        for g in sorted(pending, key=lambda x: (x["home_id"] or 0)):
            home_id = g["home_id"]
            vis_id  = g["vis_id"]
            if home_id is None or vis_id is None:
                continue

            feat = compute_game_features(
                state, home_id, vis_id, pred_date, season_year,
                home_pid=g["home_pid"], vis_pid=g["vis_pid"],
            )
            prob, label = soft_predict(models, feat)

            conf    = max(prob, 1 - prob)
            favored = "HOME" if prob >= 0.5 else "AWAY"
            note    = "***" if conf >= 0.675 else ("*" if conf >= 0.60 else "")
            sp_flag = "" if feat["sp_available"] else " (no SP)"

            away_sp = (g["vis_pname"] or "TBD")[:17]
            home_sp = (g["home_pname"] or "TBD")[:17]
            print(f"{g['vis_name']:22.22} {g['home_name']:22.22} "
                  f"{away_sp:18.18} {home_sp:18.18} "
                  f"{conf:>5.1%} {favored:>4}  {note}{sp_flag}")
            rows_md.append((g, feat, prob, conf, favored, label, note, sp_flag))

        print(divider)
        print("*** = P>=0.675 (hist acc ~69%)  * = P>=0.60  Prob = confidence (always >= 50%)\n")

        # 5. Save markdown
        if args.save:
            _save_markdown(args.date, rows_md)

    finally:
        conn.close()
        session.close()


def _save_markdown(date_str, rows_md):
    path = Path(f"predictions_{date_str.replace('-', '')}.md")
    lines = [
        f"# MLB Predictions {date_str}",
        "",
        "| Away | Home | Away SP | Home SP | Prob | Favored | Note |",
        "| --- | --- | --- | --- | ---: | --- | --- |",
    ]
    for g, feat, prob, conf, favored, label, note, sp_flag in rows_md:
        lines.append(
            f"| {g['vis_name']} | {g['home_name']} "
            f"| {g['vis_pname'] or 'TBD'} | {g['home_pname'] or 'TBD'} "
            f"| {conf:.1%} | {favored} | {note}{sp_flag} |"
        )
    lines += [
        "",
        f"*Model: Soft-regime XGBoost (max_depth=3, reg_lambda=3.0)*  ",
        "**** = P>=0.675 (historical acc ~69%)  * = P>=0.60*",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved → {path}")


if __name__ == "__main__":
    main()
