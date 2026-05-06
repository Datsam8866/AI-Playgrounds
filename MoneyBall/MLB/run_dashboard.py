"""
MLB dashboard data adapter.
Run from MLB/ directory (subprocess, cwd=MLB_DIR).
Outputs single JSON line to stdout.

Usage: python run_dashboard.py [YYYY-MM-DD]
"""
import json, sys, sqlite3, requests
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from predict_mlb_today import (
    build_live_state, compute_game_features, fetch_today_games,
    ELO_BASE, ELO_REGRESSION,
)
from train_mlb_model import build_rows, fit_models, soft_predict
from mlb_betting_ev import ev, vig_free_prob, EV_THRESHOLD
from playsport_scraper import fetch_playsport_game_map
from playsport_results_sync import sync_results_for_league

MIN_CONF_MLB = 0.52

def _conf(prob_home):
    c = max(prob_home, 1 - prob_home)
    return "HIGH" if c >= 0.675 else ("MED" if c >= 0.55 else "LOW")


def main():
    argv = sys.argv[1:]
    strict_cutoff = "--strict-cutoff" in argv
    date_str = next((arg for arg in argv if not arg.startswith("--")), date.today().isoformat())
    pred_date = date.fromisoformat(date_str)
    season_year = pred_date.year

    conn = sqlite3.connect("mlb.sqlite")
    session = requests.Session()
    session.headers.update({"User-Agent": "MLB-Dashboard/1.0", "Accept": "application/json"})

    games_out = []
    error_msg = None
    odds_ok = False
    playsport = {"ok": False, "reason": None, "games": [], "game_map": {}}

    try:
        if pred_date <= date.today():
            sync_results_for_league("mlb", pred_date)

        cutoff_date = pred_date if strict_cutoff else None
        all_rows = build_rows(conn, cutoff_date=cutoff_date)
        models = fit_models(all_rows)

        state = build_live_state(conn, cutoff_date=cutoff_date)
        if season_year > (state.get("current_season") or 0):
            for tid in list(state["current_elo"].keys()):
                state["current_elo"][tid] = (
                    ELO_BASE * ELO_REGRESSION + state["current_elo"][tid] * (1 - ELO_REGRESSION)
                )
            state["team_season_cnt"] = defaultdict(int)

        games = fetch_today_games(session, date_str)

        try:
            playsport = fetch_playsport_game_map("mlb", pred_date)
            odds_ok = playsport.get("ok", False)
        except Exception as e:
            print(f"PLAYSPORT ERROR: {e}", file=sys.stderr)

        for g in games:
            if not g["home_id"] or not g["vis_id"]:
                continue

            feat = compute_game_features(
                state, g["home_id"], g["vis_id"], pred_date, season_year,
                home_pid=g["home_pid"], vis_pid=g["vis_pid"],
            )
            prob_home, _ = soft_predict(models, feat)
            prob_away = 1.0 - prob_home

            source_game = playsport["game_map"].get((g["home_name"], g["vis_name"]))
            home_sp = (source_game or {}).get("home_sp") or g.get("home_pname") or "TBD"
            away_sp = (source_game or {}).get("away_sp") or g.get("vis_pname") or "TBD"
            odds = None
            if source_game and source_game.get("odds_home") and source_game.get("odds_away"):
                odds = (source_game["odds_home"], source_game["odds_away"])
            dec_home = dec_away = None
            ev_home = ev_away = edge_home = edge_away = None
            bet_side = None

            if odds:
                dec_home, dec_away = odds
                fair_h, fair_a = vig_free_prob(dec_home, dec_away)
                ev_home = round(ev(prob_home, dec_home), 4)
                ev_away = round(ev(prob_away, dec_away), 4)
                edge_home = round(prob_home - fair_h, 4)
                edge_away = round(prob_away - fair_a, 4)
                if prob_home >= MIN_CONF_MLB and ev_home >= EV_THRESHOLD:
                    bet_side = "HOME"
                elif prob_away >= MIN_CONF_MLB and ev_away >= EV_THRESHOLD:
                    bet_side = "AWAY"

            games_out.append({
                "id": str(g["game_pk"]),
                "home": g["home_name"],
                "away": g["vis_name"],
                "home_sp": home_sp,
                "away_sp": away_sp,
                "status": g.get("status", "Scheduled"),
                "prob_home": round(prob_home, 4),
                "confidence": _conf(prob_home),
                "route": "primary" if feat.get("sp_available") else "fallback",
                "odds_home": dec_home,
                "odds_away": dec_away,
                "ev_home": ev_home,
                "ev_away": ev_away,
                "edge_home": edge_home,
                "edge_away": edge_away,
                "bet_side": bet_side,
                "has_odds": odds is not None,
            })

    except Exception as e:
        error_msg = str(e)
    finally:
        conn.close()
        session.close()

    result = {
        "league": "MLB",
        "date": date_str,
        "last_updated": datetime.now().isoformat(),
        "odds_source": "playsport.cc",
        "has_live_odds": any(g.get("odds_home") and g.get("odds_away") for g in playsport.get("games", [])),
        "odds_fetched": odds_ok,
        "games": games_out,
        "error": error_msg,
        "cutoff_mode": "previous_day" if strict_cutoff else "live",
    }
    print(json.dumps(result, ensure_ascii=False))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
