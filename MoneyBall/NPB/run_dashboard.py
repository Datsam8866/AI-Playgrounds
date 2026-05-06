"""
NPB dashboard data adapter.
Run from NPB/ directory (subprocess, cwd=NPB_DIR).
Outputs single JSON line to stdout.

Usage: python run_dashboard.py [YYYY-MM-DD]
"""
import json, sys, sqlite3, subprocess
from datetime import date, datetime
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from predict_today_npb import (
    connect, load_training_rows, load_target_rows, validate_features,
    train_models, predict_rows, TEAM_SHORT,
)
from npb_betting_ev import ev, vig_free_prob, EV_THRESHOLD
from playsport_scraper import fetch_playsport_game_map
from playsport_results_sync import sync_results_for_league

NPB_MIN_CONF = 0.55
_HERE = Path(__file__).parent


def _team(code):
    return TEAM_SHORT.get((code or "").lower(), code or "")


def _conf(prob_home):
    c = max(prob_home, 1 - prob_home)
    return "HIGH" if c >= 0.60 else ("MED" if c >= 0.55 else "LOW")


def _run_feature_rebuild(year: int) -> list[str]:
    logs = []
    scripts = [
        ([sys.executable, str(_HERE / "build_game_features_npb.py"),
          "--year", str(year), "--include-scheduled"], "build_game_features_npb"),
        ([sys.executable, str(_HERE / "build_pitcher_features_npb.py"),
          "--year", str(year)], "build_pitcher_features_npb"),
    ]
    for cmd, label in scripts:
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=180,
                cwd=str(_HERE),
            )
            status = "ok" if result.returncode == 0 else f"exit={result.returncode}"
            logs.append(f"{label}:{status}")
        except subprocess.TimeoutExpired:
            logs.append(f"{label}:timeout")
        except Exception as exc:
            logs.append(f"{label}:error({exc})")
    return logs


def main():
    date_str = sys.argv[1] if len(sys.argv) > 1 else date.today().isoformat()
    target = date.fromisoformat(date_str)

    games_out = []
    error_msg = None
    odds_ok = False
    playsport = {"ok": False, "reason": None, "games": [], "game_map": {}}
    refresh_log = None

    try:
        if target <= date.today():
            sync_summary = sync_results_for_league("npb", target)
            refresh_log = [
                "playsport_results:"
                f"ok={sync_summary['ok']},changed={sync_summary['changed']},"
                f"inserted={sync_summary['inserted']},updated={sync_summary['updated']}"
            ]
            if sync_summary["changed"]:
                refresh_log.extend(_run_feature_rebuild(target.year))

        conn = connect()
        try:
            train_rows = load_training_rows(conn, target)
            validate_features(train_rows)
            past = target < date.today()
            target_rows = load_target_rows(conn, target, verify=past)
        finally:
            conn.close()

        predictions = []
        if target_rows:
            models = train_models(train_rows)
            predictions = predict_rows(models, target_rows)

        try:
            playsport = fetch_playsport_game_map("npb", target)
            odds_ok = playsport.get("ok", False)
        except Exception as e:
            print(f"PLAYSPORT ERROR: {e}", file=sys.stderr)

        for pred in predictions:
            home_code = str(pred.get("home_code") or "")
            away_code = str(pred.get("away_code") or "")
            prob_home = float(pred["prob_home_win"])
            prob_away = 1.0 - prob_home
            home_name = _team(home_code)
            away_name = _team(away_code)
            source_game = playsport["game_map"].get((home_name, away_name))

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
                if prob_home >= NPB_MIN_CONF and ev_home >= EV_THRESHOLD:
                    bet_side = "HOME"
                elif prob_away >= NPB_MIN_CONF and ev_away >= EV_THRESHOLD:
                    bet_side = "AWAY"

            games_out.append({
                "id": str(pred.get("game_url", "")),
                "home": home_name,
                "away": away_name,
                "home_sp": (source_game or {}).get("home_sp") or "TBD",
                "away_sp": (source_game or {}).get("away_sp") or "TBD",
                "status": "Scheduled",
                "prob_home": round(prob_home, 4),
                "confidence": _conf(prob_home),
                "route": pred.get("route", ""),
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

    result = {
        "league": "NPB",
        "date": date_str,
        "last_updated": datetime.now().isoformat(),
        "odds_source": "playsport.cc",
        "has_live_odds": any(g.get("odds_home") and g.get("odds_away") for g in playsport.get("games", [])),
        "odds_fetched": odds_ok,
        "games": games_out,
        "error": error_msg,
        "refresh_log": refresh_log,
    }
    print(json.dumps(result, ensure_ascii=False))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
