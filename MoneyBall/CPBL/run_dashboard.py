"""
CPBL dashboard data adapter.
Run from CPBL/ directory (subprocess, cwd=CPBL_DIR).
Outputs single JSON line to stdout.

Usage: python run_dashboard.py [YYYY-MM-DD]
"""
import json, sys, sqlite3, subprocess
from datetime import date, datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

# predict_today.py replaces sys.stdout at import time; importing it first avoids
# creating a competing TextIOWrapper that would close the shared buffer on GC.
from predict_today import DB_PATH, TEAM_NAMES, load_data, train_and_predict
from playsport_scraper import fetch_playsport_game_map
from playsport_results_sync import sync_results_for_league

_HERE = Path(__file__).parent


def _db_latest_date(conn) -> str | None:
    row = conn.execute("SELECT MAX(DATE(game_date)) FROM team_game_results").fetchone()
    return row[0] if row else None


def _has_games_for_date(conn, date_str: str) -> bool:
    row = conn.execute(
        "SELECT COUNT(*) FROM team_game_results WHERE DATE(game_date) = ? AND game_status IN (1,3,4,6)",
        (date_str,)
    ).fetchone()
    return (row[0] or 0) > 0


def _run_refresh(year: int) -> list[str]:
    cmd = [sys.executable, str(_HERE / "cpbl_boxscore_scraper.py"),
           "--start-year", str(year), "--end-year", str(year)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                                errors="replace", timeout=180, cwd=str(_HERE))
        status = "ok" if result.returncode == 0 else f"exit={result.returncode}"
        return [f"cpbl_boxscore_scraper:{status}"]
    except subprocess.TimeoutExpired:
        return ["cpbl_boxscore_scraper:timeout"]
    except Exception as exc:
        return [f"cpbl_boxscore_scraper:error({exc})"]


from cpbl_betting_ev import ev, vig_free_prob, EV_THRESHOLD, MIN_CONF

_CONF_THRESHOLD = 0.60


def _conf(prob_home):
    c = max(prob_home, 1 - prob_home)
    return "HIGH" if c >= 0.60 else ("MED" if c >= 0.55 else "LOW")


def main():
    date_str = sys.argv[1] if len(sys.argv) > 1 else date.today().isoformat()
    target = date.fromisoformat(date_str)

    games_out = []
    error_msg = None
    odds_fetched = False
    has_live_odds = False
    refresh_log = None
    db_latest = None
    playsport = {"ok": False, "reason": None, "games": [], "game_map": {}}

    try:
        refresh_log = []
        if target <= date.today():
            sync_summary = sync_results_for_league("cpbl", target)
            refresh_log.append(
                "playsport_results:"
                f"ok={sync_summary['ok']},changed={sync_summary['changed']},"
                f"inserted={sync_summary['inserted']},updated={sync_summary['updated']}"
            )

        # Auto-refresh: if no games for target date and DB is stale, run scraper
        conn = sqlite3.connect(DB_PATH)
        try:
            db_latest = _db_latest_date(conn)
            has_games = _has_games_for_date(conn, date_str)
        finally:
            conn.close()

        if not has_games and (db_latest is None or db_latest < date_str):
            refresh_log.extend(_run_refresh(target.year))

        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        try:
            train_rows, pred_games = load_data(conn, target)
            predictions = train_and_predict(train_rows, pred_games)
        finally:
            conn.close()

        try:
            playsport = fetch_playsport_game_map("cpbl", target)
            odds_fetched = playsport.get("ok", False)
            has_live_odds = any(g.get("odds_home") and g.get("odds_away") for g in playsport.get("games", []))
        except Exception as playsport_err:
            error_msg = f"playsport_fetch_error: {playsport_err}"

        for p in predictions:
            prob_home = float(p["prob_home_win"])
            prob_away = 1.0 - prob_home
            home_code = p["home_team"]
            away_code = p["vis_team"]
            hn = TEAM_NAMES.get(home_code, home_code)
            vn = TEAM_NAMES.get(away_code, away_code)
            source_game = playsport["game_map"].get((hn, vn))
            home_sp = (source_game or {}).get("home_sp") or p.get("home_sp_name") or "TBD"
            away_sp = (source_game or {}).get("away_sp") or p.get("vis_sp_name") or "TBD"

            odds = None
            if source_game and source_game.get("odds_home") and source_game.get("odds_away"):
                odds = (source_game["odds_home"], source_game["odds_away"])

            if odds is not None:
                dec_home, dec_away = odds
                fair_h, fair_a = vig_free_prob(dec_home, dec_away)
                ev_home = ev(prob_home, dec_home)
                ev_away = ev(prob_away, dec_away)
                edge_home = prob_home - fair_h
                edge_away = prob_away - fair_a

                # Determine best side to bet
                bet_side = None
                if prob_home >= MIN_CONF and ev_home >= EV_THRESHOLD:
                    bet_side = "HOME"
                if prob_away >= MIN_CONF and ev_away >= EV_THRESHOLD:
                    if bet_side is None or ev_away > ev_home:
                        bet_side = "AWAY"

                games_out.append({
                    "id": str(p.get("game_sno", "")),
                    "home": hn,
                    "away": vn,
                    "home_sp": home_sp,
                    "away_sp": away_sp,
                    "status": "Scheduled",
                    "prob_home": round(prob_home, 4),
                    "confidence": _conf(prob_home),
                    "route": p.get("model_used", ""),
                    "odds_home": dec_home,
                    "odds_away": dec_away,
                    "ev_home": round(ev_home, 4),
                    "ev_away": round(ev_away, 4),
                    "edge_home": round(edge_home, 4),
                    "edge_away": round(edge_away, 4),
                    "bet_side": bet_side,
                    "has_odds": True,
                })
            else:
                games_out.append({
                    "id": str(p.get("game_sno", "")),
                    "home": hn,
                    "away": vn,
                    "home_sp": home_sp,
                    "away_sp": away_sp,
                    "status": "Scheduled",
                    "prob_home": round(prob_home, 4),
                    "confidence": _conf(prob_home),
                    "route": p.get("model_used", ""),
                    "odds_home": None,
                    "odds_away": None,
                    "ev_home": None,
                    "ev_away": None,
                    "edge_home": None,
                    "edge_away": None,
                    "bet_side": None,
                    "has_odds": False,
                })

    except Exception as e:
        if error_msg is None:
            error_msg = str(e)
        else:
            error_msg = f"{error_msg}; {e}"

    result = {
        "league": "CPBL",
        "date": date_str,
        "last_updated": datetime.now().isoformat(),
        "odds_source": "playsport.cc",
        "has_live_odds": has_live_odds,
        "odds_fetched": odds_fetched,
        "games": games_out,
        "error": error_msg,
        "db_latest_date": db_latest,
        "refresh_log": refresh_log,
    }
    # Write bytes directly to avoid any TextIOWrapper closed-buffer issues
    out = (json.dumps(result, ensure_ascii=False) + "\n").encode("utf-8")
    sys.stdout.buffer.write(out)
    sys.stdout.buffer.flush()


if __name__ == "__main__":
    main()
