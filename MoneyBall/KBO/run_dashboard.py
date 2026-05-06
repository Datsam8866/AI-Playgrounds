"""
KBO dashboard data adapter.
Run from KBO/ directory (subprocess, cwd=KBO_DIR).
Outputs single JSON line to stdout.

Usage: python run_dashboard.py [YYYY-MM-DD]

Auto-refresh logic:
  If game_features has no row for the target date AND the DB is stale
  (latest game_date < target date), this script will automatically:
    1. Run kbo_boxscore_scraper.py --start-year <current_year> --end-year <current_year>
       to fetch any completed games missing from team_game_results.
    2. Run build_kbo_game_features.py to rebuild game_features.
    3. Run build_kbo_pitcher_features.py to rebuild SP features.
  Then re-query game_features for the target date.
"""
import json, sys, sqlite3, subprocess
from datetime import date, datetime
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import evaluate_kbo_predictions_regime as ev
from playsport_scraper import fetch_playsport_game_map
from playsport_results_sync import sync_results_for_league

TEAM_NAMES = {
    "LG":  "LG 트윈스",   "KT":  "KT 위즈",
    "SS":  "삼성 라이온즈", "NC":  "NC 다이노스",
    "OB":  "두산 베어스",  "KIA": "KIA 타이거즈",
    "LT":  "롯데 자이언츠","SK":  "SSG 랜더스",
    "HH":  "한화 이글스",  "HT":  "KIA 타이거즈",
    "WO":  "키움 히어로즈",
}

EV_THRESHOLD = 0.03
KBO_MIN_CONF = 0.55

# Scripts are expected to live alongside this file (KBO/ directory).
_HERE = Path(__file__).parent


def _team(code):
    return TEAM_NAMES.get(code, code)


def _conf(prob_home):
    c = max(prob_home, 1 - prob_home)
    return "HIGH" if c >= 0.60 else ("MED" if c >= 0.55 else "LOW")


def _db_latest_date(conn) -> str | None:
    """Return the latest game_date in game_features, or None if table is empty."""
    row = conn.execute("SELECT MAX(game_date) FROM game_features").fetchone()
    return row[0] if row else None


def _run_feature_rebuild(target_date_str: str) -> list[str]:
    """
    Run feature builder → pitcher feature builder.
    Returns a list of status messages (one per step).
    """
    logs = []
    scripts = [
        ([sys.executable, str(_HERE / "build_kbo_game_features.py"),
          "--include-scheduled", "--date", target_date_str],
         "build_kbo_game_features"),
        ([sys.executable, str(_HERE / "build_kbo_pitcher_features.py")],
         "build_kbo_pitcher_features"),
    ]
    for cmd, label in scripts:
        try:
            result = subprocess.run(
                cmd,
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=120, cwd=str(_HERE),
            )
            status = "ok" if result.returncode == 0 else f"exit={result.returncode}"
            logs.append(f"{label}:{status}")
        except subprocess.TimeoutExpired:
            logs.append(f"{label}:timeout")
        except Exception as exc:
            logs.append(f"{label}:error({exc})")
    return logs


def load_target_rows(conn, target):
    date_str = target.isoformat()
    rows = conn.execute(
        "SELECT * FROM game_features WHERE sr_id=0 AND game_date=? ORDER BY game_id",
        (date_str,)
    ).fetchall()
    cols = [d[1] for d in conn.execute("PRAGMA table_info(game_features)").fetchall()]
    result = []
    for r in rows:
        d = dict(zip(cols, r))
        d["early_flag"] = int(
            d.get("home_season_games_before", 0) < ev.TEAM_BURN_IN
            or d.get("away_season_games_before", 0) < ev.TEAM_BURN_IN
        )
        d["sp_available"] = int(d.get("diff_sp_era") is not None)
        result.append(d)
    return result


def main():
    argv = sys.argv[1:]
    strict_cutoff = "--strict-cutoff" in argv
    date_str = next((arg for arg in argv if not arg.startswith("--")), date.today().isoformat())
    target = date.fromisoformat(date_str)

    games_out = []
    error_msg = None
    refresh_log = None
    db_latest = None
    odds_fetched = False
    has_live_odds = False
    playsport = {"ok": False, "reason": None, "games": [], "game_map": {}}

    try:
        if target <= date.today():
            sync_summary = sync_results_for_league("kbo", target)
            refresh_log = [
                "playsport_results:"
                f"ok={sync_summary['ok']},changed={sync_summary['changed']},"
                f"inserted={sync_summary['inserted']},updated={sync_summary['updated']}"
            ]
            if sync_summary["changed"]:
                refresh_log.extend(_run_feature_rebuild(date_str))

        conn = sqlite3.connect(ev.DB_PATH)
        try:
            db_latest = _db_latest_date(conn)
            target_rows = load_target_rows(conn, target)
        finally:
            conn.close()

        # If no rows for today AND DB is behind target date → auto-refresh
        if not target_rows and (db_latest is None or db_latest < date_str):
            extra_logs = _run_feature_rebuild(date_str)
            refresh_log = (refresh_log or []) + extra_logs
            # Re-open connection after rebuild
            conn = sqlite3.connect(ev.DB_PATH)
            try:
                db_latest = _db_latest_date(conn)
                target_rows = load_target_rows(conn, target)
            finally:
                conn.close()

        try:
            playsport = fetch_playsport_game_map("kbo", target)
            odds_fetched = playsport.get("ok", False)
            has_live_odds = any(g.get("odds_home") and g.get("odds_away") for g in playsport.get("games", []))
        except Exception as playsport_err:
            error_msg = f"playsport_fetch_error: {playsport_err}"

        if target_rows:
            conn = sqlite3.connect(ev.DB_PATH)
            try:
                all_rows = ev.load_rows(conn)
            finally:
                conn.close()

            train = [
                r for r in all_rows
                if ev.TRAIN_START_YEAR <= r["season_year"] <= target.year
                and r["game_date"] < date_str
            ]
            models = ev.train_models(train)

            for row in target_rows:
                prob_home, model_used = ev.predict_one(models, row)
                prob_away = 1.0 - prob_home
                home_name = _team(row.get("home_code", ""))
                away_name = _team(row.get("away_code", ""))
                source_game = playsport["game_map"].get((home_name, away_name))
                dec_home = (source_game or {}).get("odds_home")
                dec_away = (source_game or {}).get("odds_away")
                ev_home = ev_away = edge_home = edge_away = None
                bet_side = None

                if dec_home and dec_away:
                    raw_h, raw_a = 1 / dec_home, 1 / dec_away
                    total = raw_h + raw_a
                    fair_h, fair_a = raw_h / total, raw_a / total
                    ev_home = round(prob_home * (dec_home - 1) - prob_away, 4)
                    ev_away = round(prob_away * (dec_away - 1) - prob_home, 4)
                    edge_home = round(prob_home - fair_h, 4)
                    edge_away = round(prob_away - fair_a, 4)
                    if prob_home >= KBO_MIN_CONF and ev_home >= EV_THRESHOLD:
                        bet_side = "HOME"
                    elif prob_away >= KBO_MIN_CONF and ev_away >= EV_THRESHOLD:
                        bet_side = "AWAY"

                games_out.append({
                    "id": str(row.get("game_id", "")),
                    "home": home_name,
                    "away": away_name,
                    "home_sp": (source_game or {}).get("home_sp") or "TBD",
                    "away_sp": (source_game or {}).get("away_sp") or "TBD",
                    "status": "Scheduled",
                    "prob_home": round(prob_home, 4),
                    "confidence": _conf(prob_home),
                    "route": model_used,
                    "odds_home": dec_home,
                    "odds_away": dec_away,
                    "ev_home": ev_home,
                    "ev_away": ev_away,
                    "edge_home": edge_home,
                    "edge_away": edge_away,
                    "bet_side": bet_side,
                    "has_odds": bool(dec_home and dec_away),
                })

    except Exception as e:
        error_msg = str(e)

    result = {
        "league": "KBO",
        "date": date_str,
        "last_updated": datetime.now().isoformat(),
        "odds_source": "playsport.cc",
        "has_live_odds": has_live_odds,
        "odds_fetched": odds_fetched,
        "games": games_out,
        "error": error_msg,
        "db_latest_date": db_latest,
        "refresh_log": refresh_log,
        "cutoff_mode": "previous_day" if strict_cutoff else "live",
    }
    print(json.dumps(result, ensure_ascii=False))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
