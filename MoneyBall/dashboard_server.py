"""
MoneyBall Dashboard Server
Usage:  python dashboard_server.py [--port 5555]
Open:   http://localhost:5555
"""
import argparse
import json
import re
import sqlite3
import subprocess
import sys
import threading
from datetime import date, datetime, timedelta
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_from_directory

app = Flask(__name__, static_folder="dashboard")

BASE_DIR   = Path(__file__).parent
DASH_DIR   = BASE_DIR / "dashboard"
DATA_DIR   = DASH_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

LEAGUE_DIRS = {
    "mlb":  BASE_DIR / "MLB",
    "cpbl": BASE_DIR / "CPBL",
    "kbo":  BASE_DIR / "KBO",
    "npb":  BASE_DIR / "NPB",
    "nba":  BASE_DIR / "NBA",
}

_lock    = {lg: threading.Lock() for lg in LEAGUE_DIRS}
_status  = {lg: {"running": False, "last": None, "error": None, "progress": ""} for lg in LEAGUE_DIRS}

MIN_CONF_MAP = {"mlb": 0.52, "cpbl": 0.60, "kbo": 0.55, "npb": 0.55}
EV_THRESHOLD = 0.03


def _cache_path(league: str, date_str: str) -> Path:
    return DATA_DIR / f"{league}_{date_str}.json"


def _load_cached_data(league: str, date_str: str) -> dict | None:
    cache = _cache_path(league, date_str)
    if not cache.exists():
        return None
    try:
        return json.loads(cache.read_text(encoding="utf-8"))
    except Exception:
        return None


def _run_dashboard_command(script: Path, date_str: str, strict_cutoff: bool = False) -> list[str]:
    cmd = [sys.executable, str(script), date_str]
    if strict_cutoff:
        cmd.append("--strict-cutoff")
    return cmd


def _game_key(game: dict) -> tuple:
    return (
        game.get("id"),
        game.get("home"),
        game.get("away"),
    )


def _team_key(game: dict) -> tuple:
    return (
        game.get("home"),
        game.get("away"),
    )


def _apply_odds_fields(game: dict, league: str) -> None:
    oh = game.get("odds_home")
    oa = game.get("odds_away")
    if not (oh and oa and oh > 1 and oa > 1):
        game["ev_home"] = None
        game["ev_away"] = None
        game["edge_home"] = None
        game["edge_away"] = None
        game["bet_side"] = None
        game["has_odds"] = bool(oh and oa)
        return

    ph = game.get("prob_home", 0.5)
    pa = 1.0 - ph
    raw_h, raw_a = 1 / oh, 1 / oa
    total = raw_h + raw_a
    fair_h, fair_a = raw_h / total, raw_a / total
    ev_h = round(ph * (oh - 1) - pa, 4)
    ev_a = round(pa * (oa - 1) - ph, 4)
    game["ev_home"] = ev_h
    game["ev_away"] = ev_a
    game["edge_home"] = round(ph - fair_h, 4)
    game["edge_away"] = round(pa - fair_a, 4)
    game["has_odds"] = True

    min_conf = MIN_CONF_MAP.get(league, 0.55)
    bet = None
    if ph >= min_conf and ev_h >= EV_THRESHOLD:
        bet = "HOME"
    elif pa >= min_conf and ev_a >= EV_THRESHOLD:
        bet = "AWAY"
    game["bet_side"] = bet


def _merge_cached_manual_odds(league: str, date_str: str, fresh_data: dict) -> dict:
    cached = _load_cached_data(league, date_str)
    if not cached:
        return fresh_data

    cached_games = {}
    cached_team_games = {}
    for game in cached.get("games", []):
        cached_games[_game_key(game)] = game
        cached_team_games[_team_key(game)] = game

    fresh_has_source_odds = any(
        game.get("odds_home") is not None and game.get("odds_away") is not None
        for game in fresh_data.get("games", [])
    )
    merged_cached_odds = False

    for game in fresh_data.get("games", []):
        if game.get("odds_home") is not None or game.get("odds_away") is not None:
            _apply_odds_fields(game, league)
            continue

        prior = cached_games.get(_game_key(game)) or cached_team_games.get(_team_key(game))
        if not prior:
            continue

        if (not game.get("home_sp") or game.get("home_sp") == "TBD") and prior.get("home_sp") and prior.get("home_sp") != "TBD":
            game["home_sp"] = prior.get("home_sp")
        if (not game.get("away_sp") or game.get("away_sp") == "TBD") and prior.get("away_sp") and prior.get("away_sp") != "TBD":
            game["away_sp"] = prior.get("away_sp")

        if prior.get("odds_home") is not None or prior.get("odds_away") is not None:
            game["odds_home"] = prior.get("odds_home")
            game["odds_away"] = prior.get("odds_away")
            _apply_odds_fields(game, league)
            merged_cached_odds = True

    if not fresh_has_source_odds and merged_cached_odds:
        cached_source = cached.get("odds_source")
        if cached_source:
            fresh_data["odds_source"] = f"{cached_source} (cache)"
        fresh_data["odds_fetched"] = cached.get("odds_fetched", False)
        fresh_data["has_live_odds"] = any(game.get("has_odds") for game in fresh_data.get("games", []))

    return fresh_data


# ── Static ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(DASH_DIR, "index.html")

@app.route("/dashboard/<path:filename>")
def static_files(filename):
    return send_from_directory(DASH_DIR, filename)


# ── Data API ────────────────────────────────────────────────────────────────

@app.route("/api/data/<league>")
def get_data(league):
    if league not in LEAGUE_DIRS:
        return jsonify({"error": "Unknown league"}), 400
    date_str = request.args.get("date", date.today().isoformat())
    cache = _cache_path(league, date_str)
    if cache.exists():
        return Response(cache.read_text(encoding="utf-8"), mimetype="application/json")
    return jsonify({"league": league.upper(), "date": date_str, "games": [], "last_updated": None})


@app.route("/api/refresh/<league>", methods=["POST"])
def refresh(league):
    if league not in LEAGUE_DIRS:
        return jsonify({"error": "Unknown league"}), 400
    date_str = request.args.get("date", date.today().isoformat())

    with _lock[league]:
        if _status[league]["running"]:
            return jsonify({"status": "already_running"})
        _status[league]["running"]  = True
        _status[league]["error"]    = None
        _status[league]["progress"] = "Starting…"

    def run():
        try:
            script = LEAGUE_DIRS[league] / "run_dashboard.py"
            _status[league]["progress"] = "Running prediction pipeline…"
            strict_cutoff = date_str < date.today().isoformat()
            result = subprocess.run(
                _run_dashboard_command(script, date_str, strict_cutoff=strict_cutoff),
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                cwd=str(LEAGUE_DIRS[league]),
                timeout=300,
            )
            stdout = result.stdout.strip()
            if result.returncode == 0 and stdout:
                # Grab the last JSON line (scripts may print status lines before it)
                for line in reversed(stdout.splitlines()):
                    line = line.strip()
                    if line.startswith("{"):
                        data = json.loads(line)
                        data = _merge_cached_manual_odds(league, date_str, data)
                        cache = _cache_path(league, date_str)
                        cache.write_text(
                            json.dumps(data, ensure_ascii=False, indent=2),
                            encoding="utf-8"
                        )
                        _status[league]["last"]     = datetime.now().isoformat()
                        _status[league]["progress"] = "Done"
                        break
                else:
                    _status[league]["error"] = "No JSON in output"
            else:
                err = (result.stderr or "No output")[-600:]
                _status[league]["error"]    = err
                _status[league]["progress"] = "Error"
        except subprocess.TimeoutExpired:
            _status[league]["error"]    = "Timed out (>5 min)"
            _status[league]["progress"] = "Timeout"
        except Exception as e:
            _status[league]["error"]    = str(e)
            _status[league]["progress"] = "Error"
        finally:
            _status[league]["running"] = False

    threading.Thread(target=run, daemon=True).start()
    return jsonify({"status": "started"})


@app.route("/api/status/<league>")
def status(league):
    return jsonify(_status.get(league, {}))


# ── SP overrides ────────────────────────────────────────────────────────────

SP_FILE = DATA_DIR / "sp_overrides.json"

@app.route("/api/sp", methods=["GET", "POST"])
def sp_overrides():
    if request.method == "GET":
        if SP_FILE.exists():
            return Response(SP_FILE.read_text(encoding="utf-8"), mimetype="application/json")
        return jsonify({})
    data = request.get_json(silent=True) or {}
    SP_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return jsonify({"ok": True})


@app.route("/api/odds", methods=["POST"])
def save_odds():
    """Persist manually-entered odds into the day's cached JSON and recompute EV."""
    date_str  = request.args.get("date", date.today().isoformat())
    payload   = request.get_json(silent=True) or {}
    league    = payload.get("league", "").lower()
    game_idx  = payload.get("game_idx")
    odds_home = payload.get("odds_home")
    odds_away = payload.get("odds_away")

    cache = _cache_path(league, date_str)
    if not cache.exists():
        return jsonify({"error": "No cached data for this date"}), 400

    data = json.loads(cache.read_text(encoding="utf-8"))
    games = data.get("games", [])
    if game_idx is None or not (0 <= game_idx < len(games)):
        return jsonify({"error": "Invalid game_idx"}), 400

    game = games[game_idx]
    if odds_home is not None:
        game["odds_home"] = float(odds_home)
    if odds_away is not None:
        game["odds_away"] = float(odds_away)
    _apply_odds_fields(game, league)

    cache.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return jsonify({"ok": True, "game": game})


# ── Backfill API ─────────────────────────────────────────────────────────────

_backfill_status: dict = {
    "running": False,
    "progress": "",
    "done": 0,
    "total": 0,
    "errors": [],
}
_backfill_lock = threading.Lock()


@app.route("/api/backfill", methods=["POST"])
def start_backfill():
    days   = int(request.args.get("days", 30))
    leagues_raw = request.args.get("leagues", "mlb,kbo,npb,cpbl")
    force  = request.args.get("force", "0") not in ("0", "false", "")

    leagues = [lg.strip().lower() for lg in leagues_raw.split(",") if lg.strip()]

    with _backfill_lock:
        if _backfill_status["running"]:
            return jsonify({"status": "already_running"})
        _backfill_status["running"]  = True
        _backfill_status["progress"] = "Starting…"
        _backfill_status["done"]     = 0
        _backfill_status["total"]    = 0
        _backfill_status["errors"]   = []

    def _run_backfill():
        from datetime import date as _date, timedelta as _timedelta
        today = _date.today()
        date_list = [
            (today - _timedelta(days=i)).isoformat()
            for i in range(days, 0, -1)
        ]

        valid_leagues = [lg for lg in leagues if lg in LEAGUE_DIRS]
        total = len(valid_leagues) * len(date_list)
        _backfill_status["total"] = total
        done = 0

        try:
            for league in valid_leagues:
                league_dir = LEAGUE_DIRS[league]
                script = league_dir / "run_dashboard.py"
                if not script.exists():
                    _backfill_status["errors"].append(f"{league}: run_dashboard.py not found")
                    done += len(date_list)
                    continue

                for date_str in date_list:
                    cache = DATA_DIR / f"{league}_{date_str}.json"
                    _backfill_status["progress"] = f"{league} {date_str}"

                    if cache.exists() and not force:
                        done += 1
                        _backfill_status["done"] = done
                        continue

                    try:
                        strict_cutoff = date_str < _date.today().isoformat()
                        result = subprocess.run(
                            _run_dashboard_command(script, date_str, strict_cutoff=strict_cutoff),
                            capture_output=True,
                            encoding="utf-8",
                            errors="replace",
                            cwd=str(league_dir),
                            timeout=180,
                        )
                        stdout = result.stdout.strip() if result.stdout else ""
                        json_line = None
                        for line in reversed(stdout.splitlines()):
                            line = line.strip()
                            if line.startswith("{"):
                                json_line = line
                                break

                        if result.returncode == 0 and json_line:
                            data = json.loads(json_line)
                            cache.write_text(
                                json.dumps(data, ensure_ascii=False, indent=2),
                                encoding="utf-8",
                            )
                        else:
                            err = f"{league} {date_str}: no JSON (rc={result.returncode})"
                            _backfill_status["errors"].append(err)
                    except subprocess.TimeoutExpired:
                        _backfill_status["errors"].append(f"{league} {date_str}: timeout")
                    except Exception as exc:
                        _backfill_status["errors"].append(f"{league} {date_str}: {exc}")

                    done += 1
                    _backfill_status["done"] = done
        finally:
            _backfill_status["running"]  = False
            _backfill_status["progress"] = "Done"

    threading.Thread(target=_run_backfill, daemon=True).start()
    return jsonify({"status": "started"})


@app.route("/api/backfill_status")
def get_backfill_status():
    return jsonify({
        "running":  _backfill_status["running"],
        "progress": _backfill_status["progress"],
        "done":     _backfill_status["done"],
        "total":    _backfill_status["total"],
        "errors":   _backfill_status["errors"][-20:],
    })


# ── Accuracy API ────────────────────────────────────────────────────────────

# Team name → DB lookup key mappings (reversed from run_dashboard.py TEAM_NAMES)
_CPBL_NAME_TO_CODE = {
    "中信兄弟": "ACN011", "味全龍": "AAA011", "樂天桃猿": "AJL011",
    "富邦悍將": "AEO011", "統一獅": "ADD011", "台鋼雄鷹": "AKP011",
}

_KBO_NAME_TO_CODE = {
    "LG 트윈스": "LG",    "KT 위즈": "KT",
    "삼성 라이온즈": "SS",  "NC 다이노스": "NC",
    "두산 베어스": "OB",    "KIA 타이거즈": "KIA",
    "롯데 자이언츠": "LT",  "SSG 랜더스": "SK",
    "한화 이글스": "HH",   "키움 히어로즈": "WO",
}

_NPB_NAME_TO_CODE = {
    "巨人": "g", "中日": "d", "DeNA": "db", "阪神": "t",
    "広島": "c", "ヤクルト": "s", "SB": "h", "日ハム": "f",
    "オリ": "b", "楽天": "e", "西武": "l", "ロッテ": "m",
}

# In-memory cache: {(league, date_str): {(home, away): [(home_score, away_score), ...]}}
_result_cache: dict = {}
_result_cache_lock = threading.Lock()


def _add_day_result(day_cache: dict, home: str, away: str, hs: float, vs: float) -> None:
    day_cache.setdefault((home, away), []).append((float(hs), float(vs)))


def _clone_day_cache(day_cache: dict) -> dict:
    return {key: list(values) for key, values in day_cache.items()}


def _get_cached_day_cache(league: str, game_date: str) -> dict:
    try:
        day_cache = _build_day_cache(league, game_date)
    except Exception:
        day_cache = {}

    cache_key = (league, game_date)
    with _result_cache_lock:
        _result_cache[cache_key] = day_cache

    return day_cache


def _build_day_cache(league: str, game_date: str) -> dict:
    """Return dict keyed by (home_name, away_name) -> list[(home_score, away_score)]."""
    result = {}
    league = league.lower()

    if league == "cpbl":
        db_path = LEAGUE_DIRS["cpbl"] / "cpbl.sqlite"
        conn = sqlite3.connect(str(db_path))
        try:
            rows = conn.execute(
                "SELECT home_team_code, visiting_team_code, home_score, visiting_score "
                "FROM team_game_results "
                "WHERE date(game_date) = ? AND home_score IS NOT NULL",
                (game_date,)
            ).fetchall()
            # Reverse map: code → name
            code_to_name = {v: k for k, v in _CPBL_NAME_TO_CODE.items()}
            # CPBL codes are like "ADD011"; strip trailing digits for lookup
            for home_code, vis_code, hs, vs in rows:
                home_base = home_code.rstrip("0123456789") if home_code else home_code
                vis_base  = vis_code.rstrip("0123456789")  if vis_code  else vis_code
                home_name = code_to_name.get(home_code) or code_to_name.get(home_base + "011") or home_code
                vis_name  = code_to_name.get(vis_code)  or code_to_name.get(vis_base  + "011") or vis_code
                _add_day_result(result, home_name, vis_name, hs, vs)
        finally:
            conn.close()

    elif league == "mlb":
        db_path = LEAGUE_DIRS["mlb"] / "mlb.sqlite"
        conn = sqlite3.connect(str(db_path))
        try:
            rows = conn.execute(
                "SELECT home_team_name, vis_team_name, home_score, vis_score "
                "FROM team_game_results "
                "WHERE game_date = ? AND home_score IS NOT NULL",
                (game_date,)
            ).fetchall()
            for home_name, vis_name, hs, vs in rows:
                _add_day_result(result, home_name, vis_name, hs, vs)
                if home_name == "Oakland Athletics":
                    _add_day_result(result, "Athletics", vis_name, hs, vs)
                if vis_name == "Oakland Athletics":
                    _add_day_result(result, home_name, "Athletics", hs, vs)
        finally:
            conn.close()

    elif league == "kbo":
        db_path = LEAGUE_DIRS["kbo"] / "kbo.sqlite"
        conn = sqlite3.connect(str(db_path))
        try:
            rows = conn.execute(
                "SELECT home_code, away_code, home_score, away_score "
                "FROM team_game_results "
                "WHERE game_date = ? AND home_score IS NOT NULL",
                (game_date,)
            ).fetchall()
            code_to_name = {v: k for k, v in _KBO_NAME_TO_CODE.items()}
            # HT is legacy code for KIA Tigers (Haitai era)
            code_to_name["HT"] = code_to_name.get("KIA", "KIA 타이거즈")
            for home_code, away_code, hs, vs in rows:
                home_name = code_to_name.get(home_code, home_code)
                away_name = code_to_name.get(away_code, away_code)
                _add_day_result(result, home_name, away_name, hs, vs)
        finally:
            conn.close()

    elif league == "npb":
        db_path = LEAGUE_DIRS["npb"] / "npb.sqlite"
        conn = sqlite3.connect(str(db_path))
        try:
            rows = conn.execute(
                "SELECT home_code, away_code, home_score, away_score "
                "FROM team_game_results "
                "WHERE game_date = ? AND home_score IS NOT NULL",
                (game_date,)
            ).fetchall()
            code_to_name = {v: k for k, v in _NPB_NAME_TO_CODE.items()}
            for home_code, away_code, hs, vs in rows:
                home_name = code_to_name.get(home_code.lower() if home_code else "", home_code)
                away_name = code_to_name.get(away_code.lower() if away_code else "", away_code)
                _add_day_result(result, home_name, away_name, hs, vs)
        finally:
            conn.close()

    return result


def _build_game_counts(games: list[dict]) -> dict:
    counts = {}
    for game in games:
        key = (game.get("home", ""), game.get("away", ""))
        counts[key] = counts.get(key, 0) + 1
    return counts


def _count_cache_matches(day_cache: dict, game_counts: dict) -> int:
    total = 0
    for key, count in game_counts.items():
        total += min(count, len(day_cache.get(key, [])))
    return total


def _select_result_cache_for_games(league: str, game_date: str, games: list[dict]) -> tuple[dict, str]:
    if league != "mlb":
        return _clone_day_cache(_get_cached_day_cache(league, game_date)), game_date

    game_counts = _build_game_counts(games)
    requested_date = date.fromisoformat(game_date)
    candidate_dates = [
        requested_date.isoformat(),
        (requested_date + timedelta(days=1)).isoformat(),
        (requested_date - timedelta(days=1)).isoformat(),
    ]

    best_date = game_date
    best_cache = {}
    best_score = -1
    for candidate_date in candidate_dates:
        candidate_cache = _get_cached_day_cache(league, candidate_date)
        score = _count_cache_matches(candidate_cache, game_counts)
        if score > best_score:
            best_date = candidate_date
            best_cache = candidate_cache
            best_score = score

    return _clone_day_cache(best_cache), best_date


def _pop_game_result(day_cache: dict, home: str, away: str):
    scores = day_cache.get((home, away))
    if not scores:
        return None
    return scores.pop(0)


def _compute_summary(days_data: list, window: int) -> dict:
    """Compute accuracy stats over the last `window` days from days_data list."""
    recent = days_data[-window:] if len(days_data) >= window else days_data
    totals = {"all": [0, 0], "high": [0, 0], "med": [0, 0], "low": [0, 0]}
    for d in recent:
        for conf in ("all", "high", "med", "low"):
            t = d[conf]["total"]
            c = d[conf]["correct"]
            totals[conf][0] += t
            totals[conf][1] += c
    out = {}
    for conf, (t, c) in totals.items():
        pct = round(c / t, 4) if t > 0 else None
        out[conf] = {"total": t, "correct": c, "pct": pct}
    return out


@app.route("/api/accuracy/<league>")
def get_accuracy(league):
    league_lc = league.lower()
    if league_lc not in LEAGUE_DIRS:
        return jsonify({"error": "Unknown league"}), 400

    days_param = request.args.get("days", "60")
    try:
        max_days = int(days_param)
    except ValueError:
        max_days = 60

    today_str = date.today().isoformat()

    # Scan all matching cache files
    pattern = re.compile(rf"^{re.escape(league_lc)}_(\d{{4}}-\d{{2}}-\d{{2}})\.json$")
    all_files = []
    for f in DATA_DIR.iterdir():
        m = pattern.match(f.name)
        if m:
            d_str = m.group(1)
            if d_str < today_str:   # skip today and future
                all_files.append((d_str, f))

    # Sort descending, take most recent `max_days`
    all_files.sort(key=lambda x: x[0], reverse=True)
    all_files = all_files[:max_days]
    all_files.sort(key=lambda x: x[0])  # back to ascending for output

    days_out = []
    db_errors = []

    for d_str, f in all_files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue

        games = data.get("games", [])
        if not games:
            continue

        try:
            day_result_cache, result_date = _select_result_cache_for_games(league_lc, d_str, games)
        except Exception as e:
            db_errors.append(str(e))
            continue

        counts = {
            "all":  {"total": 0, "correct": 0},
            "high": {"total": 0, "correct": 0},
            "med":  {"total": 0, "correct": 0},
            "low":  {"total": 0, "correct": 0},
        }

        for g in games:
            prob_home = g.get("prob_home")
            conf = (g.get("confidence") or "LOW").upper()
            home = g.get("home", "")
            away = g.get("away", "")

            if prob_home is None:
                continue

            actual = _pop_game_result(day_result_cache, home, away)
            if actual is None:
                continue

            hs, vs = actual
            if hs == vs:    # tie → skip
                continue

            predicted_home_win = prob_home > 0.5
            actual_home_win = hs > vs
            correct = int(predicted_home_win == actual_home_win)

            counts["all"]["total"]   += 1
            counts["all"]["correct"] += correct

            conf_key = conf.lower() if conf.lower() in ("high", "med", "low") else "low"
            counts[conf_key]["total"]   += 1
            counts[conf_key]["correct"] += correct

        # Only include days that had at least one resolved game
        if counts["all"]["total"] > 0:
            days_out.append({
                "date": d_str,
                "result_date": result_date,
                **counts,
            })

    # Build summary windows
    summary = {
        "7d":  _compute_summary(days_out, 7),
        "14d": _compute_summary(days_out, 14),
        "30d": _compute_summary(days_out, 30),
    }

    resp: dict = {
        "league": league.upper(),
        "days": days_out,
        "summary": summary,
    }
    if db_errors:
        resp["db_errors"] = db_errors[:5]   # cap at 5 for brevity

    return jsonify(resp)


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MoneyBall Dashboard Server")
    parser.add_argument("--port", type=int, default=5555)
    args = parser.parse_args()

    try:
        import flask  # noqa: F401
    except ImportError:
        print("Flask not found. Install: pip install flask")
        sys.exit(1)

    print(f"\n{'='*50}")
    print(f"  MoneyBall Dashboard → http://localhost:{args.port}")
    print(f"{'='*50}\n")
    app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)
