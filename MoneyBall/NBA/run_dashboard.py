# -*- coding: utf-8 -*-
"""
NBA Dashboard adapter.
Auto-detects regular season vs playoffs by game_id prefix.
Usage: python run_dashboard.py [YYYY-MM-DD]
Outputs single JSON line to stdout.
"""
from __future__ import annotations

import json
import sqlite3
import sys
import time
import warnings
from collections import defaultdict, deque
from datetime import date, datetime
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "nba.sqlite"

ROUND_NAMES = {1: "First Round", 2: "Second Round", 3: "Conf Finals", 4: "Finals"}


def _conf_label(conf: float) -> str:
    if conf > 0.65:
        return "HIGH"
    if conf > 0.55:
        return "MED"
    return "LOW"


def _infer_season(d: date) -> int:
    return d.year if d.month >= 10 else d.year - 1


def _fetch_scheduled(target_date: date, team_lookup: dict) -> list[dict]:
    from nba_api.stats.endpoints import ScoreboardV2
    sb = ScoreboardV2(game_date=target_date.strftime("%Y-%m-%d"), league_id="00")
    time.sleep(1.5)
    frames = sb.get_data_frames()
    if not frames or frames[0].empty:
        return []
    season_year = _infer_season(target_date)
    games = []
    for row in frames[0].to_dict("records"):
        home_id = int(row["HOME_TEAM_ID"])
        vis_id = int(row["VISITOR_TEAM_ID"])
        games.append({
            "game_id": str(row["GAME_ID"]),
            "season_year": season_year,
            "game_date": target_date.isoformat(),
            "home_team_id": home_id,
            "vis_team_id": vis_id,
            "home_team_abbr": team_lookup.get(home_id, str(home_id)),
            "vis_team_abbr": team_lookup.get(vis_id, str(vis_id)),
            "home_win": None,
            "game_status_text": str(row.get("GAME_STATUS_TEXT") or ""),
        })
    return games


def _run_regular(conn, scheduled, target_date: date) -> list[dict]:
    from build_nba_game_features import (
        ELO_BASE, WINDOW,
        is_neutral_site, load_games, load_player_season_avg,
        regress_elo, to_feature_row, update_team_state,
    )
    from train_nba_model import (
        FEATURES, MIN_CALIB_ROWS,
        apply_calibrator, fit_calibrator, fit_xgb, predict_probs,
        rolling_train_years, target_vector,
    )

    season_year = _infer_season(target_date)

    injury_map = {}
    try:
        from nba_injury_scraper import fetch_and_store_injuries
        injury_map = fetch_and_store_injuries(conn, date_str=target_date.isoformat())
    except Exception:
        pass

    player_season_avg = load_player_season_avg(conn, season_year)

    all_games = load_games(conn)
    completed = [g for g in all_games if g["game_date"] < target_date.isoformat()]
    team_history = defaultdict(lambda: deque(maxlen=WINDOW))
    elo_by_team = defaultdict(lambda: ELO_BASE)
    streak_by_team: dict = {}
    last_game_dates: dict = {}
    season_games: dict = defaultdict(int)
    current_season = None

    for game in completed:
        sy = int(game["season_year"])
        if sy != current_season:
            if current_season is not None:
                regress_elo(elo_by_team)
            current_season = sy
            streak_by_team = {}
            last_game_dates = {}
        update_team_state(
            game=game,
            team_history=team_history,
            elo_by_team=elo_by_team,
            streak_by_team=streak_by_team,
            last_game_dates=last_game_dates,
            season_games=season_games,
            neutral=is_neutral_site(game),
        )

    feature_rows = [
        to_feature_row(
            game=g,
            team_history=team_history,
            elo_by_team=elo_by_team,
            streak_by_team=streak_by_team,
            last_game_dates=last_game_dates,
            season_games=season_games,
            neutral=is_neutral_site(g),
            injury_map=injury_map,
            player_season_avg=player_season_avg,
        )
        for g in scheduled
    ]

    train_years = rolling_train_years(season_year + 1)
    placeholders = ",".join("?" for _ in train_years)
    train_rows = [
        dict(r) for r in conn.execute(
            f"SELECT season_year, game_id, game_date, home_team_abbr, vis_team_abbr, home_win,"
            f" {', '.join(FEATURES)}"
            f" FROM game_features"
            f" WHERE season_year IN ({placeholders}) AND game_date < ? AND home_win IS NOT NULL"
            f" ORDER BY season_year, game_date, game_id",
            (*train_years, target_date.isoformat()),
        )
    ]
    if not train_rows:
        raise RuntimeError("No training data for regular season model.")

    model, medians = fit_xgb(train_rows, FEATURES)
    latest = max(int(r["season_year"]) for r in train_rows)
    calib = [r for r in train_rows if int(r["season_year"]) == latest]
    pretrain = [r for r in train_rows if int(r["season_year"]) < latest]
    calibrator = None
    if latest == season_year and len(calib) >= MIN_CALIB_ROWS and pretrain:
        cm, cmd = fit_xgb(pretrain, FEATURES)
        cp = predict_probs(cm, cmd, calib, FEATURES)
        calibrator = fit_calibrator(cp.tolist(), target_vector(calib).tolist())
    elif latest < season_year and len(calib) >= MIN_CALIB_ROWS:
        fp = predict_probs(model, medians, calib, FEATURES)
        calibrator = fit_calibrator(fp.tolist(), target_vector(calib).tolist())

    raw_probs = predict_probs(model, medians, feature_rows, FEATURES)
    out = []
    for game, raw in zip(scheduled, raw_probs):
        cal = apply_calibrator(calibrator, float(raw))
        conf = cal if cal >= 0.5 else 1.0 - cal
        out.append({
            "id": game["game_id"],
            "home": game["home_team_abbr"],
            "away": game["vis_team_abbr"],
            "status": game.get("game_status_text", "Scheduled"),
            "prob_home": round(cal, 4),
            "confidence": _conf_label(conf),
            "route": "regular",
            "has_odds": False,
            "home_sp": "",
            "away_sp": "",
        })
    return out


def _run_playoffs(conn, scheduled, target_date: date) -> list[dict]:
    from build_nba_playoff_features import (
        ELO_BASE, ELO_K_PO, ELO_HOME_ADV, FIRST_GAME_REST,
        load_rs_games, load_rs_player_game_map, compute_rs_end_state,
        infer_playoff_round,
    )
    from build_nba_game_features import elo_win_prob
    from train_nba_playoff_model import (
        FEATURES, MIN_CALIB_ROWS, MIN_TRAIN_ROWS,
        apply_calibrator, can_fit, fit_calibrator, fit_xgb,
        predict_probs, rolling_train_years, target_vector,
    )

    season_year = _infer_season(target_date)

    rs_games = load_rs_games(conn)
    rs_player_map = load_rs_player_game_map(conn)
    season_end_states = compute_rs_end_state(rs_games, rs_player_map)
    rs_state = season_end_states.get(season_year, {})

    # Replay completed playoff games to build current Elo + series state
    elo_po: dict = {}
    series_state: dict = {}

    po_rows = conn.execute(
        "SELECT game_id, game_date, home_team_id, vis_team_id, home_win "
        "FROM playoff_game_results "
        "WHERE season_year = ? AND home_win IS NOT NULL "
        "ORDER BY game_date, game_id",
        (season_year,),
    ).fetchall()

    for row in po_rows:
        if str(row[1]) >= target_date.isoformat():
            continue
        game_id = str(row[0])
        home_id = int(row[2])
        vis_id = int(row[3])
        home_win = int(row[4])
        rnd = infer_playoff_round(game_id)
        sk = (season_year, rnd, frozenset({home_id, vis_id}))

        if home_id not in elo_po:
            elo_po[home_id] = rs_state.get(home_id, {}).get("elo", ELO_BASE)
        if vis_id not in elo_po:
            elo_po[vis_id] = rs_state.get(vis_id, {}).get("elo", ELO_BASE)

        if sk not in series_state:
            series_state[sk] = {
                "home_id": home_id, "home_wins": 0, "vis_wins": 0,
                "game_count": 0, "last_date": None, "homecourt_team": home_id,
            }
        ss = series_state[sk]
        ss["game_count"] += 1
        ss["last_date"] = date.fromisoformat(str(row[1]))
        if home_id == ss["home_id"]:
            ss["home_wins"] += home_win
            ss["vis_wins"] += (1 - home_win)
        else:
            ss["vis_wins"] += home_win
            ss["home_wins"] += (1 - home_win)

        exp_h = elo_win_prob(elo_po[home_id], elo_po[vis_id])
        elo_po[home_id] += ELO_K_PO * (home_win - exp_h)
        elo_po[vis_id] += ELO_K_PO * ((1 - home_win) - (1.0 - exp_h))

    # Build feature rows for today's games
    feature_rows = []
    game_meta = []

    for game in scheduled:
        home_id = int(game["home_team_id"])
        vis_id = int(game["vis_team_id"])
        game_id = str(game["game_id"])
        rnd = infer_playoff_round(game_id)
        sk = (season_year, rnd, frozenset({home_id, vis_id}))

        if home_id not in elo_po:
            elo_po[home_id] = rs_state.get(home_id, {}).get("elo", ELO_BASE)
        if vis_id not in elo_po:
            elo_po[vis_id] = rs_state.get(vis_id, {}).get("elo", ELO_BASE)

        if sk not in series_state:
            series_state[sk] = {
                "home_id": home_id, "home_wins": 0, "vis_wins": 0,
                "game_count": 0, "last_date": None, "homecourt_team": home_id,
            }
        ss = series_state[sk]
        h_wins = ss["home_wins"] if home_id == ss["home_id"] else ss["vis_wins"]
        v_wins = ss["vis_wins"] if home_id == ss["home_id"] else ss["home_wins"]
        gin = ss["game_count"] + 1
        last_d = ss.get("last_date")
        rest = (target_date - last_d).days if last_d else FIRST_GAME_REST
        hc = int(ss["homecourt_team"] == home_id)

        hr = rs_state.get(home_id, {})
        vr = rs_state.get(vis_id, {})
        elo_rs_h = hr.get("elo", ELO_BASE)
        elo_rs_v = vr.get("elo", ELO_BASE)
        elo_po_h = elo_po.get(home_id, ELO_BASE)
        elo_po_v = elo_po.get(vis_id, ELO_BASE)
        diff_elo_rs = elo_rs_h - elo_rs_v
        diff_elo_po = elo_po_h - elo_po_v
        elo_win_prob_po = 1.0 / (1 + 10 ** ((elo_po_v - (elo_po_h + ELO_HOME_ADV)) / 400))

        feature_rows.append({
            "diff_elo_rs": diff_elo_rs,
            "diff_rs_net_rtg": (hr.get("net_rtg") or 0) - (vr.get("net_rtg") or 0),
            "diff_rs_pyth_wp": (hr.get("pyth_wp") or 0.5) - (vr.get("pyth_wp") or 0.5),
            "diff_rs_lineup_pts": (hr.get("lineup_pts") or 0) - (vr.get("lineup_pts") or 0),
            "diff_elo_po": diff_elo_po,
            "elo_win_prob_po": elo_win_prob_po,
            "diff_elo_change_po": diff_elo_po - diff_elo_rs,
            "game_in_series": gin,
            "home_series_wins": h_wins,
            "vis_series_wins": v_wins,
            "series_score_diff": h_wins - v_wins,
            "is_elimination": int(max(h_wins, v_wins) == 3),
            "playoff_round": rnd,
            "series_rest_days": rest,
            "home_has_homecourt": hc,
            "home_win": None,
        })

        rnd_name = ROUND_NAMES.get(rnd, f"R{rnd}")
        series_label = f"{rnd_name} G{gin} ({h_wins}-{v_wins})"
        game_meta.append({
            "game_id": game_id,
            "home": game["home_team_abbr"],
            "away": game["vis_team_abbr"],
            "status": game.get("game_status_text", "Scheduled"),
            "playoff_round": rnd,
            "game_in_series": gin,
            "home_series_wins": h_wins,
            "vis_series_wins": v_wins,
            "series_label": series_label,
        })

    # Train playoff model
    train_years = rolling_train_years(season_year + 1)
    placeholders = ",".join("?" for _ in train_years)
    train_rows = [
        dict(r) for r in conn.execute(
            f"SELECT season_year, game_id, game_date, home_team_abbr, vis_team_abbr, home_win,"
            f" {', '.join(FEATURES)}"
            f" FROM playoff_game_features"
            f" WHERE season_year IN ({placeholders}) AND home_win IS NOT NULL"
            f" ORDER BY season_year, game_date, game_id",
            train_years,
        )
    ]
    if not train_rows or not can_fit(train_rows):
        raise RuntimeError(f"Insufficient playoff training data (n={len(train_rows)})")

    model, medians = fit_xgb(train_rows, FEATURES)
    latest = max(int(r["season_year"]) for r in train_rows)
    calib = [r for r in train_rows if int(r["season_year"]) == latest]
    pretrain = [r for r in train_rows if int(r["season_year"]) < latest]
    calibrator = None
    if latest == season_year and len(calib) >= MIN_CALIB_ROWS and pretrain:
        cm, cmd = fit_xgb(pretrain, FEATURES)
        cp = predict_probs(cm, cmd, calib, FEATURES)
        calibrator = fit_calibrator(cp.tolist(), target_vector(calib).tolist())
    elif latest < season_year and len(calib) >= MIN_CALIB_ROWS:
        fp = predict_probs(model, medians, calib, FEATURES)
        calibrator = fit_calibrator(fp.tolist(), target_vector(calib).tolist())

    raw_probs = predict_probs(model, medians, feature_rows, FEATURES)
    out = []
    for meta, raw in zip(game_meta, raw_probs):
        cal = apply_calibrator(calibrator, float(raw))
        conf = cal if cal >= 0.5 else 1.0 - cal
        out.append({
            "id": meta["game_id"],
            "home": meta["home"],
            "away": meta["away"],
            "status": meta["status"],
            "prob_home": round(cal, 4),
            "confidence": _conf_label(conf),
            "route": f"playoff_r{meta['playoff_round']}",
            "has_odds": False,
            "home_sp": meta["series_label"],
            "away_sp": "",
            "playoff_round": meta["playoff_round"],
            "game_in_series": meta["game_in_series"],
            "home_series_wins": meta["home_series_wins"],
            "vis_series_wins": meta["vis_series_wins"],
            "series_label": meta["series_label"],
        })
    return out


def main():
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    date_str = argv[0] if argv else date.today().isoformat()
    target_date = date.fromisoformat(date_str)

    games_out = []
    error_msg = None
    mode = "regular"

    try:
        from nba_api.stats.static import teams as nba_teams
        team_lookup = {int(t["id"]): t["abbreviation"] for t in nba_teams.get_teams()}
        scheduled = _fetch_scheduled(target_date, team_lookup)

        if not scheduled:
            raise RuntimeError("No games found for this date.")

        is_playoff = any(g["game_id"].startswith("004") for g in scheduled)
        mode = "playoffs" if is_playoff else "regular"

        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        try:
            if is_playoff:
                games_out = _run_playoffs(conn, scheduled, target_date)
            else:
                games_out = _run_regular(conn, scheduled, target_date)
        finally:
            conn.close()

    except Exception as e:
        error_msg = str(e)

    result = {
        "league": "NBA",
        "date": date_str,
        "last_updated": datetime.now().isoformat(),
        "mode": mode,
        "games": games_out,
        "error": error_msg,
        "has_live_odds": False,
        "odds_fetched": False,
    }
    print(json.dumps(result, ensure_ascii=False))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
