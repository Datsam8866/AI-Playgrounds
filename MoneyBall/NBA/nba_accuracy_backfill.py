# -*- coding: utf-8 -*-
"""
Fast NBA accuracy backfill: reads pre-computed features from DB,
skips NBA API entirely. Generates data/nba_YYYY-MM-DD.json files.
"""
from __future__ import annotations

import json
import sqlite3
import sys
import warnings
from datetime import date, datetime
from pathlib import Path

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR  = Path(__file__).resolve().parent
DB_PATH   = BASE_DIR / "nba.sqlite"
DATA_DIR  = BASE_DIR.parent / "dashboard" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(BASE_DIR))


def _conf_label(conf: float) -> str:
    if conf > 0.65:
        return "HIGH"
    if conf > 0.55:
        return "MED"
    return "LOW"


def backfill_regular(conn, since: str, force: bool = False) -> int:
    from train_nba_model import (
        FEATURES, MIN_CALIB_ROWS,
        apply_calibrator, fit_calibrator, fit_xgb, predict_probs,
        rolling_train_years, target_vector,
    )

    dates = [r[0] for r in conn.execute(
        "SELECT DISTINCT game_date FROM game_features "
        "WHERE game_date >= ? AND game_date < ? AND home_win IS NOT NULL "
        "ORDER BY game_date",
        (since, date.today().isoformat())
    )]

    written = 0
    for gdate in dates:
        out_path = DATA_DIR / f"nba_{gdate}.json"
        if out_path.exists() and not force:
            continue

        games = [dict(r) for r in conn.execute(
            f"SELECT game_id, game_date, home_team_abbr, vis_team_abbr, home_win,"
            f" {', '.join(FEATURES)}"
            f" FROM game_features WHERE game_date = ? AND home_win IS NOT NULL",
            (gdate,)
        )]
        if not games:
            continue

        sy = int(conn.execute(
            "SELECT season_year FROM game_features WHERE game_date = ? LIMIT 1", (gdate,)
        ).fetchone()[0])
        train_years = rolling_train_years(sy + 1)
        placeholders = ",".join("?" * len(train_years))
        train_rows = [dict(r) for r in conn.execute(
            f"SELECT season_year, game_id, game_date, home_team_abbr, vis_team_abbr, home_win,"
            f" {', '.join(FEATURES)}"
            f" FROM game_features"
            f" WHERE season_year IN ({placeholders}) AND game_date < ? AND home_win IS NOT NULL"
            f" ORDER BY season_year, game_date, game_id",
            (*train_years, gdate)
        )]
        if not train_rows:
            continue

        model, medians = fit_xgb(train_rows, FEATURES)
        latest = max(int(r["season_year"]) for r in train_rows)
        calib   = [r for r in train_rows if int(r["season_year"]) == latest]
        pretrain = [r for r in train_rows if int(r["season_year"]) < latest]
        calibrator = None
        if latest == sy and len(calib) >= MIN_CALIB_ROWS and pretrain:
            cm, cmd = fit_xgb(pretrain, FEATURES)
            cp = predict_probs(cm, cmd, calib, FEATURES)
            calibrator = fit_calibrator(cp.tolist(), target_vector(calib).tolist())
        elif latest < sy and len(calib) >= MIN_CALIB_ROWS:
            fp = predict_probs(model, medians, calib, FEATURES)
            calibrator = fit_calibrator(fp.tolist(), target_vector(calib).tolist())

        raw_probs = predict_probs(model, medians, games, FEATURES)
        game_list = []
        for g, raw in zip(games, raw_probs):
            cal  = apply_calibrator(calibrator, float(raw))
            conf = cal if cal >= 0.5 else 1.0 - cal
            game_list.append({
                "id":        g["game_id"],
                "home":      g["home_team_abbr"],
                "away":      g["vis_team_abbr"],
                "prob_home": round(cal, 4),
                "confidence": _conf_label(conf),
                "route":     "regular",
                "has_odds":  False,
            })

        payload = {
            "league": "NBA", "date": gdate,
            "last_updated": datetime.now().isoformat(),
            "mode": "regular", "games": game_list,
            "error": None, "has_live_odds": False, "odds_fetched": False,
        }
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        written += 1

    return written


def backfill_playoffs(conn, season_year: int = 2025, force: bool = False) -> int:
    from train_nba_playoff_model import (
        FEATURES, MIN_CALIB_ROWS, MIN_TRAIN_ROWS,
        apply_calibrator, can_fit, fit_calibrator, fit_xgb,
        predict_probs, rolling_train_years, target_vector,
    )

    ROUND_NAMES = {1: "First Round", 2: "Second Round", 3: "Conf Finals", 4: "Finals"}

    dates = [r[0] for r in conn.execute(
        "SELECT DISTINCT game_date FROM playoff_game_features "
        "WHERE season_year = ? AND home_win IS NOT NULL "
        "ORDER BY game_date",
        (season_year,)
    )]

    # Pre-load train rows (all prior seasons)
    train_years = rolling_train_years(season_year + 1)
    placeholders = ",".join("?" * len(train_years))

    written = 0
    for gdate in dates:
        out_path = DATA_DIR / f"nba_{gdate}.json"
        if out_path.exists() and not force:
            # Don't overwrite if already has playoff mode (RS may have written it)
            existing = json.loads(out_path.read_text(encoding="utf-8"))
            if existing.get("mode") == "playoffs":
                continue

        games = [dict(r) for r in conn.execute(
            f"SELECT game_id, game_date, home_team_abbr, vis_team_abbr, home_win,"
            f" playoff_round, game_in_series, home_series_wins, vis_series_wins,"
            f" {', '.join(FEATURES)}"
            f" FROM playoff_game_features"
            f" WHERE game_date = ? AND season_year = ? AND home_win IS NOT NULL",
            (gdate, season_year)
        )]
        if not games:
            continue

        train_rows = [dict(r) for r in conn.execute(
            f"SELECT season_year, game_id, game_date, home_team_abbr, vis_team_abbr, home_win,"
            f" {', '.join(FEATURES)}"
            f" FROM playoff_game_features"
            f" WHERE season_year IN ({placeholders}) AND game_date < ? AND home_win IS NOT NULL"
            f" ORDER BY season_year, game_date, game_id",
            (*train_years, gdate)
        )]
        if not train_rows or not can_fit(train_rows):
            continue

        model, medians = fit_xgb(train_rows, FEATURES)
        latest = max(int(r["season_year"]) for r in train_rows)
        calib   = [r for r in train_rows if int(r["season_year"]) == latest]
        pretrain = [r for r in train_rows if int(r["season_year"]) < latest]
        calibrator = None
        if latest == season_year and len(calib) >= MIN_CALIB_ROWS and pretrain:
            cm, cmd = fit_xgb(pretrain, FEATURES)
            cp = predict_probs(cm, cmd, calib, FEATURES)
            calibrator = fit_calibrator(cp.tolist(), target_vector(calib).tolist())
        elif latest < season_year and len(calib) >= MIN_CALIB_ROWS:
            fp = predict_probs(model, medians, calib, FEATURES)
            calibrator = fit_calibrator(fp.tolist(), target_vector(calib).tolist())

        raw_probs = predict_probs(model, medians, games, FEATURES)
        game_list = []
        for g, raw in zip(games, raw_probs):
            cal  = apply_calibrator(calibrator, float(raw))
            conf = cal if cal >= 0.5 else 1.0 - cal
            rnd  = int(g["playoff_round"])
            gin  = int(g["game_in_series"])
            hw   = int(g["home_series_wins"])
            vw   = int(g["vis_series_wins"])
            series_label = f"{ROUND_NAMES.get(rnd, f'R{rnd}')} G{gin} ({hw}-{vw})"
            game_list.append({
                "id":        g["game_id"],
                "home":      g["home_team_abbr"],
                "away":      g["vis_team_abbr"],
                "prob_home": round(cal, 4),
                "confidence": _conf_label(conf),
                "route":     f"playoff_r{rnd}",
                "has_odds":  False,
                "home_sp":   series_label,
                "away_sp":   "",
                "playoff_round":    rnd,
                "game_in_series":   gin,
                "home_series_wins": hw,
                "vis_series_wins":  vw,
                "series_label":     series_label,
            })

        payload = {
            "league": "NBA", "date": gdate,
            "last_updated": datetime.now().isoformat(),
            "mode": "playoffs", "games": game_list,
            "error": None, "has_live_odds": False, "odds_fetched": False,
        }
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        written += 1

    return written


if __name__ == "__main__":
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    print("Backfilling regular season (last 60 days)…")
    n_rs = backfill_regular(conn, since="2026-02-16")
    print(f"  Written: {n_rs} files")

    print("Backfilling 2025-26 playoffs…")
    n_po = backfill_playoffs(conn, season_year=2025)
    print(f"  Written: {n_po} files")

    conn.close()
    print(f"Done. Total {n_rs + n_po} new files.")
