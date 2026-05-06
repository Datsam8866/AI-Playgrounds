"""
predict_today_kbo.py

KBO daily prediction tool.
Reads pre-computed features from game_features (sr_id=0), trains on all
seasons < target_year using evaluate_kbo_predictions_regime, and predicts
all games on the target date.

Usage:
    python predict_today_kbo.py                          # today
    python predict_today_kbo.py --date 2026-04-23        # specific date
    python predict_today_kbo.py --date 2026-04-22 --verify  # verify past date

Requirements:
    - game_features must have rows for the target date.
      If missing, run build_kbo_game_features.py first.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import io
from datetime import date, datetime
from pathlib import Path

import evaluate_kbo_predictions_regime as ev

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── KBO team name lookup ──────────────────────────────────────────────────────
TEAM_NAMES: dict[str, str] = {
    "LG": "LG 트윈스",
    "KT": "KT 위즈",
    "SS": "삼성 라이온즈",
    "NC": "NC 다이노스",
    "OB": "두산 베어스",
    "KIA": "KIA 타이거즈",
    "LT": "롯데 자이언츠",
    "SK": "SSG 랜더스",   # legacy code (SK → SSG 2021)
    "HH": "한화 이글스",
    "HT": "KIA 타이거즈",  # legacy code (HT = KIA)
    "WO": "키움 히어로즈",
}


def team_display(code: str) -> str:
    return TEAM_NAMES.get(code, code)


# ── Confidence label ──────────────────────────────────────────────────────────

def confidence_label(p_winner: float) -> str:
    """p_winner is the probability of the predicted side (always >= 0.5)."""
    if p_winner >= 0.60:
        return "HIGH"
    if p_winner >= 0.55:
        return "MED"
    return "LOW"


# ── Data loading ──────────────────────────────────────────────────────────────

def load_target_rows(conn: sqlite3.Connection, target: date) -> list[dict]:
    """
    Return game_features rows for the target date (sr_id=0).
    Derived flags (early_flag, sp_available) are added by ev.load_rows(),
    so we replicate that derivation here for the target rows.
    """
    date_str = target.isoformat()
    rows = conn.execute("""
        SELECT *
        FROM game_features
        WHERE sr_id = 0
          AND game_date = ?
        ORDER BY game_id
    """, (date_str,)).fetchall()

    cols = [d[1] for d in conn.execute("PRAGMA table_info(game_features)").fetchall()]
    result = []
    for r in rows:
        d = dict(zip(cols, r))
        d["early_flag"]   = int(
            d.get("home_season_games_before", 0) < ev.TEAM_BURN_IN
            or d.get("away_season_games_before", 0) < ev.TEAM_BURN_IN
        )
        d["sp_available"] = int(d.get("diff_sp_era") is not None)
        result.append(d)
    return result


def load_actual_scores(
    conn: sqlite3.Connection, target: date
) -> dict[str, tuple[int | None, int | None]]:
    """
    Return {game_id: (away_score, home_score)} for completed games on target date.
    game_state = 3 means completed in KBO database.
    """
    date_str = target.isoformat()
    rows = conn.execute("""
        SELECT game_id, away_score, home_score
        FROM team_game_results
        WHERE game_date = ?
          AND away_score IS NOT NULL
          AND home_score IS NOT NULL
    """, (date_str,)).fetchall()
    return {r[0]: (r[1], r[2]) for r in rows}


# ── Prediction ────────────────────────────────────────────────────────────────

def run_predictions(
    all_rows: list[dict],
    target_rows: list[dict],
    target_year: int,
) -> list[dict]:
    """
    Train on all_rows with season_year < target_year, predict target_rows.
    Returns list of result dicts.
    """
    train = [r for r in all_rows if ev.TRAIN_START_YEAR <= r["season_year"] < target_year]
    if not train:
        print(f"[ERROR] No training data found for seasons < {target_year}.")
        sys.exit(1)

    models = ev.train_models(train)

    results = []
    for row in target_rows:
        prob_home, model_used = ev.predict_one(models, row)
        pred_home = int(prob_home >= 0.5)
        p_winner = prob_home if pred_home else (1.0 - prob_home)
        results.append({
            "game_id":    row["game_id"],
            "game_date":  row["game_date"],
            "away_code":  row["away_code"],
            "home_code":  row["home_code"],
            "prob_home":  prob_home,
            "pred_home":  pred_home,
            "p_winner":   p_winner,
            "confidence": confidence_label(p_winner),
            "model":      model_used,
            "early_flag": row["early_flag"],
            "sp_available": row["sp_available"],
            "diff_elo":   row.get("diff_elo", 0.0),
            "diff_w5_win_pct": row.get("diff_w5_win_pct", 0.0),
            "home_season_games_before": row.get("home_season_games_before", 0),
            "away_season_games_before": row.get("away_season_games_before", 0),
        })
    return results


# ── Console output ────────────────────────────────────────────────────────────

def print_predictions(
    target: date,
    results: list[dict],
    actuals: dict[str, tuple] | None = None,
) -> tuple[int, int]:
    """Print to console. Returns (correct, total) if verify mode, else (0, 0)."""
    sep = "=" * 68
    print(f"\n{sep}")
    print(f"KBO PREDICTIONS — {target}")
    print(sep)

    correct = total = 0
    for r in results:
        away = team_display(r["away_code"])
        home = team_display(r["home_code"])
        prob = r["prob_home"]
        pred_side = "Home" if r["pred_home"] else "Away"
        pred_team = home if r["pred_home"] else away
        conf = r["confidence"]
        model = r["model"]
        regime = " [EARLY]" if r["early_flag"] else (" [SP]" if r["sp_available"] else "")

        print(f"\n{r['away_code']} (Away)  vs  {r['home_code']} (Home)")
        print(f"  {away}  @  {home}")
        print(f"  Prob(Home): {prob:.3f}  |  Pred: {pred_side} ({pred_team})")
        print(f"  Confidence: {conf}  |  Model: {model}{regime}")
        print(f"  diff_elo={r['diff_elo']:+.1f}  diff_w5={r['diff_w5_win_pct']:+.3f}"
              f"  h_games={r['home_season_games_before']}  a_games={r['away_season_games_before']}")

        if actuals is not None:
            score = actuals.get(r["game_id"])
            if score and score[0] is not None and score[1] is not None:
                away_s, home_s = score
                actual_home_win = int(home_s > away_s)
                hit = int(r["pred_home"] == actual_home_win)
                actual_side = "Home" if actual_home_win else "Away"
                correct += hit; total += 1
                print(f"  Result: {r['away_code']} {away_s}-{home_s} {r['home_code']}"
                      f"  → {'HIT ✓' if hit else 'MISS ✗'}  (actual: {actual_side})")
            else:
                print(f"  Result: not yet available")

    if actuals is not None and total > 0:
        print(f"\n{'-'*40}")
        print(f"Verification: {correct}/{total} = {correct/total:.1%}")

    print(f"\n{sep}\n")
    return correct, total


# ── Markdown report ───────────────────────────────────────────────────────────

def write_markdown(
    target: date,
    results: list[dict],
    actuals: dict[str, tuple] | None,
    train_rows_count: int,
) -> Path:
    date_str = target.strftime("%Y-%m-%d")
    lines = [
        f"# KBO Predictions — {date_str}",
        "",
        f"- Training: seasons {ev.TRAIN_START_YEAR}–{target.year - 1}  ({train_rows_count} games)",
        f"- Regime: early model when either team < {ev.TEAM_BURN_IN} games; "
        f"primary (with SP) / fallback (no SP) otherwise",
        f"- Confidence: p≥0.60 = HIGH, p≥0.55 = MED, else LOW",
        "",
    ]

    if not results:
        lines += [f"No games found in game_features for {date_str}.", ""]
        lines += [
            "> **Note**: If today's games have not yet been processed, "
            "run `build_kbo_game_features.py` first.",
            "",
        ]
    else:
        verify_cols = " Score | Result |" if actuals is not None else ""
        lines += [
            "## Predictions",
            "",
            f"| Game | Away | Home | Prob(Home) | Pred | P(winner) | Conf | Model |{verify_cols}",
            f"| --- | --- | --- | ---: | --- | ---: | --- | --- |{'--- | --- |' if actuals is not None else ''}",
        ]

        correct = total = 0
        for r in results:
            away = team_display(r["away_code"])
            home = team_display(r["home_code"])
            pred = "Home" if r["pred_home"] else "Away"
            regime_tag = "[E]" if r["early_flag"] else ("[P]" if r["sp_available"] else "[F]")
            model_str = f"{r['model']} {regime_tag}"

            row_line = (
                f"| {r['game_id']} "
                f"| {r['away_code']} {away} "
                f"| {r['home_code']} {home} "
                f"| {r['prob_home']:.3f} "
                f"| {pred} "
                f"| {r['p_winner']:.3f} "
                f"| {r['confidence']} "
                f"| {model_str} |"
            )

            if actuals is not None:
                score = actuals.get(r["game_id"])
                if score and score[0] is not None and score[1] is not None:
                    away_s, home_s = score
                    hit = int(r["pred_home"] == int(home_s > away_s))
                    correct += hit; total += 1
                    row_line += f" {r['away_code']} {away_s}-{home_s} {r['home_code']} | {'HIT' if hit else 'MISS'} |"
                else:
                    row_line += " — | — |"

            lines.append(row_line)

        if actuals is not None and total > 0:
            lines += [
                "",
                f"**Result: {correct}/{total} = {correct/total:.1%}**",
            ]

        lines += [
            "",
            "## Feature Snapshot",
            "",
            "| Game | Away | Home | diff_elo | diff_w5 | h_games | a_games | SP? | Early? |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
        ]
        for r in results:
            sp_flag = "Y" if r["sp_available"] else "N"
            early_flag = "Y" if r["early_flag"] else "N"
            lines.append(
                f"| {r['game_id']} "
                f"| {r['away_code']} "
                f"| {r['home_code']} "
                f"| {r['diff_elo']:+.1f} "
                f"| {r['diff_w5_win_pct']:+.3f} "
                f"| {r['home_season_games_before']} "
                f"| {r['away_season_games_before']} "
                f"| {sp_flag} "
                f"| {early_flag} |"
            )

    lines += [
        "",
        f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
        "",
    ]

    report_path = Path(f"predictions_kbo_{target.strftime('%Y%m%d')}.md")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="KBO daily game predictions"
    )
    parser.add_argument(
        "--date",
        default=date.today().isoformat(),
        help="Target date YYYY-MM-DD (default: today)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Compare predictions against actual results (for completed dates)",
    )
    args = parser.parse_args()

    target = date.fromisoformat(args.date)
    target_year = target.year

    conn = sqlite3.connect(ev.DB_PATH)
    try:
        print(f"Loading game_features (sr_id=0) …")
        all_rows = ev.load_rows(conn)
        print(f"  {len(all_rows)} historical rows loaded")

        print(f"Loading target date rows for {target} …")
        target_rows = load_target_rows(conn, target)

        if not target_rows:
            print(
                f"\n[WARNING] No game_features rows found for {target}.\n"
                f"  Please run build_kbo_game_features.py first to populate features "
                f"for {target}."
            )
            # Still write an empty report so caller knows the date was attempted
            report_path = write_markdown(target, [], None, 0)
            print(f"Empty report written → {report_path}")
            return

        print(f"  {len(target_rows)} games found for {target}")

        # Training count (for report header)
        train_count = sum(
            1 for r in all_rows
            if ev.TRAIN_START_YEAR <= r["season_year"] < target_year
        )
        print(f"  Training on {train_count} games (seasons {ev.TRAIN_START_YEAR}–{target_year-1})")

        # Remove target date from all_rows to avoid data leakage
        # (in case game_features includes today's completed games)
        train_source = [r for r in all_rows if r["game_date"] < target.isoformat()]
        train_count_adj = sum(
            1 for r in train_source
            if ev.TRAIN_START_YEAR <= r["season_year"] < target_year
        )

        print(f"Training models …")
        train = [
            r for r in train_source
            if ev.TRAIN_START_YEAR <= r["season_year"] < target_year
        ]
        if not train:
            print(f"[ERROR] No training data found for seasons < {target_year}.")
            sys.exit(1)
        models = ev.train_models(train)

        # Predict
        results = []
        for row in target_rows:
            prob_home, model_used = ev.predict_one(models, row)
            pred_home = int(prob_home >= 0.5)
            p_winner = prob_home if pred_home else (1.0 - prob_home)
            results.append({
                "game_id":    row["game_id"],
                "game_date":  row["game_date"],
                "away_code":  row["away_code"],
                "home_code":  row["home_code"],
                "prob_home":  prob_home,
                "pred_home":  pred_home,
                "p_winner":   p_winner,
                "confidence": confidence_label(p_winner),
                "model":      model_used,
                "early_flag": row["early_flag"],
                "sp_available": row["sp_available"],
                "diff_elo":   row.get("diff_elo", 0.0),
                "diff_w5_win_pct": row.get("diff_w5_win_pct", 0.0),
                "home_season_games_before": row.get("home_season_games_before", 0),
                "away_season_games_before": row.get("away_season_games_before", 0),
            })

        # Load actuals for verify mode
        actuals = None
        if args.verify:
            actuals = load_actual_scores(conn, target)
            if not actuals:
                print(f"  [INFO] No completed scores found for {target} (verify mode).")

        # Console output
        correct, total = print_predictions(target, results, actuals)

        # Markdown report
        report_path = write_markdown(target, results, actuals, train_count_adj)
        print(f"Report written → {report_path.resolve()}")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
