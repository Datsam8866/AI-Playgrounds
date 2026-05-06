"""
track_high_confidence_predictions.py

Track verified high-confidence game predictions over a date range.

Confidence is defined on the predicted side:
  - if model picks home, confidence = P(home)
  - if model picks visitor, confidence = 1 - P(home)

Default threshold follows the current project convention: 0.600.
"""

import argparse
import csv
import sqlite3
from datetime import date, datetime
from pathlib import Path

from predict_today import DB_PATH, TEAM_NAMES, load_data, train_and_predict


CSV_PATH = Path("high_confidence_tracking.csv")
REPORT_PATH = Path("high_confidence_tracking.md")
TRACKING_TABLE = "prediction_tracking"


def team_name(code: str) -> str:
    return TEAM_NAMES.get(code, code)


def side_label(is_home: bool) -> str:
    return "Home" if is_home else "Vis"


def confidence_level(confidence: float) -> str:
    if confidence >= 0.600:
        return "HIGH"
    if confidence >= 0.550:
        return "MED"
    return "LOW"


def ensure_tracking_table(conn):
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {TRACKING_TABLE} (
            prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
            season_year INTEGER NOT NULL,
            kind_code TEXT NOT NULL,
            game_sno INTEGER NOT NULL,
            game_date TEXT NOT NULL,
            home_team_code TEXT NOT NULL,
            visiting_team_code TEXT NOT NULL,
            home_team_name TEXT NOT NULL,
            visiting_team_name TEXT NOT NULL,
            home_sp_name TEXT,
            visiting_sp_name TEXT,
            home_sp_acnt TEXT,
            visiting_sp_acnt TEXT,
            prob_home_win REAL NOT NULL,
            predicted_side TEXT NOT NULL,
            predicted_team_code TEXT NOT NULL,
            predicted_team_name TEXT NOT NULL,
            confidence REAL NOT NULL,
            confidence_level TEXT NOT NULL,
            is_high_confidence INTEGER NOT NULL,
            threshold REAL NOT NULL,
            model_used TEXT NOT NULL,
            sp_available INTEGER NOT NULL,
            early_season INTEGER NOT NULL,
            actual_side TEXT NOT NULL,
            actual_team_code TEXT NOT NULL,
            actual_team_name TEXT NOT NULL,
            home_score REAL NOT NULL,
            visiting_score REAL NOT NULL,
            is_correct INTEGER NOT NULL,
            verified_at TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE (season_year, kind_code, game_sno)
        )
        """
    )
    conn.execute(
        f"""
        CREATE INDEX IF NOT EXISTS idx_{TRACKING_TABLE}_date_conf
        ON {TRACKING_TABLE} (game_date, is_high_confidence, confidence)
        """
    )
    conn.execute(
        f"""
        CREATE INDEX IF NOT EXISTS idx_{TRACKING_TABLE}_model
        ON {TRACKING_TABLE} (model_used, early_season, sp_available)
        """
    )


def completed_dates(conn, start_date: str, end_date: str) -> list[str]:
    rows = conn.execute(
        """
        SELECT DISTINCT DATE(game_date) AS game_date
        FROM team_game_results
        WHERE game_status = 3
          AND home_score IS NOT NULL
          AND visiting_score IS NOT NULL
          AND visiting_score != home_score
          AND DATE(game_date) BETWEEN ? AND ?
        ORDER BY DATE(game_date)
        """,
        (start_date, end_date),
    ).fetchall()
    return [row["game_date"] for row in rows]


def actual_results_for_date(conn, target_date: str) -> dict[int, dict]:
    rows = conn.execute(
        """
        SELECT season_year, kind_code, game_sno, home_score, visiting_score
        FROM team_game_results
        WHERE game_status = 3
          AND home_score IS NOT NULL
          AND visiting_score IS NOT NULL
          AND visiting_score != home_score
          AND DATE(game_date) = ?
        """,
        (target_date,),
    ).fetchall()
    return {
        row["game_sno"]: {
            "season_year": row["season_year"],
            "kind_code": row["kind_code"],
            "home_score": float(row["home_score"]),
            "visiting_score": float(row["visiting_score"]),
            "actual_home": int(float(row["home_score"]) > float(row["visiting_score"])),
        }
        for row in rows
    }


def collect_tracking_rows(conn, start_date: str, end_date: str, threshold: float) -> tuple[list[dict], list[dict], dict]:
    all_rows = []

    for target_date in completed_dates(conn, start_date, end_date):
        target = date.fromisoformat(target_date)
        train_rows, pred_games = load_data(conn, target)
        if not pred_games:
            continue

        predictions = train_and_predict(train_rows, pred_games)
        actuals = actual_results_for_date(conn, target_date)

        for pred in predictions:
            actual = actuals.get(pred["game_sno"])
            if actual is None:
                continue

            pred_home = int(pred["prob_home_win"] >= 0.5)
            actual_home = actual["actual_home"]
            confidence = pred["prob_home_win"] if pred_home else (1.0 - pred["prob_home_win"])
            predicted_team_code = pred["home_team"] if pred_home else pred["vis_team"]
            actual_team_code = pred["home_team"] if actual_home else pred["vis_team"]

            row = {
                "season_year": pred["season_year"],
                "kind_code": actual["kind_code"],
                "game_date": target_date,
                "game_sno": pred["game_sno"],
                "home_team_code": pred["home_team"],
                "visiting_team_code": pred["vis_team"],
                "home_team_name": team_name(pred["home_team"]),
                "visiting_team_name": team_name(pred["vis_team"]),
                "home_sp_name": pred.get("home_sp_name"),
                "visiting_sp_name": pred.get("vis_sp_name"),
                "home_sp_acnt": pred.get("home_sp_acnt"),
                "visiting_sp_acnt": pred.get("vis_sp_acnt"),
                "predicted_team_code": predicted_team_code,
                "predicted_team_name": team_name(predicted_team_code),
                "pred_side": side_label(bool(pred_home)),
                "actual_team_code": actual_team_code,
                "actual_team_name": team_name(actual_team_code),
                "actual_side": side_label(bool(actual_home)),
                "prob_home_win": pred["prob_home_win"],
                "confidence": confidence,
                "confidence_level": confidence_level(confidence),
                "is_high_confidence": int(confidence >= threshold),
                "threshold": threshold,
                "home_score": actual["home_score"],
                "visiting_score": actual["visiting_score"],
                "hit": int(pred_home == actual_home),
                "model_used": pred["model_used"],
                "sp_available": int(pred.get("sp_available", 0) > 0.5),
                "early_season": int(bool(pred.get("_early_season"))),
            }
            all_rows.append(row)

    high_confidence_rows = [row for row in all_rows if row["is_high_confidence"]]
    summary = {
        "start_date": start_date,
        "end_date": end_date,
        "threshold": threshold,
        "tracked_games": len(high_confidence_rows),
        "scored_predictions": len(all_rows),
        "hits": sum(row["hit"] for row in high_confidence_rows),
        "coverage": (len(high_confidence_rows) / len(all_rows)) if all_rows else float("nan"),
        "accuracy": (sum(row["hit"] for row in high_confidence_rows) / len(high_confidence_rows))
        if high_confidence_rows else float("nan"),
    }
    return all_rows, high_confidence_rows, summary


def upsert_tracking_rows(conn, rows: list[dict]):
    ensure_tracking_table(conn)
    now = datetime.now().isoformat(timespec="seconds")
    columns = [
        "season_year",
        "kind_code",
        "game_sno",
        "game_date",
        "home_team_code",
        "visiting_team_code",
        "home_team_name",
        "visiting_team_name",
        "home_sp_name",
        "visiting_sp_name",
        "home_sp_acnt",
        "visiting_sp_acnt",
        "prob_home_win",
        "predicted_side",
        "predicted_team_code",
        "predicted_team_name",
        "confidence",
        "confidence_level",
        "is_high_confidence",
        "threshold",
        "model_used",
        "sp_available",
        "early_season",
        "actual_side",
        "actual_team_code",
        "actual_team_name",
        "home_score",
        "visiting_score",
        "is_correct",
        "verified_at",
        "created_at",
        "updated_at",
    ]
    insert_cols = ", ".join(columns)
    placeholders = ", ".join(f":{col}" for col in columns)
    update_cols = [
        col for col in columns
        if col not in {"season_year", "kind_code", "game_sno", "created_at"}
    ]
    update_clause = ", ".join(f"{col}=excluded.{col}" for col in update_cols)

    sql = f"""
        INSERT INTO {TRACKING_TABLE} ({insert_cols})
        VALUES ({placeholders})
        ON CONFLICT(season_year, kind_code, game_sno)
        DO UPDATE SET {update_clause}
    """
    for row in rows:
        params = dict(row)
        params["predicted_side"] = params.pop("pred_side")
        params["is_correct"] = params.pop("hit")
        params["verified_at"] = now
        params["created_at"] = now
        params["updated_at"] = now
        conn.execute(sql, params)
    conn.commit()


def write_csv(rows: list[dict], output_path: Path):
    fieldnames = [
        "season_year",
        "kind_code",
        "game_date",
        "game_sno",
        "home_team_name",
        "visiting_team_name",
        "home_sp_name",
        "visiting_sp_name",
        "pred_side",
        "predicted_team_name",
        "actual_side",
        "actual_team_name",
        "prob_home_win",
        "confidence",
        "confidence_level",
        "hit",
        "model_used",
        "sp_available",
        "early_season",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# 建議4: 歷史前60場各信心桶基準（2021-2025 walk-forward，p>0.60）
# 當某桶低於 BUCKET_ALERT_FLOOR 時在報告中標示警示
BUCKET_BASELINES = {
    "0.60-0.70": 0.75,
    "0.70-0.80": 0.82,
    "0.80-0.90": 0.95,
    "0.90+":     0.95,
}
BUCKET_ALERT_FLOOR = 0.65  # 任一桶低於此值觸發警示


def write_report(rows: list[dict], summary: dict, output_path: Path):
    lines = [
        "# High-Confidence Prediction Tracking",
        "",
        f"- Window: `{summary['start_date']}` to `{summary['end_date']}`",
        f"- Threshold: predicted-side confidence `>= {summary['threshold']:.3f}`",
        f"- Scored predictions in window: `{summary['scored_predictions']}`",
        f"- High-confidence games: `{summary['tracked_games']}`",
        f"- Coverage: `{summary['coverage']:.1%}`" if summary["scored_predictions"] else "- Coverage: `N/A`",
        f"- Accuracy: `{summary['hits']} / {summary['tracked_games']} = {summary['accuracy']:.1%}`" if summary["tracked_games"] else "- Accuracy: `N/A`",
        "",
    ]

    # 建議3: Platt re-fit 提醒
    if summary["scored_predictions"] >= 80:
        lines += [
            "**⚠ 建議3: 已累積 ≥80 場驗證預測，可執行 Platt re-fit 更新 A/B 參數。**",
            "",
        ]

    if rows:
        lines += [
            "## By Model",
            "",
            "| Model | Games | Hits | Accuracy |",
            "| --- | ---: | ---: | ---: |",
        ]
        for model_used in sorted({row["model_used"] for row in rows}):
            model_rows = [row for row in rows if row["model_used"] == model_used]
            hits = sum(row["hit"] for row in model_rows)
            lines.append(f"| {model_used} | {len(model_rows)} | {hits} | {(hits / len(model_rows)):.1%} |")

        # 建議4: 信心桶監控
        lines += [
            "",
            "## By Confidence Bucket (建議4監控)",
            "",
            "| Bucket | Games | Hits | Accuracy | Baseline | Status |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
        bucket_def = [
            ("0.60-0.70", 0.60, 0.70),
            ("0.70-0.80", 0.70, 0.80),
            ("0.80-0.90", 0.80, 0.90),
            ("0.90+",     0.90, 1.01),
        ]
        alerts = []
        for label, lo, hi in bucket_def:
            bucket_rows = [r for r in rows if lo <= r["confidence"] < hi]
            if not bucket_rows:
                lines.append(f"| {label} | 0 | — | — | {BUCKET_BASELINES[label]:.0%} | — |")
                continue
            b_hits = sum(r["hit"] for r in bucket_rows)
            b_acc = b_hits / len(bucket_rows)
            baseline = BUCKET_BASELINES[label]
            if b_acc < BUCKET_ALERT_FLOOR and len(bucket_rows) >= 5:
                status = "**ALERT**"
                alerts.append(f"{label}: {b_acc:.1%} (n={len(bucket_rows)}, baseline {baseline:.0%})")
            elif b_acc < baseline and len(bucket_rows) >= 5:
                status = "WARN"
            else:
                status = "OK"
            lines.append(
                f"| {label} | {len(bucket_rows)} | {b_hits} | {b_acc:.1%} | {baseline:.0%} | {status} |"
            )
        if alerts:
            lines += ["", f"> ALERT 桶: {', '.join(alerts)}", ""]

        lines += [
            "",
            "## Games",
            "",
            "| Date | SNO | Matchup | Pred | Actual | Conf | Model | SP | Data-Limited | Hit |",
            "| --- | ---: | --- | --- | --- | ---: | --- | --- | --- | --- |",
        ]
        for row in rows:
            matchup = f"{row['visiting_team_name']} @ {row['home_team_name']}"
            lines.append(
                f"| {row['game_date']} | {row['game_sno']} | {matchup} | {row['pred_side']} | "
                f"{row['actual_side']} | {row['confidence']:.3f} | {row['model_used']} | "
                f"{'Y' if row['sp_available'] else 'N'} | {'Y' if row['early_season'] else 'N'} | "
                f"{'Y' if row['hit'] else 'N'} |"
            )
    else:
        lines += [
            "No high-confidence games found in this window.",
            "",
        ]

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Track verified high-confidence predictions")
    parser.add_argument("--start-date", default="2026-04-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end-date", default=None, help="End date YYYY-MM-DD; default uses latest completed date")
    parser.add_argument("--threshold", type=float, default=0.600, help="Predicted-side confidence threshold")
    args = parser.parse_args()

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        if args.end_date is None:
            row = conn.execute(
                """
                SELECT DATE(MAX(game_date)) AS latest_completed
                FROM team_game_results
                WHERE game_status = 3
                  AND home_score IS NOT NULL
                  AND visiting_score IS NOT NULL
                """
            ).fetchone()
            end_date = row["latest_completed"]
        else:
            end_date = args.end_date

        all_rows, high_confidence_rows, summary = collect_tracking_rows(conn, args.start_date, end_date, args.threshold)
        upsert_tracking_rows(conn, all_rows)
        write_csv(high_confidence_rows, CSV_PATH)
        write_report(high_confidence_rows, summary, REPORT_PATH)

        print(f"sqlite_table={TRACKING_TABLE}")
        print(f"sqlite_rows_upserted={len(all_rows)}")
        print(f"csv={CSV_PATH.resolve()}")
        print(f"report={REPORT_PATH.resolve()}")
        print(f"start_date={summary['start_date']}")
        print(f"end_date={summary['end_date']}")
        print(f"threshold={summary['threshold']:.3f}")
        print(f"scored_predictions={summary['scored_predictions']}")
        print(f"tracked_games={summary['tracked_games']}")
        if summary["scored_predictions"]:
            print(f"coverage={summary['coverage']:.4f}")
        if summary["tracked_games"]:
            print(f"accuracy={summary['accuracy']:.4f}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
