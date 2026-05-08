"""
Predict NPB game outcomes for a given date using game_features_npb.

Usage:
    python predict_today_npb.py
    python predict_today_npb.py --date 2026-04-23
    python predict_today_npb.py --date 2026-04-23 --verify
"""

from __future__ import annotations

import argparse
import math
import sqlite3
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression


BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "npb.sqlite"
TABLE_NAME = "game_features_npb"
TRAIN_LEAGUES = ("CL", "PL")

FALLBACK_FEATURES = [
    "diff_elo",
    "home_elo",
    "vis_elo",
    "diff_win_pct",
    "diff_rd_pg",
    "diff_pyth_wp",
    "diff_w5_win_pct",
    "diff_w10_win_pct",
    "diff_w5_rd_pg",
    "diff_w10_rd_pg",
    "diff_streak",
    "home_rest",
    "vis_rest",
    "diff_rest",
    "home_season_games_before",
    "vis_season_games_before",
    "home_win_pct",
    "vis_win_pct",
    "home_rs_pg",
    "home_ra_pg",
    "vis_rs_pg",
    "vis_ra_pg",
]
PRIMARY_FEATURES = FALLBACK_FEATURES + [
    "diff_sp_era",
    "diff_sp_whip",
    "diff_sp_k9",
    "diff_sp_fip",
    "sp_available",
]

XGB_PARAMS = dict(
    n_estimators=50,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=15,
    reg_lambda=3.0,
    eval_metric="logloss",
    random_state=42,
    verbosity=0,
)

EARLY_MODEL_FEATURES = ["diff_elo", "home_elo", "vis_elo"]
EARLY_MIN_TRAIN_ROWS = 30
EARLY_BASELINE_PROBABILITY = 0.535

TEAM_SHORT = {
    "g": "巨人",
    "d": "中日",
    "db": "DeNA",
    "t": "阪神",
    "c": "広島",
    "s": "ヤクルト",
    "h": "SB",
    "f": "日ハム",
    "b": "オリ",
    "e": "楽天",
    "l": "西武",
    "m": "ロッテ",
}
LEAGUE_CODE = {
    "g": "CL",
    "d": "CL",
    "db": "CL",
    "t": "CL",
    "c": "CL",
    "s": "CL",
    "h": "PL",
    "f": "PL",
    "b": "PL",
    "e": "PL",
    "l": "PL",
    "m": "PL",
}
LEAGUE_NAMES = {
    "CL": "中央聯盟",
    "PL": "太平洋聯盟",
}

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict NPB games for a target date.")
    parser.add_argument(
        "--date",
        default=date.today().isoformat(),
        help="Target date YYYY-MM-DD (default: today).",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Compare predictions against completed games on the target date.",
    )
    return parser.parse_args()


def update_features(target_date: date) -> None:
    target_year = target_date.year
    subprocess.run(
        [
            sys.executable,
            "build_game_features_npb.py",
            "--year",
            str(target_year),
            "--include-scheduled",
        ],
        cwd=BASE_DIR,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "build_pitcher_features_npb.py",
            "--year",
            str(target_year),
        ],
        cwd=BASE_DIR,
        check=True,
    )


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def placeholders(values: tuple[str, ...]) -> str:
    return ",".join("?" for _ in values)


def load_training_rows(conn: sqlite3.Connection, target_date: date) -> list[dict]:
    query = f"""
        SELECT *
        FROM {TABLE_NAME}
        WHERE game_date < ?
          AND home_win IS NOT NULL
          AND league_code IN ({placeholders(TRAIN_LEAGUES)})
        ORDER BY game_date, game_url
    """
    params = (target_date.isoformat(), *TRAIN_LEAGUES)
    return [dict(row) for row in conn.execute(query, params).fetchall()]


def load_target_rows(conn: sqlite3.Connection, target_date: date, verify: bool) -> list[dict]:
    label_clause = "home_win IS NOT NULL" if verify else "home_win IS NULL"
    query = f"""
        SELECT *
        FROM {TABLE_NAME}
        WHERE game_date = ?
          AND {label_clause}
          AND league_code IN ({placeholders(TRAIN_LEAGUES)})
        ORDER BY league_code, game_url
    """
    params = (target_date.isoformat(), *TRAIN_LEAGUES)
    return [dict(row) for row in conn.execute(query, params).fetchall()]


def validate_features(rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError("No training rows found in game_features_npb.")
    required = sorted(set(PRIMARY_FEATURES + EARLY_MODEL_FEATURES))
    required += [
        "away_code",
        "game_date",
        "game_url",
        "home_code",
        "home_win",
        "league_code",
        "season_year",
    ]
    missing = [name for name in required if name not in rows[0]]
    if missing:
        raise RuntimeError(f"Missing required columns: {', '.join(missing)}")


def as_float(value) -> float:
    if value is None:
        return math.nan
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def fit_medians(rows: list[dict], features: list[str]) -> dict[str, float]:
    medians = {}
    for feature in features:
        clean = [as_float(row.get(feature)) for row in rows]
        clean = [value for value in clean if not math.isnan(value)]
        medians[feature] = float(np.median(clean)) if clean else 0.0
    return medians


def transform(rows: list[dict], features: list[str], medians: dict[str, float]) -> np.ndarray:
    matrix = []
    for row in rows:
        matrix.append(
            [
                medians[feature]
                if math.isnan(as_float(row.get(feature)))
                else as_float(row.get(feature))
                for feature in features
            ]
        )
    return np.array(matrix, dtype=float)


def is_early_season(row: dict) -> bool:
    return (
        int(row.get("home_season_games_before") or 0) < 10
        or int(row.get("vis_season_games_before") or 0) < 10
    )


def route_regime(row: dict) -> str:
    if is_early_season(row):
        return "early"
    return "primary" if has_primary_pitcher_features(row) else "fallback"


def has_primary_pitcher_features(row: dict) -> bool:
    return as_float(row.get("sp_available")) >= 0.5


def fit_xgb_model(rows: list[dict], features: list[str]) -> tuple[xgb.XGBClassifier, dict[str, float]]:
    medians = fit_medians(rows, features)
    x_train = transform(rows, features, medians)
    y_train = np.array([int(row["home_win"]) for row in rows], dtype=int)
    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(x_train, y_train)
    return model, medians


def fit_early_model(rows: list[dict]) -> tuple[LogisticRegression, dict[str, float]] | None:
    if len(rows) < EARLY_MIN_TRAIN_ROWS:
        return None
    y_train = np.array([int(row["home_win"]) for row in rows], dtype=int)
    if len(set(y_train.tolist())) < 2:
        return None
    medians = fit_medians(rows, EARLY_MODEL_FEATURES)
    x_train = transform(rows, EARLY_MODEL_FEATURES, medians)
    model = LogisticRegression(C=1.0, max_iter=500)
    model.fit(x_train, y_train)
    return model, medians


def train_models(train_rows: list[dict]) -> dict:
    fallback_train = [
        row
        for row in train_rows
        if not is_early_season(row)
    ]
    if not fallback_train:
        raise RuntimeError("No fallback-route training rows.")

    primary_train = [
        row
        for row in train_rows
        if route_regime(row) == "primary"
    ]

    early_train = [row for row in train_rows if route_regime(row) == "early"]
    return {
        "fallback": fit_xgb_model(fallback_train, FALLBACK_FEATURES),
        "primary": fit_xgb_model(primary_train, PRIMARY_FEATURES) if primary_train else None,
        "early": fit_early_model(early_train),
    }


def predict_rows(models: dict, rows: list[dict]) -> list[dict]:
    predictions = []
    fallback_model, fallback_medians = models["fallback"]
    primary_bundle = models["primary"]
    early_model = models["early"]

    for row in rows:
        route = route_regime(row)
        if route == "early":
            if early_model is None:
                home_probability = EARLY_BASELINE_PROBABILITY
            else:
                model, medians = early_model
                x_test = transform([row], EARLY_MODEL_FEATURES, medians)
                home_probability = float(model.predict_proba(x_test)[0, 1])
        elif route == "primary" and primary_bundle is not None:
            primary_model, primary_medians = primary_bundle
            x_test = transform([row], PRIMARY_FEATURES, primary_medians)
            home_probability = float(primary_model.predict_proba(x_test)[0, 1])
        else:
            route = "fallback"
            x_test = transform([row], FALLBACK_FEATURES, fallback_medians)
            home_probability = float(fallback_model.predict_proba(x_test)[0, 1])

        predictions.append(
            {
                **row,
                "prob_home_win": home_probability,
                "route": route,
                "predicted_home_win": int(home_probability >= 0.5),
            }
        )

    return predictions


def team_name(code: str | None) -> str:
    if code is None:
        return ""
    return TEAM_SHORT.get(code.lower(), code)


def row_league(row: dict) -> str:
    league = row.get("league_code")
    if league in LEAGUE_NAMES:
        return league
    return LEAGUE_CODE.get(str(row.get("home_code", "")).lower(), str(league or ""))


def confidence(side_probability: float) -> str:
    if side_probability >= 0.60:
        return "HIGH"
    if side_probability >= 0.55:
        return "MED"
    return "LOW"


def prediction_name(row: dict) -> str:
    if int(row["predicted_home_win"]):
        return team_name(row.get("home_code"))
    return team_name(row.get("away_code"))


def actual_name(row: dict) -> str:
    if row.get("home_win") is None:
        return ""
    if int(row["home_win"]):
        return team_name(row.get("home_code"))
    return team_name(row.get("away_code"))


def format_training_range(train_rows: list[dict]) -> str:
    years = [int(row["season_year"]) for row in train_rows if row.get("season_year") is not None]
    if not years:
        return ""
    return f"{min(years)}-{max(years)}"


def format_percent(probability: float) -> str:
    return f"{probability * 100:.1f}%"


def build_report(
    target_date: date,
    train_rows: list[dict],
    predictions: list[dict],
    verify: bool,
) -> str:
    title = "NPB 預測驗證" if verify else "NPB 預測"
    lines = [
        f"# {title} — {target_date.isoformat()}",
        "",
        f"訓練資料：{len(train_rows)} 場（{format_training_range(train_rows)}）",
        "",
    ]

    if not predictions:
        target = "已完賽結果" if verify else "scheduled"
        lines.append(f"找不到 {target_date.isoformat()} 的 {target} 場次。")
        lines.append("")
        return "\n".join(lines)

    total = 0
    correct = 0
    for league in ("CL", "PL"):
        lines.append(f"## {LEAGUE_NAMES[league]}")
        if verify:
            lines.extend(
                [
                    "| 主場 | 客場 | 勝率 | 預測 | 實際 | 命中 | 信心 | 路由 |",
                    "|------|------|------|------|------|------|------|------|",
                ]
            )
        else:
            lines.extend(
                [
                    "| 主場 | 客場 | 勝率 | 預測 | 信心 | 路由 |",
                    "|------|------|------|------|------|------|",
                ]
            )

        league_rows = [row for row in predictions if row_league(row) == league]
        if not league_rows:
            empty_cols = 8 if verify else 6
            lines.append("| " + " | ".join(["-"] * empty_cols) + " |")
        for row in league_rows:
            home_probability = float(row["prob_home_win"])
            side_probability = home_probability if home_probability >= 0.5 else 1.0 - home_probability
            base_cols = [
                team_name(row.get("home_code")),
                team_name(row.get("away_code")),
                format_percent(home_probability),
                prediction_name(row),
            ]
            if verify:
                hit = int(row["predicted_home_win"]) == int(row["home_win"])
                correct += int(hit)
                total += 1
                base_cols.extend([actual_name(row), "✓" if hit else "✗"])
            base_cols.extend([confidence(side_probability), row["route"]])
            lines.append("| " + " | ".join(base_cols) + " |")
        lines.append("")

    if verify:
        accuracy = correct / total if total else 0.0
        lines.extend([f"驗證結果：{correct}/{total} = {accuracy:.1%}", ""])

    lines.extend(
        [
            "信心等級：LOW < 55% ≤ MED < 60% ≤ HIGH",
            "",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    target_date = date.fromisoformat(args.date)

    update_features(target_date)
    with connect() as conn:
        train_rows = load_training_rows(conn, target_date)
        validate_features(train_rows)
        target_rows = load_target_rows(conn, target_date, args.verify)

    predictions = []
    if target_rows:
        models = train_models(train_rows)
        predictions = predict_rows(models, target_rows)

    report = build_report(target_date, train_rows, predictions, args.verify)
    report_path = BASE_DIR / f"predictions_npb_{target_date.strftime('%Y%m%d')}.md"
    report_path.write_text(report, encoding="utf-8")
    print(report, end="")
    print(f"\nMarkdown: {report_path.name}")


if __name__ == "__main__":
    main()
