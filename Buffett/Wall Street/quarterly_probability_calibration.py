# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss


ROOT = Path(__file__).resolve().parent
INPUT_PATH = ROOT / "walkforward_portfolio_beta_constrained_voo_alpha.csv"
SUMMARY_PATH = ROOT / "quarterly_probability_calibration_summary.csv"
DETAIL_PATH = ROOT / "quarterly_probability_calibration_detail.csv"
CURRENT_PATH = ROOT / "quarterly_probability_calibration_current.csv"
BINS_PATH = ROOT / "quarterly_probability_calibration_bins.csv"

TARGETS = [
    ("p_gt_5", "event_gt_5", "P(>5%)"),
    ("p_lt_0", "event_lt_0", "P(<0%)"),
]
MIN_TRAIN_OBS = 8
BIN_COUNT = 5
EPS = 1e-6


def clip_prob(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return np.clip(arr, EPS, 1.0 - EPS)


def fit_platt(x: np.ndarray, y: np.ndarray) -> LogisticRegression | None:
    if len(np.unique(y)) < 2:
        return None
    model = LogisticRegression(solver="lbfgs")
    model.fit(x.reshape(-1, 1), y.astype(int))
    return model


def predict_platt(model: LogisticRegression | None, x: np.ndarray) -> np.ndarray:
    if model is None:
        return clip_prob(x)
    return clip_prob(model.predict_proba(x.reshape(-1, 1))[:, 1])


def fit_isotonic(x: np.ndarray, y: np.ndarray) -> IsotonicRegression | None:
    if len(np.unique(y)) < 2:
        return None
    model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    model.fit(x, y.astype(float))
    return model


def predict_isotonic(model: IsotonicRegression | None, x: np.ndarray) -> np.ndarray:
    if model is None:
        return clip_prob(x)
    return clip_prob(model.predict(x))


def build_events(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["event_gt_5"] = (out["realized_return"] > 0.05).astype(int)
    out["event_lt_0"] = (out["realized_return"] < 0.0).astype(int)
    return out


def source_probability_column(frame: pd.DataFrame, prob_col: str) -> str:
    raw_col = f"raw_{prob_col}"
    return raw_col if raw_col in frame.columns else prob_col


def append_platt_calibration_columns(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.sort_values("forecast_start").reset_index(drop=True).copy()
    ordered["raw_p_gt_5"] = ordered[source_probability_column(ordered, "p_gt_5")].astype(float)
    ordered["raw_p_lt_0"] = ordered[source_probability_column(ordered, "p_lt_0")].astype(float)

    with_events = build_events(ordered)
    realized = with_events[with_events["realized_return"].notna()].copy().reset_index(drop=True)
    current = with_events[with_events["realized_return"].isna()].copy().reset_index(drop=True)

    for prob_col, target_col, _metric_label in TARGETS:
        source_col = source_probability_column(with_events, prob_col)
        working = with_events.copy()
        working[prob_col] = working[source_col].astype(float)
        detail = expanding_predictions(working, prob_col, target_col)
        calibrated_col = f"{prob_col}_calibrated"
        realized_calibrated = detail[["period", "platt_prob"]].rename(columns={"platt_prob": calibrated_col})
        ordered = ordered.merge(realized_calibrated, on="period", how="left")

        if not current.empty:
            current_raw = float(current.iloc[0][source_col])
            x_train = clip_prob(realized[source_col])
            y_train = realized[target_col].astype(int).to_numpy()
            model = fit_platt(x_train, y_train)
            current_calibrated = float(predict_platt(model, np.array([current_raw]))[0]) if model is not None else current_raw
            ordered.loc[ordered["realized_return"].isna(), calibrated_col] = current_calibrated

        ordered[calibrated_col] = ordered[calibrated_col].fillna(ordered[source_col])

    return ordered


def expanding_predictions(frame: pd.DataFrame, prob_col: str, target_col: str) -> pd.DataFrame:
    realized = frame[frame["realized_return"].notna()].copy().reset_index(drop=True)
    rows: list[dict[str, float | int | str]] = []

    for idx in range(len(realized)):
        row = realized.iloc[idx]
        raw_prob = float(row[prob_col])
        actual = int(row[target_col])

        if idx < MIN_TRAIN_OBS:
            platt_prob = raw_prob
            isotonic_prob = raw_prob
            stage = "warmup_raw"
        else:
            train = realized.iloc[:idx]
            x_train = clip_prob(train[prob_col])
            y_train = train[target_col].astype(int).to_numpy()
            x_pred = np.array([raw_prob], dtype=float)

            platt_model = fit_platt(x_train, y_train)
            isotonic_model = fit_isotonic(x_train, y_train)
            platt_prob = float(predict_platt(platt_model, x_pred)[0])
            isotonic_prob = float(predict_isotonic(isotonic_model, x_pred)[0])
            stage = "walkforward_calibrated"

        rows.append(
            {
                "period": row["period"],
                "metric": prob_col,
                "target": target_col,
                "actual_event": actual,
                "raw_prob": raw_prob,
                "platt_prob": platt_prob,
                "isotonic_prob": isotonic_prob,
                "stage": stage,
            }
        )

    return pd.DataFrame(rows)


def summarize_method(detail: pd.DataFrame, prob_column: str) -> dict[str, float | str]:
    y = detail["actual_event"].astype(int).to_numpy()
    p = clip_prob(detail[prob_column])
    return {
        "method": prob_column.replace("_prob", ""),
        "rows": int(len(detail)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "pred_mean": float(np.mean(p)),
        "actual_rate": float(np.mean(y)),
    }


def calibration_bins(detail: pd.DataFrame, metric_label: str, prob_column: str) -> pd.DataFrame:
    usable = detail.copy()
    usable["prob"] = usable[prob_column].astype(float)
    usable["bucket"] = pd.qcut(usable["prob"], q=min(BIN_COUNT, len(usable)), duplicates="drop")
    grouped = (
        usable.groupby("bucket", observed=False)
        .agg(
            count=("actual_event", "size"),
            pred_mean=("prob", "mean"),
            actual_rate=("actual_event", "mean"),
        )
        .reset_index()
    )
    grouped["metric"] = metric_label
    grouped["method"] = prob_column.replace("_prob", "")
    grouped["calibration_gap"] = grouped["pred_mean"] - grouped["actual_rate"]
    return grouped


def pick_best_method(summary_rows: list[dict[str, float | str]]) -> str:
    ranked = sorted(summary_rows, key=lambda row: (float(row["brier"]), float(row["log_loss"])))
    return str(ranked[0]["method"])


def main() -> None:
    frame = pd.read_csv(INPUT_PATH)
    frame = frame.sort_values("forecast_start").reset_index(drop=True)
    frame = build_events(frame)

    all_detail = []
    summary_rows = []
    bin_rows = []
    current_rows = []

    realized = frame[frame["realized_return"].notna()].copy().reset_index(drop=True)
    current = frame[frame["realized_return"].isna()].copy().reset_index(drop=True)

    for prob_col, target_col, metric_label in TARGETS:
        source_col = source_probability_column(frame, prob_col)
        working = frame.copy()
        working[prob_col] = working[source_col].astype(float)
        detail = expanding_predictions(working, prob_col, target_col)
        all_detail.append(detail)

        metric_summary_rows = [
            {"metric": metric_label, **summarize_method(detail, "raw_prob")},
            {"metric": metric_label, **summarize_method(detail, "platt_prob")},
            {"metric": metric_label, **summarize_method(detail, "isotonic_prob")},
        ]
        summary_rows.extend(metric_summary_rows)

        best_method = pick_best_method(metric_summary_rows)
        best_prob_column = f"{best_method}_prob"
        bin_rows.append(calibration_bins(detail, metric_label, best_prob_column))

        x_train = clip_prob(realized[source_col])
        y_train = realized[target_col].astype(int).to_numpy()
        current_raw = float(current.iloc[0][source_col]) if not current.empty else np.nan

        if best_method == "platt":
            model = fit_platt(x_train, y_train)
            current_calibrated = float(predict_platt(model, np.array([current_raw]))[0]) if not np.isnan(current_raw) else np.nan
        elif best_method == "isotonic":
            model = fit_isotonic(x_train, y_train)
            current_calibrated = float(predict_isotonic(model, np.array([current_raw]))[0]) if not np.isnan(current_raw) else np.nan
        else:
            current_calibrated = current_raw

        current_rows.append(
            {
                "period": current.iloc[0]["period"] if not current.empty else "",
                "metric": metric_label,
                "best_method": best_method,
                "raw_probability": current_raw,
                "calibrated_probability": current_calibrated,
            }
        )

    detail_report = pd.concat(all_detail, ignore_index=True)
    summary_report = pd.DataFrame(summary_rows)
    bins_report = pd.concat(bin_rows, ignore_index=True)
    current_report = pd.DataFrame(current_rows)

    detail_report.to_csv(DETAIL_PATH, index=False)
    summary_report.to_csv(SUMMARY_PATH, index=False)
    bins_report.to_csv(BINS_PATH, index=False)
    current_report.to_csv(CURRENT_PATH, index=False)

    display = summary_report.copy().astype(object)
    for column in ["brier", "log_loss", "pred_mean", "actual_rate"]:
        if column in ["pred_mean", "actual_rate"]:
            display.loc[:, column] = display[column].map(lambda value: f"{value * 100:.1f}%")
        else:
            display.loc[:, column] = display[column].map(lambda value: f"{value:.4f}")

    current_display = current_report.copy().astype(object)
    for column in ["raw_probability", "calibrated_probability"]:
        current_display.loc[:, column] = current_display[column].map(lambda value: "" if pd.isna(value) else f"{value * 100:.1f}%")

    print("Quarterly probability calibration")
    print(display.to_string(index=False))
    print()
    print("Current quarter calibrated probabilities")
    print(current_display.to_string(index=False))
    print()
    print(f"Saved summary CSV: {SUMMARY_PATH.name}")
    print(f"Saved detail CSV: {DETAIL_PATH.name}")
    print(f"Saved bins CSV: {BINS_PATH.name}")
    print(f"Saved current CSV: {CURRENT_PATH.name}")


if __name__ == "__main__":
    main()
