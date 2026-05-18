# -*- coding: utf-8 -*-
"""
MLB threshold sweep: accuracy & coverage for HIGH thresholds 60-65% (2020+).
Run from MLB/ directory.
"""
import sqlite3, sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_mlb_model import build_rows, walkforward

DB_PATH = Path(__file__).resolve().parent / "mlb.sqlite"
SINCE_YEAR = 2020
THRESHOLDS = [0.60, 0.61, 0.62, 0.63, 0.64, 0.65]

def main():
    conn = sqlite3.connect(str(DB_PATH))
    print("Loading MLB rows...", flush=True)
    all_rows = build_rows(conn)
    conn.close()
    print(f"Total rows: {len(all_rows)}", flush=True)

    print("\nRunning walk-forward (2020-2026)...", flush=True)
    results, _ = walkforward(all_rows, rolling_window=True)

    # Filter to 2020+
    preds = [r for r in results if int(r["season_year"]) >= SINCE_YEAR]
    print(f"\nPredictions from {SINCE_YEAR}+: {len(preds)} games\n")

    # Yearly breakdown per threshold
    years = sorted(set(int(r["season_year"]) for r in preds))

    print(f"{'Thresh':>7}  {'Year':>6}  {'n_all':>6}  {'n_high':>7}  {'cov':>7}  {'acc_all':>8}  {'acc_high':>9}")
    print("  " + "-" * 62)

    summary = {}
    for thresh in THRESHOLDS:
        summary[thresh] = {"n_all": 0, "n_high": 0, "correct_all": 0, "correct_high": 0}
        for yr in years:
            yr_rows = [r for r in preds if int(r["season_year"]) == yr]
            n_all = len(yr_rows)
            cor_all = sum(r["correct"] for r in yr_rows)
            high = [r for r in yr_rows if max(r["prob_home"], 1 - r["prob_home"]) >= thresh]
            n_high = len(high)
            cor_high = sum(r["correct"] for r in high)
            cov = n_high / n_all if n_all else 0
            acc_all = cor_all / n_all if n_all else 0
            acc_high = cor_high / n_high if n_high else 0
            cov_str = f"{cov:.1%}"
            acc_h_str = f"{acc_high:.1%}" if n_high else "—"
            print(f"  {thresh:.0%}  {yr:>6}  {n_all:>6}  {n_high:>7}  {cov_str:>7}  {acc_all:>7.1%}  {acc_h_str:>9}")
            summary[thresh]["n_all"] += n_all
            summary[thresh]["n_high"] += n_high
            summary[thresh]["correct_all"] += cor_all
            summary[thresh]["correct_high"] += cor_high
        print()

    # Aggregate 2020+
    print(f"\n{'':=<64}")
    print(f"  AGGREGATE 2020–2026")
    print(f"{'Thresh':>7}  {'n_all':>6}  {'n_high':>7}  {'coverage':>9}  {'acc_all':>8}  {'acc_high':>9}")
    print("  " + "-" * 55)
    for thresh in THRESHOLDS:
        s = summary[thresh]
        n_all = s["n_all"]
        n_high = s["n_high"]
        cov = n_high / n_all if n_all else 0
        acc_all = s["correct_all"] / n_all if n_all else 0
        acc_high = s["correct_high"] / n_high if n_high else 0
        print(f"  {thresh:.0%}  {n_all:>6}  {n_high:>7}  {cov:>9.1%}  {acc_all:>8.1%}  {acc_high:>9.1%}")

    print()

if __name__ == "__main__":
    main()
