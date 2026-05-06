"""
Backfill historical accuracy cache files.

Usage:
    python backfill_accuracy.py [--leagues mlb,kbo,npb,cpbl] [--days 30] [--force]
"""
import argparse
import json
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "dashboard" / "data"

LEAGUE_DIRS = {
    "mlb":  BASE_DIR / "MLB",
    "cpbl": BASE_DIR / "CPBL",
    "kbo":  BASE_DIR / "KBO",
    "npb":  BASE_DIR / "NPB",
}


def backfill(leagues, days, force):
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    today = date.today()
    date_list = [
        (today - timedelta(days=i)).isoformat()
        for i in range(days, 0, -1)   # oldest → newest, exclude today
    ]

    done = 0
    errors = []

    for league in leagues:
        league = league.lower()
        if league not in LEAGUE_DIRS:
            print(f"SKIP unknown league: {league}")
            continue

        league_dir = LEAGUE_DIRS[league]
        script = league_dir / "run_dashboard.py"
        if not script.exists():
            print(f"ERROR {league}: run_dashboard.py not found at {script}")
            errors.append(f"{league}: run_dashboard.py not found")
            continue

        for date_str in date_list:
            cache = DATA_DIR / f"{league}_{date_str}.json"

            if cache.exists() and not force:
                print(f"SKIP {league} {date_str}")
                continue

            print(f"RUN  {league} {date_str} …", flush=True)
            try:
                result = subprocess.run(
                    [sys.executable, str(script), date_str, "--strict-cutoff"],
                    capture_output=True,
                    encoding="utf-8",
                    errors="replace",
                    cwd=str(league_dir),
                    timeout=180,
                )
            except subprocess.TimeoutExpired:
                msg = f"{league} {date_str}: timeout (>180s)"
                print(f"ERROR {msg}")
                errors.append(msg)
                continue
            except Exception as e:
                msg = f"{league} {date_str}: {e}"
                print(f"ERROR {msg}")
                errors.append(msg)
                continue

            stdout = result.stdout.strip() if result.stdout else ""

            # Find last JSON line in stdout
            json_line = None
            for line in reversed(stdout.splitlines()):
                line = line.strip()
                if line.startswith("{"):
                    json_line = line
                    break

            if result.returncode == 0 and json_line:
                try:
                    data = json.loads(json_line)
                    cache.write_text(
                        json.dumps(data, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    games_count = len(data.get("games", []))
                    print(f"SAVED {league} {date_str} ({games_count} games)")
                    done += 1
                except json.JSONDecodeError as e:
                    msg = f"{league} {date_str}: JSON parse error: {e}"
                    print(f"ERROR {msg}")
                    errors.append(msg)
            else:
                stderr_snippet = (result.stderr or "")[-300:].strip()
                msg = f"{league} {date_str}: returncode={result.returncode}, no JSON output. stderr={stderr_snippet!r}"
                print(f"ERROR {msg}")
                errors.append(msg)

    print(f"\n=== Done: {done} saved, {len(errors)} errors ===")
    return done, errors


def main():
    parser = argparse.ArgumentParser(description="Backfill accuracy cache files")
    parser.add_argument(
        "--leagues",
        default="mlb,kbo,npb,cpbl",
        help="Comma-separated list of leagues (default: mlb,kbo,npb,cpbl)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of past days to backfill (default: 30)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run even if cache file already exists",
    )
    args = parser.parse_args()

    leagues = [lg.strip() for lg in args.leagues.split(",") if lg.strip()]
    backfill(leagues, args.days, args.force)


if __name__ == "__main__":
    main()
