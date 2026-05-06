"""
cpbl_betting_ev.py

Scrape CPBL moneyline odds from Taiwan Sports Lottery (不讓分),
compare with model predictions, and compute Expected Value (EV).

Usage:
    python cpbl_betting_ev.py
    python cpbl_betting_ev.py --date 2026-04-28
    python cpbl_betting_ev.py --save

Odds source: https://www-talo-ssb-pr.sportslottery.com.tw/sport/棒球/臺灣/中華職棒/36680.1
EV formula:  EV = model_prob * (decimal_odds - 1) - (1 - model_prob)
"""

import argparse
import sys
from datetime import date
from pathlib import Path

from playwright.sync_api import sync_playwright

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# All-baseball page: we filter to CPBL only via ZH_TO_CODE mapping.
# The CPBL-specific URL (/36680.1) does not load game data on direct navigation;
# loading the parent all-baseball page and filtering works reliably.
SPORTSLOTTERY_URL = (
    "https://www-talo-ssb-pr.sportslottery.com.tw/sport/棒球/34731.1"
)

# Taiwan Sports Lottery Chinese team names → CPBL team codes used in cpbl.sqlite
ZH_TO_CODE: dict[str, str] = {
    "中信兄弟":   "ACN011",
    "味全龍":     "AAA011",
    "樂天桃猿":   "AJL011",
    "富邦悍將":   "AEO011",
    "統一獅":     "ADD011",
    "台鋼雄鷹":   "AKP011",
}

EV_THRESHOLD = 0.03
MIN_CONF = 0.60


# ---------------------------------------------------------------------------
# Odds helpers
# ---------------------------------------------------------------------------

def vig_free_prob(dec_home: float, dec_away: float) -> tuple[float, float]:
    raw_h = 1 / dec_home
    raw_a = 1 / dec_away
    total = raw_h + raw_a
    return raw_h / total, raw_a / total


def ev(model_prob: float, decimal_odds: float) -> float:
    return model_prob * (decimal_odds - 1) - (1 - model_prob)


# ---------------------------------------------------------------------------
# Scrape Taiwan Sports Lottery
# ---------------------------------------------------------------------------

def fetch_sportslottery_odds_cpbl(url: str) -> dict[tuple[str, str], tuple[float, float]]:
    """
    Scrape CPBL 不讓分 (moneyline) odds from Taiwan Sports Lottery.

    The page uses bot-detection; headless=False + anti-fingerprint args are
    required (same approach as the NPB scraper).  CPBL games are nested under
    a collapsible '臺灣' region that must be expanded before odds appear.

    Returns {(home_code, away_code): (dec_home, dec_away)}.
    """
    _UA = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=False,
            args=["--disable-blink-features=AutomationControlled", "--no-sandbox"],
        )
        ctx = browser.new_context(
            user_agent=_UA,
            viewport={"width": 1280, "height": 800},
            locale="zh-TW",
            timezone_id="Asia/Taipei",
        )
        page = ctx.new_page()
        page.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
        )
        page.goto(url, wait_until="domcontentloaded", timeout=30000)

        # Wait for at least one 不讓分 checkbox to confirm page is rendered
        try:
            page.wait_for_selector(
                '[role="checkbox"][aria-label*="不讓分"]', timeout=25000
            )
        except Exception:
            print("  WARNING: Game data not found on page — no CPBL odds available")
            browser.close()
            return {}

        # Expand the 臺灣 (Taiwan) region which is collapsed by default.
        # The region header button has data-test-id="navigation-node-arrow-臺灣".
        try:
            taiwan_arrow = page.locator(
                '[data-test-id="navigation-node-arrow-臺灣"]'
            ).first
            if taiwan_arrow.count() > 0:
                taiwan_arrow.click()
                # Wait for CPBL event links to appear after expansion
                page.wait_for_timeout(3000)
        except Exception:
            pass  # Region may already be expanded or absent

        # Extract games using event links (a[href*="/event/"]) — same approach as NPB.
        # Each event link contains <p> tags: paragraphs[0]=home, [1]=away.
        # 不讓分 checkboxes are siblings in the same game container.
        games = page.evaluate("""() => {
            const results = [];
            const links = Array.from(document.querySelectorAll('a[href*="/event/"]'));
            for (const link of links) {
                // Only process links inside 臺灣/中華職棒 competition
                const href = link.getAttribute('href') || '';
                if (!href.includes('%e8%87%ba%e7%81%a3') &&
                    !href.includes('臺灣') &&
                    !href.includes('%e4%b8%ad%e8%8f%af%e8%81%b7%e6%a3%92') &&
                    !href.includes('36680')) continue;

                const paras = Array.from(link.querySelectorAll('p'))
                    .map(p => p.textContent.trim())
                    .filter(t => t.length > 0);
                if (paras.length < 2) continue;
                const homeTeam = paras[0];
                const awayTeam = paras[1];

                // Walk up DOM to find container with ≥2 不讓分 checkboxes
                let container = link.parentElement;
                let cbs = [];
                for (let i = 0; i < 6; i++) {
                    if (!container) break;
                    cbs = Array.from(container.querySelectorAll(
                        '[role="checkbox"][aria-label*="不讓分"]'
                    ));
                    if (cbs.length >= 2) break;
                    container = container.parentElement;
                }

                let homeOdds = null, awayOdds = null;
                for (const cb of cbs) {
                    const cl = (cb.getAttribute('aria-label') || '').replace(/\\r/g, '').trim();
                    const om = cl.match(/不讓分\\s*-\\s*(.+?)\\s*-\\s*odds\\s*([0-9.]+)/);
                    if (!om) continue;
                    const team = om[1].trim();
                    const odds = parseFloat(om[2]);
                    if (team === homeTeam) homeOdds = odds;
                    else if (team === awayTeam) awayOdds = odds;
                }

                if (homeOdds && awayOdds)
                    results.push({home: homeTeam, away: awayTeam,
                                  homeOdds, awayOdds});
            }
            return results;
        }""")

        browser.close()

    odds_map: dict[tuple[str, str], tuple[float, float]] = {}
    cpbl_count = 0
    for g in games:
        home_code = ZH_TO_CODE.get(g["home"])
        away_code = ZH_TO_CODE.get(g["away"])
        if home_code and away_code:
            odds_map[(home_code, away_code)] = (g["homeOdds"], g["awayOdds"])
            cpbl_count += 1
        else:
            print(f"  WARN: unrecognised team pair: {g['away']} @ {g['home']}")

    print(f"  SportLottery: {len(games)} total games, {cpbl_count} CPBL matched\n")
    return odds_map


# ---------------------------------------------------------------------------
# Main (standalone CLI)
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--url", default=SPORTSLOTTERY_URL)
    parser.add_argument("--save", action="store_true",
                        help="Save report to cpbl_ev_YYYYMMDD.md")
    args = parser.parse_args()

    print(f"=== CPBL Betting EV  {args.date} ===\n")

    # Import here to avoid circular dependency when imported from run_dashboard
    from predict_today import (
        DB_PATH, TEAM_NAMES, load_data, train_and_predict,
    )
    import sqlite3

    target_date = date.fromisoformat(args.date)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        train_rows, pred_games = load_data(conn, target_date)
    finally:
        conn.close()

    if not pred_games:
        print(f"No scheduled CPBL games found for {args.date}.")
        return

    print(f"Training on {len(train_rows)} rows, predicting {len(pred_games)} games...")
    predictions = train_and_predict(train_rows, pred_games)
    print(f"  {len(predictions)} predictions.\n")

    print("Fetching odds from Taiwan Sports Lottery...")
    odds_map = fetch_sportslottery_odds_cpbl(args.url)

    results = []
    for pred in predictions:
        home_code = pred["home_team"]
        away_code = pred["vis_team"]
        prob_home = float(pred["prob_home_win"])
        prob_away = 1.0 - prob_home

        odds = odds_map.get((home_code, away_code))
        if odds is None:
            results.append({
                "home_code": home_code, "away_code": away_code,
                "home_name": TEAM_NAMES.get(home_code, home_code),
                "away_name": TEAM_NAMES.get(away_code, away_code),
                "prob_home": prob_home,
                "odds": None,
            })
            continue

        dec_home, dec_away = odds
        fair_h, fair_a = vig_free_prob(dec_home, dec_away)

        results.append({
            "home_code": home_code, "away_code": away_code,
            "home_name": TEAM_NAMES.get(home_code, home_code),
            "away_name": TEAM_NAMES.get(away_code, away_code),
            "prob_home": prob_home,
            "dec_home": dec_home, "dec_away": dec_away,
            "fair_h": fair_h, "fair_a": fair_a,
            "ev_home": ev(prob_home, dec_home),
            "ev_away": ev(prob_away, dec_away),
            "edge_home": prob_home - fair_h,
            "edge_away": prob_away - fair_a,
        })

    _print_table(results)
    if args.save:
        _save_report(args.date, results)


def _print_table(results: list) -> None:
    print(f"{'Matchup':<28} {'Model':>6} {'Odds':>6} {'EV':>6} {'Edge':>6}  Rec")
    print("─" * 65)

    bets = []
    for r in results:
        matchup = f"{r['away_name']} @ {r['home_name']}"

        if "dec_home" not in r:
            print(f"  {matchup:<28}  (no odds)")
            continue

        prob_home = r["prob_home"]
        prob_away = 1.0 - prob_home

        for side, prob, dec, ev_val, edge in [
            ("HOME", prob_home, r["dec_home"], r["ev_home"], r["edge_home"]),
            ("AWAY", prob_away, r["dec_away"], r["ev_away"], r["edge_away"]),
        ]:
            rec = ""
            if prob >= MIN_CONF and ev_val >= EV_THRESHOLD:
                rec = "BET"
                bets.append((matchup, side, prob, dec, ev_val, edge))
            if ev_val > 0 or side == ("HOME" if prob_home >= 0.5 else "AWAY"):
                marker = " <--" if rec == "BET" else ""
                print(f"  {matchup:<28} {prob:>5.1%}  {dec:>4.2f}  {ev_val:>+5.1%}  {edge:>+5.1%}  {side}{marker}")

    print("─" * 65)
    if bets:
        print("\nBET RECOMMENDATIONS")
        for matchup, side, prob, dec, ev_val, edge in sorted(bets, key=lambda x: -x[4]):
            print(f"  {side} {matchup:<26}  prob={prob:.1%}  odds={dec:.2f}  ev={ev_val:+.1%}  edge={edge:+.1%}")
    else:
        print("\nNo bets meet the threshold today.")


def _save_report(date_str: str, results: list) -> None:
    path = Path(f"cpbl_ev_{date_str.replace('-', '')}.md")
    lines = [
        f"# CPBL Betting EV — {date_str}",
        "*(Odds source: Taiwan Sports Lottery 不讓分)*",
        "",
        "| Away | Home | Side | Model | Odds | EV | Edge | Rec |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for r in results:
        if "dec_home" not in r:
            lines.append(
                f"| {r['away_name']} | {r['home_name']} | — "
                f"| {r['prob_home']:.1%} | N/A | — | — | — |"
            )
            continue
        for side, prob, dec, ev_val, edge in [
            ("HOME", r["prob_home"],       r["dec_home"], r["ev_home"], r["edge_home"]),
            ("AWAY", 1.0 - r["prob_home"], r["dec_away"], r["ev_away"], r["edge_away"]),
        ]:
            if ev_val > 0 or side == ("HOME" if r["prob_home"] >= 0.5 else "AWAY"):
                rec = "**BET**" if (prob >= MIN_CONF and ev_val >= EV_THRESHOLD) else ""
                lines.append(
                    f"| {r['away_name']} | {r['home_name']} | {side} "
                    f"| {prob:.1%} | {dec:.2f} | {ev_val:+.1%} | {edge:+.1%} | {rec} |"
                )
    lines += [
        "",
        f"*EV threshold: {EV_THRESHOLD:+.0%}  |  Min confidence: {MIN_CONF:.0%}*",
        "*EV = model_prob × (decimal − 1) − (1 − model_prob)*",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nSaved → {path}")


if __name__ == "__main__":
    main()
