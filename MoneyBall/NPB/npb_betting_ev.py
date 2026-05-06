"""
npb_betting_ev.py

Scrape NPB moneyline odds from Taiwan Sports Lottery (不讓分),
compare with model predictions, and compute Expected Value (EV).

Usage:
    python npb_betting_ev.py
    python npb_betting_ev.py --date 2026-04-28
    python npb_betting_ev.py --save
    python npb_betting_ev.py --url "https://..." --save

Odds source: https://www-talo-ssb-pr.sportslottery.com.tw/sport/棒球/34731.1
EV formula:  EV = model_prob * (decimal_odds - 1) - (1 - model_prob)
"""

import argparse
import sys
from datetime import date
from pathlib import Path

from playwright.sync_api import sync_playwright

from predict_today_npb import (
    connect,
    load_training_rows,
    load_target_rows,
    validate_features,
    train_models,
    predict_rows,
    TEAM_SHORT,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Navigate directly to the iframe content URL (bypasses cross-origin iframe restriction)
SPORTSLOTTERY_URL = "https://www-talo-ssb-pr.sportslottery.com.tw/sport/棒球/34731.1"

# Taiwan Sports Lottery Chinese team names → NPB team codes used in game_features_npb
ZH_TO_CODE: dict[str, str] = {
    "讀賣巨人":           "g",
    "中日龍":             "d",
    "橫濱海灣之星":       "db",
    "橫濱DeNA海灣之星":   "db",
    "阪神虎":             "t",
    "廣島東洋鯉魚":       "c",
    "廣島鯉魚":           "c",
    "養樂多燕子":         "s",
    "東京養樂多燕子":     "s",
    "福岡軟銀鷹":         "h",
    "日本火腿鬥士":       "f",
    "北海道日本火腿鬥士": "f",
    "歐力士猛牛":         "b",
    "歐力士牛":           "b",
    "樂天金鷲":           "e",
    "楽天金鷲":           "e",
    "東北樂天金鷲":       "e",
    "西武獅":             "l",
    "埼玉西武獅":         "l",
    "羅德海洋":           "m",
    "千葉羅德海洋":       "m",
}

EV_THRESHOLD = 0.03
MIN_CONF = 0.55


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

def fetch_sportslottery_odds_npb(url: str) -> dict[tuple[str, str], tuple[float, float]]:
    """
    Scrape NPB 不讓分 (moneyline) odds from Taiwan Sports Lottery.
    Returns {(home_code, away_code): (dec_home, dec_away)}.

    Strategy: use event links (a[href*="/event/"]) for team names and
    sibling checkboxes for odds. Head to Head buttons are not reliably
    present at page load time. aria-label values may contain \\r characters
    which are stripped before regex matching.
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

        try:
            page.wait_for_selector(
                '[role="checkbox"][aria-label*="不讓分"]', timeout=25000
            )
        except Exception:
            print("  WARNING: Game data not found on page")
            browser.close()
            return {}

        # Use event links (<a href="/sport/.../event/...">): paragraphs[0]=home, [1]=away.
        # Sibling checkboxes carry aria-label "不讓分 - <team>\r - odds X.XX"
        # (note the \r embedded in the label; strip it before matching).
        games = page.evaluate("""() => {
            const results = [];
            const links = Array.from(document.querySelectorAll('a[href*="/event/"]'));
            for (const link of links) {
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
                    // Strip \\r which appears between team name and " - odds"
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
    npb_count = 0
    for g in games:
        home_code = ZH_TO_CODE.get(g["home"])
        away_code = ZH_TO_CODE.get(g["away"])
        if home_code and away_code:
            odds_map[(home_code, away_code)] = (g["homeOdds"], g["awayOdds"])
            npb_count += 1
        # Silently skip MLB teams (they won't be in ZH_TO_CODE)

    print(f"  SportLottery: {len(games)} total games, {npb_count} NPB matched\n")
    return odds_map


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--url", default=SPORTSLOTTERY_URL)
    parser.add_argument("--save", action="store_true",
                        help="Save report to npb_ev_YYYYMMDD.md")
    args = parser.parse_args()

    target_date = date.fromisoformat(args.date)

    print(f"=== NPB Betting EV  {args.date} ===\n")

    conn = connect()
    try:
        train_rows = load_training_rows(conn, target_date)
        validate_features(train_rows)
        target_rows = load_target_rows(conn, target_date, verify=False)
    finally:
        conn.close()

    if not target_rows:
        print("No scheduled NPB games found for this date.")
        print(f"  Tip: python build_game_features_npb.py --year {target_date.year} --include-scheduled")
        return

    print(f"Training model on {len(train_rows)} rows...")
    models = train_models(train_rows)
    predictions = predict_rows(models, target_rows)
    print(f"  {len(predictions)} games to predict.\n")

    print("Fetching odds from Taiwan Sports Lottery...")
    odds_map = fetch_sportslottery_odds_npb(args.url)

    results = []
    for pred in predictions:
        home_code = str(pred.get("home_code") or "")
        away_code = str(pred.get("away_code") or "")
        prob_home = float(pred["prob_home_win"])
        prob_away = 1.0 - prob_home

        odds = odds_map.get((home_code, away_code))
        if odds is None:
            results.append({
                "home_code": home_code, "away_code": away_code,
                "prob_home": prob_home, "route": pred.get("route", ""),
                "odds": None,
            })
            continue

        dec_home, dec_away = odds
        fair_h, fair_a = vig_free_prob(dec_home, dec_away)

        results.append({
            "home_code": home_code, "away_code": away_code,
            "prob_home": prob_home, "route": pred.get("route", ""),
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


def _team(code: str) -> str:
    return TEAM_SHORT.get(code.lower(), code)


def _print_table(results: list) -> None:
    print(f"{'Matchup':<28} {'Model':>6} {'Odds':>6} {'EV':>6} {'Edge':>6}  Rec")
    print("─" * 65)

    bets = []
    for r in results:
        matchup = f"{_team(r['away_code'])} @ {_team(r['home_code'])}"

        if r.get("odds") is None:
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
    path = Path(f"npb_ev_{date_str.replace('-', '')}.md")
    lines = [
        f"# NPB Betting EV — {date_str}",
        "*(Odds source: Taiwan Sports Lottery 不讓分)*",
        "",
        "| Away | Home | Side | Model | Odds | EV | Edge | Rec |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for r in results:
        home = _team(r["home_code"])
        away = _team(r["away_code"])
        if r.get("odds") is None:
            lines.append(f"| {away} | {home} | — | {r['prob_home']:.1%} | N/A | — | — | — |")
            continue
        for side, prob, dec, ev_val, edge in [
            ("HOME", r["prob_home"],       r["dec_home"], r["ev_home"], r["edge_home"]),
            ("AWAY", 1.0 - r["prob_home"], r["dec_away"], r["ev_away"], r["edge_away"]),
        ]:
            if ev_val > 0 or side == ("HOME" if r["prob_home"] >= 0.5 else "AWAY"):
                rec = "**BET**" if (prob >= MIN_CONF and ev_val >= EV_THRESHOLD) else ""
                lines.append(
                    f"| {away} | {home} | {side} "
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
