"""
mlb_betting_ev.py

Scrape MLB moneyline odds from Taiwan Sports Lottery (不讓分),
compare with model predictions, and compute Expected Value (EV).

Usage:
    python mlb_betting_ev.py
    python mlb_betting_ev.py --date 2026-04-28
    python mlb_betting_ev.py --url "https://..." --save

Odds source: https://www.sportslottery.com.tw/sportsbook/sport/%E6%A3%92%E7%90%83/34731.1
EV formula:  EV = model_prob * (decimal_odds - 1) - (1 - model_prob)
"""

import argparse
import sys
from datetime import date, datetime
from pathlib import Path
from collections import defaultdict

import requests
import sqlite3
from playwright.sync_api import sync_playwright

from predict_mlb_today import (
    build_live_state, compute_game_features, fetch_today_games,
    ELO_BASE, ELO_REGRESSION,
)
from train_mlb_model import build_rows, fit_models, soft_predict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DB_PATH = Path("mlb.sqlite")

# Navigate directly to the iframe content URL (bypasses cross-origin iframe restriction)
SPORTSLOTTERY_URL = "https://www-talo-ssb-pr.sportslottery.com.tw/sport/棒球/34731.1"

# Taiwan Sports Lottery Chinese team names → MLB English names
ZH_TO_EN = {
    "亞利桑納響尾蛇": "Arizona Diamondbacks",
    "亞利桑那響尾蛇": "Arizona Diamondbacks",
    "亞特蘭大勇士":   "Atlanta Braves",
    "巴爾的摩金鶯":   "Baltimore Orioles",
    "波士頓紅襪":     "Boston Red Sox",
    "芝加哥小熊":     "Chicago Cubs",
    "芝加哥白襪":     "Chicago White Sox",
    "辛辛那提紅人":   "Cincinnati Reds",
    "辛辛那堤紅人":   "Cincinnati Reds",
    "亞歷桑那響尾蛇": "Arizona Diamondbacks",
    "克里夫蘭守護者": "Cleveland Guardians",
    "科羅拉多落磯":   "Colorado Rockies",
    "底特律老虎":     "Detroit Tigers",
    "休士頓太空人":   "Houston Astros",
    "堪薩斯市皇家":   "Kansas City Royals",
    "洛杉磯天使":     "Los Angeles Angels",
    "洛杉磯道奇":     "Los Angeles Dodgers",
    "邁阿密馬林魚":   "Miami Marlins",
    "密爾瓦基釀酒人": "Milwaukee Brewers",
    "明尼蘇達雙城":   "Minnesota Twins",
    "紐約大都會":     "New York Mets",
    "紐約洋基":       "New York Yankees",
    "奧克蘭運動家":   "Oakland Athletics",
    "薩克拉門托運動家": "Oakland Athletics",
    "運動家":         "Oakland Athletics",
    "堪薩斯皇家":     "Kansas City Royals",
    "費城人":         "Philadelphia Phillies",
    "費城費城人":     "Philadelphia Phillies",
    "匹茲堡海盜":     "Pittsburgh Pirates",
    "聖地牙哥教士":   "San Diego Padres",
    "舊金山巨人":     "San Francisco Giants",
    "西雅圖水手":     "Seattle Mariners",
    "聖路易紅雀":     "St. Louis Cardinals",
    "坦帕灣光芒":     "Tampa Bay Rays",
    "德州遊騎兵":     "Texas Rangers",
    "多倫多藍鳥":     "Toronto Blue Jays",
    "華盛頓國民":     "Washington Nationals",
}

EV_THRESHOLD = 0.03
MIN_CONF     = 0.52


# ---------------------------------------------------------------------------
# Odds helpers
# ---------------------------------------------------------------------------

def vig_free_prob(dec_home: float, dec_away: float):
    """Remove vig and return (fair_prob_home, fair_prob_away)."""
    raw_h = 1 / dec_home
    raw_a = 1 / dec_away
    total = raw_h + raw_a
    return raw_h / total, raw_a / total


def ev(model_prob: float, decimal_odds: float) -> float:
    """EV per 1-unit stake."""
    return model_prob * (decimal_odds - 1) - (1 - model_prob)


# ---------------------------------------------------------------------------
# Scrape Taiwan Sports Lottery
# ---------------------------------------------------------------------------

def fetch_sportslottery_odds(url: str) -> dict:
    """
    Scrape 不讓分 (moneyline) odds from Taiwan Sports Lottery.
    Returns {(home_en, away_en): (dec_home, dec_away)}.

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

        # Use event links (<a href="/sport/.../event/...">): paragraphs[0]=home, [2]=away.
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


    odds_map = {}
    for g in games:
        home_en = ZH_TO_EN.get(g["home"])
        away_en = ZH_TO_EN.get(g["away"])
        if home_en and away_en:
            odds_map[(home_en, away_en)] = (g["homeOdds"], g["awayOdds"])
        elif home_en or away_en:
            # Only warn if one team mapped (likely a MLB team name variant issue)
            unmapped = [n for n, e in [(g["home"], home_en), (g["away"], away_en)] if not e]
            print(f"  UNMAPPED: {', '.join(unmapped)}")

    print(f"  SportLottery: {len(games)} games found, {len(odds_map)} matched\n")
    return odds_map


def match_odds(game_home: str, game_away: str, odds_map: dict):
    """Fuzzy-match English team names."""
    if (game_home, game_away) in odds_map:
        return odds_map[(game_home, game_away)]
    for (oh, oa), v in odds_map.items():
        if (game_home.split()[-1] in oh or oh.split()[-1] in game_home) and \
           (game_away.split()[-1] in oa or oa.split()[-1] in game_away):
            return v
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url", default=SPORTSLOTTERY_URL,
        help="Taiwan Sports Lottery baseball page URL"
    )
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--save", action="store_true",
                        help="Save report to mlb_ev_YYYYMMDD.md")
    args = parser.parse_args()

    pred_date   = datetime.fromisoformat(args.date).date()
    season_year = pred_date.year

    print(f"=== MLB Betting EV  {args.date} ===\n")

    conn    = sqlite3.connect(DB_PATH)
    session = requests.Session()
    session.headers.update({"User-Agent": "MLB-EV/1.0", "Accept": "application/json"})

    try:
        # 1. Train model
        print("Training model...")
        all_rows = build_rows(conn)
        models   = fit_models(all_rows)
        print(f"  {len(all_rows)} games trained.\n")

        # 2. Rolling state
        print("Building rolling state...")
        state = build_live_state(conn)
        last_season = state["current_season"]
        if season_year > last_season:
            for tid in list(state["current_elo"].keys()):
                state["current_elo"][tid] = (
                    ELO_BASE * ELO_REGRESSION + state["current_elo"][tid] * (1 - ELO_REGRESSION)
                )
            state["team_season_cnt"] = defaultdict(int)

        # 3. Model predictions
        print("Fetching schedule...")
        games = fetch_today_games(session, args.date)
        if not games:
            print("No games today.")
            return

        preds = {}
        for g in games:
            if not g["home_id"] or not g["vis_id"]:
                continue
            feat = compute_game_features(
                state, g["home_id"], g["vis_id"], pred_date, season_year,
                home_pid=g["home_pid"], vis_pid=g["vis_pid"],
            )
            prob, label = soft_predict(models, feat)
            preds[g["game_pk"]] = {"game": g, "prob_home": prob, "label": label}

        # 4. Scrape odds
        print("Fetching odds from Taiwan Sports Lottery...")
        odds_map = fetch_sportslottery_odds(args.url)

        # 5. EV analysis
        results = []
        for gp, p in preds.items():
            g         = p["game"]
            prob_home = p["prob_home"]
            prob_away = 1 - prob_home

            odds = match_odds(g["home_name"], g["vis_name"], odds_map)
            if odds is None:
                results.append({**p, "odds": None, "ev_home": None, "ev_away": None,
                                 "edge_home": None, "edge_away": None})
                continue

            dec_home, dec_away = odds
            fair_h, fair_a = vig_free_prob(dec_home, dec_away)
            ev_h = ev(prob_home, dec_home)
            ev_a = ev(prob_away, dec_away)

            results.append({
                "game":      g,
                "prob_home": prob_home,
                "label":     p["label"],
                "dec_home":  dec_home,
                "dec_away":  dec_away,
                "fair_h":    fair_h,
                "fair_a":    fair_a,
                "ev_home":   ev_h,
                "ev_away":   ev_a,
                "edge_home": prob_home - fair_h,
                "edge_away": prob_away - fair_a,
            })

        _print_table(results)
        if args.save:
            _save_report(args.date, results)

    finally:
        conn.close()
        session.close()


def _print_table(results):
    print(f"{'Matchup':<44} {'Model':>6} {'Odds':>6} {'EV':>6} {'Edge':>6}  Rec")
    print("─" * 80)

    bets = []
    for r in results:
        g = r["game"]
        matchup = f"{g['vis_name'][:20]} @ {g['home_name'][:20]}"

        if r.get("ev_home") is None:
            print(f"  {matchup:<44}  (no odds)")
            continue

        prob_home = r["prob_home"]
        prob_away = 1 - prob_home

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
                print(f"  {matchup:<44} {prob:>5.1%}  {dec:>4.2f}  {ev_val:>+5.1%}  {edge:>+5.1%}  {side}{marker}")

    print("─" * 80)
    if bets:
        print("\nBET RECOMMENDATIONS")
        for matchup, side, prob, dec, ev_val, edge in sorted(bets, key=lambda x: -x[4]):
            print(f"  {side} {matchup:<42}  prob={prob:.1%}  odds={dec:.2f}  ev={ev_val:+.1%}  edge={edge:+.1%}")
    else:
        print("\nNo bets meet the threshold today.")


def _save_report(date_str, results):
    path = Path(f"mlb_ev_{date_str.replace('-','')}.md")
    lines = [
        f"# MLB Betting EV — {date_str}",
        f"*(Odds source: Taiwan Sports Lottery 不讓分)*",
        "",
        "| Away | Home | Side | Model | Odds | EV | Edge | Rec |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for r in results:
        g = r["game"]
        if r.get("ev_home") is None:
            lines.append(
                f"| {g['vis_name']} | {g['home_name']} | — "
                f"| {r['prob_home']:.1%} | N/A | — | — | — |"
            )
            continue
        for side, prob, dec, ev_val, edge in [
            ("HOME", r["prob_home"],     r["dec_home"], r["ev_home"], r["edge_home"]),
            ("AWAY", 1-r["prob_home"],   r["dec_away"], r["ev_away"], r["edge_away"]),
        ]:
            if ev_val > 0 or side == ("HOME" if r["prob_home"] >= 0.5 else "AWAY"):
                rec = "**BET**" if (prob >= MIN_CONF and ev_val >= EV_THRESHOLD) else ""
                lines.append(
                    f"| {g['vis_name']} | {g['home_name']} | {side} "
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
