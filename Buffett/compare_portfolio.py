"""
compare_portfolio.py
比較「模型建議配置」vs「實際持倉」，顯示差距與調整建議。
"""
import sqlite3
import sys
import warnings
from pathlib import Path

import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DB_PATH = Path(__file__).parent / "portfolio.sqlite"
WF_CSV  = Path(__file__).parent / "Wall Street" / "walkforward_portfolio_beta_constrained_voo_alpha.csv"


def parse_model_weights(weights_str: str) -> dict[str, float]:
    """解析 'VOO 80.0%, LITE 2.0%, ...' → {ticker: weight}"""
    result = {}
    for part in weights_str.split(","):
        part = part.strip()
        if not part:
            continue
        tok = part.rsplit(" ", 1)
        if len(tok) == 2:
            ticker = tok[0].strip()
            pct = float(tok[1].replace("%", "").strip()) / 100
            result[ticker] = result.get(ticker, 0) + pct
    return result


def get_model_recommendation() -> tuple[dict[str, float], str, str]:
    """從 walk-forward CSV 取最新期 2026Q2_current 的模型建議。"""
    df = pd.read_csv(WF_CSV)
    latest = df[df["period"].str.contains("current", na=False)].iloc[-1]
    weights = parse_model_weights(latest["weights"])
    regime  = latest.get("regime_label", "unknown")
    stage   = latest.get("selection_stage", "unknown")
    return weights, regime, stage


def get_actual_weights() -> dict[str, float]:
    """從 portfolio.sqlite 取美股實際持倉，並用最近快照估算市值權重。"""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT h.ticker, h.shares,
               COALESCE(h.last_price,
                   (SELECT price FROM holdings_snapshot
                    WHERE ticker=h.ticker AND market='US'
                    ORDER BY snapshot_date DESC LIMIT 1)) AS price
        FROM holdings h
        WHERE h.market = 'US'
    """).fetchall()
    conn.close()

    total_mv = sum(r[1] * r[2] for r in rows if r[2])
    if total_mv == 0:
        return {}
    return {r[0]: (r[1] * r[2]) / total_mv for r in rows if r[2]}


def print_comparison(model: dict[str, float], actual: dict[str, float],
                     regime: str, stage: str):
    all_tickers = sorted(set(model) | set(actual))

    print(f"\n{'='*72}")
    print(f"  模型 vs 實際持倉比較")
    print(f"  Regime: {regime.upper()}  |  Stage: {stage}")
    print(f"{'='*72}")
    print(f"  {'代號':<8} {'模型%':>7} {'實際%':>7} {'差距':>8}  {'狀態'}")
    print(f"  {'─'*60}")

    over, under, missing, extra = [], [], [], []

    for t in all_tickers:
        m = model.get(t, 0.0)
        a = actual.get(t, 0.0)
        diff = a - m
        if m > 0 and a == 0:
            status = "❌ 未持有"
            missing.append(t)
        elif m == 0 and a > 0:
            status = "➕ 模型外"
            extra.append(t)
        elif diff > 0.03:
            status = "⬆ 超配"
            over.append(t)
        elif diff < -0.03:
            status = "⬇ 低配"
            under.append(t)
        else:
            status = "✅ 接近"
        print(f"  {t:<8} {m*100:>6.1f}%  {a*100:>6.1f}%  {diff*100:>+7.1f}%  {status}")

    print(f"  {'─'*60}")

    print(f"\n  模型建議 P(>VOO) 最新：見 Wall Street/README.md")
    print(f"\n  【調整建議】")
    if missing:
        print(f"  • 模型推薦但未持有（考慮加碼）：{', '.join(missing)}")
    if extra:
        print(f"  • 持有但模型未推薦（考慮減碼）：{', '.join(extra)}")
    if under:
        print(f"  • 明顯低配（>3% 差距）：{', '.join(under)}")
    if over:
        print(f"  • 明顯超配（>3% 差距）：{', '.join(over)}")
    if not (missing or extra or under or over):
        print(f"  • 持倉與模型高度一致，無需調整")
    print(f"{'='*72}\n")


if __name__ == "__main__":
    model_weights, regime, stage = get_model_recommendation()
    actual_weights = get_actual_weights()
    print_comparison(model_weights, actual_weights, regime, stage)
