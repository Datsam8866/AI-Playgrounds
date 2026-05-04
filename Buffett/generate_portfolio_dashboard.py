"""
generate_portfolio_dashboard.py
從 portfolio.sqlite + walk-forward CSV 生成 portfolio_dashboard.html。
"""
import sqlite3
import sys
import json
from datetime import date
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd

ROOT    = Path(__file__).parent
DB_PATH = ROOT / "portfolio.sqlite"
WF_CSV  = ROOT / "Wall Street" / "walkforward_portfolio_beta_constrained_voo_alpha.csv"
OUT     = ROOT / "portfolio_dashboard.html"
TODAY   = str(date.today())

# ── 1. 讀取持倉 ─────────────────────────────────────────────────────────────

def load_holdings():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT h.market, h.ticker, h.name, h.shares, h.avg_cost,
               COALESCE(h.last_price,
                   (SELECT price FROM holdings_snapshot
                    WHERE ticker=h.ticker ORDER BY snapshot_date DESC LIMIT 1)) AS price,
               h.last_price_date, h.currency
        FROM holdings h ORDER BY h.market DESC, h.ticker
    """).fetchall()
    conn.close()
    data = []
    for market, ticker, name, shares, avg_cost, price, price_date, currency in rows:
        cost  = shares * avg_cost if avg_cost else None
        mv    = shares * price if price else None
        pnl   = mv - cost if (mv and cost) else None
        pct   = pnl / cost * 100 if (pnl is not None and cost) else None
        data.append(dict(market=market, ticker=ticker, name=name or ticker,
                         shares=shares, avg_cost=avg_cost, price=price,
                         price_date=price_date, currency=currency,
                         cost=cost, mv=mv, pnl=pnl, pct=pct))
    return data

# ── 2. 讀取模型建議 ──────────────────────────────────────────────────────────

def load_model():
    df = pd.read_csv(WF_CSV)
    row = df[df["period"].str.contains("current", na=False)].iloc[-1]
    weights_str = row["weights"]
    regime  = row.get("regime_label", "unknown")
    stage   = row.get("selection_stage", "unknown")
    p_beat  = row.get("p_beat_voo_calibrated", None)

    weights = {}
    for part in weights_str.split(","):
        part = part.strip()
        if not part:
            continue
        tok = part.rsplit(" ", 1)
        if len(tok) == 2:
            t = tok[0].strip()
            w = float(tok[1].replace("%", "").strip()) / 100
            weights[t] = weights.get(t, 0) + w
    return weights, regime, stage, p_beat

# ── 3. 計算彙總 ─────────────────────────────────────────────────────────────

def summarise(holdings, market):
    h = [x for x in holdings if x["market"] == market]
    total_cost = sum(x["cost"] for x in h if x["cost"])
    total_mv   = sum(x["mv"]   for x in h if x["mv"])
    total_pnl  = total_mv - total_cost if (total_mv and total_cost) else 0
    total_pct  = total_pnl / total_cost * 100 if total_cost else 0
    return total_cost, total_mv, total_pnl, total_pct, h

# ── 4. 格式化工具 ─────────────────────────────────────────────────────────────

def fmt(v, prefix="", suffix="", decimals=2):
    if v is None:
        return "—"
    return f"{prefix}{v:,.{decimals}f}{suffix}"

def pct_class(v):
    if v is None:
        return ""
    return "pos" if v >= 0 else "neg"

# ── 5. 生成 HTML ─────────────────────────────────────────────────────────────

def rows_html(holdings):
    out = []
    for h in holdings:
        pc = pct_class(h["pct"])
        pnl_str = fmt(h["pnl"], prefix="+$" if (h["pnl"] or 0) >= 0 else "$")
        pct_str = fmt(h["pct"], suffix="%")
        out.append(f"""
        <tr>
          <td><strong>{h['ticker']}</strong></td>
          <td>{h['name']}</td>
          <td class="num">{fmt(h['shares'], decimals=0)}</td>
          <td class="num">{fmt(h['avg_cost'], prefix='$')}</td>
          <td class="num">{fmt(h['price'], prefix='$')}</td>
          <td class="num">{fmt(h['mv'], prefix='$')}</td>
          <td class="num {pc}">{pnl_str}</td>
          <td class="num {pc}">{pct_str}</td>
        </tr>""")
    return "\n".join(out)

def build_html(holdings, model_w, regime, stage, p_beat):
    us_cost, us_mv, us_pnl, us_pct, us_h = summarise(holdings, "US")
    tw_cost, tw_mv, tw_pnl, tw_pct, tw_h = summarise(holdings, "TW")

    # 實際美股配置（圓餅圖）
    us_total_mv = sum(x["mv"] for x in us_h if x["mv"])
    pie_labels  = [x["ticker"] for x in us_h if x["mv"]]
    pie_values  = [round(x["mv"] / us_total_mv * 100, 1) for x in us_h if x["mv"]]

    # 模型 vs 實際（長條圖）— 排除 VOO，聚焦衛星股
    actual_w = {x["ticker"]: x["mv"] / us_total_mv for x in us_h if x["mv"]}
    bar_tickers = sorted(t for t in set(model_w) | set(actual_w) if t != "VOO")
    bar_model  = [round(model_w.get(t, 0) * 100, 1) for t in bar_tickers]
    bar_actual = [round(actual_w.get(t, 0) * 100, 1) for t in bar_tickers]

    # Regime 顏色
    regime_color = {"risk_on": "#22c55e", "caution": "#f59e0b", "risk_off": "#ef4444"}.get(regime, "#6b7280")
    regime_label = regime.upper().replace("_", " ")

    p_beat_str = f"{p_beat*100:.1f}%" if p_beat else "—"
    us_pct_cls = "pos" if us_pct >= 0 else "neg"
    tw_pct_cls = "pos" if tw_pct >= 0 else "neg"

    return f"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Portfolio Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  :root {{
    --bg: #0f172a; --surface: #1e293b; --border: #334155;
    --text: #e2e8f0; --muted: #94a3b8;
    --pos: #22c55e; --neg: #ef4444; --accent: #38bdf8;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text); font-family: 'Segoe UI', system-ui, sans-serif; font-size: 14px; }}
  header {{ padding: 20px 28px; border-bottom: 1px solid var(--border); display: flex; justify-content: space-between; align-items: center; }}
  header h1 {{ font-size: 20px; font-weight: 700; color: var(--accent); letter-spacing: .5px; }}
  header span {{ color: var(--muted); font-size: 12px; }}
  .container {{ max-width: 1200px; margin: 0 auto; padding: 24px 20px; }}
  .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 14px; margin-bottom: 24px; }}
  .card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 16px 20px; }}
  .card-label {{ color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: .8px; margin-bottom: 6px; }}
  .card-value {{ font-size: 22px; font-weight: 700; }}
  .card-sub {{ font-size: 12px; color: var(--muted); margin-top: 4px; }}
  .regime-badge {{ display: inline-block; padding: 4px 14px; border-radius: 20px; font-weight: 700; font-size: 13px; background: {regime_color}22; color: {regime_color}; border: 1px solid {regime_color}66; }}
  .pos {{ color: var(--pos); }}
  .neg {{ color: var(--neg); }}
  .charts {{ display: grid; grid-template-columns: 1fr 1.6fr; gap: 16px; margin-bottom: 24px; }}
  .chart-box {{ background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 18px; }}
  .chart-box h3 {{ font-size: 13px; color: var(--muted); margin-bottom: 14px; text-transform: uppercase; letter-spacing: .6px; }}
  .chart-wrap {{ position: relative; }}
  table {{ width: 100%; border-collapse: collapse; background: var(--surface); border-radius: 10px; overflow: hidden; margin-bottom: 24px; }}
  thead th {{ background: #263348; color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: .6px; padding: 10px 14px; text-align: left; border-bottom: 1px solid var(--border); }}
  tbody td {{ padding: 10px 14px; border-bottom: 1px solid #1e293b; }}
  tbody tr:last-child td {{ border-bottom: none; }}
  tbody tr:hover {{ background: #263348; }}
  .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  h2 {{ font-size: 15px; font-weight: 600; margin-bottom: 12px; color: var(--accent); }}
  @media(max-width:700px) {{ .charts {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<header>
  <h1>📊 Portfolio Dashboard</h1>
  <span>更新日期：{TODAY}</span>
</header>
<div class="container">

  <!-- 摘要卡 -->
  <div class="cards">
    <div class="card">
      <div class="card-label">美股市值</div>
      <div class="card-value">${us_mv:,.0f}</div>
      <div class="card-sub">成本 ${us_cost:,.0f}</div>
    </div>
    <div class="card">
      <div class="card-label">美股損益</div>
      <div class="card-value {us_pct_cls}">${us_pnl:+,.0f}</div>
      <div class="card-sub {us_pct_cls}">{us_pct:+.1f}%</div>
    </div>
    <div class="card">
      <div class="card-label">台股市值</div>
      <div class="card-value">NT${tw_mv:,.0f}</div>
      <div class="card-sub">成本 NT${tw_cost:,.0f}</div>
    </div>
    <div class="card">
      <div class="card-label">台股損益</div>
      <div class="card-value {tw_pct_cls}">NT${tw_pnl:+,.0f}</div>
      <div class="card-sub {tw_pct_cls}">{tw_pct:+.1f}%</div>
    </div>
    <div class="card">
      <div class="card-label">市場狀態</div>
      <div class="card-value" style="font-size:16px;margin-top:4px">
        <span class="regime-badge">{regime_label}</span>
      </div>
      <div class="card-sub">P(&gt;VOO) {p_beat_str} · {stage}</div>
    </div>
  </div>

  <!-- 圖表 -->
  <div class="charts">
    <div class="chart-box">
      <h3>美股實際配置</h3>
      <div class="chart-wrap" style="height:260px">
        <canvas id="pieChart"></canvas>
      </div>
    </div>
    <div class="chart-box">
      <h3>模型建議 vs 實際配置（衛星股，不含 VOO）</h3>
      <div class="chart-wrap" style="height:260px">
        <canvas id="barChart"></canvas>
      </div>
    </div>
  </div>

  <!-- 美股持倉表 -->
  <h2>🇺🇸 美股持倉（USD）</h2>
  <table>
    <thead><tr>
      <th>代號</th><th>名稱</th><th class="num">股數</th>
      <th class="num">均價</th><th class="num">現價</th>
      <th class="num">市值</th><th class="num">損益</th><th class="num">損益%</th>
    </tr></thead>
    <tbody>{rows_html(us_h)}</tbody>
  </table>

  <!-- 台股持倉表 -->
  <h2>🇹🇼 台股持倉（TWD）</h2>
  <table>
    <thead><tr>
      <th>代號</th><th>名稱</th><th class="num">股數</th>
      <th class="num">均價</th><th class="num">現價</th>
      <th class="num">市值</th><th class="num">損益</th><th class="num">損益%</th>
    </tr></thead>
    <tbody>{rows_html(tw_h)}</tbody>
  </table>

</div>

<script>
const COLORS = [
  '#38bdf8','#818cf8','#34d399','#fb923c','#f472b6',
  '#a78bfa','#4ade80','#facc15','#60a5fa','#f87171','#2dd4bf'
];

// 圓餅圖
new Chart(document.getElementById('pieChart'), {{
  type: 'doughnut',
  data: {{
    labels: {json.dumps(pie_labels)},
    datasets: [{{ data: {json.dumps(pie_values)}, backgroundColor: COLORS,
      borderColor: '#0f172a', borderWidth: 2 }}]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{
      legend: {{ position: 'right', labels: {{ color: '#94a3b8', font: {{ size: 11 }} }} }},
      tooltip: {{ callbacks: {{ label: ctx => ` ${{ctx.label}}: ${{ctx.parsed}}%` }} }}
    }}
  }}
}});

// 長條圖
new Chart(document.getElementById('barChart'), {{
  type: 'bar',
  data: {{
    labels: {json.dumps(bar_tickers)},
    datasets: [
      {{ label: '模型建議 %', data: {json.dumps(bar_model)},
         backgroundColor: '#38bdf855', borderColor: '#38bdf8', borderWidth: 1 }},
      {{ label: '實際持倉 %', data: {json.dumps(bar_actual)},
         backgroundColor: '#818cf855', borderColor: '#818cf8', borderWidth: 1 }}
    ]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{ legend: {{ labels: {{ color: '#94a3b8' }} }} }},
    scales: {{
      x: {{ ticks: {{ color: '#94a3b8', font: {{ size: 10 }} }}, grid: {{ color: '#1e293b' }} }},
      y: {{ ticks: {{ color: '#94a3b8', callback: v => v + '%' }}, grid: {{ color: '#334155' }} }}
    }}
  }}
}});
</script>
</body>
</html>"""

# ── 6. 執行 ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("讀取持倉資料...")
    holdings = load_holdings()

    print("讀取模型建議...")
    model_w, regime, stage, p_beat = load_model()

    print("生成 HTML...")
    html = build_html(holdings, model_w, regime, stage, p_beat)
    OUT.write_text(html, encoding="utf-8")
    print(f"✅ 已輸出：{OUT}")
