"""
generate_portfolio_dashboard.py
生成三 Tab Portfolio Dashboard HTML。
Tab 1: Overview  |  Tab 2: 美股  |  Tab 3: 台股
"""
import sqlite3, sys, json
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

# ── helpers ──────────────────────────────────────────────────────────────────

COLORS = ["#38bdf8","#818cf8","#34d399","#fb923c","#f472b6",
          "#a78bfa","#4ade80","#facc15","#60a5fa","#f87171","#2dd4bf","#e879f9"]

def fmt(v, prefix="", suffix="", d=2):
    return "—" if v is None else f"{prefix}{v:,.{d}f}{suffix}"

def pc(v):
    return "pos" if (v or 0) >= 0 else "neg"

# ── data loading ──────────────────────────────────────────────────────────────

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
    out = []
    for market, ticker, name, shares, avg_cost, price, price_date, currency in rows:
        cost = shares * avg_cost if avg_cost else None
        mv   = shares * price   if price    else None
        pnl  = mv - cost        if (mv and cost) else None
        pct  = pnl / cost * 100 if (pnl is not None and cost) else None
        out.append(dict(market=market, ticker=ticker, name=name or ticker,
                        shares=shares, avg_cost=avg_cost, price=price,
                        price_date=price_date, currency=currency,
                        cost=cost, mv=mv, pnl=pnl, pct=pct))
    return out

def load_history():
    """回傳 {market: [{date, mv, pnl}]} 按日期排序"""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT snapshot_date, market,
               SUM(market_value) as mv,
               SUM(pnl)          as pnl
        FROM holdings_snapshot
        WHERE market_value IS NOT NULL
        GROUP BY snapshot_date, market
        ORDER BY snapshot_date
    """).fetchall()
    conn.close()
    hist = {"US": [], "TW": []}
    for d, m, mv, pnl in rows:
        if m in hist:
            hist[m].append({"date": d[:10], "mv": round(mv, 2),
                             "pnl": round(pnl, 2) if pnl else 0})
    return hist

def load_model():
    df  = pd.read_csv(WF_CSV)
    row = df[df["period"].str.contains("current", na=False)].iloc[-1]
    w   = {}
    for part in str(row["weights"]).split(","):
        tok = part.strip().rsplit(" ", 1)
        if len(tok) == 2:
            t, v = tok[0].strip(), float(tok[1].replace("%","")) / 100
            w[t] = w.get(t, 0) + v
    return (w,
            row.get("regime_label", "unknown"),
            row.get("selection_stage", "unknown"),
            row.get("p_beat_voo_calibrated", None))

def summarise(holdings, market):
    h  = [x for x in holdings if x["market"] == market]
    tc = sum(x["cost"] for x in h if x["cost"])
    tm = sum(x["mv"]   for x in h if x["mv"])
    tp = tm - tc if (tm and tc) else 0
    pp = tp / tc * 100 if tc else 0
    return tc, tm, tp, pp, h

# ── HTML fragments ────────────────────────────────────────────────────────────

def holdings_table(hlist):
    rows = []
    for h in hlist:
        pnl_s = fmt(h["pnl"], prefix=("+" if (h["pnl"] or 0) >= 0 else ""), suffix="", d=2)
        pct_s = fmt(h["pct"], suffix="%")
        rows.append(f"""<tr>
          <td><strong>{h['ticker']}</strong></td><td class="muted">{h['name']}</td>
          <td class="num">{fmt(h['shares'],d=0)}</td>
          <td class="num">{fmt(h['avg_cost'],prefix='$')}</td>
          <td class="num">{fmt(h['price'],prefix='$')}</td>
          <td class="num">{fmt(h['mv'],prefix='$')}</td>
          <td class="num {pc(h['pnl'])}">{pnl_s}</td>
          <td class="num {pc(h['pct'])}">{pct_s}</td>
        </tr>""")
    return "\n".join(rows)

def card(label, value, sub="", sub_class=""):
    return f"""<div class="card">
      <div class="card-label">{label}</div>
      <div class="card-value">{value}</div>
      {f'<div class="card-sub {sub_class}">{sub}</div>' if sub else ''}
    </div>"""

# ── main builder ──────────────────────────────────────────────────────────────

def build():
    holdings            = load_holdings()
    hist                = load_history()
    model_w, regime, stage, p_beat = load_model()

    us_tc, us_tm, us_tp, us_pp, us_h = summarise(holdings, "US")
    tw_tc, tw_tm, tw_tp, tw_pp, tw_h = summarise(holdings, "TW")

    # regime
    rc = {"risk_on":"#22c55e","caution":"#f59e0b","risk_off":"#ef4444"}.get(regime,"#6b7280")
    rl = regime.upper().replace("_"," ")
    pb = f"{p_beat*100:.1f}%" if p_beat else "—"

    # US pie
    us_total_mv = sum(x["mv"] for x in us_h if x["mv"]) or 1
    pie_l = [x["ticker"] for x in us_h if x["mv"]]
    pie_v = [round(x["mv"]/us_total_mv*100,1) for x in us_h if x["mv"]]

    # US bar (model vs actual, satellites only)
    actual_w   = {x["ticker"]: x["mv"]/us_total_mv for x in us_h if x["mv"]}
    bar_tickers = sorted(t for t in set(model_w)|set(actual_w) if t != "VOO")
    bar_model  = [round(model_w.get(t,0)*100,1) for t in bar_tickers]
    bar_actual = [round(actual_w.get(t,0)*100,1) for t in bar_tickers]

    # TW pie
    tw_total_mv = sum(x["mv"] for x in tw_h if x["mv"]) or 1
    tw_pie_l = [x["ticker"] for x in tw_h if x["mv"]]
    tw_pie_v = [round(x["mv"]/tw_total_mv*100,1) for x in tw_h if x["mv"]]

    # history JSON
    us_dates = json.dumps([d["date"] for d in hist["US"]])
    us_mvs   = json.dumps([d["mv"]   for d in hist["US"]])
    us_pnls  = json.dumps([d["pnl"]  for d in hist["US"]])
    tw_dates = json.dumps([d["date"] for d in hist["TW"]])
    tw_mvs   = json.dumps([d["mv"]   for d in hist["TW"]])
    tw_pnls  = json.dumps([d["pnl"]  for d in hist["TW"]])

    # overview combined dates
    all_dates = sorted(set(d["date"] for d in hist["US"]+hist["TW"]))
    us_mv_map = {d["date"]:d["mv"]  for d in hist["US"]}
    tw_mv_map = {d["date"]:d["mv"]  for d in hist["TW"]}
    ov_dates  = json.dumps(all_dates)
    ov_us_mvs = json.dumps([us_mv_map.get(d) for d in all_dates])
    ov_tw_mvs = json.dumps([tw_mv_map.get(d) for d in all_dates])

    update_cmd = "python -X utf8 update_portfolio.py &amp;&amp; python -X utf8 generate_portfolio_dashboard.py &amp;&amp; git add Buffett/portfolio_dashboard.html &amp;&amp; git commit -m &quot;更新 Dashboard&quot; &amp;&amp; git push"

    html = f"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Portfolio Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
:root{{--bg:#0f172a;--sf:#1e293b;--bd:#334155;--tx:#e2e8f0;--mu:#94a3b8;--ac:#38bdf8;--pos:#22c55e;--neg:#ef4444;}}
*{{box-sizing:border-box;margin:0;padding:0;}}
body{{background:var(--bg);color:var(--tx);font-family:'Segoe UI',system-ui,sans-serif;font-size:14px;min-height:100vh;}}
/* header */
header{{padding:14px 24px;border-bottom:1px solid var(--bd);display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap;}}
header h1{{font-size:18px;font-weight:700;color:var(--ac);}}
.header-right{{display:flex;align-items:center;gap:10px;}}
.updated{{color:var(--mu);font-size:12px;}}
.btn-update{{background:#1e3a5f;border:1px solid var(--ac);color:var(--ac);padding:6px 14px;border-radius:6px;cursor:pointer;font-size:12px;font-weight:600;}}
.btn-update:hover{{background:#1e4a7f;}}
/* tabs */
.tabs{{display:flex;border-bottom:1px solid var(--bd);background:var(--sf);}}
.tab{{padding:12px 24px;cursor:pointer;color:var(--mu);font-size:13px;font-weight:600;border-bottom:2px solid transparent;transition:.15s;user-select:none;}}
.tab.active{{color:var(--ac);border-bottom-color:var(--ac);}}
.tab:hover:not(.active){{color:var(--tx);}}
/* panels */
.panel{{display:none;padding:20px 24px;max-width:1200px;margin:0 auto;}}
.panel.active{{display:block;}}
/* cards */
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin-bottom:20px;}}
.card{{background:var(--sf);border:1px solid var(--bd);border-radius:10px;padding:14px 18px;}}
.card-label{{color:var(--mu);font-size:11px;text-transform:uppercase;letter-spacing:.7px;margin-bottom:5px;}}
.card-value{{font-size:20px;font-weight:700;}}
.card-sub{{font-size:12px;color:var(--mu);margin-top:3px;}}
/* regime */
.regime-badge{{display:inline-block;padding:3px 12px;border-radius:16px;font-weight:700;font-size:12px;background:{rc}22;color:{rc};border:1px solid {rc}66;}}
/* charts */
.charts-row{{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:20px;}}
.charts-row.wide{{grid-template-columns:1fr;}}
.chart-box{{background:var(--sf);border:1px solid var(--bd);border-radius:10px;padding:16px;}}
.chart-box h3{{font-size:12px;color:var(--mu);text-transform:uppercase;letter-spacing:.6px;margin-bottom:12px;}}
.chart-wrap{{position:relative;height:220px;}}
.chart-wrap.tall{{height:260px;}}
/* table */
table{{width:100%;border-collapse:collapse;background:var(--sf);border-radius:10px;overflow:hidden;margin-bottom:20px;}}
thead th{{background:#1a2942;color:var(--mu);font-size:11px;text-transform:uppercase;letter-spacing:.5px;padding:9px 12px;text-align:left;border-bottom:1px solid var(--bd);}}
tbody td{{padding:9px 12px;border-bottom:1px solid #192233;}}
tbody tr:last-child td{{border-bottom:none;}}
tbody tr:hover{{background:#1e2d45;}}
.num{{text-align:right;font-variant-numeric:tabular-nums;}}
.muted{{color:var(--mu);font-size:12px;}}
.pos{{color:var(--pos);}}
.neg{{color:var(--neg);}}
h2{{font-size:14px;font-weight:600;color:var(--ac);margin-bottom:12px;}}
/* modal */
.modal-overlay{{display:none;position:fixed;inset:0;background:#000a;z-index:100;align-items:center;justify-content:center;}}
.modal-overlay.open{{display:flex;}}
.modal{{background:var(--sf);border:1px solid var(--bd);border-radius:12px;padding:24px;max-width:520px;width:90%;}}
.modal h3{{color:var(--ac);margin-bottom:12px;}}
.modal code{{display:block;background:#0f172a;border:1px solid var(--bd);border-radius:6px;padding:12px;font-size:12px;color:#a5f3fc;margin:10px 0;line-height:1.6;word-break:break-all;}}
.modal-close{{margin-top:12px;background:var(--bd);border:none;color:var(--tx);padding:7px 18px;border-radius:6px;cursor:pointer;}}
@media(max-width:640px){{
  .charts-row{{grid-template-columns:1fr;}}
  .tab{{padding:10px 14px;font-size:12px;}}
  header h1{{font-size:15px;}}
}}
</style>
</head>
<body>

<header>
  <h1>📊 Portfolio Dashboard</h1>
  <div class="header-right">
    <span class="updated">更新：{TODAY}</span>
    <button class="btn-update" onclick="document.getElementById('modal').classList.add('open')">⟳ 更新資料</button>
  </div>
</header>

<!-- tabs -->
<div class="tabs">
  <div class="tab active" onclick="switchTab(0)">Overview</div>
  <div class="tab" onclick="switchTab(1)">🇺🇸 美股</div>
  <div class="tab" onclick="switchTab(2)">🇹🇼 台股</div>
</div>

<!-- ══ Tab 0: Overview ══════════════════════════════════════════════════════ -->
<div class="panel active" id="tab0">
  <div style="height:16px"></div>
  <div class="cards">
    {card("美股市值 (USD)", f"${us_tm:,.0f}", f"成本 ${us_tc:,.0f}")}
    {card("美股損益", f"${us_tp:+,.0f}", f"{us_pp:+.1f}%", pc(us_pp))}
    {card("台股市值 (TWD)", f"NT${tw_tm:,.0f}", f"成本 NT${tw_tc:,.0f}")}
    {card("台股損益", f"NT${tw_tp:+,.0f}", f"{tw_pp:+.1f}%", pc(tw_pp))}
  </div>

  <div class="charts-row wide">
    <div class="chart-box">
      <h3>美股市值走勢 (USD)</h3>
      <div class="chart-wrap"><canvas id="ovUsChart"></canvas></div>
    </div>
  </div>
  <div class="charts-row wide">
    <div class="chart-box">
      <h3>台股市值走勢 (TWD)</h3>
      <div class="chart-wrap"><canvas id="ovTwChart"></canvas></div>
    </div>
  </div>
</div>

<!-- ══ Tab 1: 美股 ══════════════════════════════════════════════════════════ -->
<div class="panel" id="tab1">
  <div style="height:16px"></div>
  <div class="cards">
    {card("美股市值", f"${us_tm:,.0f}", f"成本 ${us_tc:,.0f}")}
    {card("美股損益", f"${us_tp:+,.0f}", f"{us_pp:+.1f}%", pc(us_pp))}
    {card("市場狀態", f'<span class="regime-badge">{rl}</span>', f"P(&gt;VOO) {pb} · {stage}")}
  </div>

  <div class="charts-row">
    <div class="chart-box">
      <h3>持倉配置</h3>
      <div class="chart-wrap"><canvas id="usPie"></canvas></div>
    </div>
    <div class="chart-box">
      <h3>模型建議 vs 實際（衛星股）</h3>
      <div class="chart-wrap"><canvas id="usBar"></canvas></div>
    </div>
  </div>

  <div class="charts-row wide">
    <div class="chart-box">
      <h3>美股市值 + 損益走勢</h3>
      <div class="chart-wrap tall"><canvas id="usLine"></canvas></div>
    </div>
  </div>

  <h2>持倉明細</h2>
  <table>
    <thead><tr><th>代號</th><th>名稱</th><th class="num">股數</th><th class="num">均價</th><th class="num">現價</th><th class="num">市值</th><th class="num">損益</th><th class="num">損益%</th></tr></thead>
    <tbody>{holdings_table(us_h)}</tbody>
  </table>
</div>

<!-- ══ Tab 2: 台股 ══════════════════════════════════════════════════════════ -->
<div class="panel" id="tab2">
  <div style="height:16px"></div>
  <div class="cards">
    {card("台股市值", f"NT${tw_tm:,.0f}", f"成本 NT${tw_tc:,.0f}")}
    {card("台股損益", f"NT${tw_tp:+,.0f}", f"{tw_pp:+.1f}%", pc(tw_pp))}
    {card("持股檔數", str(len(tw_h)), "0050 核心/衛星架構")}
  </div>

  <div class="charts-row">
    <div class="chart-box">
      <h3>持倉配置</h3>
      <div class="chart-wrap"><canvas id="twPie"></canvas></div>
    </div>
    <div class="chart-box">
      <h3>台股市值走勢</h3>
      <div class="chart-wrap"><canvas id="twLine"></canvas></div>
    </div>
  </div>

  <h2>持倉明細</h2>
  <table>
    <thead><tr><th>代號</th><th>名稱</th><th class="num">股數</th><th class="num">均價</th><th class="num">現價</th><th class="num">市值</th><th class="num">損益</th><th class="num">損益%</th></tr></thead>
    <tbody>{holdings_table(tw_h)}</tbody>
  </table>
</div>

<!-- update modal -->
<div class="modal-overlay" id="modal" onclick="if(event.target===this)this.classList.remove('open')">
  <div class="modal">
    <h3>⟳ 更新資料</h3>
    <p style="color:var(--mu);font-size:13px">在本機 Buffett 目錄執行以下指令：</p>
    <code>cd "AI Playgrounds/Buffett"<br>
python -X utf8 update_portfolio.py<br>
python -X utf8 generate_portfolio_dashboard.py<br>
git add Buffett/portfolio_dashboard.html<br>
git commit -m "更新 Dashboard"<br>
git push</code>
    <p style="color:var(--mu);font-size:12px">推送後約 30 秒 GitHub Pages 自動更新。</p>
    <button class="modal-close" onclick="document.getElementById('modal').classList.remove('open')">關閉</button>
  </div>
</div>

<script>
// ── tab switch ──────────────────────────────────────────────────────────────
const tabs   = document.querySelectorAll('.tab');
const panels = document.querySelectorAll('.panel');
function switchTab(i){{
  tabs.forEach((t,j)=>t.classList.toggle('active',i===j));
  panels.forEach((p,j)=>p.classList.toggle('active',i===j));
}}

// ── chart helpers ───────────────────────────────────────────────────────────
const COLORS = {json.dumps(COLORS)};
const gridColor = '#334155', tickColor = '#94a3b8';
const baseScales = {{
  x:{{ ticks:{{color:tickColor,font:{{size:10}}}}, grid:{{color:gridColor}} }},
  y:{{ ticks:{{color:tickColor}}, grid:{{color:gridColor}} }}
}};

function lineChart(id, labels, datasets){{
  new Chart(document.getElementById(id),{{
    type:'line',
    data:{{ labels, datasets }},
    options:{{
      responsive:true, maintainAspectRatio:false,
      plugins:{{ legend:{{ labels:{{color:tickColor}} }} }},
      scales: baseScales
    }}
  }});
}}

function pieChart(id, labels, data){{
  new Chart(document.getElementById(id),{{
    type:'doughnut',
    data:{{ labels, datasets:[{{data, backgroundColor:COLORS, borderColor:'#0f172a', borderWidth:2}}] }},
    options:{{
      responsive:true, maintainAspectRatio:false,
      plugins:{{
        legend:{{position:'right', labels:{{color:tickColor, font:{{size:11}}, boxWidth:12}}}},
        tooltip:{{callbacks:{{label:c=>` ${{c.label}}: ${{c.parsed}}%`}}}}
      }}
    }}
  }});
}}

function barChart(id, labels, ds1, ds2){{
  new Chart(document.getElementById(id),{{
    type:'bar',
    data:{{ labels, datasets:[
      {{label:'模型建議%', data:ds1, backgroundColor:'#38bdf855', borderColor:'#38bdf8', borderWidth:1}},
      {{label:'實際持倉%', data:ds2, backgroundColor:'#818cf855', borderColor:'#818cf8', borderWidth:1}}
    ]}},
    options:{{
      responsive:true, maintainAspectRatio:false,
      plugins:{{ legend:{{labels:{{color:tickColor}}}} }},
      scales:{{
        x:{{ ticks:{{color:tickColor, font:{{size:10}}}}, grid:{{color:gridColor}} }},
        y:{{ ticks:{{color:tickColor, callback:v=>v+'%'}}, grid:{{color:gridColor}} }}
      }}
    }}
  }});
}}

// ── Overview charts ─────────────────────────────────────────────────────────
const ovDates = {ov_dates};
const ovUsMvs = {ov_us_mvs};
const ovTwMvs = {ov_tw_mvs};

lineChart('ovUsChart', ovDates, [{{
  label:'美股市值 (USD)', data:ovUsMvs,
  borderColor:'#38bdf8', backgroundColor:'#38bdf820', fill:true, tension:.3, pointRadius:4
}}]);
lineChart('ovTwChart', ovDates, [{{
  label:'台股市值 (TWD)', data:ovTwMvs,
  borderColor:'#34d399', backgroundColor:'#34d39920', fill:true, tension:.3, pointRadius:4
}}]);

// ── US charts ───────────────────────────────────────────────────────────────
pieChart('usPie', {json.dumps(pie_l)}, {json.dumps(pie_v)});
barChart('usBar', {json.dumps(bar_tickers)}, {json.dumps(bar_model)}, {json.dumps(bar_actual)});

const usDates = {us_dates};
const usMvs   = {us_mvs};
const usPnls  = {us_pnls};
lineChart('usLine', usDates, [
  {{label:'市值 USD', data:usMvs,  borderColor:'#38bdf8', backgroundColor:'#38bdf815', fill:true, tension:.3, pointRadius:4, yAxisID:'y'}},
  {{label:'損益 USD', data:usPnls, borderColor:'#22c55e', backgroundColor:'#22c55e10', fill:true, tension:.3, pointRadius:4, yAxisID:'y1'}}
]);
// rebuild usLine with dual axis
Chart.getChart('usLine').destroy();
new Chart(document.getElementById('usLine'),{{
  type:'line',
  data:{{ labels:usDates, datasets:[
    {{label:'市值 USD', data:usMvs,  borderColor:'#38bdf8', backgroundColor:'#38bdf815', fill:true, tension:.3, pointRadius:4, yAxisID:'y'}},
    {{label:'損益 USD', data:usPnls, borderColor:'#22c55e', backgroundColor:'#22c55e15', fill:true, tension:.3, pointRadius:4, yAxisID:'y1'}}
  ]}},
  options:{{
    responsive:true, maintainAspectRatio:false,
    plugins:{{ legend:{{labels:{{color:tickColor}}}} }},
    scales:{{
      x:{{ ticks:{{color:tickColor, font:{{size:10}}}}, grid:{{color:gridColor}} }},
      y:{{ position:'left',  ticks:{{color:'#38bdf8'}}, grid:{{color:gridColor}}, title:{{display:true, text:'市值', color:'#38bdf8'}} }},
      y1:{{ position:'right', ticks:{{color:'#22c55e'}}, grid:{{drawOnChartArea:false}}, title:{{display:true, text:'損益', color:'#22c55e'}} }}
    }}
  }}
}});

// ── TW charts ───────────────────────────────────────────────────────────────
pieChart('twPie', {json.dumps(tw_pie_l)}, {json.dumps(tw_pie_v)});

const twDates = {tw_dates};
const twMvs   = {tw_mvs};
const twPnls  = {tw_pnls};
new Chart(document.getElementById('twLine'),{{
  type:'line',
  data:{{ labels:twDates, datasets:[
    {{label:'市值 TWD', data:twMvs,  borderColor:'#34d399', backgroundColor:'#34d39915', fill:true, tension:.3, pointRadius:4, yAxisID:'y'}},
    {{label:'損益 TWD', data:twPnls, borderColor:'#facc15', backgroundColor:'#facc1510', fill:true, tension:.3, pointRadius:4, yAxisID:'y1'}}
  ]}},
  options:{{
    responsive:true, maintainAspectRatio:false,
    plugins:{{ legend:{{labels:{{color:tickColor}}}} }},
    scales:{{
      x:{{ ticks:{{color:tickColor, font:{{size:10}}}}, grid:{{color:gridColor}} }},
      y:{{ position:'left',  ticks:{{color:'#34d399'}}, grid:{{color:gridColor}}, title:{{display:true, text:'市值', color:'#34d399'}} }},
      y1:{{ position:'right', ticks:{{color:'#facc15'}}, grid:{{drawOnChartArea:false}}, title:{{display:true, text:'損益', color:'#facc15'}} }}
    }}
  }}
}});
</script>
</body>
</html>"""
    return html

if __name__ == "__main__":
    print("讀取資料...")
    html = build()
    OUT.write_text(html, encoding="utf-8")
    print(f"✅ 已輸出：{OUT}")
