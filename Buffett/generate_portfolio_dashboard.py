# -*- coding: utf-8 -*-
"""
generate_portfolio_dashboard.py
Storytelling-first Portfolio Dashboard
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
TW_WF   = ROOT / "TWSE" / "tw_0050_walkforward.csv"
OUT     = ROOT / "portfolio_dashboard.html"
TODAY   = str(date.today())

# ── helpers ───────────────────────────────────────────────────────────────────

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

def load_us_model():
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

def load_tw_model():
    try:
        df  = pd.read_csv(TW_WF)
        row = df[df["period"].str.contains("current", na=False)].iloc[-1]
        w   = {}
        for part in str(row.get("weights", "")).split(","):
            tok = part.strip().rsplit(" ", 1)
            if len(tok) == 2:
                t, v = tok[0].strip(), float(tok[1].replace("%","")) / 100
                w[t] = w.get(t, 0) + v
        p_beat = row.get("p_beat_0050", None)
        utility = row.get("utility", None)
        return w, p_beat, utility
    except Exception:
        return {}, None, None

def summarise(holdings, market):
    h  = [x for x in holdings if x["market"] == market]
    tc = sum(x["cost"] for x in h if x["cost"])
    tm = sum(x["mv"]   for x in h if x["mv"])
    tp = tm - tc if (tm and tc) else 0
    pp = tp / tc * 100 if tc else 0
    return tc, tm, tp, pp, h

# ── HTML fragments ────────────────────────────────────────────────────────────

def card(label, value, sub="", sub_class="", extra_class=""):
    return f"""<div class="card {extra_class}">
      <div class="card-label">{label}</div>
      <div class="card-value">{value}</div>
      {f'<div class="card-sub {sub_class}">{sub}</div>' if sub else ''}
    </div>"""

def holdings_table_us(hlist, model_w, total_mv):
    rows = []
    for h in sorted(hlist, key=lambda x: -(x["mv"] or 0)):
        pnl_s  = ("+" if (h["pnl"] or 0) >= 0 else "") + fmt(h["pnl"], prefix="$", d=0)
        pct_s  = fmt(h["pct"], suffix="%")
        mpct   = model_w.get(h["ticker"], 0) * 100
        if mpct > 0:
            model_cell = f'<span style="color:#22c55e">&#10003; {mpct:.1f}%</span>'
        else:
            model_cell = '<span style="color:#475569">&#8212; 自選</span>'
        rows.append(f"""<tr>
          <td><strong>{h['ticker']}</strong></td><td class="muted">{h['name']}</td>
          <td class="num">{fmt(h['shares'],d=0)}</td>
          <td class="num">{fmt(h['avg_cost'],prefix='$')}</td>
          <td class="num">{fmt(h['price'],prefix='$')}</td>
          <td class="num">{fmt(h['mv'],prefix='$',d=0)}</td>
          <td class="num {pc(h['pnl'])}">{pnl_s}</td>
          <td class="num {pc(h['pct'])}">{pct_s}</td>
          <td class="num">{model_cell}</td>
        </tr>""")
    return "\n".join(rows)

def holdings_table_tw(hlist):
    rows = []
    for h in hlist:
        pnl_s = ("+" if (h["pnl"] or 0) >= 0 else "") + fmt(h["pnl"], prefix="$", d=0)
        pct_s = fmt(h["pct"], suffix="%")
        rows.append(f"""<tr>
          <td><strong>{h['ticker']}</strong></td><td class="muted">{h['name']}</td>
          <td class="num">{fmt(h['shares'],d=0)}</td>
          <td class="num">{fmt(h['avg_cost'],prefix='$')}</td>
          <td class="num">{fmt(h['price'],prefix='$')}</td>
          <td class="num">{fmt(h['mv'],prefix='$',d=0)}</td>
          <td class="num {pc(h['pnl'])}">{pnl_s}</td>
          <td class="num {pc(h['pct'])}">{pct_s}</td>
        </tr>""")
    return "\n".join(rows)

def tw_model_grid(tw_model_w):
    items = sorted(tw_model_w.items(), key=lambda x: -x[1])
    cards = []
    for ticker, weight in items:
        is_core  = "0050" in ticker
        label    = "核心" if is_core else "衛星"
        color    = "#38bdf8" if is_core else "#e2e8f0"
        cards.append(f"""<div style="text-align:center;padding:14px 10px;background:#1a2942;border-radius:8px;">
          <div style="font-size:11px;color:#94a3b8;margin-bottom:6px;">{ticker}</div>
          <div style="font-size:22px;font-weight:700;color:{color}">{weight*100:.0f}%</div>
          <div style="font-size:11px;color:#64748b;margin-top:4px">{label}</div>
        </div>""")
    return "\n".join(cards)

# ── main builder ──────────────────────────────────────────────────────────────

def build():
    holdings                      = load_holdings()
    hist                          = load_history()
    model_w, regime, stage, p_beat = load_us_model()
    tw_model_w, tw_p_beat, tw_util = load_tw_model()

    us_tc, us_tm, us_tp, us_pp, us_h = summarise(holdings, "US")
    tw_tc, tw_tm, tw_tp, tw_pp, tw_h = summarise(holdings, "TW")

    # regime
    rc = {"risk_on":"#22c55e","caution":"#f59e0b","risk_off":"#ef4444"}.get(regime,"#6b7280")
    rl = regime.upper().replace("_"," ")
    pb = f"{p_beat*100:.1f}%" if p_beat else "—"
    tw_pb = f"{tw_p_beat*100:.2f}%" if tw_p_beat else "—"

    banner_bg     = {"risk_on":"#052e16","caution":"#451a03","risk_off":"#1f0207"}.get(regime,"#1e293b")
    regime_badge  = f'<span class="regime-badge" style="background:{rc}22;color:{rc};border:1px solid {rc}66">{rl}</span>'

    # VOO gap
    us_total_mv  = sum(x["mv"] for x in us_h if x["mv"]) or 1
    actual_w     = {x["ticker"]: x["mv"]/us_total_mv for x in us_h if x["mv"]}
    voo_actual   = actual_w.get("VOO", 0) * 100
    voo_model    = model_w.get("VOO", 0) * 100
    voo_gap      = voo_actual - voo_model
    off_model_n  = sum(1 for t in actual_w if model_w.get(t, 0) == 0)

    # US horizontal bar: all tickers sorted by actual weight desc
    bar_tickers = sorted(actual_w, key=lambda t: -actual_w[t])
    bar_actual  = [round(actual_w.get(t, 0) * 100, 1) for t in bar_tickers]
    bar_model   = [round(model_w.get(t, 0) * 100, 1) for t in bar_tickers]
    # Color: VOO = blue, model-recommended = green, off-model = gray
    bar_colors  = []
    for t in bar_tickers:
        if t == "VOO":
            bar_colors.append("#38bdf8")
        elif model_w.get(t, 0) > 0:
            bar_colors.append("#22c55e")
        else:
            bar_colors.append("#475569")

    # History JSON
    us_dates = json.dumps([d["date"] for d in hist["US"]])
    us_pnls  = json.dumps([d["pnl"]  for d in hist["US"]])
    tw_dates = json.dumps([d["date"] for d in hist["TW"]])
    tw_mvs   = json.dumps([d["mv"]   for d in hist["TW"]])

    # Overview combined
    all_dates = sorted(set(d["date"] for d in hist["US"] + hist["TW"]))
    us_mv_map = {d["date"]: d["mv"] for d in hist["US"]}
    tw_mv_map = {d["date"]: d["mv"] for d in hist["TW"]}
    ov_dates  = json.dumps(all_dates)
    ov_us_mvs = json.dumps([us_mv_map.get(d) for d in all_dates])
    ov_tw_mvs = json.dumps([tw_mv_map.get(d) for d in all_dates])

    # Divergence alert (show when gap > 10pp)
    if abs(voo_gap) > 10:
        gap_str     = f"{voo_gap:+.1f}pp"
        alert_html  = f"""<div class="alert-box">
    <span style="font-size:18px;margin-top:2px">&#9888;</span>
    <div>
      <div class="alert-title">配置偏離：VOO 實際 {voo_actual:.1f}% vs 模型建議 {voo_model:.1f}%（{gap_str}）</div>
      <div class="alert-body">{rl} 模式下建議提高 VOO 核心比重。目前有 {off_model_n} 檔模型未推薦個股在持倉中。</div>
    </div>
  </div>"""
    else:
        alert_html = ""

    # TW model grid
    if tw_model_w:
        tw_grid = tw_model_grid(tw_model_w)
        tw_util_str = f"{tw_util:.3f}" if tw_util else "—"
    else:
        tw_grid     = '<div style="color:#94a3b8;padding:12px">模型配置資料未找到</div>'
        tw_util_str = "—"

    us_table_html = holdings_table_us(us_h, model_w, us_total_mv)
    tw_table_html = holdings_table_tw(tw_h)

    hbar_labels_js = json.dumps(bar_tickers)
    hbar_actual_js = json.dumps(bar_actual)
    hbar_model_js  = json.dumps(bar_model)
    hbar_colors_js = json.dumps(bar_colors)

    html = f"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Portfolio Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
:root{{--bg:#0f172a;--sf:#1e293b;--bd:#334155;--tx:#e2e8f0;--mu:#94a3b8;--ac:#38bdf8;--pos:#22c55e;--neg:#ef4444;--warn:#f59e0b;}}
*{{box-sizing:border-box;margin:0;padding:0;}}
body{{background:var(--bg);color:var(--tx);font-family:'Segoe UI',system-ui,sans-serif;font-size:14px;min-height:100vh;}}
header{{padding:12px 24px;border-bottom:1px solid var(--bd);display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap;}}
header h1{{font-size:17px;font-weight:700;color:var(--tx);}}
header .subtitle{{font-size:12px;color:var(--mu);margin-top:2px;}}
.header-right{{display:flex;align-items:center;gap:10px;}}
.updated{{color:var(--mu);font-size:12px;}}
.btn-update{{background:transparent;border:1px solid var(--bd);color:var(--mu);padding:6px 14px;border-radius:6px;cursor:pointer;font-size:12px;font-weight:600;}}
.btn-update:hover{{border-color:var(--ac);color:var(--ac);}}
.regime-banner{{background:{banner_bg};border-bottom:2px solid {rc};padding:9px 24px;display:flex;align-items:center;gap:12px;}}
.regime-banner .msg{{font-size:13px;color:#fde68a;font-weight:600;}}
.regime-banner .detail{{font-size:12px;color:#fcd34d;margin-left:auto;}}
.tabs{{display:flex;border-bottom:1px solid var(--bd);background:var(--sf);}}
.tab{{padding:12px 24px;cursor:pointer;color:var(--mu);font-size:13px;font-weight:600;border-bottom:2px solid transparent;transition:.15s;user-select:none;}}
.tab.active{{color:var(--ac);border-bottom-color:var(--ac);}}
.tab:hover:not(.active){{color:var(--tx);}}
.panel{{display:none;padding:20px 24px;max-width:1200px;margin:0 auto;}}
.panel.active{{display:block;}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin-bottom:20px;}}
.card{{background:var(--sf);border:1px solid var(--bd);border-radius:10px;padding:14px 18px;}}
.card.warn{{border-color:var(--warn);background:#1c1507;}}
.card-label{{color:var(--mu);font-size:11px;text-transform:uppercase;letter-spacing:.7px;margin-bottom:5px;}}
.card-value{{font-size:21px;font-weight:700;}}
.card-sub{{font-size:12px;color:var(--mu);margin-top:3px;}}
.alert-box{{background:#1c1507;border:1px solid var(--warn);border-radius:10px;padding:14px 18px;margin-bottom:20px;display:flex;align-items:flex-start;gap:12px;}}
.alert-title{{font-size:13px;font-weight:700;color:#fde68a;margin-bottom:4px;}}
.alert-body{{font-size:12px;color:#fcd34d;line-height:1.6;}}
.regime-badge{{display:inline-block;padding:3px 12px;border-radius:16px;font-weight:700;font-size:12px;}}
.charts-row{{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:20px;}}
.charts-row.wide{{grid-template-columns:1fr;}}
.chart-box{{background:var(--sf);border:1px solid var(--bd);border-radius:10px;padding:16px;}}
.chart-box h3{{font-size:12px;color:var(--mu);text-transform:uppercase;letter-spacing:.6px;margin-bottom:12px;}}
.chart-wrap{{position:relative;height:220px;}}
.chart-wrap.tall{{height:300px;}}
.section-label{{font-size:11px;font-weight:600;color:var(--mu);text-transform:uppercase;letter-spacing:.6px;margin-bottom:10px;}}
table{{width:100%;border-collapse:collapse;background:var(--sf);border-radius:10px;overflow:hidden;margin-bottom:20px;}}
thead th{{background:#1a2942;color:var(--mu);font-size:11px;text-transform:uppercase;letter-spacing:.5px;padding:9px 12px;text-align:left;border-bottom:1px solid var(--bd);}}
tbody td{{padding:9px 12px;border-bottom:1px solid #192233;}}
tbody tr:last-child td{{border-bottom:none;}}
tbody tr:hover{{background:#1e2d45;}}
.num{{text-align:right;font-variant-numeric:tabular-nums;}}
.muted{{color:var(--mu);font-size:12px;}}
.pos{{color:var(--pos);}}
.neg{{color:var(--neg);}}
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
  <div>
    <h1>Portfolio Dashboard</h1>
    <div class="subtitle">Buffett 量化配置觀測系統</div>
  </div>
  <div class="header-right">
    <span class="updated">更新：{TODAY}</span>
    <button class="btn-update" onclick="document.getElementById('modal').classList.add('open')">&#8635; 更新資料</button>
  </div>
</header>

<div class="regime-banner">
  <span style="font-size:15px">&#9888;</span>
  <span class="msg">市場 Regime：{rl} &mdash; 模型建議 VOO {voo_model:.1f}%，目前實際 {voo_actual:.1f}%</span>
  <span class="detail">P(&gt;VOO) {pb} &nbsp;|&nbsp; Stage: {stage} &nbsp;|&nbsp; 2026Q2 current</span>
</div>

<div class="tabs">
  <div class="tab active" onclick="switchTab(0)">總覽</div>
  <div class="tab" onclick="switchTab(1)">&#127482;&#127480; 美股</div>
  <div class="tab" onclick="switchTab(2)">&#127481;&#127484; 台股</div>
</div>

<!-- ══ Tab 0: 總覽 ══ -->
<div class="panel active" id="tab0">
  <div style="height:16px"></div>
  {alert_html}
  <div class="cards">
    {card("美股市值 (USD)", f"${us_tm:,.0f}", f"成本 ${us_tc:,.0f} ｜ {us_pp:+.1f}%", pc(us_pp))}
    {card("台股市值 (TWD)", f"NT${tw_tm:,.0f}", f"成本 NT${tw_tc:,.0f} ｜ {tw_pp:+.1f}%", pc(tw_pp))}
    {card("美股模型信心", pb, f"P(&gt;VOO) ｜ {rl}", "", "warn" if voo_gap < -10 else "")}
    {card("台股模型信心", tw_pb, "P(&gt;0050) ｜ 2026Q2")}
  </div>
  <div class="charts-row">
    <div class="chart-box">
      <h3>美股損益走勢 (USD)</h3>
      <div class="chart-wrap"><canvas id="ovUsChart"></canvas></div>
    </div>
    <div class="chart-box">
      <h3>台股市值走勢 (TWD)</h3>
      <div class="chart-wrap"><canvas id="ovTwChart"></canvas></div>
    </div>
  </div>
</div>

<!-- ══ Tab 1: 美股 ══ -->
<div class="panel" id="tab1">
  <div style="height:16px"></div>
  <div class="cards">
    {card("美股市值", f"${us_tm:,.0f}", f"成本 ${us_tc:,.0f}")}
    {card("損益", f"${us_tp:+,.0f}", f"{us_pp:+.1f}%", pc(us_pp))}
    {card("市場狀態", regime_badge, f"VOO 模型 {voo_model:.1f}% / 實際 {voo_actual:.1f}%", "", "warn" if abs(voo_gap) > 10 else "")}
  </div>
  {alert_html}
  <div class="charts-row">
    <div class="chart-box">
      <h3>持倉配置：實際 vs 模型建議</h3>
      <div class="chart-wrap tall"><canvas id="usHBar"></canvas></div>
    </div>
    <div class="chart-box">
      <h3>損益走勢（近期，USD）</h3>
      <div class="chart-wrap tall"><canvas id="usLine"></canvas></div>
    </div>
  </div>
  <div class="section-label">持倉明細（依市值排序）</div>
  <table>
    <thead><tr>
      <th>代號</th><th>名稱</th>
      <th class="num">股數</th><th class="num">均價</th><th class="num">現價</th>
      <th class="num">市值</th><th class="num">損益</th><th class="num">損益%</th>
      <th class="num">模型</th>
    </tr></thead>
    <tbody>{us_table_html}</tbody>
  </table>
</div>

<!-- ══ Tab 2: 台股 ══ -->
<div class="panel" id="tab2">
  <div style="height:16px"></div>
  <div class="cards">
    {card("台股市值", f"NT${tw_tm:,.0f}", f"成本 NT${tw_tc:,.0f}")}
    {card("損益", f"NT${tw_tp:+,.0f}", f"{tw_pp:+.1f}%", pc(tw_pp))}
    {card("模型信心", tw_pb, f"P(&gt;0050) ｜ Utility {tw_util_str}")}
  </div>
  <div class="chart-box" style="margin-bottom:20px">
    <h3>2026Q2 模型建議配置</h3>
    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px;margin-top:12px;">
      {tw_grid}
    </div>
    <div style="margin-top:12px;padding:8px 12px;background:#0f1a2e;border-radius:6px;font-size:12px;color:var(--mu)">
      P(&gt;0050) {tw_pb} &nbsp;|&nbsp; Utility {tw_util_str} &nbsp;|&nbsp; Snapshot 2026-04-17
    </div>
  </div>
  <div class="chart-box" style="margin-bottom:20px">
    <h3>台股市值走勢 (TWD)</h3>
    <div class="chart-wrap"><canvas id="twLine"></canvas></div>
  </div>
  <div class="section-label">實際持倉</div>
  <table>
    <thead><tr>
      <th>代號</th><th>名稱</th>
      <th class="num">股數</th><th class="num">均價</th><th class="num">現價</th>
      <th class="num">市值</th><th class="num">損益</th><th class="num">損益%</th>
    </tr></thead>
    <tbody>{tw_table_html}</tbody>
  </table>
</div>

<div class="modal-overlay" id="modal" onclick="if(event.target===this)this.classList.remove('open')">
  <div class="modal">
    <h3>&#8635; 更新資料</h3>
    <p style="color:var(--mu);font-size:13px">在本機 Buffett 目錄執行：</p>
    <code>python -X utf8 update_portfolio.py<br>
python -X utf8 generate_portfolio_dashboard.py<br>
git add Buffett/portfolio_dashboard.html<br>
git commit -m "每日更新 Dashboard"<br>
git push</code>
    <p style="color:var(--mu);font-size:12px;margin-top:8px">推送後約 30 秒 GitHub Pages 自動更新。</p>
    <button class="modal-close" onclick="document.getElementById('modal').classList.remove('open')">關閉</button>
  </div>
</div>

<script>
const tabs   = document.querySelectorAll('.tab');
const panels = document.querySelectorAll('.panel');
function switchTab(i){{
  tabs.forEach((t,j)=>t.classList.toggle('active',i===j));
  panels.forEach((p,j)=>p.classList.toggle('active',i===j));
}}

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

// Overview charts
const ovDates = {ov_dates};
lineChart('ovUsChart', ovDates, [{{
  label:'損益 USD', data:{json.dumps([us_mv_map.get(d, 0) - us_tc for d in all_dates]) if us_tc else ov_us_mvs},
  borderColor:'#22c55e', backgroundColor:'#22c55e15', fill:true, tension:.3, pointRadius:3
}}]);
lineChart('ovTwChart', ovDates, [{{
  label:'市值 TWD', data:{ov_tw_mvs},
  borderColor:'#34d399', backgroundColor:'#34d39918', fill:true, tension:.3, pointRadius:3
}}]);

// US horizontal bar: actual vs model
new Chart(document.getElementById('usHBar'),{{
  type:'bar',
  data:{{
    labels: {hbar_labels_js},
    datasets:[
      {{
        label:'模型建議 %',
        data: {hbar_model_js},
        backgroundColor:'#38bdf822',
        borderColor:'#38bdf8',
        borderWidth:1,
        borderRadius:2
      }},
      {{
        label:'實際持倉 %',
        data: {hbar_actual_js},
        backgroundColor: {hbar_colors_js},
        borderRadius:2
      }}
    ]
  }},
  options:{{
    indexAxis:'y',
    responsive:true, maintainAspectRatio:false,
    plugins:{{
      legend:{{labels:{{color:tickColor,font:{{size:11}}}}}},
      tooltip:{{callbacks:{{label:c=>`${{c.dataset.label}}: ${{c.parsed.x}}%`}}}}
    }},
    scales:{{
      x:{{ticks:{{color:tickColor,callback:v=>v+'%'}},grid:{{color:gridColor}},max:100}},
      y:{{ticks:{{color:tickColor,font:{{size:11}}}},grid:{{color:gridColor}}}}
    }}
  }}
}});

// US P&L trend
lineChart('usLine', {us_dates}, [{{
  label:'損益 USD',
  data: {us_pnls},
  borderColor:'#22c55e', backgroundColor:'#22c55e12', fill:true, tension:.3, pointRadius:3
}}]);

// TW market value trend
lineChart('twLine', {tw_dates}, [{{
  label:'市值 TWD',
  data: {tw_mvs},
  borderColor:'#34d399', backgroundColor:'#34d39918', fill:true, tension:.3, pointRadius:3
}}]);
</script>
</body>
</html>"""
    return html


if __name__ == "__main__":
    print("讀取資料...")
    html = build()
    OUT.write_text(html, encoding="utf-8")
    print(f"✅ 已輸出：{OUT}")
