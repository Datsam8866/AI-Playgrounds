import argparse
import json
import sqlite3
from collections import defaultdict
from datetime import date
from html import escape
from pathlib import Path


BASE = Path(__file__).resolve().parent
DB_PATH = BASE / "house_budget.db"
OUTPUT_PATH = BASE / "house_budget_dashboard.html"


CATEGORY_ALIASES = {
    "Accommdation": "Accommodation",
    "Trasportation": "Transportation",
}


def normalize_category(category):
    if category is None:
        return "Uncategorized"
    clean = str(category).strip() or "Uncategorized"
    return CATEGORY_ALIASES.get(clean, clean)


def source_group(source):
    if source and source.startswith("au_travel/"):
        return "travel"
    return "household"


def money(value):
    return int(round(value or 0))


def fetch_transactions(db_path):
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    rows = list(
        con.execute(
            """
            SELECT id, source, date, category, item, amount, currency, amount_twd, who, note
            FROM transactions
            WHERE date IS NOT NULL AND amount_twd IS NOT NULL
            ORDER BY date, id
            """
        )
    )
    con.close()
    transactions = []
    for row in rows:
        transactions.append(
            {
                "id": row["id"],
                "source": row["source"] or "",
                "source_group": source_group(row["source"] or ""),
                "date": row["date"],
                "month": row["date"][:7],
                "category": normalize_category(row["category"]),
                "raw_category": row["category"] or "",
                "item": row["item"] or "(未命名)",
                "amount": row["amount"],
                "currency": row["currency"] or "",
                "amount_twd": money(row["amount_twd"]),
                "who": row["who"] or "(unknown)",
                "note": row["note"] or "",
            }
        )
    return transactions


def summarize_scope(transactions):
    by_month = {}
    month_names = sorted({tx["month"] for tx in transactions})
    previous_expense = None

    for month in month_names:
        month_txs = [tx for tx in transactions if tx["month"] == month]
        income = sum(tx["amount_twd"] for tx in month_txs if tx["amount_twd"] > 0)
        expense = sum(tx["amount_twd"] for tx in month_txs if tx["amount_twd"] < 0)
        net = income + expense
        category_expense = defaultdict(int)
        payer_expense = defaultdict(int)
        for tx in month_txs:
            if tx["amount_twd"] < 0:
                category_expense[tx["category"]] += tx["amount_twd"]
                payer_expense[tx["who"]] += tx["amount_twd"]

        large_expenses = sorted(
            [tx for tx in month_txs if tx["amount_twd"] < 0],
            key=lambda tx: abs(tx["amount_twd"]),
            reverse=True,
        )[:10]

        by_month[month] = {
            "income": income,
            "expense": expense,
            "net": net,
            "expense_delta": 0 if previous_expense is None else expense - previous_expense,
            "category_expense": dict(sorted(category_expense.items(), key=lambda item: item[1])),
            "payer_expense": dict(sorted(payer_expense.items())),
            "large_expenses": large_expenses,
            "transactions": month_txs,
        }
        previous_expense = expense

    return {
        "month_order": month_names,
        "latest_month": month_names[-1] if month_names else "",
        "months": by_month,
    }


def build_dashboard_data(db_path=DB_PATH):
    transactions = fetch_transactions(db_path)
    scopes = {
        "all": transactions,
        "household": [tx for tx in transactions if tx["source_group"] == "household"],
        "travel": [tx for tx in transactions if tx["source_group"] == "travel"],
    }
    data = summarize_scope(scopes["all"])
    data["scopes"] = {name: summarize_scope(rows) for name, rows in scopes.items()}
    data["generated_at"] = date.today().isoformat()
    return data


def render_html(data):
    data_json = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    template = r"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>House Budget Dashboard</title>
  <style>
    :root {
      --bg: #f4f6f9;
      --surface: #ffffff;
      --border: #e2e8f0;
      --text: #1e293b;
      --muted: #64748b;
      --good: #15803d;
      --bad: #dc2626;
      --warn: #d97706;
      --accent: #2563eb;
      --ink: #0f172a;
      --bar-default: #cbd5e1;
      --bar-top: #3b82f6;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: "Segoe UI", "Noto Sans TC", Arial, sans-serif;
      color: var(--text);
      background: var(--bg);
      font-size: 14px;
      line-height: 1.5;
    }
    header {
      position: sticky; top: 0; z-index: 10;
      background: rgba(244,246,249,0.95);
      border-bottom: 1px solid var(--border);
      backdrop-filter: blur(10px);
    }
    .nav {
      max-width: 1200px; margin: 0 auto;
      padding: 12px 20px;
      display: flex; align-items: center; gap: 20px; flex-wrap: wrap;
    }
    .nav-title { font-size: 18px; font-weight: 700; color: var(--ink); }
    .nav-sub { font-size: 12px; color: var(--muted); margin-top: 1px; }
    .tabs { display: flex; gap: 4px; margin-left: auto; }
    .tab {
      padding: 6px 16px;
      border: 1px solid var(--border);
      border-radius: 999px;
      background: var(--surface);
      color: var(--muted);
      font-size: 13px; font-weight: 600;
      cursor: pointer; transition: all .15s;
    }
    .tab.active { background: var(--ink); color: #fff; border-color: var(--ink); }
    main { max-width: 1200px; margin: 0 auto; padding: 20px; }
    .panel { display: none; }
    .panel.active { display: block; }
    /* card */
    .card { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; }
    /* kpi row */
    .kpi-row { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }
    .kpi { padding: 16px 18px; }
    .kpi-label {
      font-size: 11px; font-weight: 700;
      text-transform: uppercase; letter-spacing: .07em;
      color: var(--muted);
    }
    .kpi-value { font-size: 28px; font-weight: 700; color: var(--ink); margin-top: 8px; line-height: 1; }
    .kpi-hint { font-size: 12px; color: var(--muted); margin-top: 7px; }
    .good { color: var(--good); }
    .bad { color: var(--bad); }
    .warn { color: var(--warn); }
    /* section */
    .grid-2 { display: grid; grid-template-columns: 1.55fr 1fr; gap: 12px; margin-top: 12px; }
    .section { padding: 18px 20px; }
    .section-title {
      font-size: 11px; font-weight: 700;
      text-transform: uppercase; letter-spacing: .07em;
      color: var(--muted);
      margin-bottom: 16px;
    }
    /* insight banner */
    .insight {
      padding: 13px 16px;
      border-radius: 8px;
      border: 1px solid #bfdbfe;
      background: #eff6ff;
      font-size: 14px; color: var(--ink);
      line-height: 1.6;
      margin-bottom: 14px;
    }
    /* horizontal bars */
    .hbar-list { display: grid; gap: 11px; }
    .hbar-item {
      display: grid;
      grid-template-columns: 130px 1fr 110px;
      gap: 10px; align-items: center;
      font-size: 13px;
    }
    .hbar-name {
      color: var(--text);
      white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }
    .hbar-name.top { font-weight: 700; color: var(--ink); }
    .hbar-track { height: 8px; background: #f1f5f9; border-radius: 4px; overflow: hidden; }
    .hbar-fill { height: 100%; border-radius: 4px; background: var(--bar-default); }
    .hbar-fill.top { background: var(--bar-top); }
    .hbar-meta { text-align: right; font-variant-numeric: tabular-nums; font-size: 12px; color: var(--muted); }
    .hbar-meta.top { color: var(--ink); font-weight: 700; }
    .hbar-pct { font-size: 11px; color: #94a3b8; margin-left: 4px; }
    /* trend */
    svg { width: 100%; height: auto; display: block; overflow: visible; }
    /* toolbar */
    .toolbar { display: flex; gap: 10px; flex-wrap: wrap; align-items: center; margin-bottom: 16px; }
    .toolbar label { font-size: 13px; color: var(--muted); }
    select, input[type="number"] {
      height: 34px; border: 1px solid var(--border);
      background: var(--surface); color: var(--text);
      border-radius: 6px; padding: 0 10px; font-size: 13px;
    }
    input[type="number"] { width: 88px; }
    /* table */
    .table-wrap { overflow: auto; border-radius: 8px; border: 1px solid var(--border); max-height: 420px; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th {
      padding: 9px 10px; text-align: left;
      font-size: 11px; font-weight: 700;
      text-transform: uppercase; letter-spacing: .05em;
      color: var(--muted); background: #f8fafc;
      border-bottom: 1px solid var(--border);
      position: sticky; top: 0;
    }
    td { padding: 9px 10px; border-bottom: 1px solid #f1f5f9; vertical-align: top; }
    tr:last-child td { border-bottom: none; }
    .num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
    /* settlement callout */
    .settle-callout {
      padding: 20px 24px; border-radius: 10px;
      background: linear-gradient(135deg, #eff6ff, #fff);
      border: 1px solid #bfdbfe;
      margin-bottom: 14px;
    }
    .settle-amount { font-size: 26px; font-weight: 700; color: var(--ink); }
    .settle-note { font-size: 13px; color: var(--muted); margin-top: 5px; }
    .empty { color: var(--muted); padding: 20px; text-align: center; font-size: 13px; }
    @media (max-width: 900px) {
      .kpi-row { grid-template-columns: repeat(2, 1fr); }
      .grid-2 { grid-template-columns: 1fr; }
    }
    @media (max-width: 560px) {
      main { padding: 12px; }
      .kpi-row { grid-template-columns: 1fr; }
      .kpi-value { font-size: 22px; }
      .hbar-item { grid-template-columns: 90px 1fr 80px; }
    }
  </style>
</head>
<body>
<header>
  <nav class="nav">
    <div>
      <div class="nav-title">House Budget</div>
      <div class="nav-sub">Updated __GENERATED__ &middot; TWD</div>
    </div>
    <div class="tabs" role="tablist">
      <button class="tab active" data-tab="overview">Overview</button>
      <button class="tab" data-tab="monthly">Monthly</button>
      <button class="tab" data-tab="settlement">Sam &amp; Rita</button>
    </div>
  </nav>
</header>
<main>

  <!-- ── OVERVIEW ── -->
  <section class="panel active" data-panel="overview">
    <div class="kpi-row">
      <div class="card kpi">
        <div class="kpi-label">Total Income</div>
        <div class="kpi-value good" id="ov-income"></div>
        <div class="kpi-hint" id="ov-income-hint"></div>
      </div>
      <div class="card kpi">
        <div class="kpi-label">Total Expense</div>
        <div class="kpi-value bad" id="ov-expense"></div>
        <div class="kpi-hint" id="ov-expense-hint"></div>
      </div>
      <div class="card kpi">
        <div class="kpi-label">Net Balance</div>
        <div class="kpi-value" id="ov-net"></div>
        <div class="kpi-hint" id="ov-net-hint"></div>
      </div>
    </div>
    <div class="grid-2">
      <div class="card section">
        <div class="section-title">Monthly Trend</div>
        <div id="ov-trend"></div>
      </div>
      <div class="card section">
        <div class="section-title">Cumulative Expense by Category</div>
        <div class="hbar-list" id="ov-cats"></div>
      </div>
    </div>
    <div class="card section" style="margin-top:12px">
      <div class="section-title">Year-over-Year Expense</div>
      <div class="hbar-list" id="ov-years"></div>
    </div>
  </section>

  <!-- ── MONTHLY ── -->
  <section class="panel" data-panel="monthly">
    <div class="toolbar">
      <select id="m-scope" aria-label="資料範圍">
        <option value="all">全部</option>
        <option value="household">家庭</option>
        <option value="travel">澳洲旅遊</option>
      </select>
      <select id="m-month" aria-label="月份"></select>
    </div>
    <div class="insight" id="m-insight"></div>
    <div class="kpi-row">
      <div class="card kpi">
        <div class="kpi-label">Income</div>
        <div class="kpi-value good" id="m-income"></div>
        <div class="kpi-hint" id="m-income-hint"></div>
      </div>
      <div class="card kpi">
        <div class="kpi-label">Expense</div>
        <div class="kpi-value bad" id="m-expense"></div>
        <div class="kpi-hint" id="m-expense-hint"></div>
      </div>
      <div class="card kpi">
        <div class="kpi-label">Net</div>
        <div class="kpi-value" id="m-net"></div>
        <div class="kpi-hint" id="m-net-hint"></div>
      </div>
    </div>
    <div class="grid-2">
      <div class="card section">
        <div class="section-title">12-Month Trend</div>
        <div id="m-trend"></div>
      </div>
      <div class="card section">
        <div class="section-title">Expense by Category</div>
        <div class="hbar-list" id="m-cats"></div>
      </div>
    </div>
    <div class="card section" style="margin-top:12px">
      <div class="section-title">Top 5 Largest Transactions</div>
      <div class="table-wrap">
        <table>
          <thead><tr><th>Date</th><th>Category</th><th>Item</th><th>Who</th><th class="num">TWD</th></tr></thead>
          <tbody id="m-top5"></tbody>
        </table>
      </div>
    </div>
    <div class="card section" style="margin-top:12px">
      <div class="section-title">All Transactions</div>
      <div class="table-wrap">
        <table>
          <thead><tr><th>Date</th><th>Category</th><th>Item</th><th>Source</th><th>Who</th><th class="num">TWD</th></tr></thead>
          <tbody id="m-all"></tbody>
        </table>
      </div>
    </div>
  </section>

  <!-- ── SETTLEMENT ── -->
  <section class="panel" data-panel="settlement">
    <div class="toolbar">
      <label>Period</label>
      <select id="s-mode" aria-label="時間段">
        <option value="month">單月</option>
        <option value="range">自訂起迄</option>
      </select>
      <div id="s-month-group">
        <select id="s-month" aria-label="結算月份"></select>
      </div>
      <div id="s-range-group" style="display:none">
        <label>起</label>
        <select id="s-start" aria-label="起始月"></select>
        <label>迄</label>
        <select id="s-end" aria-label="結束月"></select>
      </div>
      <label>Sam&nbsp;%</label>
      <input id="s-sam" type="number" min="0" max="100" value="50">
      <label>Rita&nbsp;%</label>
      <input id="s-rita" type="number" min="0" max="100" value="50">
    </div>
    <div class="settle-callout" id="s-callout">
      <div class="settle-amount" id="s-callout-amount"></div>
      <div class="settle-note" id="s-callout-note"></div>
    </div>
    <div class="card section">
      <div class="section-title">Settlement Breakdown</div>
      <div class="table-wrap">
        <table>
          <thead><tr><th></th><th class="num">Sam</th><th class="num">Rita</th></tr></thead>
          <tbody id="s-rows"></tbody>
        </table>
      </div>
    </div>
    <div class="card section" style="margin-top:12px">
      <div class="section-title">By Category</div>
      <div class="table-wrap">
        <table>
          <thead><tr><th>Category</th><th class="num">Sam</th><th class="num">Rita</th></tr></thead>
          <tbody id="s-cats"></tbody>
        </table>
      </div>
    </div>
  </section>
</main>

<script>
const D = __DATA__;
const fmt = new Intl.NumberFormat('zh-TW', {maximumFractionDigits: 0});
const SHARE_KEY = 'hb-sam-share-v2';

function fmtSign(n) {
  const sign = n > 0 ? '+' : n < 0 ? '-' : '';
  return sign + fmt.format(Math.abs(Math.round(n || 0)));
}
function fmtAbs(n) { return fmt.format(Math.abs(Math.round(n || 0))); }
function cls(n) { return n > 0 ? 'good' : n < 0 ? 'bad' : ''; }
function el(id) { return document.getElementById(id); }

// ── horizontal bars ──────────────────────────────────────────
function drawHbars(targetId, data) {
  const entries = Object.entries(data)
    .filter(([, v]) => v !== 0)
    .map(([name, v]) => [name, Math.abs(v)])
    .sort((a, b) => b[1] - a[1]);
  const target = el(targetId);
  if (!entries.length) { target.innerHTML = '<div class="empty">無資料</div>'; return; }
  const maxVal = entries[0][1];
  const total = entries.reduce((s, [, v]) => s + v, 0);
  target.innerHTML = entries.map(([name, value], idx) => {
    const barW = Math.max(2, value / maxVal * 100);
    const pct = total > 0 ? (value / total * 100).toFixed(0) : 0;
    const isTop = idx === 0;
    return `<div class="hbar-item">
      <div class="hbar-name${isTop ? ' top' : ''}" title="${name}">${name}</div>
      <div class="hbar-track">
        <div class="hbar-fill${isTop ? ' top' : ''}" style="width:${barW}%"></div>
      </div>
      <div class="hbar-meta${isTop ? ' top' : ''}">${fmtAbs(value)}<span class="hbar-pct">${pct}%</span></div>
    </div>`;
  }).join('');
}

// ── trend chart ──────────────────────────────────────────────
function drawTrend(targetId, scope, chartMonths) {
  const W = 680, H = 200, padL = 8, padR = 60, topP = 24, botP = 28;
  const plotW = W - padL - padR;
  const plotH = H - topP - botP;
  const n = chartMonths.length;
  const target = el(targetId);
  if (!n) { target.innerHTML = '<div class="empty">無資料</div>'; return; }

  const incData = chartMonths.map(m => scope.months[m].income);
  const expData = chartMonths.map(m => Math.abs(scope.months[m].expense));
  const maxV = Math.max(1000, ...incData, ...expData);
  const botY = topP + plotH;

  const xPos = i => padL + (n <= 1 ? plotW / 2 : i * plotW / (n - 1));
  const yPos = v => topP + plotH - (v / maxV * plotH);

  function areaPath(arr) {
    const pts = arr.map((v, i) => `${xPos(i)},${yPos(v)}`).join(' L');
    return `M${xPos(0)},${botY} L${pts} L${xPos(n-1)},${botY}Z`;
  }
  function linePath(arr) {
    return arr.map((v, i) => `${i === 0 ? 'M' : 'L'}${xPos(i)},${yPos(v)}`).join(' ');
  }

  // avg expense reference line
  const avgExp = expData.reduce((s, v) => s + v, 0) / n;
  const avgY = yPos(avgExp);

  // x-axis labels (show at most 8)
  const step = Math.max(1, Math.ceil(n / 8));
  const xLabels = chartMonths
    .map((m, i) => (i % step === 0 || i === n - 1)
      ? `<text x="${xPos(i)}" y="${H - 6}" text-anchor="middle" font-size="10" fill="#94a3b8">${m.slice(5)}</text>`
      : '')
    .join('');

  // last-point annotations (right side)
  const lastInc = incData[n - 1];
  const lastExp = expData[n - 1];
  const annX = xPos(n - 1) + 5;
  const incAnnot = lastInc > 0
    ? `<text x="${annX}" y="${yPos(lastInc) + 4}" font-size="10" fill="#15803d" font-weight="700">${fmtAbs(lastInc)}</text>` : '';
  const expAnnot = lastExp > 0
    ? `<text x="${annX}" y="${yPos(lastExp) + 4}" font-size="10" fill="#dc2626" font-weight="700">${fmtAbs(lastExp)}</text>` : '';

  // inline legend (top-left)
  const legend = `
    <rect x="${padL}" y="4" width="10" height="3" rx="1" fill="#15803d" fill-opacity=".6"/>
    <text x="${padL + 14}" y="11" font-size="10" fill="#15803d">收入</text>
    <rect x="${padL + 44}" y="4" width="10" height="3" rx="1" fill="#dc2626" fill-opacity=".6"/>
    <text x="${padL + 58}" y="11" font-size="10" fill="#dc2626">支出</text>`;

  // avg label
  const avgLabel = `<text x="${xPos(Math.floor(n/2))}" y="${avgY - 5}" font-size="10" fill="#94a3b8" text-anchor="middle">月均 ${fmtAbs(avgExp)}</text>`;

  target.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="收支趨勢">
    <line x1="${padL}" y1="${botY}" x2="${W - padR + 4}" y2="${botY}" stroke="#e2e8f0"/>
    <line x1="${padL}" y1="${avgY}" x2="${xPos(n-1)}" y2="${avgY}" stroke="#e2e8f0" stroke-dasharray="4 3"/>
    ${avgLabel}
    <path d="${areaPath(incData)}" fill="#15803d" fill-opacity=".12"/>
    <path d="${areaPath(expData)}" fill="#dc2626" fill-opacity=".18"/>
    <path d="${linePath(incData)}" fill="none" stroke="#15803d" stroke-width="1.5" stroke-opacity=".75"/>
    <path d="${linePath(expData)}" fill="none" stroke="#dc2626" stroke-width="1.5" stroke-opacity=".75"/>
    ${incAnnot}${expAnnot}
    ${xLabels}
    ${legend}
  </svg>`;
}

// ── insight banner ───────────────────────────────────────────
function buildInsight(scope, month) {
  const d = scope.months[month];
  if (!d) return '請選擇月份。';

  const expAbs = Math.abs(d.expense);
  const net = d.net;
  const delta = d.expense_delta;

  const catEntries = Object.entries(d.category_expense).sort((a, b) => a[1] - b[1]);
  const topCat = catEntries[0];

  const parts = [];

  if (net > 0) {
    parts.push(`本月<strong class="good"> 結餘 +${fmt.format(Math.round(net))} TWD</strong>`);
  } else {
    parts.push(`本月支出 <strong class="bad">${fmt.format(expAbs)} TWD</strong>`);
  }

  if (delta !== 0) {
    const prevExpAbs = Math.abs(d.expense - delta);
    if (prevExpAbs > 0) {
      const changeAbs = Math.abs(delta);
      const pct = (changeAbs / prevExpAbs * 100).toFixed(0);
      if (delta < 0) {
        parts.push(`較上月多 <strong class="bad">${fmt.format(changeAbs)} TWD (&uarr;${pct}%)</strong>`);
      } else {
        parts.push(`較上月少 <strong class="good">${fmt.format(changeAbs)} TWD (&darr;${pct}%)</strong>`);
      }
    }
  }

  if (topCat && expAbs > 0) {
    const pct = (Math.abs(topCat[1]) / expAbs * 100).toFixed(0);
    parts.push(`最大支出 <strong>${topCat[0]}</strong>（${pct}%）`);
  }

  // style the banner based on situation
  const insightEl = el('m-insight');
  if (net > 0) {
    insightEl.style.background = '#f0fdf4';
    insightEl.style.borderColor = '#bbf7d0';
  } else if (delta < 0 && Math.abs(d.expense - delta) > 0 &&
             Math.abs(delta) / Math.abs(d.expense - delta) > 0.15) {
    insightEl.style.background = '#fff7ed';
    insightEl.style.borderColor = '#fed7aa';
  } else {
    insightEl.style.background = '#eff6ff';
    insightEl.style.borderColor = '#bfdbfe';
  }

  return parts.join('，') + '。';
}

// ── tab routing ──────────────────────────────────────────────
document.querySelectorAll('.tab').forEach(btn => {
  btn.addEventListener('click', () => {
    const tab = btn.dataset.tab;
    document.querySelectorAll('.tab').forEach(b => b.classList.toggle('active', b === btn));
    document.querySelectorAll('.panel').forEach(p => p.classList.toggle('active', p.dataset.panel === tab));
  });
});

// ── overview ─────────────────────────────────────────────────
function renderOverview() {
  const scope = D.scopes.all;
  const months = scope.month_order;
  let income = 0, expense = 0;
  const catTotals = {};
  months.forEach(m => {
    const d = scope.months[m];
    income += d.income;
    expense += d.expense;
    Object.entries(d.category_expense).forEach(([k, v]) => {
      catTotals[k] = (catTotals[k] || 0) + v;
    });
  });
  const net = income + expense;
  const avgExp = expense / (months.length || 1);
  const allTx = months.flatMap(m => scope.months[m].transactions);
  const expTxCount = allTx.filter(t => t.amount_twd < 0).length;

  el('ov-income').textContent = fmtSign(income);
  el('ov-income').className = 'kpi-value good';
  el('ov-expense').textContent = fmtSign(expense);
  el('ov-expense').className = 'kpi-value bad';
  el('ov-net').textContent = fmtSign(net);
  el('ov-net').className = 'kpi-value ' + cls(net);
  el('ov-income-hint').textContent = `${months.length} 個月份，${months[0] || ''} 起`;
  el('ov-expense-hint').textContent = `${expTxCount} 筆支出`;
  el('ov-net-hint').textContent = `月均支出 ${fmtAbs(avgExp)} TWD`;

  drawTrend('ov-trend', scope, months);
  drawHbars('ov-cats', catTotals);

  // year comparison bars
  const yearly = {};
  months.forEach(m => {
    const y = m.slice(0, 4);
    if (!yearly[y]) yearly[y] = 0;
    yearly[y] += scope.months[m].expense;
  });
  const yearEntries = Object.entries(yearly).filter(([, v]) => v !== 0).sort();
  const yearMax = Math.max(...yearEntries.map(([, v]) => Math.abs(v)));
  el('ov-years').innerHTML = yearEntries.map(([year, value]) => {
    const barW = Math.max(2, Math.abs(value) / yearMax * 100);
    return `<div class="hbar-item">
      <div class="hbar-name">${year}</div>
      <div class="hbar-track"><div class="hbar-fill" style="width:${barW}%"></div></div>
      <div class="hbar-meta">${fmtAbs(value)}</div>
    </div>`;
  }).join('') || '<div class="empty">無資料</div>';
}

// ── monthly ──────────────────────────────────────────────────
const mScopeEl = el('m-scope');
const mMonthEl = el('m-month');

function currentScope() { return D.scopes[mScopeEl.value]; }

function populateMonthSelect() {
  const scope = currentScope();
  mMonthEl.innerHTML = [...scope.month_order].reverse()
    .map(m => `<option value="${m}">${m}</option>`).join('');
  mMonthEl.value = scope.latest_month;
}

function renderMonthly() {
  const scope = currentScope();
  const month = mMonthEl.value || scope.latest_month;
  const d = scope.months[month];
  if (!d) return;

  el('m-insight').innerHTML = buildInsight(scope, month);

  el('m-income').textContent = fmtSign(d.income);
  el('m-income').className = 'kpi-value good';
  el('m-expense').textContent = fmtSign(d.expense);
  el('m-expense').className = 'kpi-value bad';
  el('m-net').textContent = fmtSign(d.net);
  el('m-net').className = 'kpi-value ' + cls(d.net);

  el('m-income-hint').textContent = `${month}`;
  el('m-expense-hint').textContent = `${d.transactions.filter(t => t.amount_twd < 0).length} 筆`;

  const deltaAbs = Math.abs(d.expense_delta);
  const deltaDir = d.expense_delta > 0 ? '↓ 較上月少' : d.expense_delta < 0 ? '↑ 較上月多' : '';
  el('m-net-hint').textContent = deltaDir
    ? `${deltaDir} ${fmt.format(deltaAbs)} TWD`
    : '首月，無比較';

  drawTrend('m-trend', scope, scope.month_order.slice(-12));
  drawHbars('m-cats', d.category_expense);

  const top5 = [...d.transactions]
    .filter(t => t.amount_twd < 0)
    .sort((a, b) => a.amount_twd - b.amount_twd)
    .slice(0, 5);
  el('m-top5').innerHTML = top5.length
    ? top5.map(t =>
        `<tr><td>${t.date}</td><td>${t.category}</td><td>${t.item}</td><td>${t.who}</td>` +
        `<td class="num bad">${fmtSign(t.amount_twd)}</td></tr>`).join('')
    : '<tr><td colspan="5" class="empty">無資料</td></tr>';

  el('m-all').innerHTML = d.transactions.length
    ? d.transactions.map(t =>
        `<tr><td>${t.date}</td><td>${t.category}</td><td>${t.item}</td>` +
        `<td>${t.source}</td><td>${t.who}</td>` +
        `<td class="num ${cls(t.amount_twd)}">${fmtSign(t.amount_twd)}</td></tr>`).join('')
    : '<tr><td colspan="6" class="empty">無資料</td></tr>';
}

mScopeEl.addEventListener('change', () => { populateMonthSelect(); renderMonthly(); });
mMonthEl.addEventListener('change', renderMonthly);

// ── settlement ───────────────────────────────────────────────
const sModeEl = el('s-mode');
const sMonthEl = el('s-month');
const sStartEl = el('s-start');
const sEndEl = el('s-end');
const sSamEl = el('s-sam');
const sRitaEl = el('s-rita');
const allScope = D.scopes.all;
const allTx = allScope.month_order.flatMap(m => allScope.months[m].transactions);

function populateSettlementMonths() {
  const months = [...allScope.month_order].reverse();
  [sMonthEl, sStartEl, sEndEl].forEach(sel => {
    sel.innerHTML = months.map(m => `<option value="${m}">${m}</option>`).join('');
  });
  const latest = allScope.latest_month;
  const earliest = allScope.month_order[0] || latest;
  sMonthEl.value = latest;
  sStartEl.value = earliest;
  sEndEl.value = latest;
}
function loadShare() {
  const saved = Number(localStorage.getItem(SHARE_KEY));
  if (Number.isFinite(saved) && saved >= 0 && saved <= 100) {
    sSamEl.value = Math.round(saved);
    sRitaEl.value = 100 - Math.round(saved);
  }
}
function syncShare(changed) {
  const raw = Math.max(0, Math.min(100, Number(changed === 'sam' ? sSamEl.value : sRitaEl.value) || 0));
  if (changed === 'sam') { sSamEl.value = Math.round(raw); sRitaEl.value = 100 - Math.round(raw); }
  else { sRitaEl.value = Math.round(raw); sSamEl.value = 100 - Math.round(raw); }
  localStorage.setItem(SHARE_KEY, sSamEl.value);
}
function getRange() {
  if (sModeEl.value === 'month') return {start: sMonthEl.value, end: sMonthEl.value};
  let [a, b] = [sStartEl.value, sEndEl.value];
  if (a > b) { [a, b] = [b, a]; sStartEl.value = a; sEndEl.value = b; }
  return {start: a, end: b};
}
function renderSettlement() {
  const {start, end} = getRange();
  const txs = allTx.filter(t => t.month >= start && t.month <= end);
  const actual = {Sam: 0, Rita: 0};
  const income = {Sam: 0, Rita: 0};
  const catMap = {};
  txs.forEach(t => {
    const p = t.who === 'Rita' ? 'Rita' : 'Sam';
    if (t.amount_twd < 0) {
      actual[p] += Math.abs(t.amount_twd);
      if (!catMap[t.category]) catMap[t.category] = {Sam: 0, Rita: 0};
      catMap[t.category][p] += Math.abs(t.amount_twd);
    } else {
      income[p] += t.amount_twd;
    }
  });
  const samRatio = Number(sSamEl.value) / 100;
  const net = {Sam: actual.Sam - income.Sam, Rita: actual.Rita - income.Rita};
  const totalNet = net.Sam + net.Rita;
  const due = {Sam: totalNet * samRatio, Rita: totalNet * (1 - samRatio)};
  const delta = {Sam: net.Sam - due.Sam, Rita: net.Rita - due.Rita};
  const transfer = Math.round(Math.abs(delta.Sam));
  const periodLabel = start === end ? start : `${start} ~ ${end}`;
  const shareLabel = `Sam ${sSamEl.value}% / Rita ${sRitaEl.value}%`;

  const callout = el('s-callout');
  if (transfer === 0) {
    el('s-callout-amount').textContent = '已平衡，無需結算';
    callout.style.background = 'linear-gradient(135deg, #f0fdf4, #fff)';
    callout.style.borderColor = '#bbf7d0';
  } else {
    const direction = delta.Sam > 0 ? 'Rita 補給 Sam' : 'Sam 補給 Rita';
    el('s-callout-amount').textContent = `${direction}　${fmt.format(transfer)} TWD`;
    callout.style.background = 'linear-gradient(135deg, #eff6ff, #fff)';
    callout.style.borderColor = '#bfdbfe';
  }
  el('s-callout-note').textContent = `${periodLabel} · ${shareLabel}`;

  el('s-rows').innerHTML = `
    <tr><td>實際支付</td><td class="num">${fmtAbs(actual.Sam)}</td><td class="num">${fmtAbs(actual.Rita)}</td></tr>
    <tr><td>收入抵扣</td><td class="num">${fmtAbs(income.Sam)}</td><td class="num">${fmtAbs(income.Rita)}</td></tr>
    <tr><td>淨支出</td><td class="num">${fmtAbs(net.Sam)}</td><td class="num">${fmtAbs(net.Rita)}</td></tr>
    <tr><td>應付（依比例）</td><td class="num">${fmtAbs(due.Sam)}</td><td class="num">${fmtAbs(due.Rita)}</td></tr>
    <tr><td><strong>差額</strong></td>
      <td class="num ${cls(delta.Sam)}"><strong>${fmtSign(delta.Sam)}</strong></td>
      <td class="num ${cls(delta.Rita)}"><strong>${fmtSign(delta.Rita)}</strong></td>
    </tr>`;

  const rows = Object.entries(catMap).sort((a, b) => (b[1].Sam + b[1].Rita) - (a[1].Sam + a[1].Rita));
  el('s-cats').innerHTML = rows.length
    ? rows.map(([cat, v]) =>
        `<tr><td>${cat}</td><td class="num">${fmtAbs(v.Sam)}</td><td class="num">${fmtAbs(v.Rita)}</td></tr>`).join('')
    : '<tr><td colspan="3" class="empty">無資料</td></tr>';
}
function toggleSettlementGroups() {
  const isMonth = sModeEl.value === 'month';
  el('s-month-group').style.display = isMonth ? '' : 'none';
  el('s-range-group').style.display = isMonth ? 'none' : '';
}

sModeEl.addEventListener('change', () => { toggleSettlementGroups(); renderSettlement(); });
sMonthEl.addEventListener('change', renderSettlement);
sStartEl.addEventListener('change', renderSettlement);
sEndEl.addEventListener('change', renderSettlement);
sSamEl.addEventListener('input', () => { syncShare('sam'); renderSettlement(); });
sRitaEl.addEventListener('input', () => { syncShare('rita'); renderSettlement(); });

// ── init ─────────────────────────────────────────────────────
populateMonthSelect();
populateSettlementMonths();
loadShare();
toggleSettlementGroups();
renderOverview();
renderMonthly();
renderSettlement();
</script>
</body>
</html>"""
    return template.replace("__GENERATED__", escape(data["generated_at"])).replace("__DATA__", data_json)


def write_dashboard(db_path=DB_PATH, output_path=OUTPUT_PATH):
    data = build_dashboard_data(db_path)
    output_path.write_text(render_html(data), encoding="utf-8")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate the House Budget HTML dashboard.")
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    output = write_dashboard(args.db, args.output)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
