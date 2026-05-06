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
    template = """<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>House Budget Dashboard</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f5f7fa;
      --panel: #ffffff;
      --line: #d9e0e8;
      --text: #1f2933;
      --muted: #64748b;
      --good: #168255;
      --bad: #c24132;
      --warn: #b45309;
      --blue: #2563eb;
      --ink: #0f172a;
      --shadow: 0 1px 2px rgba(15, 23, 42, 0.06);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Segoe UI", "Noto Sans TC", Arial, sans-serif;
      color: var(--text);
      background: var(--bg);
      letter-spacing: 0;
    }
    header {
      position: sticky;
      top: 0;
      z-index: 5;
      background: rgba(245, 247, 250, 0.94);
      border-bottom: 1px solid var(--line);
      backdrop-filter: blur(12px);
    }
    .bar {
      max-width: 1180px;
      margin: 0 auto;
      padding: 14px 18px 12px;
      display: grid;
      gap: 12px;
      align-items: center;
    }
    h1 {
      margin: 0;
      font-size: 22px;
      line-height: 1.2;
      color: var(--ink);
    }
    .meta {
      margin-top: 3px;
      color: var(--muted);
      font-size: 13px;
    }
    .tabs {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      padding-bottom: 4px;
    }
    .tab-btn {
      height: 38px;
      border: 1px solid var(--line);
      background: var(--panel);
      color: var(--muted);
      border-radius: 999px;
      padding: 0 14px;
      font-size: 14px;
      font-weight: 600;
      cursor: pointer;
    }
    .tab-btn.active {
      color: #ffffff;
      background: var(--ink);
      border-color: var(--ink);
    }
    main {
      max-width: 1180px;
      margin: 0 auto;
      padding: 18px;
    }
    .tab-panel { display: none; }
    .tab-panel.active { display: block; }
    .toolbar {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-bottom: 12px;
      align-items: center;
    }
    .toolbar-group {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      align-items: center;
    }
    .toolbar label {
      font-size: 13px;
      color: var(--muted);
    }
    select, input[type="number"] {
      height: 36px;
      border: 1px solid var(--line);
      background: var(--panel);
      color: var(--text);
      border-radius: 6px;
      padding: 0 10px;
      font-size: 14px;
    }
    input[type="number"] { width: 92px; }
    .kpis {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
    }
    .card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }
    .kpi {
      padding: 14px;
      min-height: 104px;
    }
    .label {
      color: var(--muted);
      font-size: 13px;
    }
    .value {
      margin-top: 10px;
      font-size: 28px;
      line-height: 1;
      font-weight: 700;
      color: var(--ink);
      overflow-wrap: anywhere;
    }
    .hint {
      margin-top: 10px;
      color: var(--muted);
      font-size: 12px;
    }
    .good { color: var(--good); }
    .bad { color: var(--bad); }
    .warn { color: var(--warn); }
    .grid {
      display: grid;
      grid-template-columns: minmax(0, 1.45fr) minmax(300px, 0.85fr);
      gap: 12px;
      margin-top: 12px;
    }
    .section {
      padding: 14px;
      min-width: 0;
    }
    h2 {
      margin: 0 0 12px;
      font-size: 16px;
      color: var(--ink);
    }
    svg {
      width: 100%;
      height: auto;
      display: block;
    }
    .bars {
      display: grid;
      gap: 9px;
    }
    .bar-row {
      display: grid;
      grid-template-columns: minmax(110px, 150px) 1fr minmax(86px, auto);
      gap: 10px;
      align-items: center;
      font-size: 13px;
    }
    .track {
      height: 10px;
      background: #e7edf4;
      border-radius: 999px;
      overflow: hidden;
    }
    .fill {
      height: 100%;
      background: var(--bad);
      border-radius: 999px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }
    th, td {
      border-bottom: 1px solid var(--line);
      padding: 9px 8px;
      text-align: left;
      vertical-align: top;
    }
    th {
      color: var(--muted);
      font-weight: 600;
      background: #f8fafc;
    }
    td.num, th.num {
      text-align: right;
      white-space: nowrap;
    }
    .table-wrap {
      max-height: 480px;
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
    }
    .empty {
      color: var(--muted);
      padding: 14px;
    }
    .settlement-summary {
      display: grid;
      gap: 12px;
      margin-top: 12px;
    }
    .settlement-callout {
      padding: 18px;
      border-radius: 10px;
      background: linear-gradient(135deg, #eef4ff, #ffffff);
      border: 1px solid #c8d6f8;
      font-size: 28px;
      font-weight: 700;
      color: var(--ink);
    }
    .settlement-note {
      color: var(--muted);
      font-size: 13px;
    }
    @media (max-width: 900px) {
      .kpis { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .grid { grid-template-columns: 1fr; }
    }
    @media (max-width: 560px) {
      main { padding: 12px; }
      .kpis { grid-template-columns: 1fr; }
      .value { font-size: 24px; }
      .bar-row { grid-template-columns: 1fr; gap: 4px; }
      .settlement-callout { font-size: 22px; }
      th:nth-child(4), td:nth-child(4),
      th:nth-child(6), td:nth-child(6) { display: none; }
    }
  </style>
</head>
<body>
  <header>
    <div class="bar">
      <div>
        <h1>House Budget Dashboard</h1>
        <div class="meta">Generated __GENERATED__ · TWD view</div>
      </div>
      <div class="tabs" role="tablist" aria-label="Dashboard tabs">
        <button class="tab-btn active" type="button" data-tab="overview">Overview</button>
        <button class="tab-btn" type="button" data-tab="monthly">Monthly</button>
        <button class="tab-btn" type="button" data-tab="settlement">Sam &amp; Rita</button>
      </div>
    </div>
  </header>
  <main>
    <section class="tab-panel active" data-tab-panel="overview">
      <section class="kpis">
        <div class="card kpi"><div class="label">總收入</div><div class="value good" id="overviewIncomeValue"></div><div class="hint">全期累計，不受月份篩選影響</div></div>
        <div class="card kpi"><div class="label">總支出</div><div class="value bad" id="overviewExpenseValue"></div><div class="hint" id="overviewExpenseHint"></div></div>
        <div class="card kpi"><div class="label">淨額</div><div class="value" id="overviewNetValue"></div><div class="hint" id="overviewNetHint"></div></div>
        <div class="card kpi"><div class="label">月均支出</div><div class="value bad" id="overviewAvgExpenseValue"></div><div class="hint">全期間平均每月支出</div></div>
      </section>
      <section class="grid">
        <div class="card section">
          <h2>完整月份趨勢</h2>
          <div id="overviewTrendChart"></div>
        </div>
        <div class="card section">
          <h2>各分類累計支出</h2>
          <div class="bars" id="overviewCategoryBars"></div>
        </div>
      </section>
      <section class="card section" style="margin-top:12px">
        <h2>年度對比表</h2>
        <div class="table-wrap">
          <table>
            <thead><tr><th>年度</th><th class="num">支出合計（TWD）</th></tr></thead>
            <tbody id="overviewYearRows"></tbody>
          </table>
        </div>
      </section>
    </section>

    <section class="tab-panel" data-tab-panel="monthly">
      <div class="toolbar">
        <div class="toolbar-group">
          <select id="scopeSelect" aria-label="資料範圍">
            <option value="all">全部</option>
            <option value="household">家庭</option>
            <option value="travel">澳洲旅遊</option>
          </select>
          <select id="monthSelect" aria-label="月份"></select>
        </div>
      </div>
      <section class="kpis">
        <div class="card kpi"><div class="label">本月收入</div><div class="value good" id="incomeValue"></div><div class="hint" id="incomeHint"></div></div>
        <div class="card kpi"><div class="label">本月支出</div><div class="value bad" id="expenseValue"></div><div class="hint" id="expenseHint"></div></div>
        <div class="card kpi"><div class="label">本月淨額</div><div class="value" id="netValue"></div><div class="hint" id="netHint"></div></div>
        <div class="card kpi"><div class="label">較上月支出變化</div><div class="value" id="deltaValue"></div><div class="hint">負值代表支出增加</div></div>
      </section>
      <section class="grid">
        <div class="card section">
          <h2>近 12 個月收支趨勢</h2>
          <div id="trendChart"></div>
        </div>
        <div class="card section">
          <h2>本月分類支出</h2>
          <div class="bars" id="categoryBars"></div>
        </div>
      </section>
      <section class="grid">
        <div class="card section">
          <h2>本月大筆支出 Top 10</h2>
          <div class="table-wrap"><table><thead><tr><th>日期</th><th>分類</th><th>項目</th><th>付款人</th><th class="num">金額</th></tr></thead><tbody id="largeRows"></tbody></table></div>
        </div>
        <div class="card section">
          <h2>付款人支出</h2>
          <div class="bars" id="payerBars"></div>
        </div>
      </section>
      <section class="card section" style="margin-top:12px">
        <h2>本月交易明細</h2>
        <div class="table-wrap"><table><thead><tr><th>日期</th><th>分類</th><th>項目</th><th>來源</th><th>付款人</th><th class="num">TWD</th></tr></thead><tbody id="txRows"></tbody></table></div>
      </section>
    </section>

    <section class="tab-panel" data-tab-panel="settlement">
      <div class="toolbar">
        <div class="toolbar-group">
          <label for="settlementMode">時間段</label>
          <select id="settlementMode" aria-label="時間段模式">
            <option value="month">單月</option>
            <option value="range">自訂起迄</option>
          </select>
        </div>
        <div class="toolbar-group" id="settlementMonthGroup">
          <label for="settlementMonth">月份</label>
          <select id="settlementMonth" aria-label="結算月份"></select>
        </div>
        <div class="toolbar-group" id="settlementRangeGroup" style="display:none">
          <label for="settlementStartMonth">起</label>
          <select id="settlementStartMonth" aria-label="結算起始月份"></select>
          <label for="settlementEndMonth">迄</label>
          <select id="settlementEndMonth" aria-label="結算結束月份"></select>
        </div>
        <div class="toolbar-group">
          <label for="samShareInput">Sam %</label>
          <input id="samShareInput" type="number" min="0" max="100" step="1" value="50">
          <label for="ritaShareInput">Rita %</label>
          <input id="ritaShareInput" type="number" min="0" max="100" step="1" value="50">
        </div>
      </div>
      <div class="settlement-note">比例會自動互鎖且儲存在 localStorage；時間段改變時即時重算。</div>
      <section class="settlement-summary">
        <div class="card section">
          <h2>結算結果</h2>
          <div class="table-wrap">
            <table>
              <thead><tr><th></th><th class="num">Sam</th><th class="num">Rita</th></tr></thead>
              <tbody id="settlementRows"></tbody>
            </table>
          </div>
        </div>
        <div class="settlement-callout" id="settlementCallout"></div>
      </section>
      <section class="card section" style="margin-top:12px">
        <h2>各分類明細</h2>
        <div class="table-wrap">
          <table>
            <thead><tr><th>Category</th><th class="num">Sam</th><th class="num">Rita</th></tr></thead>
            <tbody id="settlementCategoryRows"></tbody>
          </table>
        </div>
      </section>
    </section>
  </main>
  <script>
    const DASHBOARD_DATA = __DATA__;
    const fmt = new Intl.NumberFormat('zh-TW', { maximumFractionDigits: 0 });
    const scopeSelect = document.getElementById('scopeSelect');
    const monthSelect = document.getElementById('monthSelect');
    const settlementMode = document.getElementById('settlementMode');
    const settlementMonth = document.getElementById('settlementMonth');
    const settlementStartMonth = document.getElementById('settlementStartMonth');
    const settlementEndMonth = document.getElementById('settlementEndMonth');
    const settlementMonthGroup = document.getElementById('settlementMonthGroup');
    const settlementRangeGroup = document.getElementById('settlementRangeGroup');
    const samShareInput = document.getElementById('samShareInput');
    const ritaShareInput = document.getElementById('ritaShareInput');
    const STORAGE_KEY = 'house-budget-sam-share';
    const allScope = DASHBOARD_DATA.scopes.all;
    const allTransactions = allScope.month_order.flatMap(month => allScope.months[month].transactions);

    function money(n) {
      const sign = n > 0 ? '+' : n < 0 ? '-' : '';
      return sign + fmt.format(Math.abs(Math.round(n || 0)));
    }
    function moneyAbs(n) {
      return fmt.format(Math.abs(Math.round(n || 0)));
    }
    function cls(n) {
      return n > 0 ? 'good' : n < 0 ? 'bad' : '';
    }
    function currentScope() {
      return DASHBOARD_DATA.scopes[scopeSelect.value];
    }
    function populateMonths() {
      const scope = currentScope();
      monthSelect.innerHTML = '';
      [...scope.month_order].reverse().forEach(month => {
        const option = document.createElement('option');
        option.value = month;
        option.textContent = month;
        monthSelect.appendChild(option);
      });
      monthSelect.value = scope.latest_month;
    }
    function populateSettlementMonths() {
      const months = [...allScope.month_order].reverse();
      [settlementMonth, settlementStartMonth, settlementEndMonth].forEach(select => {
        select.innerHTML = '';
        months.forEach(month => {
          const option = document.createElement('option');
          option.value = month;
          option.textContent = month;
          select.appendChild(option);
        });
      });
      const latest = allScope.latest_month;
      const earliest = allScope.month_order[0] || latest;
      settlementMonth.value = latest;
      settlementStartMonth.value = earliest;
      settlementEndMonth.value = latest;
    }
    function drawTrend(targetId, scope, selectedMonth, months) {
      const chartMonths = months || scope.month_order.slice(-12);
      const width = 720, height = 260, pad = 34;
      const values = [];
      chartMonths.forEach(month => {
        const detail = scope.months[month];
        values.push(detail.income, detail.expense, detail.net);
      });
      const max = Math.max(1000, ...values.map(value => Math.abs(value)));
      const x = idx => pad + (chartMonths.length <= 1 ? 0 : idx * (width - pad * 2) / (chartMonths.length - 1));
      const y = value => height / 2 - (value / max) * (height / 2 - pad);
      const points = key => chartMonths.map((month, idx) => `${x(idx)},${y(scope.months[month][key])}`).join(' ');
      document.getElementById(targetId).innerHTML = `
        <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="收支趨勢">
          <line x1="${pad}" y1="${height / 2}" x2="${width - pad}" y2="${height / 2}" stroke="#cbd5e1" />
          ${chartMonths.map((month, idx) => `<text x="${x(idx)}" y="${height - 8}" text-anchor="middle" font-size="11" fill="#64748b">${month.slice(5)}</text>`).join('')}
          <polyline points="${points('income')}" fill="none" stroke="#168255" stroke-width="3" />
          <polyline points="${points('expense')}" fill="none" stroke="#c24132" stroke-width="3" />
          <polyline points="${points('net')}" fill="none" stroke="#2563eb" stroke-width="3" />
          ${chartMonths.map((month, idx) => `<circle cx="${x(idx)}" cy="${y(scope.months[month].net)}" r="${month === selectedMonth ? 5 : 3}" fill="#2563eb" />`).join('')}
          <text x="${pad}" y="16" font-size="12" fill="#168255">收入</text>
          <text x="${pad + 46}" y="16" font-size="12" fill="#c24132">支出</text>
          <text x="${pad + 92}" y="16" font-size="12" fill="#2563eb">淨額</text>
        </svg>`;
    }
    function drawBars(targetId, data, formatter = money) {
      const entries = Object.entries(data).filter(([, value]) => value !== 0);
      const max = Math.max(1, ...entries.map(([, value]) => Math.abs(value)));
      const target = document.getElementById(targetId);
      if (!entries.length) {
        target.innerHTML = '<div class="empty">沒有支出資料</div>';
        return;
      }
      target.innerHTML = entries.map(([name, value]) => `
        <div class="bar-row">
          <div>${name}</div>
          <div class="track"><div class="fill" style="width:${Math.max(2, Math.abs(value) / max * 100)}%"></div></div>
          <div class="num bad">${formatter(value)}</div>
        </div>`).join('');
    }
    function renderTable(targetId, rows, compact = false) {
      const target = document.getElementById(targetId);
      if (!rows.length) {
        target.innerHTML = '<tr><td colspan="6" class="empty">沒有資料</td></tr>';
        return;
      }
      target.innerHTML = rows.map(tx => compact
        ? `<tr><td>${tx.date}</td><td>${tx.category}</td><td>${tx.item}</td><td>${tx.who}</td><td class="num bad">${money(tx.amount_twd)}</td></tr>`
        : `<tr><td>${tx.date}</td><td>${tx.category}</td><td>${tx.item}</td><td>${tx.source}</td><td>${tx.who}</td><td class="num ${cls(tx.amount_twd)}">${money(tx.amount_twd)}</td></tr>`
      ).join('');
    }
    function renderOverview() {
      const months = allScope.month_order;
      let income = 0;
      let expense = 0;
      const categoryTotals = {};
      months.forEach(month => {
        const detail = allScope.months[month];
        income += detail.income;
        expense += detail.expense;
        Object.entries(detail.category_expense).forEach(([category, value]) => {
          categoryTotals[category] = (categoryTotals[category] || 0) + value;
        });
      });
      const net = income + expense;
      const monthCount = months.length || 1;
      document.getElementById('overviewIncomeValue').textContent = money(income);
      document.getElementById('overviewExpenseValue').textContent = money(expense);
      document.getElementById('overviewNetValue').textContent = money(net);
      document.getElementById('overviewNetValue').className = 'value ' + cls(net);
      document.getElementById('overviewAvgExpenseValue').textContent = money(expense / monthCount);
      document.getElementById('overviewExpenseHint').textContent = `${months.length} 個月份，${allScope.month_order[0] || '-'} 起`;
      document.getElementById('overviewNetHint').textContent = `${allTransactions.length} 筆已換算 TWD 交易`;
      drawTrend('overviewTrendChart', allScope, allScope.latest_month, months);
      drawBars('overviewCategoryBars', categoryTotals);
      const yearly = { '2024': 0, '2025': 0, '2026': 0 };
      months.forEach(month => {
        const year = month.slice(0, 4);
        if (year in yearly) yearly[year] += allScope.months[month].expense;
      });
      document.getElementById('overviewYearRows').innerHTML = Object.entries(yearly)
        .map(([year, value]) => `<tr><td>${year}</td><td class="num">${moneyAbs(value)}</td></tr>`)
        .join('');
    }
    function renderMonthly() {
      const scope = currentScope();
      const month = monthSelect.value || scope.latest_month;
      const detail = scope.months[month];
      if (!detail) return;
      document.getElementById('incomeValue').textContent = money(detail.income);
      document.getElementById('expenseValue').textContent = money(detail.expense);
      document.getElementById('netValue').textContent = money(detail.net);
      document.getElementById('netValue').className = 'value ' + cls(detail.net);
      document.getElementById('deltaValue').textContent = money(detail.expense_delta);
      document.getElementById('deltaValue').className = 'value ' + (detail.expense_delta < 0 ? 'bad' : detail.expense_delta > 0 ? 'good' : '');
      document.getElementById('incomeHint').textContent = `${month} · ${scopeSelect.options[scopeSelect.selectedIndex].textContent}`;
      document.getElementById('expenseHint').textContent = `${detail.transactions.filter(tx => tx.amount_twd < 0).length} 筆支出`;
      document.getElementById('netHint').textContent = `${detail.transactions.length} 筆已換算 TWD 交易`;
      drawTrend('trendChart', scope, month, scope.month_order.slice(-12));
      drawBars('categoryBars', detail.category_expense);
      drawBars('payerBars', detail.payer_expense);
      renderTable('largeRows', detail.large_expenses, true);
      renderTable('txRows', detail.transactions, false);
    }
    function loadSettlementShare() {
      const saved = Number(localStorage.getItem(STORAGE_KEY));
      if (Number.isFinite(saved) && saved >= 0 && saved <= 100) {
        samShareInput.value = String(Math.round(saved));
        ritaShareInput.value = String(100 - Math.round(saved));
      }
    }
    function syncShareInputs(changed) {
      const raw = Number(changed === 'sam' ? samShareInput.value : ritaShareInput.value);
      const value = Math.max(0, Math.min(100, Number.isFinite(raw) ? raw : 0));
      if (changed === 'sam') {
        samShareInput.value = String(Math.round(value));
        ritaShareInput.value = String(100 - Math.round(value));
      } else {
        ritaShareInput.value = String(Math.round(value));
        samShareInput.value = String(100 - Math.round(value));
      }
      localStorage.setItem(STORAGE_KEY, samShareInput.value);
    }
    function settlementRange() {
      if (settlementMode.value === 'month') {
        const month = settlementMonth.value;
        return { start: month, end: month };
      }
      let start = settlementStartMonth.value;
      let end = settlementEndMonth.value;
      if (start > end) {
        [start, end] = [end, start];
        settlementStartMonth.value = start;
        settlementEndMonth.value = end;
      }
      return { start, end };
    }
    function renderSettlement() {
      const range = settlementRange();
      const txs = allTransactions.filter(tx => tx.amount_twd < 0 && tx.month >= range.start && tx.month <= range.end);
      const actual = { Sam: 0, Rita: 0 };
      const categoryMap = {};
      txs.forEach(tx => {
        const payer = tx.who === 'Rita' ? 'Rita' : 'Sam';
        const paid = Math.abs(tx.amount_twd);
        actual[payer] += paid;
        if (!categoryMap[tx.category]) categoryMap[tx.category] = { Sam: 0, Rita: 0 };
        categoryMap[tx.category][payer] += paid;
      });
      const samRatio = Number(samShareInput.value) / 100;
      const totalExpense = actual.Sam + actual.Rita;
      const due = { Sam: totalExpense * samRatio, Rita: totalExpense * (1 - samRatio) };
      const delta = { Sam: actual.Sam - due.Sam, Rita: actual.Rita - due.Rita };
      document.getElementById('settlementRows').innerHTML = `
        <tr><td>實際支付（TWD）</td><td class="num">${moneyAbs(actual.Sam)}</td><td class="num">${moneyAbs(actual.Rita)}</td></tr>
        <tr><td>應付（依比例）</td><td class="num">${moneyAbs(due.Sam)}</td><td class="num">${moneyAbs(due.Rita)}</td></tr>
        <tr><td>差額</td><td class="num ${cls(delta.Sam)}">${money(delta.Sam)}</td><td class="num ${cls(delta.Rita)}">${money(delta.Rita)}</td></tr>`;
      const transfer = Math.round(Math.abs(delta.Sam));
      let callout = '已平衡，無需結算';
      if (transfer > 0) {
        callout = delta.Sam > 0
          ? `Rita 應補給 Sam ${fmt.format(transfer)} TWD`
          : `Sam 應補給 Rita ${fmt.format(transfer)} TWD`;
      }
      document.getElementById('settlementCallout').textContent = callout;
      document.getElementById('settlementCategoryRows').innerHTML = Object.entries(categoryMap)
        .sort((a, b) => (b[1].Sam + b[1].Rita) - (a[1].Sam + a[1].Rita))
        .map(([category, values]) => `<tr><td>${category}</td><td class="num">${moneyAbs(values.Sam)}</td><td class="num">${moneyAbs(values.Rita)}</td></tr>`)
        .join('') || '<tr><td colspan="3" class="empty">沒有可結算的支出資料</td></tr>';
    }
    function toggleSettlementMode() {
      const isMonth = settlementMode.value === 'month';
      settlementMonthGroup.style.display = isMonth ? '' : 'none';
      settlementRangeGroup.style.display = isMonth ? 'none' : '';
    }
    function bindTabs() {
      document.querySelectorAll('.tab-btn').forEach(button => {
        button.addEventListener('click', () => {
          const tab = button.dataset.tab;
          document.querySelectorAll('.tab-btn').forEach(item => item.classList.toggle('active', item === button));
          document.querySelectorAll('.tab-panel').forEach(panel => panel.classList.toggle('active', panel.dataset.tabPanel === tab));
        });
      });
    }

    scopeSelect.addEventListener('change', () => { populateMonths(); renderMonthly(); });
    monthSelect.addEventListener('change', renderMonthly);
    settlementMode.addEventListener('change', () => { toggleSettlementMode(); renderSettlement(); });
    settlementMonth.addEventListener('change', renderSettlement);
    settlementStartMonth.addEventListener('change', renderSettlement);
    settlementEndMonth.addEventListener('change', renderSettlement);
    samShareInput.addEventListener('input', () => { syncShareInputs('sam'); renderSettlement(); });
    ritaShareInput.addEventListener('input', () => { syncShareInputs('rita'); renderSettlement(); });

    bindTabs();
    populateMonths();
    populateSettlementMonths();
    loadSettlementShare();
    toggleSettlementMode();
    renderOverview();
    renderMonthly();
    renderSettlement();
  </script>
</body>
</html>
"""
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
