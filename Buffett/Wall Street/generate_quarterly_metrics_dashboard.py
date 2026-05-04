# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
PRIMARY_CSV = ROOT / "portfolio_pool_probability_quarterly_walkforward_combined_z_regime_constrained.csv"
FALLBACK_CSV = ROOT / "portfolio_pool_probability_quarterly_walkforward_combined_z_regime.csv"
OUTPUT_HTML = ROOT / "quarterly_metrics_dashboard.html"


SERIES_CONFIG = {
    "combined_z": {
        "label": "Combined Z",
        "column": "score_combined_z",
        "color": "#d1495b",
        "format": "number",
    },
    "sharpe": {
        "label": "Sharpe",
        "column": "historical_sharpe",
        "color": "#00798c",
        "format": "number",
    },
    "p_gt_5_cal": {
        "label": "P>5 (Cal.)",
        "column": "p_gt_5",
        "color": "#edae49",
        "format": "percent",
    },
    "p_lt_0_cal": {
        "label": "P<0 (Cal.)",
        "column": "p_lt_0",
        "color": "#30638e",
        "format": "percent",
    },
    "p_gt_5_raw": {
        "label": "P>5 (Raw)",
        "column": "raw_p_gt_5",
        "color": "#f4a261",
        "format": "percent",
    },
    "p_lt_0_raw": {
        "label": "P<0 (Raw)",
        "column": "raw_p_lt_0",
        "color": "#5c677d",
        "format": "percent",
    },
    "realized_return": {
        "label": "實際報酬",
        "column": "realized_return",
        "color": "#003d5b",
        "format": "percent",
    },
}

DEFAULT_SELECTED = ["combined_z", "sharpe", "p_gt_5_cal", "p_lt_0_cal", "realized_return"]


def load_source() -> Path:
    if PRIMARY_CSV.exists():
        return PRIMARY_CSV
    if FALLBACK_CSV.exists():
        return FALLBACK_CSV
    raise FileNotFoundError("找不到季度主線 CSV。")


def latest_snapshot(frame: pd.DataFrame) -> dict[str, object]:
    current = frame[frame["period"].astype(str).str.endswith("_current")].copy()
    if current.empty:
        current = frame.tail(1).copy()
    row = current.iloc[-1]
    return {
        "period": str(row.get("period", "")),
        "regime": str(row.get("regime_label", "")),
        "stage": str(row.get("selection_stage", "")),
        "combined_z": None if pd.isna(row.get("score_combined_z")) else float(row.get("score_combined_z")),
        "sharpe": None if pd.isna(row.get("historical_sharpe")) else float(row.get("historical_sharpe")),
        "p_gt_5_cal": None if pd.isna(row.get("p_gt_5")) else float(row.get("p_gt_5")),
        "p_lt_0_cal": None if pd.isna(row.get("p_lt_0")) else float(row.get("p_lt_0")),
        "p_gt_5_raw": None if pd.isna(row.get("raw_p_gt_5")) else float(row.get("raw_p_gt_5")),
        "p_lt_0_raw": None if pd.isna(row.get("raw_p_lt_0")) else float(row.get("raw_p_lt_0")),
        "turnover": None if pd.isna(row.get("turnover")) else float(row.get("turnover")),
        "tracking_error": None if pd.isna(row.get("tracking_error")) else float(row.get("tracking_error")),
        "theme_max": None if pd.isna(row.get("max_theme_weight")) else float(row.get("max_theme_weight")),
        "weights": str(row.get("weights", "")),
    }


def build_payload(frame: pd.DataFrame) -> dict[str, object]:
    payload: dict[str, object] = {
        "periods": frame["period"].astype(str).tolist(),
        "series": {},
        "latest": latest_snapshot(frame),
        "defaultSelected": DEFAULT_SELECTED,
    }
    for key, config in SERIES_CONFIG.items():
        values = frame[config["column"]].tolist()
        payload["series"][key] = {
            "label": config["label"],
            "color": config["color"],
            "format": config["format"],
            "values": [None if pd.isna(value) else float(value) for value in values],
        }
    return payload


def build_html(payload: dict[str, object], source_name: str) -> str:
    data_json = json.dumps(payload, ensure_ascii=False)
    return f"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Quarterly Metrics Dashboard</title>
  <style>
    :root {{
      --bg: #f5f0e8;
      --panel: #fffaf2;
      --ink: #1d1c1a;
      --muted: #6a665f;
      --grid: #d9d0c0;
      --accent: #b56576;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", "Noto Sans TC", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(181,101,118,0.15), transparent 30%),
        linear-gradient(180deg, #f8f4ec 0%, var(--bg) 100%);
      color: var(--ink);
    }}
    .wrap {{
      max-width: 1440px;
      margin: 0 auto;
      padding: 28px 24px 40px;
    }}
    .hero {{
      background: linear-gradient(135deg, rgba(255,250,242,0.96), rgba(247,238,226,0.96));
      border: 1px solid rgba(48,99,142,0.15);
      border-radius: 24px;
      padding: 24px;
      box-shadow: 0 18px 40px rgba(29,28,26,0.08);
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: 34px;
      line-height: 1.1;
      letter-spacing: -0.02em;
    }}
    p {{
      margin: 0;
      color: var(--muted);
      line-height: 1.6;
    }}
    .grid {{
      display: grid;
      grid-template-columns: 340px 1fr;
      gap: 18px;
      margin-top: 18px;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid rgba(48,99,142,0.14);
      border-radius: 20px;
      padding: 18px;
      box-shadow: 0 12px 26px rgba(29,28,26,0.06);
    }}
    .controls h2,
    .chart-panel h2,
    .table-panel h2,
    .snapshot-panel h2 {{
      margin: 0 0 14px;
      font-size: 18px;
    }}
    .toggle {{
      display: flex;
      align-items: center;
      gap: 10px;
      margin-bottom: 12px;
      padding: 10px 12px;
      border-radius: 14px;
      background: rgba(255,255,255,0.75);
      border: 1px solid rgba(48,99,142,0.10);
    }}
    .toggle input {{
      width: 18px;
      height: 18px;
      accent-color: var(--accent);
    }}
    .swatch {{
      width: 12px;
      height: 12px;
      border-radius: 999px;
      flex: 0 0 auto;
    }}
    .meta, .snapshot-meta {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      margin-top: 16px;
    }}
    .chip {{
      padding: 12px;
      border-radius: 14px;
      background: rgba(255,255,255,0.78);
      border: 1px solid rgba(48,99,142,0.10);
    }}
    .chip strong {{
      display: block;
      margin-bottom: 4px;
      font-size: 12px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }}
    #chart {{
      width: 100%;
      height: 520px;
      background: linear-gradient(180deg, rgba(255,255,255,0.7), rgba(255,255,255,0.45));
      border-radius: 16px;
      border: 1px solid rgba(48,99,142,0.10);
    }}
    .chart-note {{
      margin-top: 10px;
      font-size: 13px;
      color: var(--muted);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      overflow: hidden;
      border-radius: 14px;
      font-size: 14px;
    }}
    th, td {{
      border-bottom: 1px solid rgba(48,99,142,0.10);
      padding: 10px 12px;
      text-align: center;
    }}
    th {{
      background: rgba(48,99,142,0.08);
      font-weight: 600;
    }}
    .tooltip {{
      position: fixed;
      pointer-events: none;
      transform: translate(12px, 12px);
      background: rgba(29,28,26,0.94);
      color: white;
      padding: 10px 12px;
      border-radius: 12px;
      font-size: 13px;
      line-height: 1.5;
      box-shadow: 0 10px 28px rgba(0,0,0,0.18);
      display: none;
      z-index: 20;
      min-width: 200px;
    }}
    .snapshot-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 18px;
      margin-top: 18px;
    }}
    .weights-box {{
      margin-top: 14px;
      padding: 14px;
      border-radius: 14px;
      background: rgba(255,255,255,0.76);
      border: 1px solid rgba(48,99,142,0.10);
      color: var(--muted);
      line-height: 1.6;
      white-space: pre-wrap;
      word-break: break-word;
      font-size: 13px;
    }}
    @media (max-width: 1100px) {{
      .grid {{
        grid-template-columns: 1fr;
      }}
      .snapshot-grid {{
        grid-template-columns: 1fr;
      }}
      #chart {{
        height: 420px;
      }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <h1>季度指標關聯圖</h1>
      <p>資料來源：{source_name}。這個版本已切到 `Combined Z + regime + constraints` 主線，並同時顯示 `校準後 / 原始` 的 `P(>5%)` 與 `P(<0%)`，方便你直接看 calibration 帶來的差異。</p>
    </section>

    <div class="grid">
      <aside class="panel controls">
        <h2>指標切換</h2>
        <div id="series-controls"></div>
        <div class="toggle">
          <input id="normalize-toggle" type="checkbox" checked>
          <label for="normalize-toggle">標準化顯示（建議開啟）</label>
        </div>
        <div class="meta">
          <div class="chip">
            <strong>期間數</strong>
            <span id="period-count"></span>
          </div>
          <div class="chip">
            <strong>最新季度</strong>
            <span id="latest-period"></span>
          </div>
        </div>
      </aside>

      <section class="panel chart-panel">
        <h2>曲線圖</h2>
        <svg id="chart" viewBox="0 0 1000 520" preserveAspectRatio="none"></svg>
        <div class="chart-note">標準化模式下，每條線會各自映射到 0~100，用來比較趨勢與背離。建議同時勾選 `P>5 (Cal.) / P>5 (Raw)` 或 `P<0 (Cal.) / P<0 (Raw)` 看 calibration 差異。</div>
      </section>
    </div>

    <div class="snapshot-grid">
      <section class="panel snapshot-panel">
        <h2>最新季度風險卡</h2>
        <div class="snapshot-meta">
          <div class="chip"><strong>Period</strong><span id="snap-period"></span></div>
          <div class="chip"><strong>Regime</strong><span id="snap-regime"></span></div>
          <div class="chip"><strong>Stage</strong><span id="snap-stage"></span></div>
          <div class="chip"><strong>Combined Z</strong><span id="snap-combined-z"></span></div>
          <div class="chip"><strong>Sharpe</strong><span id="snap-sharpe"></span></div>
          <div class="chip"><strong>Turnover</strong><span id="snap-turnover"></span></div>
          <div class="chip"><strong>Tracking Error</strong><span id="snap-tracking-error"></span></div>
          <div class="chip"><strong>Theme Max</strong><span id="snap-theme-max"></span></div>
        </div>
      </section>

      <section class="panel snapshot-panel">
        <h2>校準前後機率</h2>
        <table id="current-prob-table"></table>
        <div class="weights-box" id="weights-box"></div>
      </section>
    </div>

    <section class="panel table-panel" style="margin-top: 18px;">
      <h2>相關係數矩陣</h2>
      <table id="corr-table"></table>
    </section>
  </div>

  <div id="tooltip" class="tooltip"></div>

  <script>
    const DATA = {data_json};
    const SERIES_ORDER = {json.dumps(list(SERIES_CONFIG.keys()), ensure_ascii=False)};
    const SELECTED = new Set(DATA.defaultSelected);
    const controls = document.getElementById("series-controls");
    const chart = document.getElementById("chart");
    const corrTable = document.getElementById("corr-table");
    const tooltip = document.getElementById("tooltip");
    const normalizeToggle = document.getElementById("normalize-toggle");

    document.getElementById("period-count").textContent = DATA.periods.length.toString();
    document.getElementById("latest-period").textContent = DATA.latest.period;

    function fmtValue(format, value) {{
      if (value == null || Number.isNaN(value)) return "—";
      if (format === "percent") return `${{(value * 100).toFixed(1)}}%`;
      return value.toFixed(3);
    }}

    function fmt(seriesKey, value) {{
      return fmtValue(DATA.series[seriesKey].format, value);
    }}

    function buildControls() {{
      SERIES_ORDER.forEach((key) => {{
        const wrapper = document.createElement("label");
        wrapper.className = "toggle";
        wrapper.innerHTML = `
          <input type="checkbox" data-key="${{key}}" ${{SELECTED.has(key) ? "checked" : ""}}>
          <span class="swatch" style="background:${{DATA.series[key].color}}"></span>
          <span>${{DATA.series[key].label}}</span>
        `;
        wrapper.querySelector("input").addEventListener("change", (event) => {{
          if (event.target.checked) {{
            SELECTED.add(key);
          }} else {{
            SELECTED.delete(key);
          }}
          renderAll();
        }});
        controls.appendChild(wrapper);
      }});
    }}

    function correlation(a, b) {{
      const pairs = [];
      for (let i = 0; i < a.length; i += 1) {{
        const x = a[i];
        const y = b[i];
        if (x == null || y == null || Number.isNaN(x) || Number.isNaN(y)) continue;
        pairs.push([x, y]);
      }}
      if (pairs.length < 2) return null;
      const xs = pairs.map((p) => p[0]);
      const ys = pairs.map((p) => p[1]);
      const meanX = xs.reduce((s, v) => s + v, 0) / xs.length;
      const meanY = ys.reduce((s, v) => s + v, 0) / ys.length;
      let num = 0;
      let denX = 0;
      let denY = 0;
      for (let i = 0; i < xs.length; i += 1) {{
        const dx = xs[i] - meanX;
        const dy = ys[i] - meanY;
        num += dx * dy;
        denX += dx * dx;
        denY += dy * dy;
      }}
      const den = Math.sqrt(denX * denY);
      if (den === 0) return null;
      return num / den;
    }}

    function renderCorrelationTable() {{
      const visible = SERIES_ORDER.filter((key) => SELECTED.has(key));
      let html = "<thead><tr><th>指標</th>";
      visible.forEach((key) => {{
        html += `<th>${{DATA.series[key].label}}</th>`;
      }});
      html += "</tr></thead><tbody>";
      visible.forEach((rowKey) => {{
        html += `<tr><th>${{DATA.series[rowKey].label}}</th>`;
        visible.forEach((colKey) => {{
          const corr = correlation(DATA.series[rowKey].values, DATA.series[colKey].values);
          html += `<td>${{corr == null ? "—" : corr.toFixed(2)}}</td>`;
        }});
        html += "</tr>";
      }});
      html += "</tbody>";
      corrTable.innerHTML = html;
    }}

    function normalize(values) {{
      const filtered = values.filter((value) => value != null && !Number.isNaN(value));
      if (!filtered.length) return values.map(() => null);
      const min = Math.min(...filtered);
      const max = Math.max(...filtered);
      if (Math.abs(max - min) < 1e-12) return values.map((value) => (value == null ? null : 50));
      return values.map((value) => value == null ? null : ((value - min) / (max - min)) * 100);
    }}

    function renderSnapshot() {{
      const snap = DATA.latest;
      document.getElementById("snap-period").textContent = snap.period || "—";
      document.getElementById("snap-regime").textContent = snap.regime || "—";
      document.getElementById("snap-stage").textContent = snap.stage || "—";
      document.getElementById("snap-combined-z").textContent = fmtValue("number", snap.combined_z);
      document.getElementById("snap-sharpe").textContent = fmtValue("number", snap.sharpe);
      document.getElementById("snap-turnover").textContent = fmtValue("percent", snap.turnover);
      document.getElementById("snap-tracking-error").textContent = fmtValue("percent", snap.tracking_error);
      document.getElementById("snap-theme-max").textContent = fmtValue("percent", snap.theme_max);

      document.getElementById("current-prob-table").innerHTML = `
        <thead>
          <tr><th>Metric</th><th>Calibrated</th><th>Raw</th><th>Delta</th></tr>
        </thead>
        <tbody>
          <tr>
            <td>P(>5%)</td>
            <td>${{fmtValue("percent", snap.p_gt_5_cal)}}</td>
            <td>${{fmtValue("percent", snap.p_gt_5_raw)}}</td>
            <td>${{fmtValue("percent", snap.p_gt_5_cal - snap.p_gt_5_raw)}}</td>
          </tr>
          <tr>
            <td>P(<0%)</td>
            <td>${{fmtValue("percent", snap.p_lt_0_cal)}}</td>
            <td>${{fmtValue("percent", snap.p_lt_0_raw)}}</td>
            <td>${{fmtValue("percent", snap.p_lt_0_cal - snap.p_lt_0_raw)}}</td>
          </tr>
        </tbody>
      `;
      document.getElementById("weights-box").textContent = snap.weights ? `最新配置\\n${{snap.weights}}` : "無最新配置";
    }}

    function renderChart() {{
      const visible = SERIES_ORDER.filter((key) => SELECTED.has(key));
      const width = 1000;
      const height = 520;
      const padding = {{ top: 26, right: 24, bottom: 70, left: 56 }};
      const innerWidth = width - padding.left - padding.right;
      const innerHeight = height - padding.top - padding.bottom;
      const normalizeMode = normalizeToggle.checked;

      const processed = visible.map((key) => {{
        const raw = DATA.series[key].values;
        return {{
          key,
          label: DATA.series[key].label,
          color: DATA.series[key].color,
          values: normalizeMode ? normalize(raw) : raw,
          rawValues: raw,
        }};
      }});

      const allValues = processed.flatMap((s) => s.values.filter((v) => v != null && !Number.isNaN(v)));
      const minValue = allValues.length ? Math.min(...allValues) : 0;
      const maxValue = allValues.length ? Math.max(...allValues) : 1;
      const safeMin = minValue === maxValue ? minValue - 1 : minValue;
      const safeMax = minValue === maxValue ? maxValue + 1 : maxValue;

      function xScale(index) {{
        if (DATA.periods.length === 1) return padding.left + innerWidth / 2;
        return padding.left + (index / (DATA.periods.length - 1)) * innerWidth;
      }}

      function yScale(value) {{
        return padding.top + (1 - (value - safeMin) / (safeMax - safeMin)) * innerHeight;
      }}

      let svg = "";
      for (let i = 0; i <= 5; i += 1) {{
        const y = padding.top + (innerHeight / 5) * i;
        svg += `<line x1="${{padding.left}}" y1="${{y}}" x2="${{width - padding.right}}" y2="${{y}}" stroke="rgba(48,99,142,0.12)" stroke-width="1" />`;
      }}
      for (let i = 0; i < DATA.periods.length; i += 1) {{
        const x = xScale(i);
        svg += `<line x1="${{x}}" y1="${{padding.top}}" x2="${{x}}" y2="${{height - padding.bottom}}" stroke="rgba(48,99,142,0.05)" stroke-width="1" />`;
      }}

      processed.forEach((series) => {{
        const points = [];
        series.values.forEach((value, idx) => {{
          if (value == null || Number.isNaN(value)) return;
          points.push(`${{xScale(idx)}},${{yScale(value)}}`);
        }});
        if (points.length >= 2) {{
          svg += `<polyline fill="none" stroke="${{series.color}}" stroke-width="3" points="${{points.join(" ")}}" stroke-linejoin="round" stroke-linecap="round" />`;
        }}
        series.values.forEach((value, idx) => {{
          if (value == null || Number.isNaN(value)) return;
          const cx = xScale(idx);
          const cy = yScale(value);
          const raw = series.rawValues[idx];
          svg += `<circle cx="${{cx}}" cy="${{cy}}" r="4.5" fill="${{series.color}}" data-series="${{series.key}}" data-index="${{idx}}" data-raw="${{raw}}" data-display="${{value}}" />`;
        }});
      }});

      for (let i = 0; i < DATA.periods.length; i += 1) {{
        const x = xScale(i);
        const label = DATA.periods[i];
        svg += `<text x="${{x}}" y="${{height - 22}}" fill="#6a665f" font-size="11" text-anchor="end" transform="rotate(-45 ${{x}} ${{height - 22}})">${{label}}</text>`;
      }}

      for (let i = 0; i <= 5; i += 1) {{
        const value = safeMin + ((safeMax - safeMin) / 5) * (5 - i);
        const y = padding.top + (innerHeight / 5) * i;
        const label = normalizeMode ? `${{value.toFixed(0)}}` : value.toFixed(2);
        svg += `<text x="${{padding.left - 10}}" y="${{y + 4}}" fill="#6a665f" font-size="11" text-anchor="end">${{label}}</text>`;
      }}

      chart.innerHTML = svg;
      chart.querySelectorAll("circle").forEach((node) => {{
        node.addEventListener("mouseenter", (event) => {{
          const idx = Number(event.target.dataset.index);
          const key = event.target.dataset.series;
          tooltip.style.display = "block";
          tooltip.innerHTML = `
            <strong>${{DATA.periods[idx]}}</strong><br>
            ${{DATA.series[key].label}}: ${{fmt(key, Number(event.target.dataset.raw))}}
          `;
        }});
        node.addEventListener("mousemove", (event) => {{
          tooltip.style.left = `${{event.clientX}}px`;
          tooltip.style.top = `${{event.clientY}}px`;
        }});
        node.addEventListener("mouseleave", () => {{
          tooltip.style.display = "none";
        }});
      }});
    }}

    function renderAll() {{
      renderChart();
      renderCorrelationTable();
      renderSnapshot();
    }}

    buildControls();
    normalizeToggle.addEventListener("change", renderAll);
    renderAll();
  </script>
</body>
</html>
"""


def main() -> None:
    source = load_source()
    frame = pd.read_csv(source)
    payload = build_payload(frame)
    html = build_html(payload, source.name)
    OUTPUT_HTML.write_text(html, encoding="utf-8")
    print(f"Saved HTML: {OUTPUT_HTML.name}")


if __name__ == "__main__":
    main()
