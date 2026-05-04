# Stock Market Forecast

## 主線目標

- `XGBoost regression` 做 expanded pool 截面報酬排序
- `VOO >= 50%` 核心/衛星框架；`caution >= 60%`；`risk_off >= 80%`
- 季度 walk-forward 驗證；objective：`Combined Z = z(P(>VOO)) - z(P(<0)) + 0.3 × z(Sharpe)`
- `SPY / VIX / TNX` regime filter
- `turnover / theme cap / tracking error vs VOO` 實務限制
- **Portfolio beta 約束**：`risk_on 1.40 / caution 1.20 / risk_off 1.05`

---

## Pool 異動紀錄

| 日期 | Ticker | 動作 | 原因 |
| --- | --- | --- | --- |
| 2026-04-15 | `CFLT` | **移除** | IBM 以 $31/股完成收購，2026-03-17 下市 |
| 2026-04-18 | `LITE` | **加入** | 模型 inference 排名 #1（predicted_return 0.108） |
| 2026-04-18 | `COHR` | **加入** | 模型 inference 排名 #2（predicted_return 0.057） |
| 2026-04-18 | `ARM` | **確認** | 已在 pool，排名 #10 進入組合 |
| 2026-05-05 | `PSTG` → `P` | **改名** | Pure Storage 改名 Everpure，2026-04-17 NYSE ticker 由 PSTG 改為 P；SQLite 歷史資料同步改名，config 更新 |

---

## 最新快照（2026-05-01）

最新配置（`2026Q2_current`，`caution`，`fallback_turnover_first`）：

| 標的 | 權重 |
| --- | ---: |
| `VOO` | `80.0%` |
| `LITE` | `2.0%` |
| `CLS` | `2.0%` |
| `COHR` | `2.0%` |
| `ARM` | `2.0%` |
| `MRVL` | `2.0%` |
| `DELL` | `2.0%` |
| `AMD` | `2.0%` |
| `VRT` | `2.0%` |
| `INTC` | `2.0%` |
| `MU` | `2.0%` |

指標：
- `regime = caution` ｜ `Combined Z = 1.352`
- `P(>VOO) = 72.9%`（calibrated，主指標）
- `P(>5%) = 57.9%` ｜ `P(<0%) = 21.9%`（raw-calibrated）
- `portfolio_beta = 1.280`（上限 1.20）｜ `beta_feasible = False` ✗
- `turnover = 6.0%` ｜ `tracking error = 7.0%`
- `selection_stage = fallback_turnover_first` ｜ `top_k = 10`

---

## 實際持倉快照（2026-05-03）

| 標的 | 股數 | 價格（USD） | 市值（USD） | 權重 |
| --- | ---: | ---: | ---: | ---: |
| `VOO` | 258 | 661.95 | 170,783 | 58.3% |
| `INTC` | 173 | 100.77 | 17,433 | 5.96% |
| `AMD` | 40 | 361.85 | 14,474 | 4.94% |
| `SNDK` | 11 | 1,187.94 | 13,067 | 4.46% |
| `ARM` | 60 | 211.60 | 12,696 | 4.34% |
| `AVGO` | 29 | 420.27 | 12,188 | 4.16% |
| `NVDA` | 58 | 198.12 | 11,491 | 3.92% |
| `TSM` | 27 | 397.65 | 10,737 | 3.67% |
| `LITE` | 11 | 955.30 | 10,508 | 3.59% |
| `COHR` | 30 | 332.00 | 9,960 | 3.40% |
| `TSLA` | 24 | 391.34 | 9,392 | 3.21% |
| **合計** | | | **292,729** | **100%** |

與模型配置差異（模型 vs 實際）：
- 模型有、實際**未持有**：VRT、MRVL、MU、MSTR、SMCI、CRWV、DELL、NOW
- 實際有、模型**未推薦**：AMD、AVGO、NVDA、SNDK、TSLA、TSM
- VOO 實際 58.3%，低於模型目標 70%

與上季（2026Q1）異動：

| 動作 | 標的 |
| --- | --- |
| ✅ 延續 | LITE, COHR, ARM, MRVL, DELL, VRT, INTC, MU |
| ➕ 新增 | CLS, AMD |
| ❌ 移出 | MSTR, SMCI, CRWV, NOW |

> ⚠️ **Regime 警示**：截面由 `risk_on`（4/14）轉為 `caution`（5/1），VOO 核心從 70% 升至 80%，衛星均等配 2%。beta 1.280 仍超過 caution 上限 1.20（`fallback` 模式）。

---

## 績效摘要（2016Q1~2026Q1，41 季）

| 指標 | 主線 | VOO |
| --- | ---: | ---: |
| 年化 Sharpe | **1.097** | 0.865 |
| CAGR | **24.93%** | 13.58% |
| Max Drawdown | **-29.4%** | -25.3% |
| 打贏 VOO 季度比率 | **73.2%** | — |

子期間：熊市 2022 為弱點（高 beta 本質）；近期 2024~2026Q1 Sharpe 1.337 vs VOO 1.458（改善中）。

---

## Regime / Stage 定義

`Regime`：
- `risk_on`：VIX≤25 且 SPY≥SMA200 且 TNX≤SMA20
- `caution`：上述任一不滿足
- `risk_off`：兩個以上同時不滿足

`Stage`：
- `feasible`：完全符合所有約束（含 portfolio beta）
- `fallback_turnover_first`：無完全可行解，優先壓低 turnover 違規

---

## 季度 Walk-Forward（近 12 季）

| 季度 | Regime | Stage | VOO% | K | Port.β | β✓ | P(>VOO) | P<0 | Turn | 報酬 |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| `2023Q2` | `risk_on` | `feasible` | `80%` | `2` | `1.319` | ✓ | `72.4%` | `24.2%` | `20.0%` | `+13.3%` |
| `2023Q3` | `caution` | `feasible` | `80%` | `2` | `1.076` | ✓ | `73.3%` | `23.4%` | `20.0%` | `-1.1%` |
| `2023Q4` | `caution` | `feasible` | `70%` | `4` | `1.160` | ✓ | `74.2%` | `26.0%` | `14.4%` | `+11.7%` |
| `2024Q1` | `risk_on` | `feasible` | `70%` | `4` | `1.370` | ✓ | `75.1%` | `25.1%` | `16.1%` | `+43.2%` |
| `2024Q2` | `caution` | `feasible` | `80%` | `10` | `1.199` | ✓ | `75.7%` | `24.3%` | `24.0%` | `+5.8%` |
| `2024Q3` | `caution` | `fallback` | `80%` | `10` | `1.269` | ✗ | `76.3%` | `23.4%` | `6.0%` | `+3.5%` |
| `2024Q4` | `caution` | `fallback` | `80%` | `10` | `1.318` | ✗ | `74.3%` | `22.8%` | `6.0%` | `+7.7%` |
| `2025Q1` | `caution` | `fallback` | `80%` | `8` | `1.324` | ✗ | `75.1%` | `22.3%` | `10.0%` | `-6.4%` |
| `2025Q2` | `caution` | `fallback` | `80%` | `2` | `1.341` | ✗ | `73.2%` | `24.5%` | `15.0%` | `+18.6%` |
| `2025Q3` | `risk_on` | `feasible` | `70%` | `6` | `1.378` | ✓ | `73.4%` | `23.6%` | `20.2%` | `+10.8%` |
| `2025Q4` | `caution` | `feasible` | `80%` | `10` | `1.195` | ✓ | `74.4%` | `23.2%` | `24.6%` | `+0.9%` |
| `2026Q1` | `caution` | `fallback` | `80%` | `8` | `1.223` | ✗ | `72.7%` | `22.7%` | `9.7%` | `-3.7%` |
| `2026Q2_current` | `risk_on` | `feasible` | `70%` | `13` | `1.397` | ✓ | `72.1%` | `21.6%` | `16.0%` | `—` |

> 完整 41 季資料見 `walkforward_portfolio_beta_constrained_voo_alpha.csv`

---

## 主線檔案

| 類別 | 檔案 |
| --- | --- |
| 模型 | `multi_asset_xgboost_regression.py` |
| Pool 設定 | `expanded_pool_config.py` |
| 預測 | `expanded_pool_xgboost_regression_no_leverage_v2.py` |
| Walk-Forward | `walkforward_portfolio_beta_constrained_voo_alpha.py` ← **主線** |
| Calibration | `quarterly_probability_calibration.py` |
| SQLite 更新 | `update_pool_price_history_sqlite.py` |
| 日常訊號 | `daily_signal.py` / `portfolio_daily_signal.py` |
| 查詢 | `query_sqlite.py` |
| 儀表板 | `generate_quarterly_metrics_dashboard.py` |

---

## 執行流程

```powershell
python update_pool_price_history_sqlite.py
python expanded_pool_xgboost_regression_no_leverage_v2.py
python walkforward_portfolio_beta_constrained_voo_alpha.py
python quarterly_probability_calibration.py
```

---

## 已放棄的實驗

| 實驗 | 結果 |
| --- | --- |
| risk_off VOO ≥ 90% | Sharpe -0.044、CAGR -1.56pp，放棄 |
| Regime-conditional beta 窗口 | 41 季資料量不足，零效益 |
| Caution Beta-Aware Weighting | beat-VOO 從 66.7% 降至 52.4%，放棄 |
| Caution Extreme Beta Filter | beat-VOO 61.9%，仍低於主線 66.7%，暫停 |

---

## 下一步

1. 所有支線以「提升 beat-VOO 比率」為第一判準
2. caution 支線暫停，不再擴寫
3. 考慮 ETF / 個股分層模型提升 ICIR
