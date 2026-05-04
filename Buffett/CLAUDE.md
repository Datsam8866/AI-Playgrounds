# Buffett — 量化投資系統

## 對話開始時請先讀

這個專案分為兩個子系統：
- `Wall Street/` — 美股 VOO 核心/衛星框架（XGBoost + 季度 walk-forward）
- `TWSE/` — 台股 0050 核心/衛星框架（XGBoost + 季度 walk-forward）

每個子資料夾各有自己的 README.md，記錄最新配置與進度。

---

## 系統架構

### Wall Street 主線

- **Benchmark / Core**：VOO（`>= 50%`；caution `>= 60%`；risk_off `>= 80%`）
- **Model**：XGBoost regression，截面排序 expanded pool
- **Objective**：`Combined Z = z(P(>VOO)) - z(P(<0)) + 0.3 × z(Sharpe)`
- **Regime**：SPY/VIX/TNX 三因子（risk_on / caution / risk_off）
- **Beta 約束**：risk_on 1.40 / caution 1.20 / risk_off 1.05
- **主線績效（2016Q1~2026Q1，41 季）**：Sharpe 1.097、CAGR 24.93%、beat-VOO 73.2%

### TWSE 主線

- **Benchmark / Core**：0050.TW（`>= 50%`）
- **Model**：XGBoost regression，截面排序 0050 成分股
- **Objective**：`Utility = z(P>0050) - z(P<0) + 0.3*z(Sharpe)`
- **主線績效（2016Q1~2026Q1，41 季）**：Sharpe 1.385、CAGR 累積 +1814.4%、beat-0050 68.3%

---

## 執行流程

### Wall Street

```powershell
cd "Wall Street"
python update_pool_price_history_sqlite.py
python expanded_pool_xgboost_regression_no_leverage_v2.py
python walkforward_portfolio_beta_constrained_voo_alpha.py
python quarterly_probability_calibration.py
```

### TWSE

```powershell
cd TWSE
python tw_0050_pipeline.py
```

---

## 主線檔案

### Wall Street

| 類別 | 檔案 |
| --- | --- |
| Pool 設定 | `expanded_pool_config.py` |
| 預測 | `expanded_pool_xgboost_regression_no_leverage_v2.py` |
| Walk-Forward | `walkforward_portfolio_beta_constrained_voo_alpha.py` ← **主線** |
| Calibration | `quarterly_probability_calibration.py` |
| SQLite 更新 | `update_pool_price_history_sqlite.py` |
| 日常訊號 | `daily_signal.py` / `portfolio_daily_signal.py` |
| 查詢 | `query_sqlite.py` |
| 儀表板 | `generate_quarterly_metrics_dashboard.py` |

### TWSE

| 類別 | 檔案 |
| --- | --- |
| 全流程 | `tw_0050_pipeline.py` |
| 資料庫 | `tw_stock_forecast.sqlite` |
| Universe | `tw_0050_universe.csv` |
| 預測 | `tw_0050_xgboost_predictions.csv` |
| Walk-Forward | `tw_0050_walkforward.csv` |

---

## 工作注意事項

- 程式碼修改一律交給 Codex 實作
- 主線判準：**提升 beat-VOO / beat-0050 季度比率**，不單看平均報酬
- 支線實驗放 `experiments/` 或 `archive/`，不混入主線
- SQLite 備份前先確認，避免覆蓋歷史資料
- `stock_forecast.sqlite` / `tw_stock_forecast.sqlite` 不 commit（大檔）

---

## 收工流程

對 Claude 說「**收工**」→ 自動 commit + push + 更新工作筆記
