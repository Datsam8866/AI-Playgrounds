# TWSE — 台股 0050 核心/衛星配置系統

## 主線目標

- `XGBoost regression` 對 0050 成分股做截面報酬排序
- `0050.TW >= 50%` 核心/衛星框架；季度 rebalance
- Objective：`Utility = z(P>0050) - z(P<0) + 0.3 × z(Sharpe)`
- 台股 v1 暫不加 regime filter

---

## 系統設定

| 項目 | 設定 |
| --- | --- |
| Universe | 元大 0050 成分股（50 檔） |
| Benchmark / Core | `0050.TW` |
| Core rule | `0050 >= 50%`；可為 50% / 60% / 70% / 80% |
| Rebalance | 季度 |
| Top-K | 4 / 6 / 8 / 10 |
| 單一衛星持股上限 | 10% |
| Turnover cap | 35% |
| 交易限制 | Long-only、無槓桿、無放空 |
| 買進手續費 | 0.1425% |
| 賣出手續費 | 0.1425% |
| 證交稅 | 0.3% |
| 單邊滑價 | 0.05% |

---

## 主線檔案

| 類別 | 檔案 |
| --- | --- |
| 全流程腳本 | `tw_0050_pipeline.py` |
| Universe | `tw_0050_universe.csv`（50 檔成分股） |
| 模型預測 | `tw_0050_xgboost_predictions.csv` |
| Walk-Forward | `tw_0050_walkforward.csv` |
| Walk-Forward 摘要 | `tw_0050_walkforward_summary.csv` |
| 資料庫 | `tw_stock_forecast.sqlite`（已排除於 git） |

---

## 資料來源

- **成分股**：元大投信 0050 持股比重頁（`https://www.yuantaetfs.com/product/detail/0050/ratio`）
  - 解析日期：2026-04-15，共 50 檔
- **價格**：`yfinance`，ticker 格式 `2330.TW`
  - 期間：2010-01-01 ~ 最新
  - SQLite 表：`tw_price_history`（193,182 筆）

---

## 方法

1. 解析 0050 成分股，下載歷史價格
2. 建立技術面特徵：動能、波動、均線乖離、RSI、MACD、量能、回撤、日內振幅
3. `XGBoost regression` 預測 `forward_20d_return`（expanding walk-forward，無 look-ahead）
4. 配置搜尋：0050 core + Top-K 衛星等權
5. `Paired block bootstrap`（N=20000, block=20）估計 P(>0050)、P(<0)、Sharpe
6. `Utility = z(P>0050) - z(P<0) + 0.3×z(Sharpe)` 選出最佳配置
7. 扣除實際交易成本後計算實際報酬

---

## 績效摘要（2016Q1~2026Q1，41 季）

| 指標 | 台股 v1 | 0050 |
| --- | ---: | ---: |
| 季度 Sharpe | **1.289** | 1.066 |
| Max Drawdown | -23.7% | -23.1% |
| 打贏 0050 季度比率 | **68.3%** | — |
| 平均 turnover | 27.8% | — |
| 預測 IC mean | 0.0549 | — |
| 預測 ICIR | 0.600 | — |

---

## 最新配置（2026Q2_current）

Snapshot date：2026-04-17（更新於 2026-05-08）

| 標的 | 名稱 | 權重 |
| --- | --- | ---: |
| `0050.TW` | 元大台灣50 | 60.0% |
| `6515.TW` | 穎崴 | 10.0% |
| `6919.TW` | 康霈 | 10.0% |
| `3037.TW` | 欣興 | 10.0% |
| `2360.TW` | 致茂 | 10.0% |

指標：
- `stage = feasible` ｜ `Top-K = 4`
- `Utility = 2.391` ｜ `Sharpe = 0.640`
- `P(portfolio > 0050) = 68.0%` ｜ `P(portfolio < 0) = 23.3%`
- `turnover = 25.0%`

配置異動（vs 上次 4/16）：新增 `6515` 穎崴；移除 `2383` 台光電、`3711` 日月光、`2308` 台達電、`2368` 金像電、`2345` 智邦；Top-K 8→4。

> ⚠️ 已知問題：universe CSV 含無效 ticker `p$.TW`，需清除。

---

## 已知限制

- 使用 2026-04-15 當下的 0050 成分股回測，存在 survivorship bias
- 尚未加入財報 / 基本面因子
- 尚未加入台股專用 regime filter
- 成本假設可能與實際交易有落差

---

## 執行流程

```powershell
cd TWSE
python tw_0050_pipeline.py
```

單次執行完成：資料更新 → 特徵工程 → 模型訓練 → Walk-Forward → 輸出配置建議
