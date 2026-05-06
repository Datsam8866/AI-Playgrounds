# House Budget SQLite 資料庫

> 最後更新：2026-05-06（新增宏偉婦幼診所支出）

將家庭記帳與旅遊預算 Excel 整合成單一 SQLite 資料庫，方便 SQL 查詢、月份比較與分類統計。

## 資料概況

| 項目 | 數值 |
| --- | --- |
| 總交易筆數 | 1,273 筆 |
| 資料時間範圍 | 2024-11-25 至 2026-05-06 |
| 來源 | 3 份 Excel，19 個 sheet |
| 主交易表 | `transactions` |
| 月份摘要表 | `monthly_summary` |

## 專案檔案

| 檔案 | 說明 |
| --- | --- |
| `build_db.py` | 從 Excel 重建 SQLite 的匯入腳本 |
| `generate_dashboard.py` | 從 SQLite 產生月度監控 HTML dashboard |
| `house_budget.db` | SQLite 資料庫 |
| `house_budget_dashboard.html` | 家庭收支月度監控 dashboard |
| `AU Travel Budget.xlsx` | 澳洲旅遊費用（AUD/TWD），2024-11 至 2025-01 |
| `Sam Household Budget_2025.xlsx` | 2025 年家庭月支出 |
| `Sam Household Budget_2026.xlsx` | 2026 年家庭月支出 |

## 版本管理

`House_Budget/` 已從上層 `.gitignore` 排除清單移除，可被 AI Playgrounds 主 repo 正常追蹤。`__pycache__/`、`.pyc` 等暫存檔仍由全域 Python 規則忽略。

## 重建資料庫

```bash
python build_db.py
```

需要：`pip install openpyxl`（Python 3.10+）

## 產生 Dashboard

```bash
python generate_dashboard.py
```

輸出：`house_budget_dashboard.html`

Dashboard 分為 3 個 Tab：
- **Overview**：全期 KPI（總收入/支出/淨額/月均）、完整月份趨勢圖、分類累計支出、2024/2025/2026 年度對比表
- **Monthly**：月份篩選、資料範圍切換、月收入/支出/淨額 KPI、近 12 個月趨勢、本月分類支出、大筆支出 Top 10、付款人支出、交易明細
- **Sam & Rita**：比例輸入（互鎖，預設 50/50）、即時結算計算、分類分攤明細、localStorage 記住比例

## 資料表結構

### `transactions`

| 欄位 | 類型 | 說明 |
| --- | --- | --- |
| `id` | INTEGER PK | 自動編號 |
| `source` | TEXT | 來源，格式 `來源代碼/sheet名稱` |
| `date` | TEXT | 日期 `YYYY-MM-DD` |
| `category` | TEXT | 分類 |
| `item` | TEXT | 項目描述 |
| `amount` | REAL | 原幣金額（正=收入，負=支出） |
| `currency` | TEXT | `TWD` / `AUD` / `Yen` |
| `amount_twd` | REAL | TWD 換算金額（無換算時為 NULL） |
| `who` | TEXT | 付費人（Sam / Rita） |
| `note` | TEXT | 備註（旅遊地點等） |

### `monthly_summary`

聚合維度：`source × month × category × currency`
欄位：`month`、`category`、`currency`、`tx_count`、`total_amount`、`total_twd`

## 常用查詢

```sql
-- 某月所有交易
SELECT date, category, item, amount, currency, who
FROM transactions WHERE date LIKE '2026-04%' ORDER BY date, id;

-- 各月 TWD 淨收支
SELECT substr(date, 1, 7) AS month, ROUND(SUM(amount_twd), 0) AS net_twd
FROM transactions
WHERE amount_twd IS NOT NULL
GROUP BY month ORDER BY month;

-- 各分類累計支出
SELECT category, ROUND(SUM(amount_twd), 0) AS total_twd
FROM transactions WHERE amount_twd < 0
GROUP BY category ORDER BY total_twd;
```

## 幣別統計

| 幣別 | 筆數 | 原幣總額 | TWD 總額 |
| --- | ---: | ---: | ---: |
| AUD | 167 | -6,097.99 | -124,842 |
| TWD | 1,011 | -547,231 | -547,231 |
| Yen | 95 | -562,196 | -115,568 |

## 月份 TWD 淨收支

| 月份 | TWD 淨額 | 月份 | TWD 淨額 |
| --- | ---: | --- | ---: |
| 2024-11 | -1,313 | 2025-09 | -37,059 |
| 2024-12 | -172,370 | 2025-10 | -9,228 |
| 2025-01 | -144,991 | 2025-11 | -21,023 |
| 2025-02 | -65,233 | 2025-12 | -17,406 |
| 2025-03 | +10,944 | 2026-01 | -94,176 |
| 2025-04 | -57,787 | 2026-02 | -6,041 |
| 2025-05 | -33,323 | 2026-03 | -18,820 |
| 2025-06 | -62,098 | 2026-04 | -13,750 |
| 2025-07 | -12,620 | 2026-05 | +27,049 |
| 2025-08 | -58,396 | | |

## 匯入規則

- 支援兩種欄位格式：`Date|Category|Item|Amount|Currency|Who` 及含 `TWD|Note` 的擴充版
- `who` 正規化：`阿Sam` → `Sam`，`cynical Tsai` → `Rita`
- 未計算公式儲存格匯入為 NULL；AUD 交易已依歷史 AUD/TWD 匯率換算為 TWD
- 月份淨收支以 `transactions.amount_twd` 計算；若直接修改交易資料，請重跑 `build_db.py` 以同步 `monthly_summary`
- 分類已統一為 7 類：`Accommodation`、`Entertainment`、`Food`、`Income`、`Others`、`Refund`、`Transportation`
- 重跑 `build_db.py` 會覆蓋既有 `house_budget.db`

## 新增來源時

在 `build_db.py` 的 `FILES` 字典加入檔名與來源代碼，維持欄位命名：
`Date, Category, Item, Amount, Currency, TWD, Who, Note`
