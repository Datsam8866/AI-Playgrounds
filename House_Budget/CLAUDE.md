# House Budget — 專案指示

## 專案用途

本 session 作為 House Budget 的即時記帳入口。
使用者會透過**照片、文字或語音**輸入花費或收入項目，Claude 負責：

1. 解讀輸入內容（辨識金額、項目、日期、幣別、付費人）
2. 自動分類（參照下方分類表）
3. 若有缺漏欄位，**立即追問**，確認後再寫入
4. 將資料寫入 `house_budget.db` 的 `transactions` 表
5. 每次寫入後回報確認摘要

---

## 資料庫位置

```
house_budget.db   （同資料夾）
```

寫入工具：`C:/Python314/python.exe` + `sqlite3`

---

## 欄位對照

| 欄位 | 說明 | 缺漏時處理 |
| --- | --- | --- |
| `date` | 日期 YYYY-MM-DD | 缺則問「哪一天？」，今天預設 today |
| `category` | 分類（見下表） | 自動判斷，不確定時列選項讓使用者選 |
| `item` | 項目描述 | 直接用使用者輸入的名稱 |
| `amount` | 金額（負=支出，正=收入） | 支出自動加負號 |
| `currency` | TWD / AUD / Yen / 其他 | **必問**，無預設值 |
| `amount_twd` | TWD 換算值 | 若非 TWD 則問換算率，或直接問 TWD 等值 |
| `who` | Sam / Rita | **必問**，無預設值 |
| `source` | 自動填 `household_2026/月份` | 自動處理 |
| `note` | 備註 | 選填，使用者自願提供 |

---

## 分類表

| Category | 適用場景 |
| --- | --- |
| `Food` | 餐廳、外帶、超市、飲料、食材 |
| `Transportation` | 停車、加油、大眾運輸、租車 |
| `Accommodation` | 房租、飯店、民宿 |
| `Entertainment` | 景點、票券、娛樂活動 |
| `Others` | 訂閱服務、藥局、購物、其他雜費 |
| `Income` | 薪資、退款、獎金 |

> 分類若不確定，列出 2-3 個選項讓使用者確認，不要自行猜測。

---

## 互動流程

```
使用者輸入
  → Claude 解讀
  → 若有缺漏欄位 → 追問，等使用者補齊後再寫入
  → 資料完整 → 直接寫入 DB，無需使用者確認
  → 回報 table：[日期] [分類] [項目] [金額] [幣別] [付費人]
```

---

## 寫入方式

每次寫入執行：

```python
C:/Python314/python.exe -c "
import sqlite3, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
conn = sqlite3.connect('house_budget.db')
conn.execute('''INSERT INTO transactions (source, date, category, item, amount, currency, amount_twd, who, note)
               VALUES (?,?,?,?,?,?,?,?,?)''',
             (SOURCE, DATE, CATEGORY, ITEM, AMOUNT, CURRENCY, AMOUNT_TWD, WHO, NOTE))
conn.commit()
conn.close()
print('OK')
"
```

寫入後同步更新 `monthly_summary`（DROP + 重建）。

---

## 注意事項

- **不確定就問**，不要自行填假資料
- 照片輸入：描述照片內容（收據、帳單截圖等），Claude 解讀後確認再寫入
- 批次輸入多筆時，逐筆列出讓使用者一次確認
- 每次對話結束前更新 README.md 的最新資料範圍與筆數

## Checkpoint：重複金額偵測

寫入前先查詢同一週內是否已有**相同金額**的交易：

```sql
SELECT id, date, category, item, amount, who
FROM transactions
WHERE amount = ?
  AND date BETWEEN date(?, '-6 days') AND date(?, '+6 days')
```

若有符合筆數 → **暫停寫入**，列出重複疑似筆，詢問使用者確認是否為新的一筆。
使用者確認後才寫入。
