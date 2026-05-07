# House Budget - AGENTS.md

## 專案入口

專案名稱：House Budget
專案用途：家庭收支 SQLite、月度監控 dashboard、即時記帳入口
主要工作目錄：C:\Users\dat_s\OneDrive\Documents\AI Playgrounds\House_Budget
GitHub repo：https://github.com/Datsam8866/AI-Playgrounds
預設 branch：main
互動規則來源：CLAUDE.md

## Obsidian 對應筆記

Obsidian vault：C:\Users\dat_s\OneDrive\Documents\Obsidian Vault
專案駕駛艙：AI Playgrounds/HouseBudget工作筆記.md
收工時優先更新：AI Playgrounds/HouseBudget工作筆記.md

> 注意：專案駕駛艙是 Obsidian vault 裡的一篇筆記，不是工作資料夾裡的 Markdown 檔。

## 同步規則

開工時：
- 使用 `startup-sync` 流程
- 只讀本檔作為 House_Budget 的開工規則入口
- 再讀 `AI Playgrounds/HouseBudget工作筆記.md` 取得最新進度
- 檢查 `House_Budget/` 自身 Git 狀態，避免把其他子專案變更混進來
- 回報「上次做到哪」與下一步
- 不自動 pull / commit / push

收工時：
- 使用 `shutdown-sync` 流程
- 更新 `AI Playgrounds/HouseBudget工作筆記.md`
- 如規則、路徑、專案邊界改變才更新本檔
- 若本次有實質變更，更新 `README.md`
- commit 時只納入 `House_Budget/` 相關檔案

## 主要檔案

入口檔：README.md、CLAUDE.md、AGENTS.md
資料與腳本：house_budget.db、build_db.py、generate_dashboard.py
輸出檔：house_budget_dashboard.html
測試檔：test_generate_dashboard.py

## 專案工作規則

- `house_budget.db` 是目前操作基準；若直接新增或修改交易，需同步維護 `monthly_summary`
- 若資料只存在 SQLite、尚未回寫 Excel，避免直接重跑 `build_db.py`，以免覆蓋手動新增資料
- 使用者輸入缺欄位時要先追問；`currency` 與 `who` 不可自行預設
- 寫入前先檢查同週內相同金額交易；有疑似重複時先停下來確認
- 非 TWD 交易要確認匯率或 TWD 等值
- 新增中文交易時，避免直接用 PowerShell `sqlite3.exe` 寫入，優先用 Python 參數化寫入
- 只要交易資料或 dashboard 邏輯有變動，視情況重跑 `generate_dashboard.py`
- `CLAUDE.md` 保留即時記帳互動細節；本檔只記穩定規則與路徑

## 不要做

- 不要把 House_Budget 與其他子專案改動混在同一個 commit
- 不要把 API key、token、密碼寫進 repo 或 Obsidian 筆記
- 不要把每日進度寫進本檔
- 不要假設 Excel 一定比 SQLite 新；先確認哪一份才是最新來源
