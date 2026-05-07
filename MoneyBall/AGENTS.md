# MoneyBall - AGENTS.md

## 專案入口

專案名稱：MoneyBall
專案用途：多聯盟棒球勝負預測與 Dashboard 追蹤系統
主要工作目錄：C:\Users\dat_s\OneDrive\Documents\AI Playgrounds\MoneyBall
上層 repo：https://github.com/Datsam8866/AI-Playgrounds
預設 branch：main

## Obsidian 對應筆記

Obsidian vault：C:\Users\dat_s\OneDrive\Documents\Obsidian Vault
專案駕駛艙：AI Playgrounds/MoneyBall 工作筆記.md
收工時優先更新：AI Playgrounds/MoneyBall 工作筆記.md

> 注意：專案駕駛艙是 Obsidian vault 裡的一篇筆記，不是 MoneyBall 工作資料夾裡的 Markdown 檔。

## 工作桌 + 三個家

- 工作桌：C:\Users\dat_s\OneDrive\Documents\AI Playgrounds\MoneyBall
- GitHub：AI-Playgrounds repo 的 `MoneyBall/` 子目錄
- Obsidian：C:\Users\dat_s\OneDrive\Documents\Obsidian Vault + AI Playgrounds/MoneyBall 工作筆記.md
- Firebase：目前不使用

## 同步規則

開工時：
- 使用 `startup-sync` 流程
- 讀本檔
- 讀 Obsidian 駕駛艙
- 檢查 Git 狀態
- 回報「上次做到哪」與下一步
- 不自動 pull / commit / push

收工時：
- 使用 `shutdown-sync` 流程
- 更新 Obsidian 駕駛艙
- 如規則、路徑、專案邊界改變才更新本檔
- 需要時只提交 MoneyBall 相關檔案
- commit 訊息要寫清楚做了什麼與為什麼

## 主要檔案

入口檔：README.md
固定規則：AGENTS.md
本機忽略：.gitignore
Dashboard：dashboard_server.py、dashboard/index.html
共用工具：playsport_scraper.py、playsport_results_sync.py、backfill_accuracy.py
聯盟目錄：CPBL/、MLB/、KBO/、NPB/

## 日常操作

- Dashboard：在 MoneyBall 根目錄執行 `python dashboard_server.py`，再開啟 `http://localhost:5555`
- Accuracy 回填：`python backfill_accuracy.py --days 30 --leagues cpbl,mlb,kbo,npb`
- CPBL 每日追蹤：進入 `CPBL/` 後執行 `predict_today.py --verify` 與 `track_high_confidence_predictions.py`
- NPB 歷史結果更新需使用 `npb_schedule_scraper.py`，不是 `npb_boxscore_scraper.py`
- NPB 特徵重建必須加 `--include-scheduled`

## 不要做

- 不要把每日進度寫進 AGENTS.md
- 不要提交 `.sqlite`、log、cache、Playwright session 或本機暫存
- 不要把 API key、token、密碼或 Firebase Admin 憑證寫進 repo
- 不要儲存學生姓名；若未來加入教學資料，正式資料只用座號與班級代號
- 不要一次提交 AI Playgrounds 其他子專案的無關變更
