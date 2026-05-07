# Buffett - AGENTS.md

## 專案入口

專案名稱：Buffett
專案用途：量化投資輔助、個人持倉 dashboard、美股 VOO 核心/衛星框架、台股 0050 核心/衛星框架
主要工作目錄：C:\Users\dat_s\OneDrive\Documents\AI Playgrounds\Buffett
Repo 根目錄：C:\Users\dat_s\OneDrive\Documents\AI Playgrounds
GitHub repo：https://github.com/Datsam8866/AI-Playgrounds
預設 branch：main
互動規則來源：CLAUDE.md

> 這份檔案是 Buffett 子專案的完整開工入口。從 `Buffett/` 目錄開工時，只需讀本檔，不需要再回讀上層 `AI Playgrounds/AGENTS.md`。

## Obsidian 對應筆記

Obsidian vault：C:\Users\dat_s\OneDrive\Documents\Obsidian Vault
專案駕駛艙：AI Playgrounds/Buffett工作筆記.md
收工時優先更新：AI Playgrounds/Buffett工作筆記.md

> 注意：專案駕駛艙是 Obsidian vault 裡的一篇筆記，不是工作資料夾裡的 Markdown 檔。

## 同步規則

開工時：
- 使用 `startup-sync` 流程
- 先讀本檔，再讀 `AI Playgrounds/Buffett工作筆記.md`
- Git 狀態從 repo 根目錄檢查，但範圍限定 `Buffett/`，避免把其他子專案變更混進來
- 回報「上次做到哪」與下一步
- 不自動 pull / commit / push

收工時：
- 使用 `shutdown-sync` 流程
- 更新 `AI Playgrounds/Buffett工作筆記.md`
- 如規則、路徑、專案邊界改變才更新本檔
- 若本次有實質變更，視情況更新子系統 README 或 dashboard
- commit 時只納入 `Buffett/` 相關檔案

## 主要檔案

入口檔：CLAUDE.md、AGENTS.md
持倉資料：portfolio.sqlite
每日更新：update_portfolio.py、daily_update.ps1、daily_update.bat
Dashboard：generate_portfolio_dashboard.py、portfolio_dashboard.html
持倉比較：compare_portfolio.py
均價修正：update_avg_cost.py
美股模型：Wall Street/
台股模型：TWSE/

## 專案工作規則

- `portfolio.sqlite` 是目前持倉資料基準；修改前先確認資料來源與券商 App 是否一致
- `Wall Street/` 與 `TWSE/` 各有自己的 README，模型配置與績效說明優先看子系統 README
- Wall Street 主線判準是提升 beat-VOO 季度比率，不單看平均報酬
- TWSE 主線判準是提升 beat-0050 季度比率，不單看平均報酬
- 支線實驗放 `experiments/` 或 `archive/`，不要混入主線
- `daily_update.log` 是本機執行紀錄，不納入 Git
- SQLite 大檔與模型資料庫不強制加入 Git，先看 `.gitignore`
- 投資配置建議需標明是模型訊號或實際交易動作，避免混淆
- 若需要 Git 指令，從 repo 根目錄 `C:\Users\dat_s\OneDrive\Documents\AI Playgrounds` 執行，並以 `Buffett/` 作為檔案範圍

## 不要做

- 不要把 Buffett 與其他子專案改動混在同一個 commit
- 不要因為 Buffett 開工而重新讀取或依賴上層 `AI Playgrounds/AGENTS.md`
- 不要把 API key、token、密碼寫進 repo 或 Obsidian 筆記
- 不要把每日進度寫進本檔
- 不要在未確認資料來源時覆蓋 SQLite 歷史資料
- 不要把模型訊號直接當成實際下單建議
