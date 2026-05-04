# AI Playgrounds - AGENTS.md

## 專案入口

專案名稱：AI Playgrounds
專案用途：所有 AI 相關專案的總工作區
主要工作目錄：C:\Users\dat_s\OneDrive\Documents\AI Playgrounds
GitHub repo：https://github.com/Datsam8866/AI-Playgrounds
預設 branch：main

## Obsidian 對應筆記

Obsidian vault：C:\Users\dat_s\OneDrive\Documents\Obsidian Vault
專案駕駛艙：AI Playgrounds/專案工作流程.md
收工時優先更新：AI Playgrounds/專案工作流程.md

> 注意：專案駕駛艙是 Obsidian vault 裡的一篇筆記，不是工作資料夾裡的 Markdown 檔。

## 工作桌 + 三個家

- 工作桌：C:\Users\dat_s\OneDrive\Documents\AI Playgrounds
- GitHub：https://github.com/Datsam8866/AI-Playgrounds
- Obsidian：C:\Users\dat_s\OneDrive\Documents\Obsidian Vault + AI Playgrounds/專案工作流程.md
- Firebase：ai-playground-e1f62

## 同步規則

開工時：
- 使用 `startup-sync` 流程
- 讀本檔
- 讀 Obsidian 駕駛艙
- 檢查 Git 狀態
- 不自動 pull / commit / push

收工時：
- 使用 `shutdown-sync` 流程
- 更新 Obsidian 駕駛艙
- 如規則、路徑、專案邊界改變才更新本檔
- 需要時 commit + push GitHub

新專案初始化時：
- 使用 `project-init-sync` 流程

## 主要檔案

入口檔：README.md
設定檔：AGENTS.md、.gitignore、firebase.json、firestore.rules、.firebaserc
部署位置：GitHub Pages 視 repo 權限與方案支援而定

## 不要做

- 不要把每日進度寫進 AGENTS.md
- 不要自動納入無關 git 變更
- 不要把 API key、token、密碼寫進 repo
- 不要儲存學生姓名；正式資料只用座號與班級代號
- 不要一次提交所有子專案，除非使用者明確指定範圍
