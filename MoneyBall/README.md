# MoneyBall — 多聯盟棒球勝負預測系統

**最後更新：2026-05-15（NBA neutral-site + isotonic calibration 完成）**

---

## 專案概覽

MoneyBall 是多聯盟運動勝負預測與 Dashboard 追蹤系統，整合 `CPBL`、`MLB`、`KBO`、`NPB`、`NBA` 的歷史資料、即時預測、賠率 EV 與 Accuracy 追蹤。

| 聯盟 | 球隊數 | 資料範圍 | Walk-forward 準確率 | 高信心（最佳閾值） | 狀態 |
|---|---:|---|---:|---:|---|
| **CPBL** 中華職棒 | 6 | 2011–2026 | **55.72%** | 待重新校準 | 維運，2026 Platt re-fit 待辦 |
| **MLB** 美國大聯盟 | 30 | 2003–2025 | **56.7%** | p≥0.675 → 70.3% | 完成，含賠率 EV，校準驗證完畢 |
| **KBO** 韓國職棒 | 10 | 2011–2026 | **55.77%** | p≥0.60 → 64.55% | 維運，每日工具已建立 |
| **NPB** 日本職棒 | 12 | 2006–2026 | **54.48%** | p≥0.60 → 61.57% | 維運，每日追蹤中 |
| **NBA** 美國職籃 | 30 | 2011–2025 | **63.98%** | p>0.65 → 73.23% | neutral-site / isotonic 已驗證，待調整閾值 |

---

## 最新進度

### 2026-05-15：NBA 傷兵特徵接入完成

- 新增 `NBA/nba_injury_scraper.py`，使用 ESPN teams / injuries API 寫入 `NBA/nba.sqlite` 的 `player_injuries`，並提供 `fetch_and_store_injuries(conn, scraped_date)` 給預測流程直接重用。
- `NBA/build_nba_game_features.py` 新增 `home_injury_pts / vis_injury_pts / diff_injury_pts` 三欄，會依 `player_injuries` 中 `out` / `doubtful` 球員與當季 `player_game_stats` 平均 PPG 估算缺陣火力損失。
- `NBA/predict_today_nba.py` 現在會在當日預測前先抓 ESPN 傷兵並寫回 sqlite；`NBA/train_nba_model.py` 已同步擴充到 31 欄 features。
- 本機驗證已重建 `17,878` 筆 NBA features，最新 walk-forward 結果為整體 calibrated accuracy `64.6%`、`p>0.65` accuracy `75.0%`（`N=4,963`, `cov=41.5%`）。

### 2026-05-15：NBA neutral-site + isotonic calibration 完成

- `NBA/build_nba_game_features.py` 新增 `is_neutral_site`，將 2020 泡泡賽（`season_year=2019` 且 `game_date >= 2020-07-30`）標成中立場；builder 與 Elo update 在 neutral game 不再套用 `ELO_HOME_ADV=100`。
- `NBA/train_nba_model.py` 移除冗餘 `home_elo` / `vis_elo` 特徵，改用 25 欄 feature schema，並把 calibration 從 Platt scaling 升級為 Isotonic Regression；報告新增 `Brier Score` 與 `ECE`。
- `NBA/predict_today_nba.py` 已同步切到新的 calibrator API 與 neutral-site feature 流。
- 本次全量回測結果：整體 calibrated accuracy `63.98%`、Brier Score `0.2210`、ECE `0.0142`；`2020` 季 calibrated accuracy 由 `61.67%` 提升到 `63.15%`。

### 2026-05-15：NBA 每日預測腳本完成

- 新增 `NBA/predict_today_nba.py`，會先重播 `game_results` 建立當日最新 `GameState`，再用 `ScoreboardV2` 抓指定日期賽程並即時計算 pre-game features。
- 模型訓練沿用 `train_nba_model.py` 的 `FEATURES` / `XGB_PARAMS` / Platt calibration，預設以最近 5 季 `game_features` 訓練並將結果 UPSERT 到 `prediction_tracking`。
- 支援 `python predict_today_nba.py --date YYYY-MM-DD` 與 `python predict_today_nba.py --verify YYYY-MM-DD`；`prediction_tracking` table 會在首次執行時自動建立。

### 2026-05-15：NBA features builder 完成

- 新增 `NBA/build_nba_game_features.py`，將 `game_results` 的 2011–2025 正規賽完賽資料重播為 pre-game 特徵，寫入 `NBA/nba.sqlite` 的 `game_features`。
- 已完成 `Elo`、`rolling win% / 得失分 / net rating / Pythagorean WP`、`rest / back-to-back`、`streak`、`season context`，並遵守無前瞻洩漏。
- 全量重建結果為 `17,878` 筆 features，對應目前 `game_results` 中 `home_win IS NOT NULL` 的完賽場次；另有 1 筆未完賽結果未納入。

### 2026-05-15：NBA scraper 初始化

- 新增 `NBA/nba_scraper.py`，使用 `nba_api.stats.endpoints.LeagueGameLog` 抓取 `2011-12` 到 `2025-26` 例行賽結果。
- `NBA/nba.sqlite` 會在首次執行時自動建立，僅保存 `game_results` 與 `seasons_fetched` 兩張表，不存 `raw_json`。
- 支援 `--season YYYY` 單季更新、`--force` 強制重抓與 `seasons_fetched` 增量跳過，方便後續接特徵建表與模型訓練。

### 2026-05-10：Dashboard 四聯盟快取刷新

- CPBL 同步 `2026-05-10` 新增 3 場完賽資料，NPB schedule 補入 6 場、boxscore 補入 12 筆 SP。
- 四聯盟快取建立：MLB 15 場（HIGH=1, bet=2）、KBO 5 場、NPB 6 場、CPBL 3 場（HIGH=1）。
- MLB 今日有賠率（playsport.cc），KBO / NPB / CPBL 賠率待盤口開放後更新。

### 2026-05-09：Dashboard 四聯盟快取刷新

- 透過 Dashboard API 重新刷新 `2026-05-09` 四聯盟快取：MLB 15 場、KBO 5 場、NPB 6 場、CPBL 3 場。
- 四聯盟快取皆已建立於 `dashboard/data/*_2026-05-09.json`，Odds Source 皆為 `playsport.cc`，且 `has_live_odds=True`。
- 已在本機 Dashboard `http://localhost:5555` 驗證 `2026-05-09` 不再顯示「尚未建立快取」，四聯盟頁面皆可載入今日預測。

### 2026-05-08：Dashboard 資料刷新 + MLB live 特徵同步修正

- 執行 `update_all.ps1` 更新四聯盟每日資料；MLB / KBO 無新增完賽資料，CPBL 最近 15 天 playsport 同步成功，NPB schedule 已確認 2026 排程。
- 透過 Dashboard API 重新刷新 `2026-05-08` 四聯盟快取：MLB 15 場、KBO 5 場、NPB 6 場、CPBL 3 場。
- 修正 MLB Dashboard 即時預測特徵不同步：`predict_mlb_today.py` 補上 `is_pitch_clock_era`，與 `train_mlb_model.py` 的訓練特徵一致。
- 已在本機 Dashboard `http://localhost:5555` 驗證 MLB 分頁載入 `2026-05-08` 最新資料，Pipeline Status 顯示 Updated。

### 2026-05-07：SQLite 每日自動更新 + Task Scheduler

- **MLB / KBO / NPB**：手動更新至 2026-05-06 完賽資料（MLB 新增 12 場 + 24 SP、NPB 補入 6 場）。
- **CPBL / KBO / NPB 歷史資料補齊**：CPBL 英文站 `box/getlive` 補回 2011–2025 例行賽；KBO 官方 API 補回 2011–2025；NPB BIS archive 補回 2006–2015、NPB schedule 補回 2016–2025。
- **訓練特徵重建**：`CPBL/game_features` 目前 2011–2026 共 3,964 筆；`KBO/game_features` 目前 2011–2026 共 10,093 筆；`NPB/game_features_npb` 目前 2011–2026 共 11,974 筆。
- **SP 資料補足**：CPBL 從官方 raw_json 回填 2011–2025 SP；KBO / NPB 重建 pitcher features 並補爬 NPB 2016–2025 boxscore。完賽 training SP 覆蓋率：CPBL 64.6%、KBO 72.5%、NPB 80.2%、MLB 85.7%。
- **Walk-forward 已重新評估**：補足資料後重跑 CPBL / KBO / NPB regime 評估；CPBL 2016–2025 為 55.72%，KBO 2016–2025 為 55.77%，NPB 2016–2026 為 54.48%。
- **NPB regime 路由一致化**：NPB 改為與其他聯盟一致的 `early / fallback / primary` 三段式；成熟場次若 SP 資料不足會走 fallback，不再併入 primary。
- **CPBL 改用 playsport.cc**：CPBL 官網 `/home/getdetaillist`、`/home/gamedetail` 連真瀏覽器站內互動都回 307，日常結果同步改走 `playsport_results_sync.py --league cpbl`。
- **CPBL 2026-05-06 已補入**：playsport livescore 成功同步 3 場完賽資料，`CPBL/cpbl.sqlite` 最新完賽日為 `2026-05-06`。
- **Windows Task Scheduler**：`update_all.ps1` 的 CPBL 區塊改為回補最近 15 天 playsport 結果；`MoneyBall_Update_14`（14:00）和 `MoneyBall_Update_22`（22:00）維持執行同一腳本，日誌存 `logs/update_YYYY-MM-DD.log`。

### 2026-05-06：專案初始化

- 新增 MoneyBall 專屬 `AGENTS.md`，固定專案邊界、日常流程、Obsidian 筆記位置與安全規則。
- 新增 MoneyBall 專屬 `.gitignore`，排除 `.sqlite`、log、cache、Playwright session、環境檔與本機產物。
- 建立 Obsidian 駕駛艙：`AI Playgrounds/MoneyBall 工作筆記.md`。
- 解除上層 repo 對 `MoneyBall/` 的整包忽略，準備納入 `AI-Playgrounds` repo。
- 精簡專案資料夾，只保留 Dashboard、playsport 共用工具與四聯盟主線 scraper / feature / prediction / EV / run_dashboard 腳本；移除重複支線、舊報告、輸出檔、Clippings 與 Obsidian 本地設定。

### 2026-05-03：CPBL / KBO Accuracy history 修復

- 修正 `dashboard_server.py` 的 Accuracy 日結果快取策略，不再沿用已快取的空結果。
- 補齊 `playsport_scraper.py` 的 `KBO` 隊名 alias，支援短中文名稱同步到 canonical KBO team names。
- 重新同步 `KBO 2026-05-01`、`2026-05-02` 賽果到 sqlite，並重建 historical dashboard cache。

已驗證：
- `CPBL Accuracy` 已延伸到 `2026-05-02`
- `KBO Accuracy` 已延伸到 `2026-05-02`
- `CPBL 2026-05-01 / 2026-05-02`、`KBO 2026-05-01 / 2026-05-02` 皆可在 Accuracy 分頁顯示

### 2026-05-03：Dashboard UX 第二輪

- `dashboard/index.html` 新增 mobile bottom action bar，行動版固定提供 `Update` / `Parlay` 操作。
- 強化 league tab badge、league banner、league accent color、matchup team monogram。
- Accuracy 表格維持精簡 7 欄統計：`Date / HIGH Total / HIGH Correct / HIGH % / ALL Total / ALL Correct / ALL %`。

---

## Next Action

| 優先 | 聯盟 | 項目 | 狀態 |
|---|---|---|---|
| **A** | CPBL | 每日追蹤：`predict_today.py --verify` + `track_high_confidence_predictions.py` | 持續進行 |
| **B** | CPBL | Platt re-fit：累積 ≥80 場後重算 A/B | 待辦 |
| **C** | CPBL | SP multi-season prior：SP < 5 場時加入前一年/生涯 rolling | 待辦 |
| **D** | MLB | 觀察 `schedule date ↔ result date` 是否仍有跨日邊界案例 | 追蹤中 |
| **E** | All | Dashboard 累積 Accuracy 歷史，追蹤各聯盟高信心準確率趨勢 | 持續進行 |
| **F** | NBA | 依 isotonic 後的新 coverage / accuracy 重新評估高信心閾值（`0.60` / `0.65` / 動態 cut） | 待辦 |
| **G** | All | 評估更早歷史 odds / SP 回填來源 | 待辦 |

---

## Dashboard 使用方式

```bash
pip install flask
python dashboard_server.py
```

開啟瀏覽器：

```text
http://localhost:5555
```

各聯盟 Tab 點「Update Predictions」會觸發背景預測流程。Accuracy Tab 顯示近 30 天各聯盟預測準確率折線圖。

### 歷史資料回填

```bash
python backfill_accuracy.py --days 30 --leagues cpbl,mlb,kbo,npb
```

也可使用 Dashboard 的 Accuracy 分頁「Backfill 30d」按鈕。

---

## 共同架構

```text
Regime routing:
  early    -> Logistic Regression on Elo
  primary  -> XGBoost with team/SP rolling features
  fallback -> Elo + HFA + team rolling stats
```

主要特徵：
- Elo 評分差與主場優勢
- SP rolling ERA / WHIP / K9
- Team rolling stats：得分、失分、連勝/連敗、休息日

### 四聯盟關鍵差異

| | CPBL | KBO | MLB | NPB |
|---|---|---|---|---|
| 準確率 | 55.72% | 55.77% | 56.7% | 54.48% |
| Platt scaling | A=1.45（待 re-fit） | 無 | 無 | 無 |
| ELO_K | 52 | 48 | 8 | 52 |
| home_adv | 20 | 10 | 25 | 10 |
| 台灣運彩 EV | `cpbl_betting_ev.py` | 無 | `mlb_betting_ev.py` | `npb_betting_ev.py` |

---

## 主要檔案

| 路徑 | 說明 |
|---|---|
| `dashboard_server.py` | Flask server（port 5555） |
| `dashboard/index.html` | 多聯盟 Dashboard UI |
| `backfill_accuracy.py` | 歷史 Accuracy 快取回填工具 |
| `playsport_scraper.py` | playsport.cc 賽程、先發、賠率擷取 |
| `playsport_results_sync.py` | playsport.cc 結果同步 |
| `CPBL/` | 中華職棒模型、追蹤、EV 工具 |
| `MLB/` | MLB 模型、賠率 EV、Dashboard 資料介面 |
| `KBO/` | KBO 模型、特徵、Dashboard 資料介面 |
| `NPB/` | NPB 模型、特徵、賠率 EV、Dashboard 資料介面 |
| `NBA/` | NBA scraper、feature builder、walk-forward training 工作區 |

---

## 聯盟日常提醒

### CPBL

- `predict_today.py`：即時預測
- `track_high_confidence_predictions.py`：高信心追蹤與信心桶監控
- 2026 校準仍需追蹤；累積 ≥80 場後評估 Platt re-fit

### MLB

- Accuracy 會依 cache 當日賽程自動挑選最佳結果日，處理 Stats API schedule date 與 sqlite `game_date` 的跨日落差。
- Accuracy 結果查找支援 doubleheader，同一組主客隊不再被單筆覆蓋。

### KBO

- `run_dashboard.py` 可自動觸發 DB 更新：scraper → features → pitcher。
- 若來源缺賠率，refresh / backfill 會保留 cache 或手動輸入的 odds / EV。
- p≥0.70 目前是 raw uncalibrated，不能直接當 calibrated probability。

### NPB

正確日常更新順序：

```bash
cd NPB
python npb_schedule_scraper.py --year 2026
python build_game_features_npb.py --include-scheduled
python build_pitcher_features_npb.py
```

注意：
- `npb_schedule_scraper.py` 負責比賽結果。
- `npb_boxscore_scraper.py` 只抓先發投手，不負責結果。
- 特徵重建必須加 `--include-scheduled`，否則未來賽程特徵會消失。
- NPB 預測路由已分為 `early / fallback / primary`；若 `sp_available < 0.5`，會使用不含 SP 欄位的 fallback model。

---

## 注意事項

- `.sqlite` 資料庫均由 `.gitignore` 排除，需本機重建或另行搬移。
- CPBL train rows 只含 `kind_code='A'`（例行賽）。
- NPB BIS 2006–2015 的 home/away 依標題解析，不可用位置判斷。
- 台灣運彩 Playwright scraper 需 `headless=False`，避免 Cloudflare bot detection。
- Dashboard historical accuracy / backfill 固定鎖 `target_date` 前一天以前；當日 live refresh 保留即時更新模式。
- `playsport.cc` 目前只提供 yesterday / today / tomorrow；更早歷史日期若無既有 cache，需另接可回溯來源。
