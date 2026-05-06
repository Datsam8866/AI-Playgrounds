# MoneyBall — 多聯盟棒球勝負預測系統

**最後更新：2026-05-06（專案初始化 / 主線精簡 / Git 納管準備）**

---

## 專案概覽

MoneyBall 是多聯盟棒球勝負預測與 Dashboard 追蹤系統，整合 `CPBL`、`MLB`、`KBO`、`NPB` 四個聯盟的歷史資料、即時預測、賠率 EV 與 Accuracy 追蹤。

| 聯盟 | 球隊數 | 資料範圍 | Walk-forward 準確率 | 高信心（最佳閾值） | 狀態 |
|---|---:|---|---:|---:|---|
| **CPBL** 中華職棒 | 6 | 2011–2026 | **72.7%** | p≥0.70 → 85.3% | 維運，2026 Platt re-fit 待辦 |
| **MLB** 美國大聯盟 | 30 | 2003–2025 | **56.7%** | p≥0.675 → 70.3% | 完成，含賠率 EV，校準驗證完畢 |
| **KBO** 韓國職棒 | 10 | 2011–2026 | **55.77%** | p≥0.60 → 64.55% | 維運，每日工具已建立 |
| **NPB** 日本職棒 | 12 | 2006–2026 | **54.65%** | p≥0.60 → 60.38% | 維運，每日追蹤中 |

---

## 最新進度

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
| **F** | All | 評估更早歷史 odds / SP 回填來源 | 待辦 |

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
| 準確率 | 72.7% | 55.77% | 56.7% | 54.65% |
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

---

## 注意事項

- `.sqlite` 資料庫均由 `.gitignore` 排除，需本機重建或另行搬移。
- CPBL train rows 只含 `kind_code='A'`（例行賽）。
- NPB BIS 2006–2015 的 home/away 依標題解析，不可用位置判斷。
- 台灣運彩 Playwright scraper 需 `headless=False`，避免 Cloudflare bot detection。
- Dashboard historical accuracy / backfill 固定鎖 `target_date` 前一天以前；當日 live refresh 保留即時更新模式。
- `playsport.cc` 目前只提供 yesterday / today / tomorrow；更早歷史日期若無既有 cache，需另接可回溯來源。
