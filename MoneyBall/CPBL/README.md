# CPBL Data Pipeline 進度

## 目前目標

- 持續更新 CPBL 歷史球員 / 球隊資料
- 建立預測模型：**season-level 戰績預測** 與 **game-level 逐場勝敗預測**
- game-level 分層目標：全場 73–78%、p >= 0.70 達 82–85%、p >= 0.80 達 88%+

---

## 最新進度（2026-04-27）

### Walk-Forward 年度準確率（soft-regime v2，最新模型）

| 年份 | 場次 | 全場準確率 | p>0.70 場次 | p>0.70 準確率 |
| --- | ---: | ---: | ---: | ---: |
| 2021 | 276 | 71.7% | 127 | 89.0% |
| 2022 | 276 | 69.6% | 135 | 88.9% |
| 2023 | 282 | 70.2% | 164 | 81.7% |
| 2024 | 340 | 74.7% | 240 | 84.6% |
| **2025** | **348** | **83.0%** | **238** | **90.3%** |
| 2016–2025 合計 | 2652 | 72.7% | 1478 | 85.3% |
| **2026**（04-26 止）| **50** | **53.2%** | **31** | **51.6%** ⚠️ |

> 2026 p>0.70 只有 51.6%（歷史 81–90%），統計檢定 z=-3.74，p<0.0001，**確認為系統性問題**。  
> 根本原因：2026 開季 Elo 分布最極端（spread=214，歷史 131-158）+ Platt A=1.45 放大 → 假高信心。

### 2026 即時追蹤（prediction_tracking，04-01 到 04-26）

| 指標 | 數值 |
| --- | ---: |
| 已評分場次 | 50 |
| 高信心場次（p >= 0.60） | 34 |
| 高信心命中 | 19 / 34 |
| 高信心準確率 | **55.9%** |
| ALERT：0.70-0.80 桶 | 13 場，46.2%（基準 82%） |
| ALERT：0.80-0.90 桶 | 17 場，58.8%（基準 95%） |

### 2026-04-27 主要異動（Agent team 分析後執行）

| 項目 | 變更 | 依據 |
| --- | --- | --- |
| `CONFIDENCE_CAP` | **新增 0.85** | 歷史前60場從未出現 0.90+；2026 出現 9 場全是假信心 |
| `SP_FULL_STARTS` | **新增 15** | SP < 15 出賽時 advanced+SP 路由仍套用 70% 信心收縮 |
| `SP_FULL_SHRINK` | **新增 0.70** | advanced+SP (45.5%) 比 soft_blend (57.1%) 更差的修正 |
| `PLATT_REFIT_THRESHOLD` | **新增 80** | 累積 ≥80 場時報告自動提示 re-fit Platt A/B |
| 信心桶監控 | **新增** `track_high_confidence_predictions.py` | 各桶低於 65% 自動標示 ALERT |
| `ELO_HOME_ADVANTAGE` | **10 → 20** | A/B 實驗（experiment_home_adv.py）確認 adv=20 的 ECE 0.0562 優於 adv=10 的 0.0608（2026-04-28） |

> 2026 開季 Elo 極端化：樂天(1625) vs 味全(1412)，spread=214，歷史均值 138。  
> Platt A=1.45 將 Elo 驅動的高信心進一步放大，是假 0.90+ 信心的根源。

---

## 模型改善進度

| 項目 | 狀態 |
| --- | --- |
| hard routing → soft routing | 完成 |
| EARLY_PROB_SHRINK 調整 | 完成（0.50） |
| Platt scaling 實作 | 完成（歷史擬合，待 2026 re-fit） |
| CONFIDENCE_CAP 0.85 | **完成**（建議1，04-27） |
| SP_FULL_STARTS/SHRINK 保護 | **完成**（建議2，04-27） |
| 信心桶即時監控 | **完成**（建議4，04-27） |
| SP multi-season prior | **待辦** |
| 2026 re-fit Platt calibration（≥80 場後） | **待辦**（建議3） |
| ELO_REGRESSION 調整（2026 spread=214 異常大） | **待辦**（建議5） |
| 牛棚疲勞 / 打線強度特徵 | **待辦** |

---

## 目前不要採用

- 不要把 postseason adjustment layer 接進 production
- 不要直接用 `(season_year, game_sno)` join postseason（必須帶 `kind_code`）
- 2026 Platt calibration 不應直接用舊 A/B 值做高信心篩選依據（待 re-fit）

---

## 下一步優先順序

| 優先 | 項目 |
| --- | --- |
| A | 每日追蹤：`predict_today.py --verify`、`track_high_confidence_predictions.py` |
| B | 2026 Platt re-fit：累積 ≥80 場後重算 A/B |
| C | 建立 SP multi-season prior：SP < 5 場時加入前一年 / 生涯 rolling 指標 |
| D | 分析 early model 2026 失準原因：Elo 更新是否過慢、前一年特徵是否漂移 |

---

## 重跑指令

```powershell
# 更新資料
python cpbl_boxscore_scraper.py --start-year 2026 --end-year 2026 --refresh-range
python scrape_starting_pitchers.py --year 2026

# 即時預測與驗證
python predict_today.py --date 2026-04-26
python predict_today.py --date 2026-04-26 --verify

# regime 回測（含預測記錄輸出）
python evaluate_game_predictions_regime.py --recent-start 2026-04-08
python evaluate_game_predictions_regime.py --save-predictions walkforward_all.csv

# 校準報告
python analyze_game_prediction_calibration.py --start-year 2020 --end-year 2025 --live-start-date 2026-04-01 --live-end-date 2026-04-25

# 高信心追蹤 / EV
python track_high_confidence_predictions.py --start-date 2026-04-01 --end-date 2026-04-25
python analyze_betting_ev.py --date 2026-04-26
```

---

## 主要腳本

| 腳本 | 說明 |
| --- | --- |
| `cpbl_boxscore_scraper.py` | Box score 爬蟲（中文站 `/box/getlive`） |
| `scrape_starting_pitchers.py` | 先發投手爬蟲（`--kind-code A/C/E`） |
| `build_game_features.py` | Pre-game rolling team stats |
| `evaluate_game_predictions_regime.py` | Soft-regime walk-forward；`--save-predictions` 輸出 CSV |
| `predict_today.py` | 即時預測（soft-regime v2 + Platt calibration） |
| `track_high_confidence_predictions.py` | 高信心預測追蹤 |
| `analyze_game_prediction_calibration.py` | ECE / Brier / confidence bucket 診斷 |
| `analyze_betting_ev.py` | 投注 EV 分析 |

---

## SQLite 現況（cpbl.sqlite）

| 資料表 | 筆數 | 說明 |
| --- | --- | --- |
| `team_game_results` | 4,173+ | 2026：58 完賽（SNO 8–64） |
| `game_starting_pitchers` | 4,075+ | key=(season_year, kind_code, game_sno) |
| `game_features` | 3,856 | 最新至 2026-04-25（SNO 61） |
| `prediction_tracking` | 50 | 2026 已驗證 50 場，正確 26/50 |

---

## 注意事項

- `cpbl.sqlite` 由 `.gitignore` 排除；`train_rows` 只含 `kind_code='A'`
- `predict_today.py` 輸出的 prob 為 Platt 校準後值；`PLATT_A/B` 基於 2016–2025 OOS 預測
- SP lookup key = `(season_year, kind_code, game_sno)`；walk-forward 的 `sp_available` 需 `build_pitcher_features.py` 更新才有效
