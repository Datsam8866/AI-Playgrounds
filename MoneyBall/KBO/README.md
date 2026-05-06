# Baseball Prediction Projects — CPBL + KBO

**最後更新：2026-04-28**

## 專案概覽

| 聯盟 | 資料庫 | 資料範圍 | 整體準確率 | 高信心（p≥0.60） | 狀態 |
| --- | --- | --- | ---: | ---: | --- |
| **CPBL 中華職棒** | `cpbl.sqlite` | 2011–2026 | **73.15%** | 79.87% | ELO_HOME_ADV 已更新為 20 |
| **KBO 韓國職棒** | `kbo.sqlite` | 2011–2026 | **55.77%** | 64.55% | predict_today_kbo.py 已建立 |

## 關鍵差異（CPBL vs KBO）

| 維度 | CPBL | KBO |
| --- | --- | --- |
| 球隊數 | 6 | 10 |
| Walk-forward 準確率 | **72.7%** | 55.77% |
| 實用高信心閾值 | p≥0.70 → 85.3% | p≥0.60 → 64.55% |
| p≥0.70 可行 | ✅ | ❌（raw uncalibrated，校準後消失）|
| ELO_K | 52 | 48 |
| ELO_HOME_ADV | **20**（A/B 驗證後更新）| 10 |
| ELO_REGRESSION | 0.45 | 0.50 |
| XGBoost n_estimators | 50 | 30 |
| XGBoost min_child_weight | 20 | 30 |
| Platt scaling | ✅ A=1.45（待 re-fit）| ❌（模型辨別力不足，加了也無效）|
| 每日即時預測 | ✅ predict_today.py | ✅ predict_today_kbo.py（2026-04-28 新建）|
| 高信心追蹤工具 | ✅ | ❌（待建）|

---

## CPBL 最新進度（2026-04-24）

### 核心模型結果（Walk-Forward 2016–2025）

| 模型 | 準確率 | 2026 YTD |
| --- | ---: | ---: |
| Home baseline | 51.46% | — |
| **Regime model** | **73.15%** | 50.0%（高信心 14/28） |

### 高信心門檻（regime model，2020+）

| 門檻 | 場次 | Coverage | 準確率 |
| ---: | ---: | ---: | ---: |
| 全部 | 1,837 | 100% | 73.76% |
| p ≥ 0.60 | 1,401 | 76.3% | 79.87% |
| p ≥ 0.70 | 1,123 | 61.1% | 84.59% |
| p ≥ 0.80 | 279 | 15.2% | 90.32% |

### CPBL 重跑指令

```powershell
python cpbl_boxscore_scraper.py --start-year 2026 --end-year 2026 --refresh-range
python scrape_starting_pitchers.py --year 2026
python predict_today.py --date 2026-04-24
python track_high_confidence_predictions.py --start-date 2026-04-01
```

### CPBL 主要腳本

| 腳本 | 說明 |
| --- | --- |
| `cpbl_boxscore_scraper.py` | Box score 爬蟲 |
| `scrape_starting_pitchers.py` | 先發投手爬蟲 |
| `build_game_features.py` | Pre-game rolling 特徵 |
| `build_pitcher_features.py` | 先發投手 rolling ERA/WHIP/K9 |
| `evaluate_game_predictions_regime.py` | Regime model walk-forward |
| `predict_today.py` | 即時預測（`--date / --verify`） |
| `track_high_confidence_predictions.py` | 高信心追蹤 |

### CPBL SQLite 現況（cpbl.sqlite）

| 資料表 | 筆數 | 說明 |
| --- | ---: | --- |
| `team_game_results` | 4,430+ | 2011–2026，kind_code A/C/E |
| `game_starting_pitchers` | ~4,050 | 例行賽+季後賽 |
| `game_features` | 3,802 | Pre-game rolling stats |
| `prediction_tracking` | 40+ | 高信心追蹤 |

---

## KBO 最新進度（2026-04-24）

### 核心模型結果（Walk-Forward 2016–2025）

| 指標 | 數值 |
| --- | ---: |
| 總場次 | 7,050 |
| 整體準確率 | **55.77%** |
| Home baseline | 52.71% |
| Delta vs baseline | +3.06 pp |

### 高信心門檻（2016–2025）

| 門檻 | 場次 | Coverage | 準確率 |
| ---: | ---: | ---: | ---: |
| p ≥ 0.55 | 3,334 | 47.3% | 58.91% |
| p ≥ 0.60 | 725 | 10.3% | **64.55%** |
| p ≥ 0.70 | 0 | 0.0% | 0.00% |

### 逐年準確率（2016–2025）

| 年份 | 場次 | 準確率 | early | primary | fallback |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2016 | 713 | 54.98% | 53 | 465 | 195 |
| 2017 | 709 | 54.44% | 50 | 522 | 137 |
| 2018 | 714 | 53.50% | 50 | 541 | 123 |
| 2019 | 713 | 58.06% | 50 | 507 | 156 |
| 2020 | 707 | 57.85% | 52 | 502 | 153 |
| 2021 | 670 | 54.78% | 51 | 461 | 158 |
| 2022 | 708 | 54.52% | 50 | 546 | 112 |
| 2023 | 708 | 57.06% | 54 | 515 | 139 |
| 2024 | 710 | 56.34% | 52 | 487 | 171 |
| 2025 | 698 | 56.16% | 53 | 493 | 152 |

### 2026 YTD（截至 04-24）

| 場次 | 正確 | 準確率 |
| ---: | ---: | ---: |
| 106 | 55 | 51.89% |

> 開季樣本小，信心區間寬，持續觀察中。

### Elo 參數（Gemini 審查確認）

`K=48, home_adv=10, regression=0.50`

### XGBoost 參數

`max_depth=3, min_child_weight=30, n_estimators=30`

### KBO P≥0.70 掃描結論

已建立 `analyze_kbo_p70_sweep.py`、`analyze_kbo_calibration_primary.py`、`kbo_p70_sweep.md`、`kbo_calibration_primary.md`。目前不建議直接改 production 參數：

| 候選 | 整體準確率 | Primary P≥0.70 | Holdout Primary P≥0.70 |
| --- | ---: | ---: | ---: |
| production `d3_mcw30_n30_l3` | **55.77%** | 0 | 0 |
| `d2_mcw5_n160_lr03_l05` | 55.39% | 111 場 / 75.68% | 37 場 / 75.68% |
| `d2_mcw5_n200_lr03_l05` | 55.29% | 155 場 / 74.19% | 45 場 / 75.56% |
| `d3_mcw15_n75_l3` | 55.02% | 114 場 / 71.93% | 37 場 / 72.97% |

結論：P≥0.70 場次可增加，但目前未達 80% 穩定命中率；P≥0.75 的 80% bucket 樣本太小，不足以 production 化。

Calibration guardrail 結論：

| 候選 | 校準方式 | Primary P≥0.70 | Brier | Logloss | ECE |
| --- | --- | ---: | ---: | ---: | ---: |
| production | raw | 0 | **0.2453** | **0.6837** | **0.0058** |
| precision `d2_mcw5_n160` | raw | 111 場 / 75.68% | 0.2460 | 0.6853 | 0.0166 |
| precision `d2_mcw5_n160` | Platt | 0 | 0.2460 | 0.6850 | 0.0083 |
| coverage `d2_mcw5_n200` | raw | 155 場 / 74.19% | 0.2467 | 0.6867 | 0.0212 |
| coverage `d2_mcw5_n200` | Platt | 0 | 0.2462 | 0.6855 | 0.0077 |

結論：`n160` / `n200` 可作為 raw high-confidence ranking bucket，但校準後 P≥0.70 會消失；不能把 raw P≥0.70 解讀成已校準的 70% 勝率。

### KBO 重跑指令

```powershell
# 更新資料
python kbo_boxscore_scraper.py --start-year 2026 --end-year 2026 --refresh-range
python build_kbo_game_features.py
python build_kbo_pitcher_features.py

# 每日預測（新）
python predict_today_kbo.py                         # 今日
python predict_today_kbo.py --date 2026-04-27 --verify  # 驗證昨日

# 評估
python evaluate_kbo_predictions_regime.py
```

### KBO 主要腳本

| 腳本 | 說明 |
| --- | --- |
| `kbo_boxscore_scraper.py` | Box score 爬蟲（GetMonthSchedule + GetScoreBoardScroll） |
| `build_kbo_game_features.py` | Elo + 29 rolling 特徵 → `game_features`（10,044 rows，含 early regime） |
| `build_kbo_pitcher_features.py` | SP rolling ERA/WHIP/K9 → `game_features`（72.6% coverage） |
| `evaluate_kbo_predictions_regime.py` | Regime walk-forward → `kbo_regime_benchmark.md` |
| **`predict_today_kbo.py`** | **每日即時預測（2026-04-28 新建）；需先執行 build_kbo_game_features.py** |

### KBO SQLite 現況（kbo.sqlite）

| 資料表 | 筆數 | 說明 |
| --- | ---: | --- |
| `team_game_results` | 10,976+ | 2011–2026 例行賽+季後賽 |
| `game_starting_pitchers` | ~10,000 | SP stats，side=home/away |
| `game_features` | 10,044 | Pre-game features（sr_id=0 例行賽，含 early rows） |

---

## 注意事項

- `cpbl.sqlite` / `kbo.sqlite` 由 `.gitignore` 排除
- KBO early regime：`game_features` 保留例行賽前 10 場，rolling / rest / streak / season_games_before 只用 `sr_id=0`
- CPBL train_rows 只含 `kind_code='A'`（例行賽）
- KBO `4사구` = BB+HBP 合併欄位，WHIP 計算方式一致

---

## 下一步

| 優先 | 項目 |
| --- | --- |
| A | 建立 `predict_today_kbo.py` — 每日 KBO 即時預測 |
| B | CPBL 2026 例行賽持續追蹤（每日 `predict_today.py --verify`） |
| C | KBO 高信心下一輪：建立 raw ranking bucket 追蹤，不把 P≥0.70 當 calibrated probability |
