# 棒球預測系統

---

# CPBL（維護中）

**模型：** Soft-regime XGBoost ensemble v2 + Platt scaling（A=1.450, B=−0.057）

## Walk-Forward 準確率

| 年份 | 場次 | 全場 | p>0.70 場次 | p>0.70 準確率 |
| ---: | ---: | ---: | ---: | ---: |
| 2016–2025 合計 | 2652 | 72.7% | 1478 | 85.3% |
| **2025** | 348 | **83.0%** | 238 | **90.3%** |
| **2026**（04-25 止）| 47 | 53.2% | 25 | 56.0% |

> 2026 近期 34 場（04-08 後）已回升至 64.71%；開季前 20 場 35% 是主要拖累。

## 2026 即時（04-01 到 04-25）

高信心（p≥0.60）：15/28（53.6%）；early=1：34 場 58.8%；early=0：13 場 38.5%

## 待辦

- **A** 每日：`predict_today.py --verify` + `track_high_confidence_predictions.py`
- **B** 累積 ≥80 場後 re-fit Platt A/B
- **C** SP multi-season prior（< 5 場時用前一年/生涯 rolling）
- **D** 診斷 early model 2026 失準（Elo 更新速度 / prev-season 特徵漂移）

## 常用指令

```bash
python cpbl_boxscore_scraper.py --start-year 2026 --end-year 2026 --refresh-range
python predict_today.py --date 2026-04-26 --verify
python evaluate_game_predictions_regime.py --save-predictions walkforward_all.csv
python track_high_confidence_predictions.py --start-date 2026-04-01 --end-date 2026-04-25
```

## 注意

- `cpbl.sqlite` 排除於 git；`train_rows` 只含 `kind_code='A'`
- SP key = `(season_year, kind_code, game_sno)`；`sp_available` 需先跑 `build_pitcher_features.py`
- Platt A/B 基於 2016–2025；postseason adjustment layer 不接進 production
- 詳細技術文件見 `CPBL_Project_Documentation.md`

---

---

# MLB 戰績預測系統（進行中）

## 目標

以 CPBL soft-regime XGBoost ensemble 為藍圖，建立 MLB 30 隊、162 場/季的逐場勝敗預測系統。

| 項目 | CPBL | MLB |
| --- | --- | --- |
| 隊伍數 | 6 | 30 |
| 每季場次 | ~120/隊 | 162/隊 |
| 歷史資料 | 2011–2026 | 2003–2025（建議 burn-in 3 年） |
| 資料來源 | 官網 `/box/getlive` | MLB Stats API（免費公開） |

---

## 目前進度（2026-04-28）

| 步驟 | 說明 | 狀態 |
| --- | --- | --- |
| STEP 1 | 特徵遷移分析（Elo 參數、SP 特徵、MLB 結構差異） | **完成** |
| STEP 2 | 爬蟲建立（`mlb_boxscore_scraper.py` + `mlb.sqlite`） | **完成**（34,829 場，2011–2025） |
| STEP 3 | 特徵工程（`build_mlb_game_features.py`） | **完成**（34,829 筆，ELO_K=8，SP_WINDOW=8） |
| STEP 4 | Soft-regime XGBoost 模型 | **完成** |
| STEP 5 | Walk-forward 回測與特徵優化 | **完成**（56.7% 準確率，2014–2025） |
| STEP 6 | 即時預測工具（`predict_mlb_today.py`） | **完成** |
| STEP 7 | 賠率整合（The Odds API）與 EV 分析 | **完成** |
| STEP 8 | Platt/Isotonic 校準實驗（`mlb_calibration_analysis.py`）| **完成（2026-04-28）→ 不需校準** |

**系統狀態：維運期。每日流程：scraper → features → predict → EV。**

## 校準分析結論（2026-04-28）

| 指標 | Raw | Platt（fit 2014–2020）| Isotonic |
| --- | ---: | ---: | ---: |
| ECE（holdout 2021–2025）| **0.0039** | 0.0072 | 0.0056 |
| Brier | **0.2427** | 0.2427 | 0.2428 |
| P≥0.675 場次 / 命中率 | **589 場 / 70.3%** | 201 場 / 72.1% | 426 場 / 70.9% |

> **結論**：Raw ECE=0.0039 屬極佳校準水準（業界標準 0.02–0.05 才需校準）。Platt A=0.8375 施加後 ECE 惡化，P≥0.675 場次大幅萎縮。**維持 raw prob，不加校準層。**

## 常用指令（MLB）

```bash
# === 每日流程 ===
# 1. 更新昨日結果
python mlb_boxscore_scraper.py --start-year 2026 --end-year 2026 --refresh-range
python build_mlb_game_features.py

# 2. 今日預測 + 登錄追蹤
python predict_mlb_today.py --save
python mlb_track_predictions.py --record

# 3. 驗證昨日
python mlb_track_predictions.py --verify --date YYYY-MM-DD

# 4. 賠率 EV 分析（台灣運彩 不讓分）
python mlb_betting_ev.py --save
# 若頁面 URL 變動：python mlb_betting_ev.py --url "https://..." --save

# === 查詢 ===
python mlb_track_predictions.py --summary
```

## Walk-forward 準確率（2014–2025）

| 年份 | 場次 | 準確率 |
| ---: | ---: | ---: |
| 2014 | 2426 | 53.8% |
| 2015–2019 | ~12120 | 55.5–58.3% |
| 2020 | 896 | 57.8% |
| 2021–2025 | ~12120 | 55.0–59.9% |
| **合計** | **27,561** | **56.7%** |

> MLB 主場勝率 ~53%，模型 56.7% 屬正常水準；CPBL 可預測性更高（72.7%）
>
> | 門檻 | Coverage (2020+) | 準確率 (2020+) |
> |---|---|---|
> | P≥0.650 | 11.4% | 66.9% |
> | **P≥0.675**（甜蜜點） | **5.8%** | **69.3%** |
> | **P≥0.700** | **2.0%** | **70.5%** |

---

## MLB 模型參數（與 CPBL 對照）

| 常數 | CPBL | MLB 建議 | 理由 |
| --- | --- | --- | --- |
| `ELO_K` | 52 | **8** | 162 場/季，FiveThirtyEight 建議值，降震盪 |
| `ELO_HOME_ADV` | 10 | **25** | MLB 主場勝率 ~54%，約 35 點差；25 保守設定 |
| `ELO_REGRESSION` | 0.45 | **0.35** | MLB 球隊連續性更高，回歸不需太大 |
| `TEAM_BURN_IN` | 10 | **10** | 不變，佔 162 場比例更小 |
| `STARTER_BURN_IN` | 5 | **4** | 5 天輪換，4 場約 20 天已足夠 |
| `EARLY_PROB_SHRINK` | 0.50 | **0.55** | 30 隊 burn-in 較輕，收縮不需那麼強 |
| `TRAIN_START_YEAR` | 2013 | **2003** | Elo burn-in 3 年（2000–2002）後開始 |

**MLB 新增特徵（優先）：**
- `diff_sp_fip`：FIP 差值（去除守備噪音，比 ERA 更穩定）
- `universal_dh_era`：2022+ DH 制度 flag
- `is_interleague`：跨聯賽 flag
- `coors_field_factor`：洛磯主場海拔修正

---

## 最大風險點

| 風險 | 說明 | 對策 |
| --- | --- | --- |
| **MLB Stats API 限速** | 2015–2025 共 ~27,000 場 × 2 次 API call | sleep 0.3–0.5s；分年爬取；預估 4–6 小時 |
| **Probable Pitchers 時效** | 先發名單賽前 24h 才確定 | 提前一天爬一次，當天再更新 |
| **Interleague 主客場制度** | DH rule 2022 前 NL/AL 不對稱 | 加入 `universal_dh_era` flag |
| **Coors Field 離群值** | 高海拔使得分極高，模型易誤判 | 加入 `coors_field_factor` 特徵 |
| **開季 30 隊 Elo 收斂** | 2000 年全員從 1500 出發，前 3 年不穩定 | 2003 年後才納入訓練（同 CPBL burn-in 設計） |
| **SP FIP 資料取得** | MLB Stats API 不直接回傳 FIP，需自行計算 | 用 boxscore HR/BB/K/IP 自算；或爬 FanGraphs |
