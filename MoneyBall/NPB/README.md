# Baseball 預測專案（CPBL + NPB）

## 專案概覽

| 聯盟 | 球隊數 | 資料範圍 | Walk-forward 準確率 | Baseline |
| --- | ---: | --- | ---: | ---: |
| CPBL（中華職棒） | 6 | 2011–2026 | **73.15%** | 51.46% |
| NPB（日本職棒） | 12 | 2006–2026 | **54.65%** | 53.50% |

---

## NPB 專案

### 最新進度（2026-05-08）

#### SP FIP 特徵落地（`build_game_features_npb.py`）

- 已新增 `home_sp_fip_roll`、`vis_sp_fip_roll`、`diff_sp_fip`，採最近 10 場有效先發（`ip_outs >= 3`）、不足 3 場則 `NULL`
- FIP constant 改為每季使用 prior-season 歷史聯盟資料計算，避免看見當季未來資訊
- rebuild 後 `game_features_npb` 共 `11,979` 筆、可用 `diff_sp_fip` 共 `6,272` 筆（完成賽事）
- 2016–2026 的 `diff_sp_fip` 年均值大致接近 0，Spearman 對 `home_win` 多數為小幅負相關（方向符合「主場 SP FIP 較差時，主場勝率不應被錯誤抬高」）
- 2011–2015 的 BIS 歷史先發資料存在大量純數字投手名（如 `0`、`1`），builder 已在載入時排除，避免把假投手歷史污染 FIP roll；因此早期年度的 FIP 覆蓋率仍有限

### 最新進度（2026-04-24）

#### 模型結果（Walk-Forward 2016–2026，逐場滾動）

| 指標 | 數值 |
| --- | ---: |
| 整體準確率 | 54.65% |
| vs Baseline (+53.5%) | +1.15pp |
| p≥0.60 場次 | 1,426 |
| p≥0.60 準確率 | 60.38% |
| 執行時間 | ~96 秒 |

#### 各年準確率（primary route）

| 年份 | 整體 | primary |
| ---: | ---: | ---: |
| 2016 | 55.29% | 55.94% |
| 2017 | 58.06% | 58.61% |
| 2018 | 53.21% | 53.94% |
| 2019 | 55.23% | 54.49% |
| 2020 | 54.59% | 53.85% |
| 2021 | 54.26% | 54.21% |
| 2022 | 52.28% | 51.76% |
| 2023 | 55.30% | 54.97% |
| 2024 | 53.19% | 54.14% |
| 2025 | 54.31% | 55.56% |
| 2026 | 58.59% | 57.35% |

#### 高信心分析（2020–2026）

| 門檻 | 場次 | 準確率 |
| --- | ---: | ---: |
| p ≥ 50% | 4,399 | 54.1% |
| p ≥ 60% | 732 | 59.3% |
| p ≥ 70% | 0 | — |

> NPB 12 隊競爭均衡，模型機率集中在 0.50–0.67，p≥70% 極少出現。p≥60% 為實用閾值。

### 資料管線

#### 資料庫（npb.sqlite）

| 資料表 | 筆數 | 說明 |
| --- | --- | --- |
| `team_game_results` | 25,000+ | 2006–2026 含 BIS 歷史資料 |
| `game_starting_pitchers` | 33,000+ | 先發投手逐場成績 |
| `game_features_npb` | 11,979 | Pre-game rolling stats（CL/PL 2011–2026，含新 SP FIP 欄位） |

#### 爬蟲腳本

| 腳本 | 範圍 | 說明 |
| --- | --- | --- |
| `npb_schedule_scraper.py` | 2016–2026 | 官網排程爬蟲 |
| `npb_boxscore_scraper.py` | 2016–2026 | Box score / SP 爬蟲 |
| `npb_bis_scraper.py` | 2006–2015 | 歷史 BIS archive 爬蟲 |

#### 特徵建構

```
build_game_features_npb.py --year 2026 --include-scheduled
build_pitcher_features_npb.py --year 2026
```

> 訓練起始年：2011（TRAIN_START_YEAR=2011）；測試窗口：2016–2026

### 模型架構

- **Regime Routing**：early（季初<10場，Logistic on Elo）/ primary（XGBoost 26特徵）
- **Walk-forward**：逐場滾動（game-by-game），對齊 CPBL 架構，日期快取加速
- **XGBoost params**：n=50, depth=3, min_child_weight=15, reg_lambda=3.0, lr=0.05
- **SP coverage**：82.6%（跨季滾動歷史，career-level pitcher_history）

### Elo 參數（build_game_features_npb.py）

| 參數 | 數值 | 備註 |
| --- | ---: | --- |
| `ELO_K` | **52** | 與 CPBL 相同；12 隊聯盟可考慮降至 20–30 |
| `ELO_HOME_ADV` | **10** | 對應主場勝率 +1.44%；實際 NPB 主場約 +2–3% |
| `ELO_REGRESSION` | **0.45** | 與 CPBL 相同 |
| `TRAIN_START_YEAR` | 2011 | 測試窗口 2016–2026 |

> ⚠️ 目前 NPB Elo 參數直接沿用 CPBL 設定，未針對 12 隊結構調整。建議未來實驗降低 K 值（K=20–30）與提高 home_adv（15–20），觀察對 ECE 的影響。

### 每日預測

```bash
python predict_today_npb.py                     # 今日預測
python predict_today_npb.py --date 2026-04-24   # 指定日期
python predict_today_npb.py --date 2026-04-23 --verify  # 驗證昨日
```

輸出存為 `predictions_npb_YYYYMMDD.md`

### 台灣運彩賠率 EV 分析

```bash
python npb_betting_ev.py               # 今日 EV（需 predict_today_npb 先建好 features）
python npb_betting_ev.py --date 2026-04-28 --save   # 指定日期並存 .md
```

- 賠率來源：Taiwan Sports Lottery 不讓分（`https://www-talo-ssb-pr.sportslottery.com.tw/sport/棒球/34731.1`）
- EV 計算：`model_prob × (decimal − 1) − (1 − model_prob)`
- BET 門檻：model_prob ≥ 55%，EV ≥ +3%

### 2026-04-23 驗證結果

**5/6 = 83.3%**（6 場例行賽，1 場 LOW 信心錯誤）

### 已排除方向（超參數實驗 branch: hyperparam-tuning）

降低正則化（depth=5, mcw=5, λ=0.5）雖可產生 p≥70% 預測，但 2024–2025 p70 準確率僅 50–52%，近年崩壞。**維持 baseline 超參數。**

---

## CPBL 專案

### 最新進度（2026-04-23）

#### 核心模型結果

| 模型 | 2016–2025 benchmark | 2025 單季 |
| --- | ---: | ---: |
| 主場 baseline | 51.46% | 55.75% |
| advanced ensemble (XGBoost) | 73.27% | 82.18% |
| **regime model** | **73.15%** | — |

#### 高信心門檻（2020+，regime model）

| 門檻 | 場次 | 準確率 |
| --- | ---: | ---: |
| p ≥ 0.60 | 1,401 | 79.87% |
| p ≥ 0.70 | 1,123 | 84.59% |
| p ≥ 0.80 | 279 | 90.32% |

#### 2026 即時追蹤（04-01 到 04-22）

- 可驗證：40 場；高信心（p≥0.60）：27 場；命中 14/27 = **51.9%**

### 每日預測

```bash
python predict_today.py --date 2026-04-24
python predict_today.py --date 2026-04-23 --verify
python track_high_confidence_predictions.py --start-date 2026-04-01 --end-date 2026-04-24
```

### 資料庫（cpbl.sqlite）

| 資料表 | 筆數 | 說明 |
| --- | --- | --- |
| `team_game_results` | 4,430+ | 含 2026，kind_code A/C/E |
| `game_starting_pitchers` | ~4,050 | key=(season_year, kind_code, game_sno) |
| `game_features` | 3,799 | Pre-game rolling stats（例行賽 A） |
| `prediction_tracking` | 40+ | 高信心追蹤 |

---

## 注意事項

- `cpbl.sqlite` / `npb.sqlite` 均由 `.gitignore` 排除，須本機重建
- NPB BIS 2006–2015 資料：home/away 判斷依標題解析（score box 為勝方優先，不可用位置判斷）
- CPBL train_rows 只含 kind_code='A'；GameState 仍處理所有場次（Elo/rolling 不受影響）
- CPBL SP lookup key：(season_year, kind_code, game_sno)
