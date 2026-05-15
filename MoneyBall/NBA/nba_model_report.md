# NBA Walk-Forward XGBoost

Backtest years: 2016–2025
Rolling window: 5 seasons
Features: 28

## Per-Season Accuracy

| Year | Train Window | Train N | Test N | Raw Accuracy | Cal Accuracy |
| ---: | --- | ---: | ---: | ---: | ---: |
| 2016 | 2011-2015 | 5,909 | 1,230 | 63.82% | 64.55% |
| 2017 | 2012-2016 | 6,149 | 1,230 | 64.31% | 63.82% |
| 2018 | 2013-2017 | 6,150 | 1,230 | 65.77% | 65.85% |
| 2019 | 2014-2018 | 6,150 | 1,059 | 64.02% | 64.59% |
| 2020 | 2015-2019 | 5,979 | 1,080 | 60.93% | 61.48% |
| 2021 | 2016-2020 | 5,829 | 1,230 | 64.07% | 63.74% |
| 2022 | 2017-2021 | 5,829 | 1,230 | 63.58% | 63.01% |
| 2023 | 2018-2022 | 5,829 | 1,230 | 65.69% | 66.26% |
| 2024 | 2019-2023 | 5,829 | 1,225 | 66.37% | 67.10% |
| 2025 | 2020-2024 | 5,995 | 1,225 | 66.94% | 67.10% |

## Overall Accuracy

- Raw: 64.60%
- Calibrated: 64.79%

## Calibration Metrics (Isotonic, calibrated probs)

- Brier Score: 0.2193  (lower is better; random = 0.25)
- ECE (10 bins): 0.0190  (lower is better; perfect = 0.00)

## High-Confidence Subsets (calibrated confidence)

| Threshold | Games | Coverage | Accuracy |
| ---: | ---: | ---: | ---: |
| p_cal > 0.60 | 7,719 | 64.5% | 70.03% |
| p_cal > 0.65 | 4,786 | 40.0% | 75.26% |

## Feature Importance (top 10, mean across folds)

| Feature | Importance |
| --- | ---: |
| elo_win_prob | 0.155043 |
| diff_elo | 0.120233 |
| diff_pyth_wp | 0.093647 |
| diff_net_rtg | 0.050243 |
| home_pyth_wp | 0.032929 |
| diff_lineup_pts | 0.031452 |
| vis_b2b | 0.030894 |
| diff_rest | 0.030099 |
| home_net_rtg | 0.029400 |
| vis_rest | 0.027779 |

## Model Params

- max_depth=3
- min_child_weight=10
- n_estimators=300
- learning_rate=0.05
- subsample=0.8
- colsample_bytree=0.7
- reg_lambda=2.0
- reg_alpha=0.5
