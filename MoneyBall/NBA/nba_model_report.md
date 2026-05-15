# NBA Walk-Forward XGBoost

Backtest years: 2016–2025
Rolling window: 5 seasons
Features: 31

## Per-Season Accuracy

| Year | Train Window | Train N | Test N | Raw Accuracy | Cal Accuracy |
| ---: | --- | ---: | ---: | ---: | ---: |
| 2016 | 2011-2015 | 5,909 | 1,230 | 63.90% | 63.82% |
| 2017 | 2012-2016 | 6,149 | 1,230 | 64.23% | 64.63% |
| 2018 | 2013-2017 | 6,150 | 1,230 | 66.18% | 65.69% |
| 2019 | 2014-2018 | 6,150 | 1,059 | 65.06% | 64.87% |
| 2020 | 2015-2019 | 5,979 | 1,080 | 61.76% | 62.41% |
| 2021 | 2016-2020 | 5,829 | 1,230 | 64.07% | 63.74% |
| 2022 | 2017-2021 | 5,829 | 1,230 | 63.01% | 63.66% |
| 2023 | 2018-2022 | 5,829 | 1,230 | 65.04% | 65.04% |
| 2024 | 2019-2023 | 5,829 | 1,225 | 66.37% | 65.47% |
| 2025 | 2020-2024 | 5,995 | 1,225 | 67.27% | 66.69% |

## Overall Accuracy

- Raw: 64.72%
- Calibrated: 64.63%

## Calibration Metrics (Isotonic, calibrated probs)

- Brier Score: 0.2188  (lower is better; random = 0.25)
- ECE (10 bins): 0.0171  (lower is better; perfect = 0.00)

## High-Confidence Subsets (calibrated confidence)

| Threshold | Games | Coverage | Accuracy |
| ---: | ---: | ---: | ---: |
| p_cal > 0.60 | 7,203 | 60.2% | 70.75% |
| p_cal > 0.65 | 4,963 | 41.5% | 74.97% |

## Feature Importance (top 10, mean across folds)

| Feature | Importance |
| --- | ---: |
| diff_elo | 0.160686 |
| elo_win_prob | 0.122761 |
| diff_pyth_wp | 0.062285 |
| diff_net_rtg | 0.059070 |
| home_pyth_wp | 0.034352 |
| diff_lineup_pts | 0.032549 |
| diff_rest | 0.031147 |
| home_net_rtg | 0.030912 |
| vis_b2b | 0.030894 |
| vis_rest | 0.029265 |

## Model Params

- max_depth=3
- min_child_weight=10
- n_estimators=300
- learning_rate=0.05
- subsample=0.8
- colsample_bytree=0.7
- reg_lambda=2.0
- reg_alpha=0.5
