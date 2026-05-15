# NBA Walk-Forward XGBoost

Backtest years: 2016–2025
Rolling window: 5 seasons
Features: 25

## Per-Season Accuracy

| Year | Train Window | Train N | Test N | Raw Accuracy | Cal Accuracy |
| ---: | --- | ---: | ---: | ---: | ---: |
| 2016 | 2011-2015 | 5,909 | 1,230 | 64.15% | 64.23% |
| 2017 | 2012-2016 | 6,149 | 1,230 | 64.72% | 62.93% |
| 2018 | 2013-2017 | 6,150 | 1,230 | 66.34% | 65.53% |
| 2019 | 2014-2018 | 6,150 | 1,059 | 64.59% | 64.21% |
| 2020 | 2015-2019 | 5,979 | 1,080 | 61.85% | 63.15% |
| 2021 | 2016-2020 | 5,829 | 1,230 | 63.90% | 62.52% |
| 2022 | 2017-2021 | 5,829 | 1,230 | 63.09% | 63.01% |
| 2023 | 2018-2022 | 5,829 | 1,230 | 65.20% | 63.25% |
| 2024 | 2019-2023 | 5,829 | 1,225 | 65.31% | 64.49% |
| 2025 | 2020-2024 | 5,995 | 1,225 | 66.94% | 66.45% |

## Overall Accuracy

- Raw: 64.64%
- Calibrated: 63.98%

## Calibration Metrics (Isotonic, calibrated probs)

- Brier Score: 0.2210  (lower is better; random = 0.25)
- ECE (10 bins): 0.0142  (lower is better; perfect = 0.00)

## High-Confidence Subsets (calibrated confidence)

| Threshold | Games | Coverage | Accuracy |
| ---: | ---: | ---: | ---: |
| p_cal > 0.60 | 7,776 | 65.0% | 69.80% |
| p_cal > 0.65 | 5,208 | 43.5% | 73.23% |

## Feature Importance (top 10, mean across folds)

| Feature | Importance |
| --- | ---: |
| elo_win_prob | 0.156740 |
| diff_elo | 0.145797 |
| diff_net_rtg | 0.071830 |
| diff_pyth_wp | 0.051939 |
| vis_b2b | 0.035209 |
| home_pyth_wp | 0.034705 |
| diff_rest | 0.033776 |
| home_net_rtg | 0.033648 |
| vis_rest | 0.031194 |
| diff_win_pct_20 | 0.029931 |

## Model Params

- max_depth=3
- min_child_weight=10
- n_estimators=300
- learning_rate=0.05
- subsample=0.8
- colsample_bytree=0.7
- reg_lambda=2.0
- reg_alpha=0.5
