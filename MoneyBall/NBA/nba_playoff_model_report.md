# NBA Playoff Walk-Forward XGBoost

Backtest years: 2016–2025
Rolling window: 5 seasons
Features: 15

> Note: Each playoff season has ~70-100 games. Single-season accuracy has high variance (±10pp CI). Evaluate cumulative multi-season trends.

## Per-Season Accuracy

| Year | Train Window | Train N | Test N | Raw Accuracy | Cal Accuracy |
| ---: | --- | ---: | ---: | ---: | ---: |
| 2016 | 2011-2015 | 425 | 79 | 60.76% | 63.29% |
| 2017 | 2012-2016 | 420 | 82 | 60.98% | 69.51% |
| 2018 | 2013-2017 | 417 | 82 | 63.41% | 56.10% |
| 2019 | 2014-2018 | 410 | 83 | 65.06% | 59.04% |
| 2020 | 2015-2019 | 412 | 85 | 57.65% | 57.65% |
| 2021 | 2016-2020 | 411 | 87 | 58.62% | 66.67% |
| 2022 | 2017-2021 | 419 | 84 | 61.90% | 60.71% |
| 2023 | 2018-2022 | 421 | 82 | 57.32% | 58.54% |
| 2024 | 2019-2023 | 421 | 84 | 55.95% | 55.95% |
| 2025 | 2020-2024 | 422 | 66 | 65.15% | 65.15% |

## Overall Accuracy

- Raw: 60.57%
- Calibrated: 61.18%

## Calibration Metrics (Isotonic, calibrated probs)

- Brier Score: 0.2490  (lower is better; random = 0.25)
- ECE (10 bins): 0.0815  (lower is better; perfect = 0.00)

## High-Confidence Subsets (calibrated confidence)

| Threshold | Games | Coverage | Accuracy |
| ---: | ---: | ---: | ---: |
| p_cal > 0.60 | 653 | 80.2% | 62.94% |
| p_cal > 0.65 | 405 | 49.8% | 62.22% |

## Feature Importance (top 10, mean across folds)

| Feature | Importance |
| --- | ---: |
| home_has_homecourt | 0.166614 |
| diff_elo_rs | 0.088496 |
| diff_elo_po | 0.069604 |
| diff_elo_change_po | 0.064771 |
| elo_win_prob_po | 0.060561 |
| diff_rs_pyth_wp | 0.059883 |
| diff_rs_lineup_pts | 0.059869 |
| vis_series_wins | 0.059537 |
| diff_rs_net_rtg | 0.058489 |
| playoff_round | 0.056508 |

## Model Params

- max_depth=3
- min_child_weight=5
- n_estimators=200
- learning_rate=0.05
- subsample=0.8
- colsample_bytree=0.7
- reg_lambda=2.0
- reg_alpha=0.5
