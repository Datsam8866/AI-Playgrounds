# KBO Regime Model — Walk-Forward Benchmark

Train start: 2013 | Backtest: 2016–2025
Elo: K=48, home_adv=10, regression=0.50
XGBoost: max_depth=3, min_child_weight=30, n_estimators=30

## Walk-Forward Results (2016–2025)

| Metric | Value |
| --- | ---: |
| Total games | 7050 |
| Correct | 3932 |
| **Accuracy** | **55.77%** |
| Home baseline | 52.71% |

## Per-Year Breakdown

| Year | Games | Accuracy | early | primary | fallback |
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

## High-Confidence Subset (2016–2025)

| Threshold | Games | Coverage | Accuracy |
| ---: | ---: | ---: | ---: |
| p >= 0.55 | 3334 | 47.3% | 58.91% |
| p >= 0.60 | 725 | 10.3% | 64.55% |
| p >= 0.70 | 0 | 0.0% | 0.00% |
| p >= 0.80 | 0 | 0.0% | 0.00% |

## 2026 YTD

| Games | Correct | Accuracy |
| ---: | ---: | ---: |
| 155 | 78 | 50.32% |

