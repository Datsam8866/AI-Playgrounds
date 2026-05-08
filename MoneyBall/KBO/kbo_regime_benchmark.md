# KBO Regime Model — Walk-Forward Benchmark

Train start: 2013 | Backtest: 2016–2025
Elo: K=48, home_adv=10, regression=0.50
XGBoost: max_depth=3, min_child_weight=30, n_estimators=30

## Walk-Forward Results (2016–2025)

| Metric | Value |
| --- | ---: |
| Total games | 7050 |
| Correct | 3849 |
| **Accuracy** | **54.60%** |
| Home baseline | 52.71% |

## Per-Year Breakdown

| Year | Games | Accuracy | early | primary | fallback |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2016 | 713 | 54.70% | 53 | 465 | 195 |
| 2017 | 709 | 55.57% | 50 | 522 | 137 |
| 2018 | 714 | 53.22% | 50 | 541 | 123 |
| 2019 | 713 | 55.12% | 50 | 507 | 156 |
| 2020 | 707 | 57.85% | 52 | 502 | 153 |
| 2021 | 670 | 52.09% | 51 | 461 | 158 |
| 2022 | 708 | 53.11% | 50 | 546 | 112 |
| 2023 | 708 | 54.80% | 54 | 515 | 139 |
| 2024 | 710 | 55.21% | 52 | 487 | 171 |
| 2025 | 698 | 54.15% | 53 | 493 | 152 |

## High-Confidence Subset (2016–2025)

| Threshold | Games | Coverage | Accuracy |
| ---: | ---: | ---: | ---: |
| p >= 0.55 | 3329 | 47.2% | 57.40% |
| p >= 0.60 | 1119 | 15.9% | 62.11% |
| p >= 0.70 | 82 | 1.2% | 65.85% |
| p >= 0.80 | 2 | 0.0% | 100.00% |

## 2026 YTD

| Games | Correct | Accuracy |
| ---: | ---: | ---: |
| 155 | 78 | 50.32% |

