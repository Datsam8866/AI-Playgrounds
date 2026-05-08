# KBO Regime Model — Walk-Forward Benchmark

Train start: 2013 | Backtest: 2016–2025
Elo: K=48, home_adv=10, regression=0.50
XGBoost: max_depth=3, min_child_weight=30, n_estimators=30

## Walk-Forward Results (2016–2025)

| Metric | Value |
| --- | ---: |
| Total games | 7050 |
| Correct | 3890 |
| **Accuracy** | **55.18%** |
| Home baseline | 52.71% |

## Per-Year Breakdown

| Year | Games | Accuracy | early | primary | fallback |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2016 | 713 | 55.54% | 53 | 465 | 195 |
| 2017 | 709 | 54.87% | 50 | 522 | 137 |
| 2018 | 714 | 52.94% | 50 | 541 | 123 |
| 2019 | 713 | 56.52% | 50 | 507 | 156 |
| 2020 | 707 | 57.57% | 52 | 502 | 153 |
| 2021 | 670 | 54.03% | 51 | 461 | 158 |
| 2022 | 708 | 53.11% | 50 | 546 | 112 |
| 2023 | 708 | 55.79% | 54 | 515 | 139 |
| 2024 | 710 | 55.07% | 52 | 487 | 171 |
| 2025 | 698 | 56.30% | 53 | 493 | 152 |

## High-Confidence Subset (2016–2025)

| Threshold | Games | Coverage | Accuracy |
| ---: | ---: | ---: | ---: |
| p >= 0.55 | 3367 | 47.8% | 58.87% |
| p >= 0.60 | 798 | 11.3% | 64.16% |
| p >= 0.70 | 0 | 0.0% | 0.00% |
| p >= 0.80 | 0 | 0.0% | 0.00% |

## 2026 YTD

| Games | Correct | Accuracy |
| ---: | ---: | ---: |
| 155 | 75 | 48.39% |

