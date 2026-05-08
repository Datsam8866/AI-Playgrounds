# KBO Regime Model — Walk-Forward Benchmark

Train start: 2013 | Backtest: 2016–2025
Elo: K=48, home_adv=10, regression=0.50
XGBoost: max_depth=3, min_child_weight=30, n_estimators=30

## Walk-Forward Results (2016–2025)

| Metric | Value |
| --- | ---: |
| Total games | 7050 |
| Correct | 3830 |
| **Accuracy** | **54.33%** |
| Home baseline | 52.71% |

## Per-Year Breakdown

| Year | Games | Accuracy | early | primary | fallback |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2016 | 713 | 53.72% | 53 | 0 | 660 |
| 2017 | 709 | 54.44% | 50 | 0 | 659 |
| 2018 | 714 | 54.34% | 50 | 0 | 664 |
| 2019 | 713 | 56.94% | 50 | 0 | 663 |
| 2020 | 707 | 56.58% | 52 | 0 | 655 |
| 2021 | 670 | 53.73% | 51 | 0 | 619 |
| 2022 | 708 | 54.10% | 50 | 0 | 658 |
| 2023 | 708 | 53.53% | 54 | 0 | 654 |
| 2024 | 710 | 51.41% | 52 | 0 | 658 |
| 2025 | 698 | 54.44% | 53 | 0 | 645 |

## High-Confidence Subset (2016–2025)

| Threshold | Games | Coverage | Accuracy |
| ---: | ---: | ---: | ---: |
| p >= 0.55 | 2898 | 41.1% | 56.80% |
| p >= 0.60 | 288 | 4.1% | 53.82% |
| p >= 0.70 | 0 | 0.0% | 0.00% |
| p >= 0.80 | 0 | 0.0% | 0.00% |

## 2026 YTD

| Games | Correct | Accuracy |
| ---: | ---: | ---: |
| 155 | 77 | 49.68% |

