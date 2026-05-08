# MLB Soft-regime XGBoost Walk-forward

Walk-forward: 2014–2026  |  Train start: 2011  |  Three models: early / fallback / primary+SP

## Per-Year Accuracy

| Year | Games | Accuracy |
| ---: | ---: | ---: |
| 2014 | 2426 | 0.546 |
| 2015 | 2425 | 0.559 |
| 2016 | 2421 | 0.558 |
| 2017 | 2428 | 0.559 |
| 2018 | 2426 | 0.575 |
| 2019 | 2419 | 0.569 |
| 2020 | 896 | 0.556 |
| 2021 | 2422 | 0.585 |
| 2022 | 2421 | 0.583 |
| 2023 | 2427 | 0.554 |
| 2024 | 2425 | 0.571 |
| 2025 | 2425 | 0.544 |
| 2026 | 534 | 0.549 |
| **ALL** | **28095** | **0.563** |

## Model Parameters

- ELO_K=20, HOME_ADV=25, REGRESSION=0.35
- TEAM_BURN_IN=10, STARTER_BURN_IN=4
- EARLY_PROB_SHRINK=0.55
- XGBoost: max_depth=3, n_estimators=200, lr=0.04, reg_lambda=3.0, min_child_weight=15
- Post-processing: Platt scaling via train-only OOF probabilities
- P>=0.65 coverage: 9.1%

