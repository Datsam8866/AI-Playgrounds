# MLB Soft-regime XGBoost Walk-forward

Walk-forward: 2014–2026  |  Train start: 2023  |  Three models: early / fallback / primary+SP

## Per-Year Accuracy

| Year | Games | Accuracy |
| ---: | ---: | ---: |
| 2024 | 2425 | 0.542 |
| 2025 | 2425 | 0.536 |
| 2026 | 534 | 0.521 |
| **ALL** | **5384** | **0.537** |

## Model Parameters

- ELO_K=20, HOME_ADV=25, REGRESSION=0.35
- TEAM_BURN_IN=10, STARTER_BURN_IN=4
- EARLY_PROB_SHRINK=0.55
- XGBoost: max_depth=3, n_estimators=200, lr=0.04, reg_lambda=3.0, min_child_weight=15
- Post-processing: Platt scaling via train-only OOF probabilities
- P>=0.65 coverage: 9.1%

