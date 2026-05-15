# Team Record Predictions (Regime Model)

_As of latest completed game date: `2026-05-09`_

## Soft Routing Rule

- Early model weight fades out as both teams reach `10` prior games
- SP model weight fades in as both starters reach `5` prior starts
- Early model probability is shrunk toward `0.500` with multiplier `0.50` before blending
- Fallback advanced model absorbs weight when SP data is incomplete

## Early-Season Features

- Elo state: `diff_elo`, `home_elo`, `vis_elo`, `elo_home_prob`
- Previous-season priors: `prev_diff_win_pct`, `prev_diff_rd_pg`, `prev_diff_pyth`
- Light context only: `home_rest`, `vis_rest`, `diff_rest`, `diff_streak`
- Burn-in counters: `home_season_games_before`, `vis_season_games_before`

## 2026 Opening Stretch

| Model | Games | Correct | Accuracy |
| --- | ---: | ---: | ---: |
| current advanced ensemble | 0 | 0 | nan% |
| regime model | 0 | 0 | nan% |

- Games with non-zero early weight in this window: `0 / 0`

## 2026 Through Latest Completed Date

Window: `2026-04-01` to `2026-05-09`

| Model | Games | Correct | Accuracy |
| --- | ---: | ---: | ---: |
| current advanced ensemble | 61 | 25 | 40.98% |
| regime model | 61 | 27 | 44.26% |

- Games with non-zero early weight in this window: `30 / 61`

## Recent Window

Window: `2026-04-08` to `2026-05-09`

| Model | Games | Correct | Accuracy |
| --- | ---: | ---: | ---: |
| current advanced ensemble | 61 | 25 | 40.98% |
| regime model | 61 | 27 | 44.26% |

- Games with non-zero early weight in this window: `30 / 61`

## 2016–2025 Benchmark

| Metric | Value |
| --- | ---: |
| Games | 2771 |
| Accuracy | 55.50% |
| Games with non-zero early weight | 243 |

