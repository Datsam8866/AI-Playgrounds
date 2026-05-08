# High-Confidence Prediction Tracking

- Window: `2026-04-01` to `2026-05-06`
- Threshold: predicted-side confidence `>= 0.600`
- Scored predictions in window: `55`
- High-confidence games: `5`
- Coverage: `9.1%`
- Accuracy: `0 / 5 = 0.0%`

## By Model

| Model | Games | Hits | Accuracy |
| --- | ---: | ---: | ---: |
| advanced+SP | 5 | 0 | 0.0% |

## By Confidence Bucket (建議4監控)

| Bucket | Games | Hits | Accuracy | Baseline | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| 0.60-0.70 | 5 | 0 | 0.0% | 75% | **ALERT** |
| 0.70-0.80 | 0 | — | — | 82% | — |
| 0.80-0.90 | 0 | — | — | 95% | — |
| 0.90+ | 0 | — | — | 95% | — |

> ALERT 桶: 0.60-0.70: 0.0% (n=5, baseline 75%)


## Games

| Date | SNO | Matchup | Pred | Actual | Conf | Model | SP | Data-Limited | Hit |
| --- | ---: | --- | --- | --- | ---: | --- | --- | --- | --- |
| 2026-04-24 | 33 | 台鋼雄鷹 @ 統一獅 | Vis | Home | 0.654 | advanced+SP | Y | N | N |
| 2026-04-25 | 37 | 富邦悍將 @ 中信兄弟 | Vis | Home | 0.604 | advanced+SP | Y | N | N |
| 2026-04-25 | 38 | 味全龍 @ 樂天桃猿 | Home | Vis | 0.626 | advanced+SP | Y | N | N |
| 2026-04-29 | 41 | 統一獅 @ 味全龍 | Vis | Home | 0.625 | advanced+SP | Y | N | N |
| 2026-05-02 | 48 | 樂天桃猿 @ 統一獅 | Vis | Home | 0.623 | advanced+SP | Y | N | N |
