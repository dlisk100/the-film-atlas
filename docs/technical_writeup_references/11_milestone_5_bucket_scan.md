# The Film Atlas - Milestone 5 Bucket Scan

Local scan over cached hybrid_tone_status embeddings. This does not call OpenAI and does not generate new labels.

## Summary

- Combos tested: 8
- Recommended by current quantitative score: 10 / 60 / 160
- Interpretation: use this as a pressure test, not an automatic replacement for human label QA.

## Metrics

| Macro | Neighborhood | Micro | Score | Same macro top7 | Same neighborhood top7 | Same micro top7 | Coherence avg | Tiny micro | Micro min/median/max |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 10 | 60 | 160 | 57.166 | 84.7% | 75.1% | 64.4% | 0.640 | 6 | 3 / 12.0 / 42 |
| 12 | 75 | 200 | 56.505 | 85.5% | 74.6% | 62.6% | 0.651 | 20 | 2 / 9.0 / 28 |
| 14 | 84 | 224 | 56.279 | 84.5% | 73.6% | 61.5% | 0.655 | 35 | 1 / 8.0 / 25 |
| 10 | 70 | 200 | 55.751 | 84.1% | 73.2% | 61.4% | 0.649 | 22 | 2 / 9.0 / 27 |
| 16 | 96 | 256 | 53.650 | 83.9% | 71.2% | 57.5% | 0.664 | 57 | 1 / 7.0 / 24 |
| 12 | 84 | 240 | 53.600 | 84.8% | 71.7% | 57.8% | 0.659 | 40 | 2 / 8.0 / 24 |
| 12 | 90 | 300 | 52.970 | 84.2% | 69.9% | 53.7% | 0.670 | 93 | 1 / 6.0 / 23 |
| 14 | 98 | 280 | 50.990 | 83.7% | 70.7% | 56.6% | 0.665 | 82 | 1 / 7.0 / 19 |

## Recommendation Notes

The current 12 / 75 / 200 setting remains competitive and avoids the extra tiny-cluster pressure of 240-300 microclusters. Human QA rejected switching to 10 / 60 / 160 despite its slightly higher quantitative score, because known-case labels regressed during spot checks. The final export keeps 12 / 75 / 200.
