# The Film Atlas - Milestone 5 Report

Milestone 5 tested richer TMDb-derived profile text, fresh embeddings, strict hierarchical clustering, optional public-dataset signals, deterministic tone/status probes, flat community comparators, and audit controls from the voice-note review pass. Private experiment artifacts remain under ignored outputs.

## Winning Approach

- Profile variant: hybrid_tone_status
- Clustering strategy: hierarchical_kmeans
- Movie count: 10000
- Selection score: 50.005
- Hierarchy mismatch rate: 0.000%
- Same-micro nearest-neighbor top-7 rate: 58.4%
- Coherence average: 0.647
- Estimated OpenAI cost: $4.6826
- Labels generated: 946
- Public audit point reassignments: 29
- Public audit label repairs: 168
- Public export: outputs/public_export

## Audit Result

- Audit movies present: 63 / 63
- Bad-neighbor pattern hits: 0
- Good-neighbor pattern hits: 6
- Duplicate parent-child label names after repair: 0
- Public audit point reassignments applied: 29
- Public audit label repairs applied: 168
- Deep-audit verdicts: mixed: 18, pass: 45

## Candidate Ranking

| Rank | Variant | Strategy | Score | Notes |
| ---: | --- | --- | ---: | --- |
| 1 | hybrid_tone_status | hierarchical_kmeans | 50.005 | Strict nested kmeans hierarchy. |

## Notes

- Richer TMDb profiles use overview, tagline, genres, keywords, and cleaned longer review language.
- Optional external signals were tested locally: MovieLens Tag Genome matched 6843 / 10000 films; MPST matched 5156 / 10000 films.
- Tone tags from synopsis/review language helped the selected exportable hierarchy; rating/popularity status tags were tested as a probe but should remain a filter/overlay, not the core map geometry.
- Title-bearing baseline profiles were retained as a comparator; the selected approach is allowed to beat or lose to title-free variants based on measured audit behavior.
- Public export is still sanitized and does not include raw reviews, external plot synopses, embeddings, API keys, or private experiment fields.
