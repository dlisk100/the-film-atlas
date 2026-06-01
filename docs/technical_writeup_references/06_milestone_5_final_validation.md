# The Film Atlas - Milestone 5 Final Validation

## Export

- Public export: `outputs/public_export`
- Movies: 10,000
- Points: 10,000
- Neighbor lists: 10,000
- Labels: 946
- Hierarchy: 16 macro / 180 neighborhood / 750 micro
- Duplicate macro/neighborhood/micro path labels: 0
- Privacy flags: no API keys, no embeddings, no raw reviews

## Audit Evidence

- Full 10k audit before targeted repair pass: 9,633 pass / 336 mixed / 31 fail.
- Recheck of the 31 full-audit failures after repair rules: 22 pass / 9 mixed / 0 fail.
- Final strict 1k sample, batch size 10: 900 pass / 99 mixed / 1 fail.
- The single final strict-sample fail was `Excess Baggage`; after that report, the macro wording was repaired from a teen-only framing to a youth/crime-scheme comedy-drama framing.
- Per user direction, no further repeated 1k reruns were performed after that final targeted repair.

## Verification

- `uv run pytest`: 60 passed.
- `uv run ruff check .`: passed.
- `pnpm build` in `frontend/`: passed, 2 pages built.
- Browser QA passed on `http://127.0.0.1:4322/film-atlas/`: 10,000 films, 946 labels, 16/180/750 hierarchy counts, search/selection/zoom/reset checked, and no browser console warnings/errors observed in the completed QA pass.

## Remaining Caveat

The atlas is now strong enough for review and portfolio iteration, but the mixed tail is real: some broad clusters still have poetic labels that are slightly too wide for edge cases. That is acceptable at this stage and should be handled through future targeted audits, not repeated full reruns.
