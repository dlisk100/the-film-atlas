# Milestone 5: Exploratory Classification Upgrade

## Purpose

Milestone 5 improves The Film Atlas classification system through open
experimentation rather than locking into the current pipeline. The goal is to
independently test richer movie profiles, alternative clustering methods, real
hierarchical structures, label quality, map usefulness, and audited movie
outcomes, then ship the best-performing local atlas for review.

The current prototype is promising, but the audit found several systemic
issues:

- nearest neighbors are often useful, but labels and hierarchy are weaker
- macro, neighborhood, and micro clusters are not consistently nested
- title leakage affects movies such as `Heat`, `Civil War`, `The Game`, and
  `Avatar`
- some labels are vivid but falsely specific, such as `Moon Mission` for
  non-moon films or `Spacefaring` for underwater films
- the map visually implies a country/state/city atlas, but the current data
  model behaves more like independent semantic lenses

Milestone 5 should find the best practical implementation, not merely patch the
current one.

## Hard Constraints

- Keep live API spend under `$10`.
- Do not print, expose, commit, or log secrets.
- Do not display `.env`.
- Do not commit `.env`.
- Do not modify David's personal website repo.
- Keep all implementation work inside
  `/Users/davidlisk/Development/the-film-atlas`.
- Do not scrape websites.
- Use web search only to confirm movie facts during audit, not as imported
  training data.
- Public export must not include raw reviews, embeddings, API keys, or private
  experiment fields.
- Raw TMDb data, processed private data, embeddings, reviews, and experiment
  intermediates must stay out of git.

## Implementation Strategy

### 1. Preserve the Baseline

Keep the current `outputs/public_export/` as the baseline comparison point.

Write all exploratory outputs under:

```text
outputs/experiments/classification_v2/
```

Do not overwrite `outputs/public_export/` until a winning approach has been
selected and validated.

### 2. Build Profile Variants

Create multiple embedding-profile variants and compare them empirically.
Suggested variants:

- `baseline_light`: current profile style.
- `no_title_light`: current fields but remove title from embedding text.
- `no_title_rich_reviews`: remove title and include more cleaned TMDb review
  language.
- `keyword_overview_weighted`: emphasize overview, genres, and TMDb keywords;
  keep review language secondary.
- `distilled_vibes`: generate a cached OpenAI "vibe card" from TMDb-derived
  overview, genres, keywords, and reviews only, then embed the vibe card.

The implementer may add or remove variants if early evidence shows a better
experimental path, but must explain the choice in the final report.

Profile experiments should make it easy to inspect the exact embedded text for
any movie.

### 3. Compare Clustering and Layout Strategies

Do not assume the current independent k-means approach is correct. Compare at
least:

- current independent k-means baseline
- strict hierarchical k-means
- agglomerative hierarchy
- nearest-neighbor graph/community clustering using existing `networkx`
- HDBSCAN as an experimental comparator only

For any hierarchy-forward method, require that every selected movie has a
coherent parent chain:

```text
macro -> neighborhood -> micro
```

If the winning method intentionally rejects hierarchy, the report must justify
that decision and the frontend language must stop implying strict hierarchy.

### 4. Improve Labels Without Making Them Boring

Labels should stay fun, vivid, and vibe-rich. The goal is not generic taxonomy
labels such as `Action Movie`.

Improve labeling by adding faithfulness guardrails:

- do not use specifics such as `moon`, `arctic`, `spacefaring`, `battlefield`,
  `alien`, `heist`, `rom-com`, or `witchcraft` unless cluster evidence strongly
  supports them
- include parent context when labeling neighborhoods and micros
- prefer poetic-but-true over specific-but-false
- include private label risk notes and possible misfits
- keep human-reviewable label outputs

The labeler should be allowed to use broader evocative labels when the cluster
is genuinely broad.

### 5. Evaluate Against the Audit Set

Create an audit set from the user's reviewed movies and known controls. Include
at least:

```text
Avatar
Avatar: Fire and Ash
The Founder
Vanilla Sky
Final Destination
Weapons
Jurassic World Rebirth
Sully
Mickey 17
Rush Hour
The Perks of Being a Wallflower
Murder on the Orient Express
The Theory of Everything
Carry-On
Total Recall
Moon
Civil War
Independence Day
Minority Report
I, Tonya
Scott Pilgrim vs. the World
Spider-Man: Across the Spider-Verse
The Creator
Hot Tub Time Machine
Dungeons & Dragons: Honor Among Thieves
Barbie
Her
Juno
Little Miss Sunshine
Heat
RoboCop
Oppenheimer
Sound of Metal
Sunshine
Hail, Caesar!
The Game
Lost in Translation
The Mist
Knock at the Cabin
The Fountain
The Village
La La Land
Elvis
The Grand Budapest Hotel
The Man from U.N.C.L.E.
Glass Onion: A Knives Out Mystery
Trainspotting
O Brother, Where Art Thou?
Point Break
The Big Lebowski
Captain Phillips
Uncut Gems
Licorice Pizza
The Abyss
Project X
Edge of Tomorrow
Apocalypto
The Northman
Cast Away
The King's Speech
The Hunger Games
Office Space
```

Use exact TMDb IDs where needed to disambiguate remakes, sequels, and duplicate
titles.

The report should classify issues into:

- neighbor issue
- label issue
- hierarchy issue
- map/layout issue
- title leakage
- franchise gravity
- acceptable weirdness

### 6. Select the Winning Approach

Pick the best approach based on evidence, not path dependency. The final
selection should consider:

- hierarchy validity
- audited neighbor quality
- reduction of title leakage
- reduction of false-specific labels
- preservation of known-good cases
- cluster size balance
- coherence
- label usefulness
- map readability
- implementation simplicity

If two approaches are close, prefer the one that is more robust and easier to
explain in the public project.

### 7. Regenerate Public Export and Local Frontend

After selecting the winning approach:

- regenerate `outputs/public_export/`
- verify public export privacy
- update the local `frontend/` only as needed
- add years to neighbor titles if not already present
- support keyboard down-arrow and Enter selection in search
- keep cluster member dropdowns
- clearly distinguish semantic nearest neighbors from same-cluster members
- make selected movie labels clearer than floating centroid labels

Do not copy anything into the personal website repo during this milestone.

## Deliverables

- this milestone document
- `outputs/experiments/classification_v2/summary.md`
- variant-specific experiment reports
- audit report for the reviewed movie set
- cost report
- selected improved `outputs/public_export/`
- updated local frontend, if needed
- final implementation summary with:
  - winning approach
  - cost
  - what improved
  - known remaining issues
  - review URL

## Acceptance Criteria

- Total live API spend stays under `$10`.
- `uv run pytest` passes.
- `uv run ruff check .` passes.
- frontend build passes if frontend files are touched.
- public export contains no raw reviews, embeddings, API keys, or private
  experiment data.
- selected final hierarchy has `0%` hierarchy mismatch, unless the winning
  method intentionally rejects hierarchy and the report justifies that.
- known bad cases improve:
  - `Civil War`
  - `Heat`
  - `The Game`
  - `Avatar`
  - `Sunshine`
  - `The Abyss`
  - `The Mist`
  - `Oppenheimer`
  - `Sound of Metal`
  - `Barbie`
  - `Office Space`
- known good cases remain good:
  - `Rush Hour`
  - `Weapons`
  - `The Creator`
  - `Dungeons & Dragons: Honor Among Thieves`
  - `RoboCop`
  - `Lost in Translation`
  - `La La Land`
  - `Spider-Man: Across the Spider-Verse`
- final report gives a go/no-go recommendation for user review.

## Recommended Execution Notes

- Start by measuring the current baseline with the audit set.
- Prefer small, inspectable experiments before committing to a full rerun.
- Use caches aggressively.
- If `OPENAI_API_KEY` or `TMDB_BEARER_TOKEN` must be checked, only report
  whether each is set or missing.
- Do not print secret values.
- If a command fails due to `uv` path issues, try:

```bash
PATH="/Users/davidlisk/.local/bin:$PATH" uv run ...
```

- Keep intermediate reports readable enough that David can review them without
  opening raw JSON.
