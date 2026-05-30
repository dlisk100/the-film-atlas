# The Film Atlas

The Film Atlas is a non-commercial portfolio project for discovering emergent
"vibe genres" in English-language films. Milestone 1 builds the offline data
pipeline scaffold and a small TMDb-based data-quality proof. It does not build
the final website, call OpenAI, scrape websites, or create final cluster labels.

## Setup

Install dependencies with uv:

```bash
uv sync
```

If `uv` is installed but not on your shell path, add the directory that contains
it, for example:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

Copy the environment template:

```bash
cp .env.example .env
```

Add a TMDb API bearer token to `.env` before running live fetch commands:

```bash
TMDB_BEARER_TOKEN=your_tmdb_read_access_token
```

`OPENAI_API_KEY` is used only for Milestone 2 embedding commands. Do not paste
API keys into chat, commit `.env`, or place secrets in reports, tests, or docs.
`OPENAI_LABEL_MODEL` can optionally override the Milestone 3 draft-label model.

Optional local path controls:

```bash
FILM_ATLAS_DATA_DIR=data
FILM_ATLAS_OUTPUT_DIR=outputs
```

## Milestone 1 Commands

Run the full local pipeline after adding `TMDB_BEARER_TOKEN`:

```bash
film-atlas quickstart --limit 100
```

Or run each step:

```bash
film-atlas fetch-discover --limit 500 --min-votes 500
film-atlas fetch-details
film-atlas normalize
film-atlas build-profiles --review-weight light --max-review-chars 180
film-atlas make-sample-map
film-atlas report
```

The exact command to run after adding credentials is:

```bash
uv run film-atlas quickstart --limit 100
```

## What Milestone 1 Does

- Fetches a controlled sample from TMDb `/discover/movie`.
- Filters for original English-language films.
- Excludes adult content and videos.
- Excludes future/unreleased films by default.
- Applies a default minimum vote count of 500.
- Requests a minimum runtime of 60 minutes through the discover endpoint.
- Supports primary release date bounds with `--release-date-gte` and
  `--release-date-lte`.
- Supports decade-balanced sampling with `fetch-balanced`.
- Fetches details, keywords, reviews, credits, and external IDs for discovered
  movie IDs.
- Normalizes records into JSON and Parquet.
- Builds semantic text profiles from title, overview, genres, keywords, and
  conservative, cleaned, capped review-language snippets.
- Creates a rough TF-IDF plus TruncatedSVD 2D sample map.
- Writes a data-quality report to `outputs/reports/milestone_1_report.md`.

## What Milestone 1 Does Not Do

- No OpenAI API calls.
- No scraping Letterboxd, IMDb, TMDb pages, or any other website.
- No final semantic embeddings.
- No final clustering or cluster labeling.
- No Astro frontend integration.
- No changes to David's personal website repo.

## Data And Outputs

Gitignored local data:

- `data/cache/`: cached raw TMDb API responses.
- `data/raw/`: raw discovery and detail files.
- `data/processed/`: normalized JSON, Parquet, and profile files.
- `outputs/private/` and `outputs/intermediate/`: scratch outputs.

Tracked output folders:

- `outputs/reports/`: report, sample map CSV/JSON, nearest-neighbor JSON.
- `outputs/figures/`: simple sample map HTML.

Raw review content may appear in local TMDb cache files. Export-ready semantic
profiles only use truncated review-language snippets.

## Milestone 1.5 Sampling Controls

The default `fetch-discover` and `quickstart` commands now cap release dates at
today so future/unreleased films do not enter the sample unless
`--include-future` is passed.

Date-bound example:

```bash
uv run film-atlas fetch-discover --limit 500 --min-votes 500 --release-date-gte 1980-01-01 --release-date-lte 2026-12-31
```

Recommended balanced sampling command before Milestone 2:

```bash
uv run film-atlas fetch-balanced --per-decade 100 --start-year 1980 --end-year 2026
uv run film-atlas fetch-details
uv run film-atlas normalize
uv run film-atlas build-profiles --review-weight light --max-review-chars 180
uv run film-atlas make-sample-map
uv run film-atlas report
```

`fetch-balanced` writes the same active discover file as `fetch-discover`
(`data/raw/discover_movies.json`) and dedupes movies by TMDb ID.

## Profile Review Controls

Review language is useful for vibe discovery, but it can overwhelm plot and
keyword signals. Milestone 1.5 keeps reviews conservative by default:

```bash
uv run film-atlas build-profiles --include-reviews --max-review-chars 180 --review-weight light
```

Available controls:

- `--include-reviews / --no-include-reviews`
- `--max-review-chars`
- `--review-weight light|medium|heavy`

Profile building strips obvious review noise such as URLs, hashtags, repeated
punctuation, and excessive repeated tokens. It also redacts known
production-context values from non-title profile text.

## Milestone 2 Semantic Neighborhoods

Milestone 2 turns the existing semantic profiles into OpenAI embeddings, projects
movies into 2D, clusters them into emergent vibe neighborhoods, computes nearest
neighbors, and writes an inspection report. It still does not build the final
website, generate final AI microgenre labels, export public website JSON, scrape
websites, or modify David's Astro personal website repo.

Before running live embeddings, estimate cost:

```bash
uv run film-atlas estimate-embeddings --limit 100
```

Small live run:

```bash
uv run film-atlas milestone-2 --limit 100
```

Milestone 2 writes generated private/intermediate artifacts under
`outputs/intermediate/`, which is gitignored:

- `embeddings.jsonl`
- `embedding_manifest.json`
- `coordinates.json`
- `cluster_assignments.json`
- `neighbors.json`
- `cluster_evidence.json`

The human-readable inspection report is:

```text
outputs/reports/milestone_2_report.md
```

Embedding cache behavior:

- Embeddings are cached by TMDb ID, embedding model, and profile hash.
- Re-running unchanged profiles reuses cached vectors.
- Changed profile text gets a new hash and is embedded again.
- Live embedding commands support `--limit`; start with `--limit 100`.
- If the estimated live API cost exceeds `$1`, the CLI stops before calling
  OpenAI.

Useful individual commands:

```bash
uv run film-atlas embed-profiles --limit 100
uv run film-atlas reduce-embeddings
uv run film-atlas cluster-movies
uv run film-atlas compute-neighbors
uv run film-atlas inspect-clusters
```

## Milestone 2.5 Cluster Granularity Sweep

Milestone 2.5 compares several local k-means cluster counts over the existing
embedding file. It does not call OpenAI, re-embed profiles, generate AI labels,
export public JSON, scrape websites, or modify the Astro personal website repo.

```bash
uv run film-atlas sweep-clusters --ks 15,25,35,50
```

Outputs:

- `outputs/intermediate/cluster_sweep.json`
- `outputs/reports/cluster_sweep_report.md`

## Milestone 2.6 Clustering Method Comparison

Milestone 2.6 compares local clustering methods over the existing full embedding
vectors. It does not call OpenAI, re-embed profiles, generate AI labels, export
public website JSON, scrape websites, or modify the Astro personal website repo.
PCA/2D coordinates remain visualization-only.

```bash
uv run film-atlas compare-clustering-methods
```

Useful controls:

- `--methods kmeans,agglomerative,graph,hdbscan`
- `--kmeans-k 35`
- `--agglomerative-k 35`
- `--graph-neighbors 10`
- `--min-cluster-size 5`
- `--limit 100`

Outputs:

- `outputs/intermediate/clustering_method_comparison.json`
- `outputs/reports/clustering_method_comparison.md`

## Milestone 3 Draft Microgenre Labels

Milestone 3 generates draft, human-reviewable labels for the existing k-means
`k=35` clusters. It reuses existing embeddings and cluster evidence, does not
re-embed profiles, does not scrape anything, does not export final public
website JSON, and does not modify the Astro personal website repo.

Estimate cost before live label calls:

```bash
uv run film-atlas estimate-labeling
```

Generate draft labels if the estimate is below `$1` and `OPENAI_API_KEY` is set:

```bash
uv run film-atlas label-clusters --method kmeans --k 35
uv run film-atlas render-label-review
```

Outputs:

- `outputs/reports/cluster_label_candidates.md`
- `outputs/reports/cluster_label_review.md`
- `outputs/intermediate/cluster_label_candidates.json`
- `outputs/intermediate/human_editable_cluster_labels.json`
- `outputs/intermediate/cluster_label_cache.json`

## Milestone 3.25 Review-Weight Ablation

Milestone 3.25 compares `no_reviews`, `light_reviews`, and `medium_reviews`
over the same movie set with the same embedding model, k-means `k=35`, and the
same draft-labeling style. It does not fetch new TMDb data, scrape anything,
export final public website JSON, or modify the Astro personal website repo.

```bash
uv run film-atlas review-ablation --variants no_reviews,light_reviews,medium_reviews --k 35 --limit 500
```

Variant-private outputs are written under:

```text
outputs/intermediate/review_ablation/
```

The comparison outputs are:

- `outputs/reports/review_ablation_report.md`
- `outputs/intermediate/review_ablation/review_ablation_summary.json`

## Milestone 4 Scaled Atlas Export

Milestone 4 scales the local dataset toward a frontend-ready atlas. It fetches
English-language TMDb feature films since 1980, builds `light_reviews` semantic
profiles, embeds them with OpenAI, creates hierarchical k-means atlas layers,
drafts cluster labels, computes neighbors and 2D coordinates, then writes
sanitized static JSON. It does not build the frontend, modify the Astro website
repo, scrape anything, expose API keys, or include raw reviews/embeddings in
the public export.

Run the full milestone:

```bash
uv run film-atlas milestone-4 --target 2000
```

Or run the steps separately:

```bash
uv run film-atlas scale-dataset --target 2000 --since-year 1980 --min-votes 100
uv run film-atlas embed-profiles
uv run film-atlas build-hierarchy --macro-k 12 --neighborhood-k 75 --micro-k 200
uv run film-atlas export-atlas-data
```

Milestone 4 uses a `$5` OpenAI cost gate. It estimates embedding and labeling
costs before live OpenAI calls and stops if the estimate is above that limit.

Private/generated data remains under ignored paths such as `data/raw/`,
`data/cache/`, `data/processed/`, and `outputs/intermediate/`.

Sanitized frontend-ready files are written to:

```text
outputs/public_export/
```

Expected export files:

- `manifest.json`
- `movies.json`
- `points.json`
- `macro_clusters.json`
- `neighborhood_clusters.json`
- `micro_clusters.json`
- `neighbors.json`
- `labels.json`

The Milestone 4 report is written to:

- `outputs/reports/milestone_4_report.md`

## Testing And Linting

Tests use fixtures and mocks, not live API calls:

```bash
uv run pytest
uv run ruff check .
```

## Next Milestone Outline

Later milestones can add human-readable cluster labels, final public JSON
exports, and integration into the Astro personal site. Those steps are
intentionally outside Milestone 2.
