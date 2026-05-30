# Milestone 2 — Semantic Embeddings, Projection, Clustering, and Nearest Neighbors

## Objective

Turn the Milestone 1/1.5 semantic movie profiles into real semantic embeddings, project films into 2D, cluster them into emergent vibe neighborhoods, compute nearest neighbors, and generate an inspection report.

Milestone 2 should prove that the data can support meaningful semantic neighborhoods.

This milestone should not:

- build the final website
- modify David’s Astro personal website repo
- generate final AI microgenre labels
- export final public website JSON
- scrape Letterboxd, IMDb, TMDb pages, or any website
- proceed to Milestone 3

## Current Project Context

The Film Atlas is a non-commercial portfolio project for discovering emergent “vibe genres” in English-language films.

Milestone 1 built:

- TMDb data ingestion
- movie detail fetching
- normalization
- semantic profile generation
- local TF-IDF/SVD sample maps
- data-quality reporting
- fixture-based tests
- linting

Milestone 1.5 added:

- future/unreleased film exclusion by default
- release-date bounds
- decade-balanced sampling
- TMDb ID dedupe
- conservative review profile controls
- review snippet cleaning/truncation/weighting
- production-context redaction from non-title profile text
- expanded report diagnostics

The larger balanced validation run completed successfully with:

- sampling strategy: `balanced_by_decade`
- discovered movies: 500
- detail records fetched: 500
- normalized movies: 500
- profiles: 500
- decade buckets: `1980s=100`, `1990s=100`, `2000s=100`, `2010s=100`, `2020s=100`
- overviews: 100.0%
- keywords: 99.8%
- reviews: 98.2%
- 2024 or later: 2.2%
- future release years: 0.0%

## Recommended Input Preparation

Before running Milestone 2, use the balanced sample pipeline:

```bash
uv run film-atlas fetch-balanced --per-decade 100 --start-year 1980 --end-year 2026
uv run film-atlas fetch-details
uv run film-atlas normalize
uv run film-atlas build-profiles --review-weight light --max-review-chars 180
uv run film-atlas make-sample-map
uv run film-atlas report
```

Milestone 2 should consume the existing normalized movie records and semantic profiles produced by this flow.

## Semantic Profile Rules

Semantic profile text should include:

- title
- overview
- official genres
- TMDb keywords
- conservative, cleaned, capped review-language snippets

Semantic profile text should continue excluding:

- year
- decade
- country
- language
- cast names
- director names
- producer names
- production companies
- production context

These excluded fields may be used for display/reporting later, but they should not define the semantic “vibe map.”

## Core Product Principle

Do not treat individual movie-level `top_terms` as final genre labels.

Movie-level `top_terms` from the Milestone 1 TF-IDF sample map are diagnostics only. They help debug whether the profile text contains meaningful signal, but they are too noisy and title-specific to become genres directly.

Final genre discovery should happen at the cluster level.

For each cluster, Milestone 2 should generate evidence that can later support human/AI microgenre naming:

- representative movies nearest the cluster centroid
- top official genres across the cluster
- top TMDb keywords across the cluster
- aggregated profile terms across the cluster
- sample nearest-neighbor pairs inside the cluster
- cluster size
- optional coherence score
- warnings about noisy or suspicious terms

## OpenAI Usage

Use `OPENAI_API_KEY` from `.env`.

Use `OPENAI_EMBEDDING_MODEL` from `.env`, defaulting to:

```text
text-embedding-3-large
```

if the existing config supports a default.

Security rules:

- Never print, log, cat, echo, inspect, or display `.env`.
- Never print or expose the OpenAI API key.
- Never commit API keys.
- Never commit raw embeddings unless they are tiny mocked test fixtures.
- Tests must mock OpenAI calls.
- Live embedding commands should use environment variables only.
- Do not paste secrets into code, tests, reports, README, or logs.

Cost-safety rules:

- Add an estimate command before live embedding calls.
- Show approximate token count and approximate expected cost.
- Add `--limit` to live embedding commands.
- Cache embeddings so reruns do not re-embed unchanged profiles.
- Start with small runs: 100 first, then 500 later.
- If estimated live API cost for a requested run exceeds $1, pause and ask David before running it.

## Required Functionality

Implement:

1. Profile validation before embedding
2. Embedding token/cost estimation
3. Batched OpenAI embedding generation
4. Local embedding cache keyed by movie/profile identity and profile hash
5. Cosine-similarity nearest-neighbor computation
6. 2D projection from embeddings
7. Movie clustering
8. Cluster-level evidence generation
9. Human-readable Milestone 2 inspection report

## Modeling Choices

### Embeddings

Use OpenAI embeddings.

Default model should come from env/config, ideally:

```text
text-embedding-3-large
```

Allow model override through env/CLI if consistent with existing project style.

### Projection

Prefer UMAP if dependency setup is clean.

If UMAP causes dependency friction, use a reliable fallback such as PCA, SVD, or another scikit-learn-based method.

Record projection method in the report.

### Clustering

Prefer HDBSCAN if dependency setup is clean.

If HDBSCAN causes dependency friction, use k-means as a fallback.

Record clustering method and parameters in the report.

### Nearest Neighbors

Compute cosine similarity in embedding space.

Precompute top N neighbors per movie.

Exclude self-matches.

Include sample nearest-neighbor examples in the report.

### Cluster Evidence

For each cluster, compute:

- `cluster_id`
- `cluster_size`
- representative movies
- top official genres
- top TMDb keywords
- aggregated profile terms
- representative nearest-neighbor pairs within the cluster
- optional coherence score if easy
- notes/warnings for noisy terms

## Recommended New Files

Add files as appropriate while following existing project style:

- `film_atlas/embedding.py`
- `film_atlas/embedding_cache.py`
- `film_atlas/neighbors.py`
- `film_atlas/reduce.py`
- `film_atlas/cluster.py`
- `film_atlas/inspect_clusters.py`
- `tests/test_embedding_cache.py`
- `tests/test_neighbors.py`
- `tests/test_embedding_estimate.py`
- `tests/test_cluster_evidence.py`

Do not over-engineer. Prefer simple, readable, maintainable code.

## CLI Commands

Add commands like these, adapting names if needed to match the existing CLI style:

```bash
film-atlas estimate-embeddings
film-atlas embed-profiles --limit 100
film-atlas reduce-embeddings
film-atlas cluster-movies
film-atlas compute-neighbors
film-atlas inspect-clusters
film-atlas milestone-2 --limit 100
```

Expected usage after implementation:

```bash
uv run film-atlas estimate-embeddings --limit 100
uv run film-atlas milestone-2 --limit 100
```

A later larger run can use:

```bash
uv run film-atlas milestone-2 --limit 500
```

Do not run large or expensive embedding jobs automatically.

## Outputs

Write generated/private/intermediate outputs under gitignored locations, such as:

- `outputs/intermediate/embeddings.jsonl`
- `outputs/intermediate/embedding_manifest.json`
- `outputs/intermediate/coordinates.json`
- `outputs/intermediate/cluster_assignments.json`
- `outputs/intermediate/neighbors.json`
- `outputs/intermediate/cluster_evidence.json`

Generate a human-readable report:

```bash
outputs/reports/milestone_2_report.md
```

Do not commit large generated data files.

## Milestone 2 Report Requirements

The report should include:

- number of profiles available
- number of movies embedded
- embedding model used
- estimated token count
- estimated cost
- number of cached embeddings reused
- number of new embeddings generated
- projection method used
- clustering method used
- number of clusters
- number/percent of outliers if applicable
- cluster size distribution
- 20 sample nearest-neighbor examples
- nearest neighbors for named quality-check movies, if present
- 10 sample clusters with representative movies
- top official genres per sample cluster
- top TMDb keywords per sample cluster
- aggregated cluster terms per sample cluster
- notes on whether neighborhoods feel semantically coherent
- warnings about data quality or suspicious clustering behavior
- recommendation on whether to proceed to Milestone 3

## Quality-Check Movies

If present in the dataset, report nearest neighbors for:

- No Country for Old Men
- The Social Network
- Mean Girls
- Her
- Get Out
- The Matrix
- Before Sunrise
- The Big Short
- Mad Max: Fury Road
- Lost in Translation
- The Devil Wears Prada
- Whiplash
- Nightcrawler
- Paddington 2
- The Godfather
- Pulp Fiction
- The Shawshank Redemption
- Interstellar
- The Dark Knight

## Tests

Add unit tests using fixtures and mocks, not live OpenAI calls.

Tests should cover:

- embedding cost estimation
- embedding cache key/profile hash behavior
- no re-embedding unchanged profiles
- mocked OpenAI response parsing
- nearest-neighbor self-match exclusion
- projection/reduction output shape
- clustering output shape
- cluster evidence generation from fixture data
- report generation with fixture data

Required validation:

```bash
uv run pytest
uv run ruff check .
```

## README Updates

Update `README.md` with:

- Milestone 2 overview
- setup notes for `OPENAI_API_KEY`
- security note: do not paste keys into chat or commit `.env`
- estimate command
- small live run command
- cache behavior
- output files
- what Milestone 2 does
- what Milestone 2 does not do
- next milestone outline

## Stopping Condition

Stop after Milestone 2 is implemented, tests pass, README is updated, and a real small run can generate:

```bash
outputs/reports/milestone_2_report.md
```

Do not proceed to:

- AI cluster labeling
- final public JSON export
- Astro frontend integration
- final website design
- LinkedIn writeup
