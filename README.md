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

`OPENAI_API_KEY` may exist in the environment for future milestones, but
Milestone 1 never reads or uses it.

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
film-atlas build-profiles
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
- Applies a default minimum vote count of 500.
- Requests a minimum runtime of 60 minutes through the discover endpoint.
- Fetches details, keywords, reviews, credits, and external IDs for discovered
  movie IDs.
- Normalizes records into JSON and Parquet.
- Builds semantic text profiles from title, overview, genres, keywords, and
  short review-language snippets.
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

## Testing And Linting

Tests use fixtures and mocks, not live API calls:

```bash
uv run pytest
uv run ruff check .
```

## Next Milestone Outline

Later milestones can add higher-quality semantic embeddings, final clustering,
human-readable cluster labels, static public JSON exports, and integration into
the Astro personal site. Those steps are intentionally outside Milestone 1.
