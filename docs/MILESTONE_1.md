# Milestone 1 — Data Pipeline Proof

Objective:
Create a working Python project that can fetch a controlled sample of English-language films from TMDb, normalize the data, build semantic text profiles, generate a cheap local sample map, and produce a data-quality report.

Do not use OpenAI API calls in this milestone.
Do not scrape Letterboxd, IMDb, or any website.
Use only TMDb’s official API.

Technical preferences:
- Python 3.12
- uv for dependency management
- Typer or Click for CLI
- python-dotenv for env loading
- httpx or requests for HTTP
- pydantic or dataclasses for typed records
- pandas / pyarrow for local processed outputs
- scikit-learn for TF-IDF and dimensionality reduction
- UMAP is allowed only if setup is clean; otherwise use TruncatedSVD/PCA
- pytest for tests
- ruff for linting

Required structure:
- README.md
- AGENTS.md
- pyproject.toml
- .env.example
- .gitignore
- film_atlas/
    - __init__.py
    - config.py
    - tmdb_client.py
    - models.py
    - fetch.py
    - normalize.py
    - profiles.py
    - sample_map.py
    - report.py
    - cli.py
- tests/
    - fixtures/
    - test_profiles.py
    - test_normalize.py
    - test_tmdb_client.py
- data/
    - .gitkeep
- outputs/
    - .gitkeep
    - reports/
    - figures/

Environment variables:
- TMDB_BEARER_TOKEN is required for real TMDb fetches.
- OPENAI_API_KEY may exist later but must not be used in Milestone 1.

TMDb sample fetch requirements:
- Use /discover/movie.
- Filter for original English-language films.
- Exclude adult content.
- Exclude videos.
- Use a minimum vote count threshold, default 500.
- Use minimum runtime >= 60 minutes if feasible through the endpoint.
- Sort by popularity or vote count.
- Provide CLI args for limit, min-votes, and output dir.
- Respect rate limits politely with retries/backoff.
- Cache raw TMDb API responses under data/cache/, which must remain gitignored.

Movie detail requirements:
Fetch details for discovered movie IDs and include:
- tmdb_id
- imdb_id if available
- title
- original_title
- release_date
- year
- runtime
- overview
- genres
- keywords if available
- poster_path
- backdrop_path
- vote_average
- vote_count
- popularity
- reviews if available

Raw review content may be cached locally, but do not include raw review text in any public/export-ready site data.

Profile-building requirements:
Create one semantic text profile per movie.

Include:
- title
- overview
- genres
- keywords
- short/truncated review-language snippets if available

Exclude:
- year
- decade
- country
- language
- cast names
- director names
- production companies

Add a test that ensures forbidden production-context fields are not accidentally included in the semantic profile text.

Sample map requirements:
- No OpenAI calls.
- Use TF-IDF plus dimensionality reduction to produce a rough local 2D preview.
- Output CSV or JSON with:
    - tmdb_id
    - title
    - year
    - x
    - y
    - top terms or simple local cluster if implemented
- Produce a simple HTML visualization or static PNG under outputs/reports/ or outputs/figures/ if practical.
- This map is only a sanity check, not the final visual design.

Data quality report:
Generate outputs/reports/milestone_1_report.md containing:
- number of discovered movies
- number of detail records fetched
- percentage with overview
- percentage with keywords
- percentage with reviews
- year distribution
- top official genres
- top keywords
- movies missing important fields
- 20 sample movie text profiles
- 10 example nearest-neighbor pairs from the local sample method, if implemented
- notes explaining that final semantic embeddings will happen in a later milestone

CLI commands:
Provide commands like:
- film-atlas fetch-discover --limit 500 --min-votes 500
- film-atlas fetch-details
- film-atlas build-profiles
- film-atlas make-sample-map
- film-atlas report
- film-atlas quickstart --limit 100

README:
Include:
- project overview
- setup instructions
- required API key instructions
- environment variable setup
- milestone 1 commands
- what data is gitignored
- what Milestone 1 does and does not do
- next milestone outline

Testing:
- Add pytest tests using fixtures, not live API calls.
- Add command to run tests.
- Add command to run linting.
- Ensure these pass:
    - uv run pytest
    - uv run ruff check .

Stopping condition:
Stop after Milestone 1 is implemented, tests pass, and README explains how to run the pipeline locally. Do not proceed to OpenAI embeddings, final clustering, cluster labeling, or Astro frontend integration.
