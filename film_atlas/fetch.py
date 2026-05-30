"""Fetch raw TMDb discovery and detail data."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from film_atlas.tmdb_client import TMDbClient

RAW_DIR_NAME = "raw"
DISCOVER_FILENAME = "discover_movies.json"
DETAILS_FILENAME = "movie_details.json"


def write_json(path: Path, payload: Any) -> Path:
    """Write JSON with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def read_json(path: Path) -> Any:
    """Read JSON from disk."""
    return json.loads(path.read_text(encoding="utf-8"))


def fetch_discover(
    client: TMDbClient,
    *,
    limit: int = 500,
    min_votes: int = 500,
    output_dir: str | Path = "data",
    sort_by: str = "popularity.desc",
    min_runtime: int = 60,
    release_date_gte: str | None = None,
    release_date_lte: str | None = None,
    exclude_future: bool = True,
    refresh: bool = False,
) -> Path:
    """Fetch and save a TMDb discover/movie sample."""
    movies = client.discover_movies(
        limit=limit,
        min_votes=min_votes,
        sort_by=sort_by,
        min_runtime=min_runtime,
        release_date_gte=release_date_gte,
        release_date_lte=release_date_lte,
        exclude_future=exclude_future,
        refresh=refresh,
    )
    payload = {
        "source": "tmdb:/discover/movie",
        "sampling_strategy": "discover",
        "limit": limit,
        "min_votes": min_votes,
        "sort_by": sort_by,
        "min_runtime": min_runtime,
        "release_date_gte": release_date_gte,
        "release_date_lte": release_date_lte,
        "effective_release_date_lte": _effective_release_date_lte(release_date_lte, exclude_future),
        "exclude_future": exclude_future,
        "movie_count": len(movies),
        "results": movies,
    }
    return write_json(Path(output_dir) / RAW_DIR_NAME / DISCOVER_FILENAME, payload)


def fetch_balanced(
    client: TMDbClient,
    *,
    per_decade: int = 100,
    start_year: int = 1980,
    end_year: int = 2026,
    min_votes: int = 500,
    output_dir: str | Path = "data",
    sort_by: str = "vote_count.desc",
    min_runtime: int = 60,
    exclude_future: bool = True,
    refresh: bool = False,
) -> Path:
    """Fetch a decade-balanced sample and save it as the active discover file."""
    buckets: list[dict[str, Any]] = []
    movies: list[dict[str, Any]] = []

    for decade_start in range(start_year, end_year + 1, 10):
        decade_end = min(decade_start + 9, end_year)
        release_date_gte = f"{decade_start}-01-01"
        release_date_lte = f"{decade_end}-12-31"
        bucket_movies = client.discover_movies(
            limit=per_decade,
            min_votes=min_votes,
            sort_by=sort_by,
            min_runtime=min_runtime,
            release_date_gte=release_date_gte,
            release_date_lte=release_date_lte,
            exclude_future=exclude_future,
            refresh=refresh,
        )
        movies = dedupe_movies([*movies, *bucket_movies])
        buckets.append(
            {
                "decade_start": decade_start,
                "decade_end": decade_end,
                "release_date_gte": release_date_gte,
                "release_date_lte": release_date_lte,
                "effective_release_date_lte": _effective_release_date_lte(
                    release_date_lte, exclude_future
                ),
                "movie_count": len(bucket_movies),
            }
        )

    payload = {
        "source": "tmdb:/discover/movie",
        "sampling_strategy": "balanced_by_decade",
        "per_decade": per_decade,
        "start_year": start_year,
        "end_year": end_year,
        "min_votes": min_votes,
        "sort_by": sort_by,
        "min_runtime": min_runtime,
        "exclude_future": exclude_future,
        "movie_count": len(movies),
        "buckets": buckets,
        "results": movies,
    }
    return write_json(Path(output_dir) / RAW_DIR_NAME / DISCOVER_FILENAME, payload)


def fetch_details(
    client: TMDbClient,
    *,
    discover_path: str | Path = "data/raw/discover_movies.json",
    output_dir: str | Path = "data",
    limit: int | None = None,
    refresh: bool = False,
) -> Path:
    """Fetch and save details for discovered movie IDs."""
    discover_payload = read_json(Path(discover_path))
    discovered_movies = discover_payload.get("results") or []
    discovered_movies = dedupe_movies(discovered_movies)
    if limit is not None:
        discovered_movies = discovered_movies[:limit]

    details = []
    for movie in discovered_movies:
        tmdb_id = int(movie["id"])
        details.append(client.movie_details(tmdb_id, refresh=refresh))

    payload = {
        "source": "tmdb:/movie/{movie_id}",
        "discover_path": str(discover_path),
        "detail_count": len(details),
        "results": details,
    }
    return write_json(Path(output_dir) / RAW_DIR_NAME / DETAILS_FILENAME, payload)


def dedupe_movies(movies: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Dedupe TMDb movie dictionaries by id while preserving order."""
    deduped: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    for movie in movies:
        tmdb_id = int(movie["id"])
        if tmdb_id in seen_ids:
            continue
        seen_ids.add(tmdb_id)
        deduped.append(movie)
    return deduped


def _effective_release_date_lte(release_date_lte: str | None, exclude_future: bool) -> str | None:
    if not exclude_future:
        return release_date_lte
    today = date.today().isoformat()
    if not release_date_lte:
        return today
    return min(release_date_lte, today)
