"""Fetch raw TMDb discovery and detail data."""

from __future__ import annotations

import json
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
    refresh: bool = False,
) -> Path:
    """Fetch and save a TMDb discover/movie sample."""
    movies = client.discover_movies(
        limit=limit,
        min_votes=min_votes,
        sort_by=sort_by,
        min_runtime=min_runtime,
        refresh=refresh,
    )
    payload = {
        "source": "tmdb:/discover/movie",
        "limit": limit,
        "min_votes": min_votes,
        "sort_by": sort_by,
        "min_runtime": min_runtime,
        "movie_count": len(movies),
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
