"""Small TMDb API client with polite retries and JSON response caching."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

from film_atlas.config import MissingCredentialsError

TMDB_BASE_URL = "https://api.themoviedb.org/3"
RETRY_STATUSES = {429, 500, 502, 503, 504}


class TMDbClient:
    """Client for the official TMDb API used in Milestone 1."""

    def __init__(
        self,
        bearer_token: str | None,
        *,
        cache_dir: str | Path = "data/cache",
        base_url: str = TMDB_BASE_URL,
        timeout: float = 30.0,
        max_retries: int = 3,
        backoff_seconds: float = 1.0,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.bearer_token = bearer_token
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self.sleep = sleep
        self.client = httpx.Client(base_url=base_url, timeout=timeout, transport=transport)

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self.client.close()

    def __enter__(self) -> TMDbClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def discover_movies(
        self,
        *,
        limit: int,
        min_votes: int = 500,
        sort_by: str = "popularity.desc",
        min_runtime: int = 60,
        refresh: bool = False,
    ) -> list[dict[str, Any]]:
        """Fetch a controlled sample of original English-language films."""
        results: list[dict[str, Any]] = []
        page = 1
        total_pages = 1

        while len(results) < limit and page <= total_pages:
            payload = self.get_json(
                "/discover/movie",
                params={
                    "include_adult": "false",
                    "include_video": "false",
                    "language": "en-US",
                    "page": page,
                    "sort_by": sort_by,
                    "vote_count.gte": min_votes,
                    "with_original_language": "en",
                    "with_runtime.gte": min_runtime,
                },
                refresh=refresh,
            )
            total_pages = int(payload.get("total_pages") or 1)
            results.extend(payload.get("results") or [])
            page += 1

        return results[:limit]

    def movie_details(self, tmdb_id: int, *, refresh: bool = False) -> dict[str, Any]:
        """Fetch movie details with keywords, reviews, credits, and external IDs."""
        return self.get_json(
            f"/movie/{tmdb_id}",
            params={
                "append_to_response": "keywords,reviews,credits,external_ids",
                "language": "en-US",
            },
            refresh=refresh,
        )

    def get_json(
        self,
        endpoint: str,
        *,
        params: dict[str, Any] | None = None,
        refresh: bool = False,
    ) -> dict[str, Any]:
        """Fetch JSON from TMDb or return a cached response."""
        cache_path = self._cache_path(endpoint, params or {})
        if cache_path.exists() and not refresh:
            return json.loads(cache_path.read_text(encoding="utf-8"))

        if not self.bearer_token:
            raise MissingCredentialsError(
                "TMDB_BEARER_TOKEN is required for live TMDb fetches. "
                "Copy .env.example to .env, add the token, then run the command again."
            )

        response = self._request_with_retries(endpoint, params=params or {})
        payload = response.json()
        cache_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return payload

    def _request_with_retries(self, endpoint: str, *, params: dict[str, Any]) -> httpx.Response:
        headers = {
            "Authorization": f"Bearer {self.bearer_token}",
            "Accept": "application/json",
        }
        last_response: httpx.Response | None = None

        for attempt in range(self.max_retries + 1):
            response = self.client.get(endpoint, params=params, headers=headers)
            last_response = response
            if response.status_code not in RETRY_STATUSES:
                response.raise_for_status()
                return response

            if attempt < self.max_retries:
                retry_after = response.headers.get("Retry-After")
                delay = float(retry_after) if retry_after else self.backoff_seconds * 2**attempt
                self.sleep(delay)

        if last_response is None:
            raise RuntimeError("TMDb request failed before receiving a response.")
        last_response.raise_for_status()
        return last_response

    def _cache_path(self, endpoint: str, params: dict[str, Any]) -> Path:
        normalized_endpoint = endpoint.strip("/").replace("/", "__") or "root"
        serialized_params = json.dumps(params, sort_keys=True, default=str)
        digest = hashlib.sha1(f"{endpoint}:{serialized_params}".encode("utf-8")).hexdigest()[:16]
        return self.cache_dir / f"{normalized_endpoint}__{digest}.json"
