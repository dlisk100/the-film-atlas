from __future__ import annotations

import json
from pathlib import Path

import httpx

from film_atlas.tmdb_client import TMDbClient


def test_discover_movies_uses_required_filters_and_caches(tmp_path: Path) -> None:
    fixture = json.loads(Path("tests/fixtures/tmdb_discover_page.json").read_text())
    seen_requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_requests.append(request)
        return httpx.Response(200, json=fixture)

    transport = httpx.MockTransport(handler)
    client = TMDbClient(
        "test-token",
        cache_dir=tmp_path,
        transport=transport,
        sleep=lambda _seconds: None,
    )

    movies = client.discover_movies(limit=1, min_votes=500)

    assert movies[0]["id"] == 101
    assert len(seen_requests) == 1
    request = seen_requests[0]
    assert request.url.path == "/3/discover/movie"
    assert request.headers["Authorization"] == "Bearer test-token"
    assert request.url.params["with_original_language"] == "en"
    assert request.url.params["include_adult"] == "false"
    assert request.url.params["include_video"] == "false"
    assert request.url.params["vote_count.gte"] == "500"
    assert request.url.params["with_runtime.gte"] == "60"

    cached_client = TMDbClient(
        None,
        cache_dir=tmp_path,
        transport=httpx.MockTransport(lambda _request: httpx.Response(500)),
    )
    cached_movies = cached_client.discover_movies(limit=1, min_votes=500)
    assert cached_movies[0]["id"] == 101
