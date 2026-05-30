from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from film_atlas.fetch import fetch_balanced


class FakeTMDbClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def discover_movies(self, **kwargs: Any) -> list[dict[str, Any]]:
        self.calls.append(kwargs)
        decade = kwargs["release_date_gte"][:4]
        return [
            {"id": int(decade), "title": f"{decade} Movie"},
            {"id": 42, "title": "Duplicate Across Buckets"},
        ]


def test_fetch_balanced_writes_deduped_discover_payload(tmp_path: Path) -> None:
    client = FakeTMDbClient()

    path = fetch_balanced(
        client,  # type: ignore[arg-type]
        per_decade=2,
        start_year=1980,
        end_year=1990,
        output_dir=tmp_path,
    )

    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["sampling_strategy"] == "balanced_by_decade"
    assert [call["release_date_gte"] for call in client.calls] == ["1980-01-01", "1990-01-01"]
    assert [call["release_date_lte"] for call in client.calls] == ["1989-12-31", "1990-12-31"]
    assert payload["movie_count"] == 3
    assert [movie["id"] for movie in payload["results"]] == [1980, 42, 1990]
