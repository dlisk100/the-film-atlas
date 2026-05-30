from __future__ import annotations

import json
from pathlib import Path

from film_atlas.normalize import normalize_movie_detail


def test_normalize_movie_detail_extracts_required_fields() -> None:
    detail = json.loads(Path("tests/fixtures/tmdb_movie_detail.json").read_text())

    movie = normalize_movie_detail(detail)

    assert movie.tmdb_id == 101
    assert movie.imdb_id == "tt0101001"
    assert movie.title == "Moon Harbor"
    assert movie.year == 2019
    assert movie.runtime == 111
    assert movie.genres == ["Science Fiction", "Adventure"]
    assert movie.keywords == ["moon colony", "distress signal", "salvage crew"]
    assert movie.production_companies == ["Northstar Pictures"]
    assert movie.production_countries == ["United States of America"]
    assert movie.cast == ["Avery Stone", "Mira Vale"]
    assert movie.directors == ["June Archer"]
    assert movie.reviews == [
        "Dreamy, eerie science fiction about isolation, grief, and impossible signals."
    ]
