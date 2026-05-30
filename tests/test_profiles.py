from __future__ import annotations

from film_atlas.models import MovieRecord
from film_atlas.profiles import build_semantic_profile


def test_semantic_profile_excludes_forbidden_production_context_fields() -> None:
    movie = MovieRecord(
        tmdb_id=101,
        imdb_id="tt0101001",
        title="Moon Harbor",
        original_title="Moon Harbor",
        release_date="2019-03-15",
        year=2019,
        runtime=111,
        overview="A lonely salvager follows a signal into a haunted lunar port.",
        genres=["Science Fiction", "Adventure"],
        keywords=["moon colony", "distress signal", "salvage crew"],
        poster_path="/moon-poster.jpg",
        backdrop_path="/moon-backdrop.jpg",
        vote_average=7.4,
        vote_count=860,
        popularity=42.5,
        reviews=["Dreamy, eerie science fiction about isolation and impossible signals."],
        original_language="forbidden-language-marker",
        production_countries=["United States of America"],
        production_companies=["Northstar Pictures"],
        cast=["Avery Stone"],
        directors=["June Archer"],
    )

    profile = build_semantic_profile(movie)

    assert "Moon Harbor" in profile.profile_text
    assert "haunted lunar port" in profile.profile_text
    assert "moon colony" in profile.profile_text
    assert "2019" not in profile.profile_text
    assert "United States" not in profile.profile_text
    assert "forbidden-language-marker" not in profile.profile_text
    assert "Northstar Pictures" not in profile.profile_text
    assert "Avery Stone" not in profile.profile_text
    assert "June Archer" not in profile.profile_text
