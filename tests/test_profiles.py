from __future__ import annotations

from film_atlas.models import MovieRecord
from film_atlas.profiles import build_semantic_profile


def _movie_with_reviews(reviews: list[str]) -> MovieRecord:
    return MovieRecord(
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
        reviews=reviews,
        original_language="forbidden-language-marker",
        production_countries=["United States of America"],
        production_companies=["Northstar Pictures"],
        cast=["Avery Stone"],
        directors=["June Archer"],
    )


def test_semantic_profile_excludes_forbidden_production_context_fields() -> None:
    movie = _movie_with_reviews(
        [
            "Dreamy 2019 science fiction with Avery Stone, June Archer, "
            "and Northstar Pictures energy."
        ]
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


def test_semantic_profile_can_disable_reviews() -> None:
    movie = _movie_with_reviews(["Dreamy review language."])

    profile = build_semantic_profile(movie, include_reviews=False)

    assert "Review language" not in profile.profile_text
    assert "Dreamy review language" not in profile.profile_text


def test_review_snippets_are_cleaned_truncated_and_light_by_default() -> None:
    movie = _movie_with_reviews(
        [
            "WOW!!! WOW!!! WOW!!! See https://example.com #MoonHarbor "
            "Avery Stone makes this eerie eerie eerie and dreamy.",
            "Second review should not appear at light weight.",
        ]
    )

    profile = build_semantic_profile(movie, max_review_chars=80, review_weight="light")

    assert "Review language" in profile.profile_text
    assert "https://" not in profile.profile_text
    assert "#MoonHarbor" not in profile.profile_text
    assert "!!!" not in profile.profile_text
    assert "Avery Stone" not in profile.profile_text
    assert "eerie eerie eerie" not in profile.profile_text
    assert "Second review" not in profile.profile_text
    review_line = next(line for line in profile.profile_text.splitlines() if line.startswith("Review"))
    assert len(review_line.removeprefix("Review language: ")) <= 80


def test_review_weight_medium_allows_more_review_snippets() -> None:
    movie = _movie_with_reviews(["First mood.", "Second mood."])

    profile = build_semantic_profile(movie, max_review_chars=80, review_weight="medium")

    assert "First mood" in profile.profile_text
    assert "Second mood" in profile.profile_text
