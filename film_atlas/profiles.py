"""Build semantic text profiles for local sample mapping."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal

from film_atlas.models import MovieRecord, SemanticProfile
from film_atlas.normalize import load_movie_records

PROFILES_FILENAME = "profiles.json"
DEFAULT_MAX_REVIEW_CHARS = 180
ReviewWeight = Literal["light", "medium", "heavy"]
REVIEW_WEIGHT_SNIPPETS: dict[ReviewWeight, int] = {
    "light": 1,
    "medium": 2,
    "heavy": 4,
}

FORBIDDEN_PROFILE_FIELDS = {
    "year",
    "decade",
    "country",
    "language",
    "cast",
    "director",
    "production_company",
    "production_companies",
}

COUNTRY_ADJECTIVES = {
    "Australia": "Australian",
    "Canada": "Canadian",
    "France": "French",
    "Germany": "German",
    "India": "Indian",
    "Ireland": "Irish",
    "Italy": "Italian",
    "Japan": "Japanese",
    "New Zealand": "New Zealander",
    "Spain": "Spanish",
    "United Kingdom": "British",
    "United States of America": "American",
}


def build_semantic_profile(
    movie: MovieRecord,
    *,
    include_reviews: bool = True,
    max_review_chars: int = DEFAULT_MAX_REVIEW_CHARS,
    review_weight: ReviewWeight = "light",
) -> SemanticProfile:
    """Build profile text without production-context fields."""
    forbidden_values = _forbidden_values(movie)
    review_language = _build_review_language(
        movie.reviews,
        forbidden_values=forbidden_values,
        include_reviews=include_reviews,
        max_review_chars=max_review_chars,
        review_weight=review_weight,
    )
    parts = [
        _section("Title", [movie.title]),
        _section("Overview", [_redact_forbidden_values(movie.overview, forbidden_values)]),
        _section("Genres", movie.genres),
        _section(
            "Keywords",
            [_redact_forbidden_values(keyword, forbidden_values) for keyword in movie.keywords],
        ),
        _section(
            "Review language",
            review_language,
        ),
    ]
    profile_text = "\n".join(part for part in parts if part)
    return SemanticProfile(
        tmdb_id=movie.tmdb_id,
        title=movie.title,
        year=movie.year,
        profile_text=profile_text,
        genres=movie.genres,
        keywords=movie.keywords,
    )


def build_profiles_file(
    *,
    movies_path: str | Path = "data/processed/movies.json",
    output_dir: str | Path = "data",
    include_reviews: bool = True,
    max_review_chars: int = DEFAULT_MAX_REVIEW_CHARS,
    review_weight: ReviewWeight = "light",
) -> Path:
    """Build all semantic profiles and write them to data/processed/profiles.json."""
    profiles = [
        build_semantic_profile(
            movie,
            include_reviews=include_reviews,
            max_review_chars=max_review_chars,
            review_weight=review_weight,
        )
        for movie in load_movie_records(movies_path)
    ]
    output_path = Path(output_dir) / "processed" / PROFILES_FILENAME
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps([profile.to_dict() for profile in profiles], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output_path


def load_profiles(path: str | Path = "data/processed/profiles.json") -> list[SemanticProfile]:
    """Load semantic profiles from JSON."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [SemanticProfile.from_dict(item) for item in payload]


def _section(label: str, values: list[str | None]) -> str:
    cleaned = [" ".join(str(value).split()) for value in values if value]
    if not cleaned:
        return ""
    return f"{label}: {'; '.join(cleaned)}"


def _build_review_language(
    reviews: list[str],
    *,
    forbidden_values: list[str],
    include_reviews: bool,
    max_review_chars: int,
    review_weight: ReviewWeight,
) -> list[str]:
    if not include_reviews or max_review_chars <= 0:
        return []

    snippet_limit = REVIEW_WEIGHT_SNIPPETS[review_weight]
    snippets: list[str] = []
    remaining_chars = max_review_chars
    for review in reviews[:snippet_limit]:
        cleaned = _clean_review_snippet(review)
        cleaned = _redact_forbidden_values(cleaned, forbidden_values)
        if not cleaned:
            continue
        snippet = cleaned[:remaining_chars].strip()
        if snippet:
            snippets.append(snippet)
            remaining_chars -= len(snippet)
        if remaining_chars <= 0:
            break

    return snippets


def _clean_review_snippet(review: str) -> str:
    cleaned = _strip_review_noise(review)
    cleaned = re.sub(r"([!?.,;:])\1{1,}", r"\1", cleaned)
    cleaned = _collapse_repeated_tokens(cleaned)
    return " ".join(cleaned.split())


def _forbidden_values(movie: MovieRecord) -> list[str]:
    values: list[str] = []
    values.extend(_production_country_values(movie.production_countries))
    values.extend(movie.production_companies)
    values.extend(movie.cast)
    values.extend(movie.directors)
    if movie.original_language:
        values.append(movie.original_language)
    if movie.year:
        values.append(str(movie.year))
    if movie.release_date:
        values.append(movie.release_date)
    decade = _decade_label(movie.year)
    if decade:
        values.append(decade)

    cleaned = {" ".join(value.split()) for value in values if len(value.strip()) > 2}
    return sorted(cleaned, key=len, reverse=True)


def _redact_forbidden_values(value: str | None, forbidden_values: list[str]) -> str | None:
    if not value:
        return value

    redacted = _strip_review_noise(value)
    for forbidden_value in forbidden_values:
        for variant in _redaction_variants(forbidden_value):
            pattern = re.compile(re.escape(variant), flags=re.IGNORECASE)
            redacted = pattern.sub("", redacted)
    return " ".join(redacted.split()) or None


def _collapse_repeated_tokens(value: str) -> str:
    tokens = value.split()
    collapsed: list[str] = []
    previous_normalized = ""
    repeat_count = 0

    for token in tokens:
        normalized = re.sub(r"\W+", "", token).lower()
        if normalized and normalized == previous_normalized:
            repeat_count += 1
        else:
            repeat_count = 1
            previous_normalized = normalized

        if repeat_count <= 2:
            collapsed.append(token)

    return " ".join(collapsed)


def _decade_label(year: int | None) -> str | None:
    if year is None:
        return None
    return f"{year // 10 * 10}s"


def _production_country_values(countries: list[str]) -> list[str]:
    values = list(countries)
    for country in countries:
        adjective = COUNTRY_ADJECTIVES.get(country)
        if adjective:
            values.append(adjective)
    return values


def _redaction_variants(value: str) -> set[str]:
    variants = {value}
    compact = re.sub(r"\W+", "", value)
    if len(compact) > 2:
        variants.add(compact)
    return variants


def _strip_review_noise(value: str) -> str:
    without_urls = re.sub(r"https?://\S+|www\.\S+", "", value)
    without_hashtags = re.sub(r"#\S+", "", without_urls)
    return without_hashtags
