"""Build semantic text profiles for local sample mapping."""

from __future__ import annotations

import json
from pathlib import Path

from film_atlas.models import MovieRecord, SemanticProfile
from film_atlas.normalize import load_movie_records

PROFILES_FILENAME = "profiles.json"
MAX_PROFILE_REVIEW_CHARS = 300

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


def build_semantic_profile(movie: MovieRecord) -> SemanticProfile:
    """Build profile text without production-context fields."""
    parts = [
        _section("Title", [movie.title]),
        _section("Overview", [movie.overview]),
        _section("Genres", movie.genres),
        _section("Keywords", movie.keywords),
        _section("Review language", [_truncate_review(review) for review in movie.reviews]),
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
) -> Path:
    """Build all semantic profiles and write them to data/processed/profiles.json."""
    profiles = [build_semantic_profile(movie) for movie in load_movie_records(movies_path)]
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


def _truncate_review(review: str) -> str:
    cleaned = " ".join(review.split())
    return cleaned[:MAX_PROFILE_REVIEW_CHARS]
