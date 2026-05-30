"""Typed records used by the Milestone 1 pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class MovieRecord:
    """Normalized movie detail record.

    Production-context fields are kept as metadata for future non-semantic modes
    and tests, but profile builders must not include them in semantic text.
    """

    tmdb_id: int
    imdb_id: str | None
    title: str
    original_title: str
    release_date: str | None
    year: int | None
    runtime: int | None
    overview: str | None
    genres: list[str]
    keywords: list[str]
    poster_path: str | None
    backdrop_path: str | None
    vote_average: float | None
    vote_count: int | None
    popularity: float | None
    reviews: list[str] = field(default_factory=list)
    original_language: str | None = None
    production_countries: list[str] = field(default_factory=list)
    production_companies: list[str] = field(default_factory=list)
    cast: list[str] = field(default_factory=list)
    directors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the record for JSON and tabular output."""
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> MovieRecord:
        """Hydrate a record from a JSON dictionary."""
        return cls(
            tmdb_id=int(value["tmdb_id"]),
            imdb_id=value.get("imdb_id"),
            title=str(value.get("title") or ""),
            original_title=str(value.get("original_title") or ""),
            release_date=value.get("release_date"),
            year=value.get("year"),
            runtime=value.get("runtime"),
            overview=value.get("overview"),
            genres=list(value.get("genres") or []),
            keywords=list(value.get("keywords") or []),
            poster_path=value.get("poster_path"),
            backdrop_path=value.get("backdrop_path"),
            vote_average=value.get("vote_average"),
            vote_count=value.get("vote_count"),
            popularity=value.get("popularity"),
            reviews=list(value.get("reviews") or []),
            original_language=value.get("original_language"),
            production_countries=list(value.get("production_countries") or []),
            production_companies=list(value.get("production_companies") or []),
            cast=list(value.get("cast") or []),
            directors=list(value.get("directors") or []),
        )


@dataclass(frozen=True, slots=True)
class SemanticProfile:
    """A single movie's semantic profile text plus non-semantic metadata."""

    tmdb_id: int
    title: str
    year: int | None
    profile_text: str
    genres: list[str]
    keywords: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the profile for JSON output."""
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> SemanticProfile:
        """Hydrate a semantic profile from a JSON dictionary."""
        return cls(
            tmdb_id=int(value["tmdb_id"]),
            title=str(value.get("title") or ""),
            year=value.get("year"),
            profile_text=str(value.get("profile_text") or ""),
            genres=list(value.get("genres") or []),
            keywords=list(value.get("keywords") or []),
        )


@dataclass(frozen=True, slots=True)
class MapPoint:
    """A rough 2D coordinate from the local TF-IDF preview method."""

    tmdb_id: int
    title: str
    year: int | None
    x: float
    y: float
    top_terms: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the point for CSV or JSON output."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class NeighborPair:
    """A nearest-neighbor pair from the local sample method."""

    source_tmdb_id: int
    source_title: str
    neighbor_tmdb_id: int
    neighbor_title: str
    similarity: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize the pair for report output."""
        return asdict(self)
