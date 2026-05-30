"""Normalize raw TMDb movie details into stable local records."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from film_atlas.models import MovieRecord

PROCESSED_DIR_NAME = "processed"
MOVIES_JSON_FILENAME = "movies.json"
MOVIES_PARQUET_FILENAME = "movies.parquet"
REVIEW_SNIPPET_CHARS = 500


def normalize_movie_detail(detail: dict[str, Any]) -> MovieRecord:
    """Normalize a TMDb movie detail payload."""
    release_date = _clean_optional_text(detail.get("release_date"))
    return MovieRecord(
        tmdb_id=int(detail["id"]),
        imdb_id=_clean_optional_text(detail.get("imdb_id") or _external_id(detail, "imdb_id")),
        title=_clean_text(detail.get("title")),
        original_title=_clean_text(detail.get("original_title")),
        release_date=release_date,
        year=_year_from_release_date(release_date),
        runtime=_optional_int(detail.get("runtime")),
        overview=_clean_optional_text(detail.get("overview")),
        genres=_extract_names(detail.get("genres")),
        keywords=_extract_keywords(detail.get("keywords")),
        poster_path=_clean_optional_text(detail.get("poster_path")),
        backdrop_path=_clean_optional_text(detail.get("backdrop_path")),
        vote_average=_optional_float(detail.get("vote_average")),
        vote_count=_optional_int(detail.get("vote_count")),
        popularity=_optional_float(detail.get("popularity")),
        reviews=_extract_review_snippets(detail.get("reviews")),
        original_language=_clean_optional_text(detail.get("original_language")),
        production_countries=_extract_names(detail.get("production_countries")),
        production_companies=_extract_names(detail.get("production_companies")),
        cast=_extract_cast_names(detail.get("credits")),
        directors=_extract_director_names(detail.get("credits")),
    )


def normalize_details_file(
    *,
    details_path: str | Path = "data/raw/movie_details.json",
    output_dir: str | Path = "data",
) -> Path:
    """Normalize a raw TMDb details file and write JSON plus Parquet outputs."""
    payload = json.loads(Path(details_path).read_text(encoding="utf-8"))
    records = [normalize_movie_detail(detail) for detail in payload.get("results", [])]

    processed_dir = Path(output_dir) / PROCESSED_DIR_NAME
    processed_dir.mkdir(parents=True, exist_ok=True)

    json_path = processed_dir / MOVIES_JSON_FILENAME
    json_path.write_text(
        json.dumps([record.to_dict() for record in records], indent=2, sort_keys=True),
        encoding="utf-8",
    )

    parquet_path = processed_dir / MOVIES_PARQUET_FILENAME
    frame = pd.DataFrame([record.to_dict() for record in records])
    frame.to_parquet(parquet_path, index=False)

    return json_path


def load_movie_records(path: str | Path = "data/processed/movies.json") -> list[MovieRecord]:
    """Load normalized movie records from JSON."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [MovieRecord.from_dict(item) for item in payload]


def _extract_names(items: Any) -> list[str]:
    if not isinstance(items, list):
        return []
    names = []
    for item in items:
        if isinstance(item, dict):
            name = _clean_optional_text(item.get("name"))
            if name:
                names.append(name)
    return names


def _extract_keywords(payload: Any) -> list[str]:
    if not isinstance(payload, dict):
        return []
    return _extract_names(payload.get("keywords"))


def _extract_review_snippets(payload: Any) -> list[str]:
    if not isinstance(payload, dict):
        return []
    snippets = []
    for review in payload.get("results") or []:
        if not isinstance(review, dict):
            continue
        content = _clean_optional_text(review.get("content"))
        if content:
            snippets.append(content[:REVIEW_SNIPPET_CHARS])
    return snippets


def _extract_cast_names(payload: Any, *, limit: int = 10) -> list[str]:
    if not isinstance(payload, dict):
        return []
    return _extract_names(payload.get("cast"))[:limit]


def _extract_director_names(payload: Any) -> list[str]:
    if not isinstance(payload, dict):
        return []
    names = []
    for crew_member in payload.get("crew") or []:
        if not isinstance(crew_member, dict):
            continue
        if crew_member.get("job") == "Director":
            name = _clean_optional_text(crew_member.get("name"))
            if name:
                names.append(name)
    return names


def _external_id(detail: dict[str, Any], key: str) -> str | None:
    external_ids = detail.get("external_ids")
    if isinstance(external_ids, dict):
        return _clean_optional_text(external_ids.get(key))
    return None


def _year_from_release_date(release_date: str | None) -> int | None:
    if not release_date:
        return None
    match = re.match(r"^(\d{4})", release_date)
    return int(match.group(1)) if match else None


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _clean_optional_text(value: Any) -> str | None:
    cleaned = _clean_text(value)
    return cleaned or None
