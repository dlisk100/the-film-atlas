"""Optional public-dataset signal loaders for Film Atlas experiments."""

from __future__ import annotations

import csv
import heapq
import io
import re
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from film_atlas.models import MovieRecord


@dataclass(frozen=True, slots=True)
class ExternalMovieSignals:
    """Non-TMDb optional signals matched to a movie."""

    tmdb_id: int
    movie_lens_tags: list[str] = field(default_factory=list)
    mpst_tags: list[str] = field(default_factory=list)
    mpst_synopsis: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ExternalSignalCoverage:
    """Coverage summary for optional external signal sources."""

    movie_count: int
    movie_lens_matched_count: int
    mpst_matched_count: int
    movie_lens_path: str | None
    mpst_path: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ExternalSignalBundle:
    """Loaded optional signal set and coverage metadata."""

    signals_by_tmdb_id: dict[int, ExternalMovieSignals]
    coverage: ExternalSignalCoverage


def load_external_signals(
    movies: list[MovieRecord],
    *,
    external_dir: str | Path = "data/external",
    max_movie_lens_tags: int = 24,
    min_movie_lens_relevance: float = 0.32,
    max_mpst_synopsis_chars: int = 1400,
) -> ExternalSignalBundle:
    """Load MovieLens Tag Genome and MPST signals when local files are available."""
    external_path = Path(external_dir)
    movie_lens_path = external_path / "ml-25m.zip"
    mpst_path = external_path / "mpst_full_data.csv"

    movie_lens_tags = (
        load_movielens_tag_genome(
            movies,
            movie_lens_path=movie_lens_path,
            max_tags=max_movie_lens_tags,
            min_relevance=min_movie_lens_relevance,
        )
        if movie_lens_path.exists()
        else {}
    )
    mpst_rows = (
        load_mpst_signals(
            movies,
            mpst_path=mpst_path,
            max_synopsis_chars=max_mpst_synopsis_chars,
        )
        if mpst_path.exists()
        else {}
    )

    merged: dict[int, ExternalMovieSignals] = {}
    for movie in movies:
        mpst = mpst_rows.get(movie.tmdb_id)
        merged[movie.tmdb_id] = ExternalMovieSignals(
            tmdb_id=movie.tmdb_id,
            movie_lens_tags=movie_lens_tags.get(movie.tmdb_id, []),
            mpst_tags=mpst.mpst_tags if mpst else [],
            mpst_synopsis=mpst.mpst_synopsis if mpst else None,
        )

    coverage = ExternalSignalCoverage(
        movie_count=len(movies),
        movie_lens_matched_count=sum(1 for item in merged.values() if item.movie_lens_tags),
        mpst_matched_count=sum(1 for item in merged.values() if item.mpst_tags or item.mpst_synopsis),
        movie_lens_path=str(movie_lens_path) if movie_lens_path.exists() else None,
        mpst_path=str(mpst_path) if mpst_path.exists() else None,
    )
    return ExternalSignalBundle(signals_by_tmdb_id=merged, coverage=coverage)


def load_movielens_tag_genome(
    movies: list[MovieRecord],
    *,
    movie_lens_path: str | Path,
    max_tags: int = 24,
    min_relevance: float = 0.32,
) -> dict[int, list[str]]:
    """Load top MovieLens Tag Genome tags keyed by TMDb id from the MovieLens 25M zip."""
    target_tmdb_ids = {movie.tmdb_id for movie in movies}
    movie_lens_file = Path(movie_lens_path)
    with zipfile.ZipFile(movie_lens_file) as archive:
        links_name = _zip_member(archive, "links.csv")
        tags_name = _zip_member(archive, "genome-tags.csv")
        scores_name = _zip_member(archive, "genome-scores.csv")
        if not (links_name and tags_name and scores_name):
            return {}

        tmdb_by_movie_lens_id: dict[int, int] = {}
        with archive.open(links_name) as file:
            reader = csv.DictReader(io.TextIOWrapper(file, encoding="utf-8"))
            for row in reader:
                tmdb_id = _optional_int(row.get("tmdbId"))
                movie_lens_id = _optional_int(row.get("movieId"))
                if tmdb_id in target_tmdb_ids and movie_lens_id is not None:
                    tmdb_by_movie_lens_id[movie_lens_id] = tmdb_id

        if not tmdb_by_movie_lens_id:
            return {}

        tag_by_id: dict[int, str] = {}
        with archive.open(tags_name) as file:
            reader = csv.DictReader(io.TextIOWrapper(file, encoding="utf-8"))
            for row in reader:
                tag_id = _optional_int(row.get("tagId"))
                tag = _clean_text(row.get("tag"))
                if tag_id is not None and tag:
                    tag_by_id[tag_id] = tag

        heaps: dict[int, list[tuple[float, int]]] = {movie_id: [] for movie_id in tmdb_by_movie_lens_id}
        with archive.open(scores_name) as file:
            reader = csv.DictReader(io.TextIOWrapper(file, encoding="utf-8"))
            for row in reader:
                movie_lens_id = _optional_int(row.get("movieId"))
                if movie_lens_id not in heaps:
                    continue
                tag_id = _optional_int(row.get("tagId"))
                relevance = _optional_float(row.get("relevance"))
                if tag_id is None or relevance is None or relevance < min_relevance:
                    continue
                heap = heaps[movie_lens_id]
                item = (relevance, tag_id)
                if len(heap) < max_tags:
                    heapq.heappush(heap, item)
                elif item > heap[0]:
                    heapq.heapreplace(heap, item)

    output: dict[int, list[str]] = {}
    for movie_lens_id, heap in heaps.items():
        tmdb_id = tmdb_by_movie_lens_id[movie_lens_id]
        ranked = sorted(heap, reverse=True)
        tags = [tag_by_id[tag_id] for _score, tag_id in ranked if tag_id in tag_by_id]
        if tags:
            output[tmdb_id] = tags
    return output


def load_mpst_signals(
    movies: list[MovieRecord],
    *,
    mpst_path: str | Path,
    max_synopsis_chars: int = 1400,
) -> dict[int, ExternalMovieSignals]:
    """Load MPST tags and synopsis text keyed by TMDb id through IMDb id."""
    tmdb_by_imdb = {
        str(movie.imdb_id): movie.tmdb_id
        for movie in movies
        if movie.imdb_id
    }
    output: dict[int, ExternalMovieSignals] = {}
    with Path(mpst_path).open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            imdb_id = str(row.get("imdb_id") or "")
            tmdb_id = tmdb_by_imdb.get(imdb_id)
            if tmdb_id is None:
                continue
            tags = [
                _clean_text(part)
                for part in str(row.get("tags") or "").split(",")
                if _clean_text(part)
            ]
            synopsis = _clean_synopsis(row.get("plot_synopsis"), max_chars=max_synopsis_chars)
            output[tmdb_id] = ExternalMovieSignals(
                tmdb_id=tmdb_id,
                mpst_tags=tags,
                mpst_synopsis=synopsis,
            )
    return output


def _zip_member(archive: zipfile.ZipFile, suffix: str) -> str | None:
    for name in archive.namelist():
        if name.endswith(suffix):
            return name
    return None


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _clean_synopsis(value: Any, *, max_chars: int) -> str | None:
    cleaned = _clean_text(value)
    if not cleaned:
        return None
    cleaned = re.sub(r"\\s+", " ", cleaned)
    return cleaned[:max_chars].strip()


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").split())
