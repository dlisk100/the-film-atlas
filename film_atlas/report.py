"""Generate the Milestone 1 data-quality report."""

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

from film_atlas.models import MovieRecord, NeighborPair, SemanticProfile
from film_atlas.normalize import load_movie_records
from film_atlas.profiles import load_profiles

REPORT_FILENAME = "milestone_1_report.md"
CURRENT_RELEASE_YEAR = 2024
FRANCHISE_KEYWORD_TERMS = {
    "aftercreditsstinger",
    "based on comic",
    "based on video game",
    "duringcreditsstinger",
    "marvel cinematic universe (mcu)",
    "reboot",
    "remake",
    "sequel",
    "superhero",
    "superhero team",
}
REVIEW_NOISE_TERMS = {
    "http",
    "https",
    "www",
    "spoiler",
    "review",
    "rating",
    "stars",
    "full",
    "letterboxd",
}


def generate_report_file(
    *,
    data_dir: str | Path = "data",
    output_dir: str | Path = "outputs",
) -> Path:
    """Generate outputs/reports/milestone_1_report.md."""
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    movies_path = data_path / "processed" / "movies.json"
    profiles_path = data_path / "processed" / "profiles.json"
    discover_path = data_path / "raw" / "discover_movies.json"
    neighbors_path = output_path / "reports" / "nearest_neighbors.json"

    movies = load_movie_records(movies_path) if movies_path.exists() else []
    profiles = load_profiles(profiles_path) if profiles_path.exists() else []
    discovered_count = _discovered_count(discover_path)
    neighbor_pairs = _load_neighbor_pairs(neighbors_path)

    report_path = output_path / "reports" / REPORT_FILENAME
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        render_report(
            discovered_count=discovered_count,
            movies=movies,
            profiles=profiles,
            neighbor_pairs=neighbor_pairs,
        ),
        encoding="utf-8",
    )
    return report_path


def render_report(
    *,
    discovered_count: int,
    movies: list[MovieRecord],
    profiles: list[SemanticProfile],
    neighbor_pairs: list[NeighborPair],
) -> str:
    """Render the Milestone 1 report body."""
    detail_count = len(movies)
    lines = [
        "# The Film Atlas - Milestone 1 Data Quality Report",
        "",
        "Milestone 1 uses only TMDb API data, local processing, and a TF-IDF sample map. "
        "Final semantic embeddings, final clustering, cluster labels, and OpenAI calls are "
        "intentionally left for later milestones.",
        "",
        "## Summary",
        "",
        f"- Discovered movies: {discovered_count}",
        f"- Detail records fetched: {detail_count}",
        f"- With overview: {_percent_with(movies, lambda movie: bool(movie.overview))}",
        f"- With keywords: {_percent_with(movies, lambda movie: bool(movie.keywords))}",
        f"- With reviews: {_percent_with(movies, lambda movie: bool(movie.reviews))}",
        f"- From {CURRENT_RELEASE_YEAR} or later: {_percent_with(movies, _is_current_release)}",
        f"- From future release years: {_percent_with(movies, _is_future_release_year)}",
        "",
        "## Year Distribution",
        "",
        _counter_table(Counter(movie.year for movie in movies if movie.year), "Year"),
        "",
        "## Top Official Genres",
        "",
        _counter_table(Counter(genre for movie in movies for genre in movie.genres), "Genre"),
        "",
        "## Top Keywords",
        "",
        _counter_table(Counter(keyword for movie in movies for keyword in movie.keywords), "Keyword"),
        "",
        "## Sampling Bias Diagnostics",
        "",
        _sampling_bias_section(movies),
        "",
        "## Review Noise Diagnostics",
        "",
        _review_noise_section(movies),
        "",
        "## Movies Missing Important Fields",
        "",
        _missing_fields_table(movies),
        "",
        "## Sample Movie Text Profiles",
        "",
        _sample_profiles(profiles, limit=20),
        "",
        "## Example Nearest-Neighbor Pairs",
        "",
        _neighbor_pairs_table(neighbor_pairs, limit=10),
        "",
        "## Notes",
        "",
        "- This report is a data-quality proof, not the final public website output.",
        "- Raw TMDb responses live under data/cache/ and data/raw/, which are gitignored.",
        "- Review language in profiles is truncated for semantic experimentation.",
        "- Recommended balanced sampling command before Milestone 2: "
        "`uv run film-atlas fetch-balanced --per-decade 100 --start-year 1980 --end-year 2026`.",
        "- OpenAI embeddings and cluster labeling are out of scope for Milestone 1.",
        "",
    ]
    return "\n".join(lines)


def _discovered_count(path: Path) -> int:
    if not path.exists():
        return 0
    payload = json.loads(path.read_text(encoding="utf-8"))
    return int(payload.get("movie_count") or len(payload.get("results") or []))


def _load_neighbor_pairs(path: Path) -> list[NeighborPair]:
    if not path.exists():
        return []
    payload: list[dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))
    return [
        NeighborPair(
            source_tmdb_id=int(item["source_tmdb_id"]),
            source_title=str(item["source_title"]),
            neighbor_tmdb_id=int(item["neighbor_tmdb_id"]),
            neighbor_title=str(item["neighbor_title"]),
            similarity=float(item["similarity"]),
        )
        for item in payload
    ]


def _percent_with(movies: list[MovieRecord], predicate: Any) -> str:
    if not movies:
        return "0.0% (0/0)"
    count = sum(1 for movie in movies if predicate(movie))
    return f"{count / len(movies) * 100:.1f}% ({count}/{len(movies)})"


def _is_current_release(movie: MovieRecord) -> bool:
    return bool(movie.year and movie.year >= CURRENT_RELEASE_YEAR)


def _is_future_release_year(movie: MovieRecord) -> bool:
    return bool(movie.year and movie.year > date.today().year)


def _counter_table(counter: Counter[Any], label: str, *, limit: int = 25) -> str:
    if not counter:
        return "_No data available._"

    lines = [f"| {label} | Count |", "| --- | ---: |"]
    for value, count in counter.most_common(limit):
        lines.append(f"| {value} | {count} |")
    return "\n".join(lines)


def _sampling_bias_section(movies: list[MovieRecord]) -> str:
    if not movies:
        return "_No detail records available._"

    current_count = sum(1 for movie in movies if _is_current_release(movie))
    future_count = sum(1 for movie in movies if _is_future_release_year(movie))
    franchise_counter = Counter(
        keyword
        for movie in movies
        for keyword in movie.keywords
        if _is_franchise_keyword(keyword)
    )
    franchise_movie_count = sum(
        1 for movie in movies if any(_is_franchise_keyword(keyword) for keyword in movie.keywords)
    )
    current_pct = current_count / len(movies) * 100
    franchise_pct = franchise_movie_count / len(movies) * 100

    warnings = []
    if current_pct >= 25:
        warnings.append(
            f"Current-release concentration is high: {current_pct:.1f}% of movies are "
            f"from {CURRENT_RELEASE_YEAR} or later."
        )
    if future_count:
        warnings.append(f"Future release years are present: {future_count} movies.")
    if franchise_pct >= 25:
        warnings.append(
            f"Franchise/sequel concentration is high: {franchise_pct:.1f}% of movies have "
            "franchise-related keywords."
        )

    lines = [
        f"- Movies from {CURRENT_RELEASE_YEAR} or later: {current_pct:.1f}% "
        f"({current_count}/{len(movies)})",
        f"- Movies from future release years: {future_count / len(movies) * 100:.1f}% "
        f"({future_count}/{len(movies)})",
        f"- Movies with franchise/sequel keywords: {franchise_pct:.1f}% "
        f"({franchise_movie_count}/{len(movies)})",
        "",
        "### Franchise/Sequel Keywords",
        "",
        _counter_table(franchise_counter, "Keyword", limit=15),
        "",
        "### Warnings",
        "",
    ]
    if warnings:
        lines.extend(f"- Warning: {warning}" for warning in warnings)
    else:
        lines.append("_No high-concentration warnings for this sample._")
    lines.extend(
        [
            "",
            "### Recommended Balanced Command",
            "",
            "```bash",
            "uv run film-atlas fetch-balanced --per-decade 100 --start-year 1980 --end-year 2026",
            "uv run film-atlas fetch-details",
            "uv run film-atlas normalize",
            "uv run film-atlas build-profiles --review-weight light --max-review-chars 180",
            "uv run film-atlas make-sample-map",
            "uv run film-atlas report",
            "```",
        ]
    )
    return "\n".join(lines)


def _review_noise_section(movies: list[MovieRecord]) -> str:
    if not movies:
        return "_No detail records available._"

    counter = Counter()
    examples = []
    for movie in movies:
        for review in movie.reviews:
            lowered = review.lower()
            for term in REVIEW_NOISE_TERMS:
                if term in lowered:
                    counter[term] += 1
            if len(examples) < 5 and _looks_noisy(review):
                examples.append((movie.title, _shorten_review(review)))

    lines = [
        "### Suspicious Review Terms",
        "",
        _counter_table(counter, "Term", limit=15),
        "",
        "### Review Noise Examples",
        "",
    ]
    if not examples:
        lines.append("_No obvious review-noise examples found._")
    else:
        for title, example in examples:
            lines.append(f"- {title}: {example}")
    return "\n".join(lines)


def _is_franchise_keyword(keyword: str) -> bool:
    lowered = keyword.lower()
    return any(term in lowered for term in FRANCHISE_KEYWORD_TERMS)


def _looks_noisy(review: str) -> bool:
    lowered = review.lower()
    return any(term in lowered for term in REVIEW_NOISE_TERMS) or "#" in review


def _shorten_review(review: str, *, limit: int = 180) -> str:
    cleaned = re.sub(r"https?://\S+|www\.\S+|#\S+", "", review)
    cleaned = " ".join(cleaned.split())
    return cleaned[:limit].strip()


def _missing_fields_table(movies: list[MovieRecord]) -> str:
    if not movies:
        return "_No detail records available._"

    rows = []
    for movie in movies:
        missing = []
        if not movie.overview:
            missing.append("overview")
        if not movie.keywords:
            missing.append("keywords")
        if not movie.runtime:
            missing.append("runtime")
        if not movie.release_date:
            missing.append("release_date")
        if missing:
            rows.append((movie.tmdb_id, movie.title, ", ".join(missing)))

    if not rows:
        return "_No important fields are missing from the normalized sample._"

    lines = ["| TMDb ID | Title | Missing Fields |", "| ---: | --- | --- |"]
    for tmdb_id, title, missing in rows[:50]:
        lines.append(f"| {tmdb_id} | {title} | {missing} |")
    return "\n".join(lines)


def _sample_profiles(profiles: list[SemanticProfile], *, limit: int) -> str:
    if not profiles:
        return "_No profiles available._"

    sections = []
    for profile in profiles[:limit]:
        text = profile.profile_text[:900]
        sections.append(f"### {profile.title}\n\n```text\n{text}\n```")
    return "\n\n".join(sections)


def _neighbor_pairs_table(pairs: list[NeighborPair], *, limit: int) -> str:
    if not pairs:
        return (
            "_No nearest-neighbor pairs available. Run `film-atlas make-sample-map` "
            "with at least two profiles._"
        )

    lines = ["| Source | Neighbor | Similarity |", "| --- | --- | ---: |"]
    for pair in pairs[:limit]:
        lines.append(
            f"| {pair.source_title} | {pair.neighbor_title} | {pair.similarity:.3f} |"
        )
    return "\n".join(lines)
