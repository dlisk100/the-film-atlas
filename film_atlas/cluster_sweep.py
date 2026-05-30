"""Compare k-means clustering granularities over existing embeddings."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from film_atlas.cluster import cluster_embedding_records
from film_atlas.embedding import load_embedding_records
from film_atlas.inspect_clusters import ClusterEvidence, build_cluster_evidence
from film_atlas.models import MovieRecord, SemanticProfile
from film_atlas.neighbors import MovieNeighbors, load_neighbors
from film_atlas.normalize import load_movie_records
from film_atlas.profiles import load_profiles

CLUSTER_SWEEP_JSON_FILENAME = "cluster_sweep.json"
CLUSTER_SWEEP_REPORT_FILENAME = "cluster_sweep_report.md"


class ClusterSweepError(RuntimeError):
    """Raised when a local cluster sweep cannot run."""


@dataclass(frozen=True, slots=True)
class SweepClusterSample:
    cluster_id: int
    cluster_size: int
    representative_movies: list[str]
    top_genres: list[tuple[str, int]]
    top_keywords: list[tuple[str, int]]
    top_terms: list[str]
    coherence_score: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SweepKResult:
    requested_k: int
    cluster_count: int
    movie_count: int
    average_cluster_size: float
    largest_cluster_size: int
    smallest_cluster_size: int
    coherence_average: float | None
    coherence_min: float | None
    coherence_max: float | None
    tiny_cluster_count: int
    notes: str
    clusters: list[SweepClusterSample]
    sample_clusters: list[SweepClusterSample]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ClusterSweepResult:
    embedding_count: int
    ks: list[int]
    results: list[SweepKResult]
    recommended_k: int | None
    recommendation_note: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_ks(value: str) -> list[int]:
    """Parse a comma-separated list of positive k values."""
    ks = []
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        try:
            k = int(part)
        except ValueError as exc:
            raise ClusterSweepError(f"Invalid k value: {part!r}. Use comma-separated integers.") from exc
        if k < 2:
            raise ClusterSweepError("Cluster k values must be at least 2.")
        ks.append(k)
    if not ks:
        raise ClusterSweepError("At least one cluster k value is required.")
    return sorted(dict.fromkeys(ks))


def sweep_clusters_file(
    *,
    ks: list[int],
    embeddings_path: str | Path = "outputs/intermediate/embeddings.jsonl",
    movies_path: str | Path = "data/processed/movies.json",
    profiles_path: str | Path = "data/processed/profiles.json",
    neighbors_path: str | Path = "outputs/intermediate/neighbors.json",
    output_dir: str | Path = "outputs",
) -> tuple[Path, Path]:
    """Run a local cluster sweep and write JSON plus Markdown report artifacts."""
    resolved_embeddings_path = Path(embeddings_path)
    if not resolved_embeddings_path.exists():
        raise ClusterSweepError(
            f"No embeddings found at {resolved_embeddings_path}. "
            "Run Milestone 2 first; sweep-clusters reuses existing embeddings and does not call OpenAI."
        )

    embeddings = load_embedding_records(resolved_embeddings_path)
    if not embeddings:
        raise ClusterSweepError(
            f"No embedding records found at {resolved_embeddings_path}. "
            "Run Milestone 2 first; sweep-clusters reuses existing embeddings and does not call OpenAI."
        )

    movies = load_movie_records(movies_path)
    profiles = load_profiles(profiles_path)
    neighbors = load_neighbors(neighbors_path) if Path(neighbors_path).exists() else []

    result = sweep_clusters(
        ks=ks,
        embeddings=embeddings,
        movies=movies,
        profiles=profiles,
        neighbors=neighbors,
    )

    output_path = Path(output_dir)
    json_path = output_path / "intermediate" / CLUSTER_SWEEP_JSON_FILENAME
    report_path = output_path / "reports" / CLUSTER_SWEEP_REPORT_FILENAME
    json_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    report_path.write_text(render_cluster_sweep_report(result), encoding="utf-8")
    return json_path, report_path


def sweep_clusters(
    *,
    ks: list[int],
    embeddings: list[Any],
    movies: list[MovieRecord],
    profiles: list[SemanticProfile],
    neighbors: list[MovieNeighbors] | None = None,
) -> ClusterSweepResult:
    """Compute sweep metrics for multiple k-means granularities."""
    if not embeddings:
        raise ClusterSweepError("At least one embedding record is required for a cluster sweep.")

    normalized_ks = sorted(dict.fromkeys(ks))
    results = [
        _sweep_one_k(
            requested_k=k,
            embeddings=embeddings,
            movies=movies,
            profiles=profiles,
            neighbors=neighbors or [],
        )
        for k in normalized_ks
    ]
    recommended = _recommend_k(results)
    recommended_k = recommended.requested_k if recommended else None
    recommendation_note = (
        f"Use k={recommended.requested_k} for Milestone 3 labeling: {recommended.notes}"
        if recommended
        else "No cluster count could be recommended from this sweep."
    )
    return ClusterSweepResult(
        embedding_count=len(embeddings),
        ks=normalized_ks,
        results=results,
        recommended_k=recommended_k,
        recommendation_note=recommendation_note,
    )


def render_cluster_sweep_report(result: ClusterSweepResult) -> str:
    """Render the cluster sweep report as Markdown."""
    lines = [
        "# The Film Atlas - Milestone 2.5 Cluster Sweep Report",
        "",
        "This report compares local k-means granularities over existing embeddings. "
        "It does not call OpenAI, generate final AI labels, export public JSON, or touch frontend code.",
        "",
        "## Summary",
        "",
        f"- Embedded movies inspected: {result.embedding_count}",
        f"- k values tested: {', '.join(map(str, result.ks))}",
        f"- Recommended k: {result.recommended_k if result.recommended_k is not None else 'n/a'}",
        f"- Recommendation note: {result.recommendation_note}",
        "",
        "## Sweep Metrics",
        "",
        "| k | Clusters | Avg Size | Largest | Smallest | Coherence Avg | Coherence Range | Tiny <5 | Notes |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |",
    ]
    for item in result.results:
        lines.append(
            (
                f"| {item.requested_k} | {item.cluster_count} | "
                f"{item.average_cluster_size:.1f} | {item.largest_cluster_size} | "
                f"{item.smallest_cluster_size} | {_fmt_float(item.coherence_average)} | "
                f"{_fmt_range(item.coherence_min, item.coherence_max)} | "
                f"{item.tiny_cluster_count} | {item.notes} |"
            )
        )

    lines.extend(["", "## Sample Clusters", ""])
    for item in result.results:
        lines.extend(
            [
                f"### k={item.requested_k}",
                "",
                _sample_cluster_table(item.sample_clusters),
                "",
            ]
        )

    lines.extend(
        [
            "## Recommendation For Milestone 3",
            "",
            result.recommendation_note,
            "",
        ]
    )
    return "\n".join(lines)


def _sweep_one_k(
    *,
    requested_k: int,
    embeddings: list[Any],
    movies: list[MovieRecord],
    profiles: list[SemanticProfile],
    neighbors: list[MovieNeighbors],
) -> SweepKResult:
    assignments = cluster_embedding_records(embeddings, n_clusters=requested_k)
    evidence = build_cluster_evidence(
        movies=movies,
        profiles=profiles,
        embeddings=embeddings,
        assignments=assignments,
        neighbors=neighbors,
    )
    sizes = Counter(assignment.cluster_id for assignment in assignments if assignment.cluster_id >= 0)
    coherence_values = [
        cluster.coherence_score for cluster in evidence if cluster.coherence_score is not None
    ]
    cluster_count = len(sizes)
    average_size = len(assignments) / cluster_count if cluster_count else 0
    largest = max(sizes.values(), default=0)
    smallest = min(sizes.values(), default=0)
    tiny_count = sum(1 for size in sizes.values() if size < 5)
    coherence_average = (
        sum(coherence_values) / len(coherence_values) if coherence_values else None
    )
    notes = _granularity_note(
        average_size=average_size,
        largest_size=largest,
        tiny_count=tiny_count,
        cluster_count=cluster_count,
        coherence_average=coherence_average,
    )
    return SweepKResult(
        requested_k=requested_k,
        cluster_count=cluster_count,
        movie_count=len(assignments),
        average_cluster_size=average_size,
        largest_cluster_size=largest,
        smallest_cluster_size=smallest,
        coherence_average=coherence_average,
        coherence_min=min(coherence_values, default=None),
        coherence_max=max(coherence_values, default=None),
        tiny_cluster_count=tiny_count,
        notes=notes,
        clusters=_cluster_samples(evidence),
        sample_clusters=_cluster_samples(evidence, limit=6),
    )


def _cluster_samples(
    evidence: list[ClusterEvidence],
    *,
    limit: int | None = None,
) -> list[SweepClusterSample]:
    ranked = sorted(evidence, key=lambda item: item.cluster_size, reverse=True)
    if limit is not None:
        ranked = ranked[:limit]
    return [
        SweepClusterSample(
            cluster_id=cluster.cluster_id,
            cluster_size=cluster.cluster_size,
            representative_movies=cluster.representative_movies[:6],
            top_genres=cluster.top_official_genres[:5],
            top_keywords=cluster.top_tmdb_keywords[:6],
            top_terms=[term for term, _score in cluster.aggregated_profile_terms[:8]],
            coherence_score=cluster.coherence_score,
        )
        for cluster in ranked
    ]


def _granularity_note(
    *,
    average_size: float,
    largest_size: int,
    tiny_count: int,
    cluster_count: int,
    coherence_average: float | None,
) -> str:
    tiny_ratio = tiny_count / cluster_count if cluster_count else 0
    if average_size > 25 or largest_size > 60:
        return "Too broad for final microgenre labels; useful as a baseline."
    if tiny_ratio > 0.2:
        return "Too fragmented; many clusters have fewer than five movies."
    if coherence_average is not None and coherence_average < 0.4:
        return "Weak coherence; inspect sample clusters before labeling."
    if average_size < 8:
        return "Fine-grained and potentially fragmented; useful if samples still look stable."
    return "Promising granularity for Milestone 3 microgenre labeling."


def _recommend_k(results: list[SweepKResult]) -> SweepKResult | None:
    if not results:
        return None

    def score(item: SweepKResult) -> tuple[float, int]:
        tiny_ratio = item.tiny_cluster_count / item.cluster_count if item.cluster_count else 1
        coherence = item.coherence_average or 0
        size_penalty = abs(item.average_cluster_size - 14) / 14
        largest_penalty = max(0, item.largest_cluster_size - 40) / 40
        tiny_penalty = tiny_ratio * 2
        value = coherence - size_penalty - largest_penalty - tiny_penalty
        return (value, item.requested_k)

    return max(results, key=score)


def _sample_cluster_table(samples: list[SweepClusterSample]) -> str:
    if not samples:
        return "_No sample clusters available._"
    lines = [
        "| Cluster | Size | Coherence | Representatives | Evidence terms |",
        "| ---: | ---: | ---: | --- | --- |",
    ]
    for sample in samples:
        representatives = ", ".join(sample.representative_movies)
        terms = ", ".join(sample.top_terms)
        lines.append(
            f"| {sample.cluster_id} | {sample.cluster_size} | "
            f"{_fmt_float(sample.coherence_score)} | {representatives} | {terms} |"
        )
    return "\n".join(lines)


def _fmt_float(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _fmt_range(low: float | None, high: float | None) -> str:
    if low is None or high is None:
        return "n/a"
    return f"{low:.3f}-{high:.3f}"
