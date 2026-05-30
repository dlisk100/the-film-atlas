"""Generate the Milestone 2 semantic-neighborhood inspection report."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from film_atlas.cluster import load_cluster_assignments
from film_atlas.embedding import load_embedding_records
from film_atlas.inspect_clusters import ClusterEvidence, load_cluster_evidence
from film_atlas.neighbors import MovieNeighbors, load_neighbors
from film_atlas.profiles import load_profiles

MILESTONE_2_REPORT_FILENAME = "milestone_2_report.md"
QUALITY_CHECK_MOVIES = [
    "No Country for Old Men",
    "The Social Network",
    "Mean Girls",
    "Her",
    "Get Out",
    "The Matrix",
    "Before Sunrise",
    "The Big Short",
    "Mad Max: Fury Road",
    "Lost in Translation",
    "The Devil Wears Prada",
    "Whiplash",
    "Nightcrawler",
    "Paddington 2",
    "The Godfather",
    "Pulp Fiction",
    "The Shawshank Redemption",
    "Interstellar",
    "The Dark Knight",
]


def generate_milestone_2_report_file(
    *,
    profiles_path: str | Path = "data/processed/profiles.json",
    embeddings_path: str | Path = "outputs/intermediate/embeddings.jsonl",
    manifest_path: str | Path = "outputs/intermediate/embedding_manifest.json",
    assignments_path: str | Path = "outputs/intermediate/cluster_assignments.json",
    neighbors_path: str | Path = "outputs/intermediate/neighbors.json",
    evidence_path: str | Path = "outputs/intermediate/cluster_evidence.json",
    output_dir: str | Path = "outputs",
) -> Path:
    """Write outputs/reports/milestone_2_report.md."""
    profiles = load_profiles(profiles_path)
    embeddings = load_embedding_records(embeddings_path)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    assignments = load_cluster_assignments(assignments_path)
    neighbors = load_neighbors(neighbors_path)
    evidence = load_cluster_evidence(evidence_path)
    path = Path(output_dir) / "reports" / MILESTONE_2_REPORT_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        render_milestone_2_report(
            profile_count=len(profiles),
            embedded_count=len(embeddings),
            manifest=manifest,
            assignments=assignments,
            neighbors=neighbors,
            evidence=evidence,
        ),
        encoding="utf-8",
    )
    return path


def render_milestone_2_report(
    *,
    profile_count: int,
    embedded_count: int,
    manifest: dict[str, Any],
    assignments: list[Any],
    neighbors: list[MovieNeighbors],
    evidence: list[ClusterEvidence],
) -> str:
    """Render the Milestone 2 report."""
    cluster_counter = Counter(
        assignment.cluster_id for assignment in assignments if assignment.cluster_id >= 0
    )
    outliers = sum(1 for assignment in assignments if assignment.cluster_id < 0)
    cluster_count = len(cluster_counter)
    lines = [
        "# The Film Atlas - Milestone 2 Semantic Neighborhood Report",
        "",
        "Milestone 2 uses OpenAI embeddings, local projection, local clustering, "
        "and local inspection only. It does not generate final AI microgenre labels, "
        "public website JSON, or frontend integration.",
        "",
        "## Summary",
        "",
        f"- Profiles available: {profile_count}",
        f"- Movies embedded: {embedded_count}",
        f"- Embedding model: {manifest.get('model')}",
        f"- Estimated tokens: {manifest.get('estimated_tokens')}",
        f"- Estimated cost: ${float(manifest.get('estimated_cost_usd') or 0):.4f}",
        f"- Cached embeddings reused: {manifest.get('cached_reused_count')}",
        f"- New embeddings generated: {manifest.get('new_embedding_count')}",
        "- Projection method: pca",
        "- Clustering method: kmeans",
        f"- Cluster count: {cluster_count}",
        f"- Outliers: {outliers} ({_percent(outliers, len(assignments))})",
        "",
        "## Cluster Size Distribution",
        "",
        _counter_table(cluster_counter, "Cluster ID"),
        "",
        "## Sample Nearest Neighbors",
        "",
        _sample_neighbors(neighbors, limit=20),
        "",
        "## Quality-Check Movie Neighbors",
        "",
        _quality_check_neighbors(neighbors),
        "",
        "## Sample Clusters",
        "",
        _sample_clusters(evidence, limit=10),
        "",
        "## Warnings",
        "",
        _warnings(profile_count, embedded_count, evidence),
        "",
        "## Recommendation For Milestone 3",
        "",
        "Review the sample clusters and quality-check neighbors. If the neighborhoods feel "
        "semantically coherent, proceed to Milestone 3 with human-guided cluster naming or "
        "mocked label prompts before any final public export.",
        "",
    ]
    return "\n".join(lines)


def _counter_table(counter: Counter[Any], label: str) -> str:
    if not counter:
        return "_No data available._"
    lines = [f"| {label} | Count |", "| --- | ---: |"]
    for key, count in counter.most_common():
        lines.append(f"| {key} | {count} |")
    return "\n".join(lines)


def _sample_neighbors(neighbors: list[MovieNeighbors], *, limit: int) -> str:
    lines = ["| Movie | Neighbor | Similarity |", "| --- | --- | ---: |"]
    count = 0
    for entry in neighbors:
        for neighbor in entry.neighbors[:1]:
            lines.append(f"| {entry.title} | {neighbor.title} | {neighbor.similarity:.3f} |")
            count += 1
            if count >= limit:
                return "\n".join(lines)
    return "\n".join(lines) if count else "_No neighbors available._"


def _quality_check_neighbors(neighbors: list[MovieNeighbors]) -> str:
    by_title = {entry.title.lower(): entry for entry in neighbors}
    sections = []
    for title in QUALITY_CHECK_MOVIES:
        entry = by_title.get(title.lower())
        if not entry:
            continue
        neighbor_text = ", ".join(
            f"{neighbor.title} ({neighbor.similarity:.3f})" for neighbor in entry.neighbors[:5]
        )
        sections.append(f"- {entry.title}: {neighbor_text}")
    return "\n".join(sections) if sections else "_None of the quality-check movies are present._"


def _sample_clusters(evidence: list[ClusterEvidence], *, limit: int) -> str:
    if not evidence:
        return "_No cluster evidence available._"
    sections = []
    for cluster in sorted(evidence, key=lambda item: item.cluster_size, reverse=True)[:limit]:
        genres = ", ".join(f"{name} ({count})" for name, count in cluster.top_official_genres[:6])
        keywords = ", ".join(f"{name} ({count})" for name, count in cluster.top_tmdb_keywords[:8])
        terms = ", ".join(term for term, _score in cluster.aggregated_profile_terms[:10])
        representatives = ", ".join(cluster.representative_movies[:8])
        coherence = (
            f"{cluster.coherence_score:.3f}" if cluster.coherence_score is not None else "n/a"
        )
        sections.append(
            "\n".join(
                [
                    f"### Cluster {cluster.cluster_id} ({cluster.cluster_size} movies)",
                    "",
                    f"- Representative movies: {representatives}",
                    f"- Top official genres: {genres}",
                    f"- Top TMDb keywords: {keywords}",
                    f"- Aggregated profile terms: {terms}",
                    f"- Coherence score: {coherence}",
                    f"- Warnings: {', '.join(cluster.warnings) if cluster.warnings else 'none'}",
                ]
            )
        )
    return "\n\n".join(sections)


def _warnings(
    profile_count: int,
    embedded_count: int,
    evidence: list[ClusterEvidence],
) -> str:
    warnings = []
    if embedded_count < profile_count:
        warnings.append("Only a subset of available profiles was embedded for this run.")
    small_clusters = [cluster.cluster_id for cluster in evidence if cluster.cluster_size < 3]
    if small_clusters:
        warnings.append(f"Small clusters present: {', '.join(map(str, small_clusters[:10]))}.")
    noisy_clusters = [cluster.cluster_id for cluster in evidence if cluster.warnings]
    if noisy_clusters:
        warnings.append(f"Cluster evidence warnings present: {', '.join(map(str, noisy_clusters[:10]))}.")
    return "\n".join(f"- Warning: {warning}" for warning in warnings) if warnings else "_No major warnings._"


def _percent(value: int, total: int) -> str:
    if total == 0:
        return "0.0%"
    return f"{value / total * 100:.1f}%"
