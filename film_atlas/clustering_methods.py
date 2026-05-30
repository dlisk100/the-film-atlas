"""Compare clustering methods over existing full embedding vectors."""

from __future__ import annotations

import json
import statistics
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from film_atlas.cluster import ClusterAssignment, cluster_embedding_records
from film_atlas.embedding import load_embedding_records
from film_atlas.inspect_clusters import ClusterEvidence, build_cluster_evidence
from film_atlas.milestone2_report import QUALITY_CHECK_MOVIES
from film_atlas.models import MovieRecord, SemanticProfile
from film_atlas.normalize import load_movie_records
from film_atlas.profiles import load_profiles

METHOD_COMPARISON_JSON_FILENAME = "clustering_method_comparison.json"
METHOD_COMPARISON_REPORT_FILENAME = "clustering_method_comparison.md"
FULL_EMBEDDING_INPUT_SPACE = "full_embedding_vectors"


class ClusteringMethodComparisonError(RuntimeError):
    """Raised when method comparison cannot run locally."""


@dataclass(frozen=True, slots=True)
class MovieClusterAssignment:
    tmdb_id: int
    title: str
    cluster_id: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class MethodClusterSample:
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
class QualityCheckClusterAssignment:
    title: str
    present: bool
    cluster_id: int | None
    cluster_size: int | None
    representative_movies: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ClusteringMethodResult:
    method: str
    status: str
    note: str
    input_space: str
    movie_count: int
    cluster_count: int
    average_cluster_size: float | None
    median_cluster_size: float | None
    largest_cluster_size: int | None
    smallest_cluster_size: int | None
    tiny_cluster_count: int
    outlier_count: int
    coherence_average: float | None
    coherence_min: float | None
    coherence_max: float | None
    silhouette_score: float | None
    labelability_score: float | None
    assignments: list[MovieClusterAssignment]
    clusters: list[MethodClusterSample]
    sample_clusters: list[MethodClusterSample]
    quality_check_assignments: list[QualityCheckClusterAssignment]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ClusteringMethodComparison:
    embedding_count: int
    methods_requested: list[str]
    input_space: str
    current_clustering_check: str
    results: list[ClusteringMethodResult]
    recommended_method: str | None
    recommendation_note: str
    milestone_3_readiness: str
    ablation_recommendation: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_methods(value: str) -> list[str]:
    """Parse a comma-separated method list."""
    valid = {"kmeans", "agglomerative", "graph", "hdbscan"}
    methods = []
    for raw_part in value.split(","):
        method = raw_part.strip().lower()
        if not method:
            continue
        if method not in valid:
            raise ClusteringMethodComparisonError(
                f"Unknown clustering method {method!r}. Choose from: {', '.join(sorted(valid))}."
            )
        methods.append(method)
    if not methods:
        raise ClusteringMethodComparisonError("At least one clustering method is required.")
    return list(dict.fromkeys(methods))


def compare_clustering_methods_file(
    *,
    methods: list[str],
    embeddings_path: str | Path = "outputs/intermediate/embeddings.jsonl",
    movies_path: str | Path = "data/processed/movies.json",
    profiles_path: str | Path = "data/processed/profiles.json",
    output_dir: str | Path = "outputs",
    kmeans_k: int = 35,
    agglomerative_k: int = 35,
    graph_neighbors: int = 10,
    min_cluster_size: int = 5,
    limit: int | None = None,
) -> tuple[Path, Path]:
    """Compare clustering methods and write JSON plus Markdown artifacts."""
    resolved_embeddings_path = Path(embeddings_path)
    if not resolved_embeddings_path.exists():
        raise ClusteringMethodComparisonError(
            f"No embeddings found at {resolved_embeddings_path}. "
            "Run Milestone 2 first; compare-clustering-methods reuses existing embeddings "
            "and does not call OpenAI."
        )

    embeddings = load_embedding_records(resolved_embeddings_path)
    if limit is not None:
        embeddings = embeddings[:limit]
    if not embeddings:
        raise ClusteringMethodComparisonError(
            f"No embedding records found at {resolved_embeddings_path}. "
            "Run Milestone 2 first; compare-clustering-methods reuses existing embeddings "
            "and does not call OpenAI."
        )

    movies = load_movie_records(movies_path)
    profiles = load_profiles(profiles_path)
    comparison = compare_clustering_methods(
        methods=methods,
        embeddings=embeddings,
        movies=movies,
        profiles=profiles,
        kmeans_k=kmeans_k,
        agglomerative_k=agglomerative_k,
        graph_neighbors=graph_neighbors,
        min_cluster_size=min_cluster_size,
    )

    output_path = Path(output_dir)
    json_path = output_path / "intermediate" / METHOD_COMPARISON_JSON_FILENAME
    report_path = output_path / "reports" / METHOD_COMPARISON_REPORT_FILENAME
    json_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(comparison.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    report_path.write_text(render_clustering_method_comparison(comparison), encoding="utf-8")
    return json_path, report_path


def compare_clustering_methods(
    *,
    methods: list[str],
    embeddings: list[Any],
    movies: list[MovieRecord],
    profiles: list[SemanticProfile],
    kmeans_k: int = 35,
    agglomerative_k: int = 35,
    graph_neighbors: int = 10,
    min_cluster_size: int = 5,
) -> ClusteringMethodComparison:
    """Compare requested methods on full embedding vectors."""
    if not embeddings:
        raise ClusteringMethodComparisonError(
            "At least one embedding record is required for method comparison."
        )

    normalized_methods = parse_methods(",".join(methods))
    matrix = _embedding_matrix(embeddings)
    results = [
        _run_method(
            method=method,
            embeddings=embeddings,
            movies=movies,
            profiles=profiles,
            matrix=matrix,
            kmeans_k=kmeans_k,
            agglomerative_k=agglomerative_k,
            graph_neighbors=graph_neighbors,
            min_cluster_size=min_cluster_size,
        )
        for method in normalized_methods
    ]
    recommended = _recommended_method(results)
    recommendation_note = _recommendation_note(recommended)
    return ClusteringMethodComparison(
        embedding_count=len(embeddings),
        methods_requested=normalized_methods,
        input_space=FULL_EMBEDDING_INPUT_SPACE,
        current_clustering_check=(
            "Current clustering uses normalized full embedding vectors from embeddings.jsonl. "
            "PCA/2D coordinates are produced separately for visualization only."
        ),
        results=results,
        recommended_method=recommended.method if recommended else None,
        recommendation_note=recommendation_note,
        milestone_3_readiness=_milestone_3_readiness(recommended),
        ablation_recommendation=(
            "Recommended after method selection: run a small profile ablation or review-weight "
            "experiment to verify that reviews add useful vibe signal without adding noise."
        ),
    )


def render_clustering_method_comparison(comparison: ClusteringMethodComparison) -> str:
    """Render the method comparison report."""
    lines = [
        "# The Film Atlas - Milestone 2.6 Clustering Method Comparison",
        "",
        "This report compares local clustering methods over existing embeddings. It does not call "
        "OpenAI, re-embed profiles, generate final AI labels, export public website JSON, scrape "
        "websites, or touch frontend code.",
        "",
        "## Critical Input-Space Check",
        "",
        f"- Method comparison input space: {comparison.input_space}",
        f"- Current clustering check: {comparison.current_clustering_check}",
        "- Best-practice alignment: nearest neighbors use full embeddings; clustering uses full "
        "embeddings; PCA/2D projection is visualization only.",
        "",
        "## Summary",
        "",
        f"- Embedded movies inspected: {comparison.embedding_count}",
        f"- Methods requested: {', '.join(comparison.methods_requested)}",
        f"- Recommended method: {comparison.recommended_method or 'n/a'}",
        f"- Recommendation note: {comparison.recommendation_note}",
        f"- Milestone 3 readiness: {comparison.milestone_3_readiness}",
        f"- Profile ablation recommendation: {comparison.ablation_recommendation}",
        "",
        "## Method Metrics",
        "",
        "| Method | Status | Clusters | Avg | Median | Largest | Smallest | Tiny <5 | Outliers | Coherence Avg | Coherence Range | Silhouette | Score | Notes |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |",
    ]
    for result in comparison.results:
        lines.append(
            f"| {result.method} | {result.status} | {result.cluster_count} | "
            f"{_fmt_float(result.average_cluster_size)} | {_fmt_float(result.median_cluster_size)} | "
            f"{_fmt_int(result.largest_cluster_size)} | {_fmt_int(result.smallest_cluster_size)} | "
            f"{result.tiny_cluster_count} | {result.outlier_count} | "
            f"{_fmt_float(result.coherence_average)} | "
            f"{_fmt_range(result.coherence_min, result.coherence_max)} | "
            f"{_fmt_float(result.silhouette_score)} | {_fmt_float(result.labelability_score)} | "
            f"{_escape_table(result.note)} |"
        )

    lines.extend(
        [
            "",
            "## Direct Answers",
            "",
            _direct_answers(comparison),
            "",
            "## Quality-Check Movie Cluster Assignments",
            "",
        ]
    )
    for result in comparison.results:
        lines.extend([f"### {result.method}", "", _quality_check_table(result), ""])

    lines.extend(["## Sample Clusters", ""])
    for result in comparison.results:
        lines.extend([f"### {result.method}", "", _sample_cluster_table(result.sample_clusters), ""])

    lines.extend(["## Recommendation For Milestone 3", "", comparison.recommendation_note, ""])
    return "\n".join(lines)


def _run_method(
    *,
    method: str,
    embeddings: list[Any],
    movies: list[MovieRecord],
    profiles: list[SemanticProfile],
    matrix: np.ndarray,
    kmeans_k: int,
    agglomerative_k: int,
    graph_neighbors: int,
    min_cluster_size: int,
) -> ClusteringMethodResult:
    try:
        assignments = _assignments_for_method(
            method=method,
            embeddings=embeddings,
            matrix=matrix,
            kmeans_k=kmeans_k,
            agglomerative_k=agglomerative_k,
            graph_neighbors=graph_neighbors,
            min_cluster_size=min_cluster_size,
        )
    except _SkippedMethod as exc:
        return _skipped_result(method, str(exc), len(embeddings))

    return _evaluate_method(
        method=method,
        note=_method_note(method),
        embeddings=embeddings,
        movies=movies,
        profiles=profiles,
        matrix=matrix,
        assignments=assignments,
    )


def _assignments_for_method(
    *,
    method: str,
    embeddings: list[Any],
    matrix: np.ndarray,
    kmeans_k: int,
    agglomerative_k: int,
    graph_neighbors: int,
    min_cluster_size: int,
) -> list[ClusterAssignment]:
    if method == "kmeans":
        return cluster_embedding_records(embeddings, n_clusters=kmeans_k)
    if method == "agglomerative":
        return _agglomerative_assignments(embeddings, matrix, n_clusters=agglomerative_k)
    if method == "graph":
        return _graph_assignments(embeddings, matrix, graph_neighbors=graph_neighbors)
    if method == "hdbscan":
        return _hdbscan_assignments(embeddings, matrix, min_cluster_size=min_cluster_size)
    raise ClusteringMethodComparisonError(f"Unsupported method: {method}")


def _agglomerative_assignments(
    embeddings: list[Any],
    matrix: np.ndarray,
    *,
    n_clusters: int,
) -> list[ClusterAssignment]:
    if len(embeddings) == 1:
        return [ClusterAssignment(embeddings[0].tmdb_id, embeddings[0].title, 0)]
    cluster_count = min(max(2, n_clusters), len(embeddings))
    model = AgglomerativeClustering(
        n_clusters=cluster_count,
        metric="cosine",
        linkage="average",
    )
    labels = model.fit_predict(matrix)
    return _labels_to_assignments(embeddings, labels)


def _graph_assignments(
    embeddings: list[Any],
    matrix: np.ndarray,
    *,
    graph_neighbors: int,
) -> list[ClusterAssignment]:
    if len(embeddings) == 1:
        return [ClusterAssignment(embeddings[0].tmdb_id, embeddings[0].title, 0)]
    similarities = cosine_similarity(matrix)
    graph = nx.Graph()
    graph.add_nodes_from(range(len(embeddings)))
    neighbor_count = min(max(1, graph_neighbors), len(embeddings) - 1)
    for source_index in range(len(embeddings)):
        ranked = np.argsort(similarities[source_index])[::-1]
        added = 0
        for neighbor_index in ranked:
            if int(neighbor_index) == source_index:
                continue
            graph.add_edge(
                source_index,
                int(neighbor_index),
                weight=float(similarities[source_index, int(neighbor_index)]),
            )
            added += 1
            if added >= neighbor_count:
                break

    if hasattr(nx.community, "louvain_communities"):
        communities = nx.community.louvain_communities(graph, weight="weight", seed=42)
    else:
        communities = nx.community.greedy_modularity_communities(graph, weight="weight")
    labels = _communities_to_labels(communities, len(embeddings))
    return _labels_to_assignments(embeddings, labels)


def _hdbscan_assignments(
    embeddings: list[Any],
    matrix: np.ndarray,
    *,
    min_cluster_size: int,
) -> list[ClusterAssignment]:
    try:
        import hdbscan  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise _SkippedMethod(
            "Skipped because hdbscan is not installed. It is an optional compiled dependency, "
            "so Milestone 2.6 does not spend time fighting setup."
        ) from exc

    model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean")
    labels = model.fit_predict(matrix)
    return _labels_to_assignments(embeddings, labels)


def _evaluate_method(
    *,
    method: str,
    note: str,
    embeddings: list[Any],
    movies: list[MovieRecord],
    profiles: list[SemanticProfile],
    matrix: np.ndarray,
    assignments: list[ClusterAssignment],
) -> ClusteringMethodResult:
    inlier_assignments = [assignment for assignment in assignments if assignment.cluster_id >= 0]
    evidence = build_cluster_evidence(
        movies=movies,
        profiles=profiles,
        embeddings=embeddings,
        assignments=inlier_assignments,
        neighbors=[],
    )
    sizes = Counter(assignment.cluster_id for assignment in inlier_assignments)
    size_values = list(sizes.values())
    coherence_values = [
        cluster.coherence_score for cluster in evidence if cluster.coherence_score is not None
    ]
    outlier_count = sum(1 for assignment in assignments if assignment.cluster_id < 0)
    cluster_count = len(sizes)
    average_size = statistics.mean(size_values) if size_values else None
    median_size = statistics.median(size_values) if size_values else None
    tiny_count = sum(1 for size in size_values if size < 5)
    silhouette = _silhouette(matrix, assignments)
    labelability_score = _labelability_score(
        average_size=average_size,
        largest_size=max(size_values, default=0),
        tiny_count=tiny_count,
        cluster_count=cluster_count,
        outlier_count=outlier_count,
        movie_count=len(assignments),
        coherence_average=statistics.mean(coherence_values) if coherence_values else None,
        silhouette=silhouette,
    )
    clusters = _cluster_samples(evidence)
    sample_clusters = clusters[:8]
    quality_assignments = _quality_check_assignments(assignments, evidence)
    return ClusteringMethodResult(
        method=method,
        status="completed",
        note=note,
        input_space=FULL_EMBEDDING_INPUT_SPACE,
        movie_count=len(assignments),
        cluster_count=cluster_count,
        average_cluster_size=float(average_size) if average_size is not None else None,
        median_cluster_size=float(median_size) if median_size is not None else None,
        largest_cluster_size=max(size_values, default=None),
        smallest_cluster_size=min(size_values, default=None),
        tiny_cluster_count=tiny_count,
        outlier_count=outlier_count,
        coherence_average=statistics.mean(coherence_values) if coherence_values else None,
        coherence_min=min(coherence_values, default=None),
        coherence_max=max(coherence_values, default=None),
        silhouette_score=silhouette,
        labelability_score=labelability_score,
        assignments=[
            MovieClusterAssignment(
                assignment.tmdb_id,
                assignment.title,
                assignment.cluster_id,
            )
            for assignment in assignments
        ],
        clusters=clusters,
        sample_clusters=sample_clusters,
        quality_check_assignments=quality_assignments,
    )


def _skipped_result(method: str, note: str, movie_count: int) -> ClusteringMethodResult:
    return ClusteringMethodResult(
        method=method,
        status="skipped",
        note=note,
        input_space=FULL_EMBEDDING_INPUT_SPACE,
        movie_count=movie_count,
        cluster_count=0,
        average_cluster_size=None,
        median_cluster_size=None,
        largest_cluster_size=None,
        smallest_cluster_size=None,
        tiny_cluster_count=0,
        outlier_count=0,
        coherence_average=None,
        coherence_min=None,
        coherence_max=None,
        silhouette_score=None,
        labelability_score=None,
        assignments=[],
        clusters=[],
        sample_clusters=[],
        quality_check_assignments=[],
    )


def _embedding_matrix(embeddings: list[Any]) -> np.ndarray:
    return normalize(np.array([record.embedding for record in embeddings], dtype=float))


def _labels_to_assignments(embeddings: list[Any], labels: Any) -> list[ClusterAssignment]:
    return [
        ClusterAssignment(
            tmdb_id=record.tmdb_id,
            title=record.title,
            cluster_id=int(labels[index]),
        )
        for index, record in enumerate(embeddings)
    ]


def _communities_to_labels(communities: Any, count: int) -> list[int]:
    labels = [-1] * count
    ordered = sorted((sorted(community) for community in communities), key=lambda group: (-len(group), group))
    for cluster_id, community in enumerate(ordered):
        for index in community:
            labels[int(index)] = cluster_id
    return labels


def _cluster_samples(evidence: list[ClusterEvidence]) -> list[MethodClusterSample]:
    return [
        MethodClusterSample(
            cluster_id=cluster.cluster_id,
            cluster_size=cluster.cluster_size,
            representative_movies=cluster.representative_movies[:8],
            top_genres=cluster.top_official_genres[:6],
            top_keywords=cluster.top_tmdb_keywords[:8],
            top_terms=[term for term, _score in cluster.aggregated_profile_terms[:10]],
            coherence_score=cluster.coherence_score,
        )
        for cluster in sorted(evidence, key=lambda item: item.cluster_size, reverse=True)
    ]


def _quality_check_assignments(
    assignments: list[ClusterAssignment],
    evidence: list[ClusterEvidence],
) -> list[QualityCheckClusterAssignment]:
    assignments_by_title = {assignment.title.lower(): assignment for assignment in assignments}
    evidence_by_id = {cluster.cluster_id: cluster for cluster in evidence}
    output = []
    for title in QUALITY_CHECK_MOVIES:
        assignment = assignments_by_title.get(title.lower())
        if assignment is None:
            output.append(QualityCheckClusterAssignment(title, False, None, None, []))
            continue
        cluster = evidence_by_id.get(assignment.cluster_id)
        output.append(
            QualityCheckClusterAssignment(
                title=title,
                present=True,
                cluster_id=assignment.cluster_id,
                cluster_size=cluster.cluster_size if cluster else None,
                representative_movies=cluster.representative_movies[:6] if cluster else [],
            )
        )
    return output


def _silhouette(matrix: np.ndarray, assignments: list[ClusterAssignment]) -> float | None:
    inlier_indexes = [
        index for index, assignment in enumerate(assignments) if assignment.cluster_id >= 0
    ]
    if len(inlier_indexes) < 3:
        return None
    labels = np.array([assignments[index].cluster_id for index in inlier_indexes])
    unique_labels = set(labels.tolist())
    if len(unique_labels) < 2 or len(unique_labels) >= len(inlier_indexes):
        return None
    try:
        return float(silhouette_score(matrix[inlier_indexes], labels, metric="cosine"))
    except ValueError:
        return None


def _labelability_score(
    *,
    average_size: float | None,
    largest_size: int,
    tiny_count: int,
    cluster_count: int,
    outlier_count: int,
    movie_count: int,
    coherence_average: float | None,
    silhouette: float | None,
) -> float | None:
    if average_size is None or cluster_count == 0:
        return None
    tiny_ratio = tiny_count / cluster_count
    outlier_ratio = outlier_count / movie_count if movie_count else 0
    size_penalty = abs(average_size - 14) / 14
    broad_penalty = max(0, largest_size - 45) / 45
    score = (coherence_average or 0) + ((silhouette or 0) * 0.5)
    score -= size_penalty + broad_penalty + tiny_ratio * 1.5 + outlier_ratio * 2
    return float(score)


def _recommended_method(results: list[ClusteringMethodResult]) -> ClusteringMethodResult | None:
    completed = [
        result
        for result in results
        if result.status == "completed" and result.labelability_score is not None
    ]
    if not completed:
        return None
    return max(completed, key=lambda result: result.labelability_score or -999)


def _recommendation_note(result: ClusteringMethodResult | None) -> str:
    if result is None:
        return "No completed method had enough evidence to recommend for Milestone 3."
    return (
        f"{result.method} appears most labelable: {result.cluster_count} clusters, "
        f"average size {_fmt_float(result.average_cluster_size)}, largest "
        f"{_fmt_int(result.largest_cluster_size)}, {result.tiny_cluster_count} tiny clusters, "
        f"and coherence average {_fmt_float(result.coherence_average)}."
    )


def _milestone_3_readiness(result: ClusteringMethodResult | None) -> str:
    if result is None:
        return "Not ready; no completed clustering method could be recommended."
    if result.cluster_count < 10 or (result.largest_cluster_size or 0) > 70:
        return "Not quite ready; neighborhoods still look too broad for final labeling."
    if result.cluster_count > 80 or result.tiny_cluster_count > result.cluster_count * 0.25:
        return "Not quite ready; neighborhoods look too fragmented for stable labels."
    return "Sufficient for a Milestone 3 labeling pass, with human review of edge clusters."


def _method_note(method: str) -> str:
    if method == "kmeans":
        return "Fixed-k baseline on normalized full embedding vectors."
    if method == "agglomerative":
        return "Average-linkage agglomerative clustering with cosine distance."
    if method == "graph":
        return "k-nearest-neighbor graph over cosine similarities with NetworkX community detection."
    return "Optional density clustering over normalized full embedding vectors."


def _direct_answers(comparison: ClusteringMethodComparison) -> str:
    completed = [result for result in comparison.results if result.status == "completed"]
    best = _recommended_method(comparison.results)
    if not completed:
        return "_No methods completed._"
    avoids_broad = min(
        completed,
        key=lambda result: (
            result.largest_cluster_size or 10_000,
            abs((result.average_cluster_size or 0) - 14),
        ),
    )
    avoids_fragmentation = min(
        completed,
        key=_fragmentation_balance_score,
    )
    preserves_known = max(
        completed,
        key=lambda result: (
            result.coherence_average or 0,
            result.silhouette_score or -1,
        ),
    )
    return "\n".join(
        [
            f"- Most labelable neighborhoods: {best.method if best else 'n/a'}.",
            f"- Best at avoiding overly broad clusters: {avoids_broad.method}.",
            "- Best at avoiding excessive tiny/franchise-only fragmentation: "
            f"{avoids_fragmentation.method}.",
            f"- Best preservation signal for sensible neighborhoods: {preserves_known.method}.",
            f"- Data sufficiency for Milestone 3: {comparison.milestone_3_readiness}",
            f"- Later ablation/review-weight experiment: {comparison.ablation_recommendation}",
        ]
    )


def _quality_check_table(result: ClusteringMethodResult) -> str:
    if result.status != "completed":
        return f"_Method skipped: {result.note}_"
    lines = [
        "| Movie | Present | Cluster | Cluster Size | Cluster Representatives |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for item in result.quality_check_assignments:
        lines.append(
            f"| {_escape_table(item.title)} | {'yes' if item.present else 'no'} | "
            f"{_fmt_int(item.cluster_id)} | {_fmt_int(item.cluster_size)} | "
            f"{_escape_table(', '.join(item.representative_movies))} |"
        )
    return "\n".join(lines)


def _fragmentation_balance_score(result: ClusteringMethodResult) -> float:
    tiny_ratio = result.tiny_cluster_count / result.cluster_count if result.cluster_count else 1
    outlier_ratio = result.outlier_count / result.movie_count if result.movie_count else 1
    average_size = result.average_cluster_size or 0
    largest_size = result.largest_cluster_size or 10_000
    size_penalty = abs(average_size - 14) / 14
    broad_penalty = max(0, largest_size - 45) / 45
    return size_penalty + broad_penalty + tiny_ratio * 1.5 + outlier_ratio * 2


def _sample_cluster_table(samples: list[MethodClusterSample]) -> str:
    if not samples:
        return "_No sample clusters available._"
    lines = [
        "| Cluster | Size | Coherence | Representatives | Genres | Keywords | Terms |",
        "| ---: | ---: | ---: | --- | --- | --- | --- |",
    ]
    for sample in samples:
        genres = ", ".join(f"{name} ({count})" for name, count in sample.top_genres)
        keywords = ", ".join(f"{name} ({count})" for name, count in sample.top_keywords)
        lines.append(
            f"| {sample.cluster_id} | {sample.cluster_size} | "
            f"{_fmt_float(sample.coherence_score)} | "
            f"{_escape_table(', '.join(sample.representative_movies))} | "
            f"{_escape_table(genres)} | {_escape_table(keywords)} | "
            f"{_escape_table(', '.join(sample.top_terms))} |"
        )
    return "\n".join(lines)


def _fmt_float(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _fmt_int(value: int | None) -> str:
    return "n/a" if value is None else str(value)


def _fmt_range(low: float | None, high: float | None) -> str:
    if low is None or high is None:
        return "n/a"
    return f"{low:.3f}-{high:.3f}"


def _escape_table(value: str) -> str:
    return value.replace("|", "\\|")


class _SkippedMethod(RuntimeError):
    """Internal signal for optional methods that should appear as skipped."""
