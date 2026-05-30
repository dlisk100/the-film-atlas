"""Generate cluster-level evidence for Milestone 2 inspection."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from film_atlas.cluster import ClusterAssignment, load_cluster_assignments
from film_atlas.embedding import load_embedding_records
from film_atlas.models import MovieRecord, SemanticProfile
from film_atlas.neighbors import MovieNeighbors, load_neighbors
from film_atlas.normalize import load_movie_records
from film_atlas.profiles import load_profiles

CLUSTER_EVIDENCE_FILENAME = "cluster_evidence.json"
CLUSTER_STOP_WORDS = {
    "genres",
    "keywords",
    "language",
    "overview",
    "profile",
    "review",
    "reviews",
    "title",
    "rating",
    "stars",
    "full",
    "spoiler",
    "http",
    "https",
    "www",
}


@dataclass(frozen=True, slots=True)
class ClusterEvidence:
    cluster_id: int
    cluster_size: int
    representative_movies: list[str]
    top_official_genres: list[tuple[str, int]]
    top_tmdb_keywords: list[tuple[str, int]]
    aggregated_profile_terms: list[tuple[str, float]]
    in_cluster_neighbor_pairs: list[dict[str, Any]]
    coherence_score: float | None
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def inspect_clusters_file(
    *,
    movies_path: str | Path = "data/processed/movies.json",
    profiles_path: str | Path = "data/processed/profiles.json",
    embeddings_path: str | Path = "outputs/intermediate/embeddings.jsonl",
    assignments_path: str | Path = "outputs/intermediate/cluster_assignments.json",
    neighbors_path: str | Path = "outputs/intermediate/neighbors.json",
    output_dir: str | Path = "outputs",
) -> Path:
    """Build cluster evidence and write outputs/intermediate/cluster_evidence.json."""
    evidence = build_cluster_evidence(
        movies=load_movie_records(movies_path),
        profiles=load_profiles(profiles_path),
        embeddings=load_embedding_records(embeddings_path),
        assignments=load_cluster_assignments(assignments_path),
        neighbors=load_neighbors(neighbors_path),
    )
    path = Path(output_dir) / "intermediate" / CLUSTER_EVIDENCE_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([entry.to_dict() for entry in evidence], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path


def build_cluster_evidence(
    *,
    movies: list[MovieRecord],
    profiles: list[SemanticProfile],
    embeddings: list[Any],
    assignments: list[ClusterAssignment],
    neighbors: list[MovieNeighbors],
) -> list[ClusterEvidence]:
    """Compute human-readable evidence for each cluster."""
    movies_by_id = {movie.tmdb_id: movie for movie in movies}
    profiles_by_id = {profile.tmdb_id: profile for profile in profiles}
    embeddings_by_id = {record.tmdb_id: record for record in embeddings}
    neighbors_by_id = {entry.tmdb_id: entry for entry in neighbors}
    assignments_by_cluster: dict[int, list[ClusterAssignment]] = defaultdict(list)
    for assignment in assignments:
        assignments_by_cluster[assignment.cluster_id].append(assignment)

    output = []
    for cluster_id, cluster_assignments in sorted(assignments_by_cluster.items()):
        cluster_ids = [assignment.tmdb_id for assignment in cluster_assignments]
        cluster_movies = [movies_by_id[tmdb_id] for tmdb_id in cluster_ids if tmdb_id in movies_by_id]
        cluster_profiles = [
            profiles_by_id[tmdb_id] for tmdb_id in cluster_ids if tmdb_id in profiles_by_id
        ]
        cluster_embeddings = [
            embeddings_by_id[tmdb_id] for tmdb_id in cluster_ids if tmdb_id in embeddings_by_id
        ]

        terms = _aggregated_profile_terms(cluster_profiles)
        warnings = _cluster_warnings(terms, cluster_movies)
        output.append(
            ClusterEvidence(
                cluster_id=cluster_id,
                cluster_size=len(cluster_assignments),
                representative_movies=_representative_movies(cluster_embeddings),
                top_official_genres=Counter(
                    genre for movie in cluster_movies for genre in movie.genres
                ).most_common(10),
                top_tmdb_keywords=Counter(
                    keyword for movie in cluster_movies for keyword in movie.keywords
                ).most_common(12),
                aggregated_profile_terms=terms,
                in_cluster_neighbor_pairs=_in_cluster_neighbor_pairs(
                    cluster_ids, neighbors_by_id, limit=8
                ),
                coherence_score=_coherence_score(cluster_embeddings),
                warnings=warnings,
            )
        )
    return output


def load_cluster_evidence(
    path: str | Path = "outputs/intermediate/cluster_evidence.json",
) -> list[ClusterEvidence]:
    """Load cluster evidence from JSON."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        ClusterEvidence(
            cluster_id=int(item["cluster_id"]),
            cluster_size=int(item["cluster_size"]),
            representative_movies=list(item.get("representative_movies") or []),
            top_official_genres=[tuple(row) for row in item.get("top_official_genres") or []],
            top_tmdb_keywords=[tuple(row) for row in item.get("top_tmdb_keywords") or []],
            aggregated_profile_terms=[
                (str(row[0]), float(row[1])) for row in item.get("aggregated_profile_terms") or []
            ],
            in_cluster_neighbor_pairs=list(item.get("in_cluster_neighbor_pairs") or []),
            coherence_score=item.get("coherence_score"),
            warnings=list(item.get("warnings") or []),
        )
        for item in payload
    ]


def _representative_movies(embeddings: list[Any], *, limit: int = 8) -> list[str]:
    if not embeddings:
        return []
    matrix = np.array([record.embedding for record in embeddings], dtype=float)
    centroid = matrix.mean(axis=0).reshape(1, -1)
    scores = cosine_similarity(matrix, centroid).ravel()
    ranked = scores.argsort()[::-1][:limit]
    return [embeddings[int(index)].title for index in ranked]


def _aggregated_profile_terms(
    profiles: list[SemanticProfile],
    *,
    limit: int = 15,
) -> list[tuple[str, float]]:
    if not profiles:
        return []
    texts = [profile.profile_text for profile in profiles]
    vectorizer = TfidfVectorizer(
        stop_words=list(ENGLISH_STOP_WORDS.union(CLUSTER_STOP_WORDS)),
        max_features=3000,
        ngram_range=(1, 2),
    )
    matrix = vectorizer.fit_transform(texts)
    values = np.asarray(matrix.mean(axis=0)).ravel()
    names = np.array(vectorizer.get_feature_names_out())
    indexes = values.argsort()[::-1][:limit]
    return [(str(names[index]), float(values[index])) for index in indexes if values[index] > 0]


def _in_cluster_neighbor_pairs(
    cluster_ids: list[int],
    neighbors_by_id: dict[int, MovieNeighbors],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    cluster_id_set = set(cluster_ids)
    pairs = []
    seen: set[tuple[int, int]] = set()
    for source_id in cluster_ids:
        entry = neighbors_by_id.get(source_id)
        if not entry:
            continue
        for neighbor in entry.neighbors:
            if neighbor.tmdb_id not in cluster_id_set:
                continue
            key = tuple(sorted((entry.tmdb_id, neighbor.tmdb_id)))
            if key in seen:
                continue
            seen.add(key)
            pairs.append(
                {
                    "source_title": entry.title,
                    "neighbor_title": neighbor.title,
                    "similarity": neighbor.similarity,
                }
            )
            if len(pairs) >= limit:
                return pairs
    return pairs


def _coherence_score(embeddings: list[Any]) -> float | None:
    if len(embeddings) < 2:
        return None
    matrix = np.array([record.embedding for record in embeddings], dtype=float)
    similarities = cosine_similarity(matrix)
    upper_indexes = np.triu_indices(len(embeddings), k=1)
    return float(np.mean(similarities[upper_indexes]))


def _cluster_warnings(
    terms: list[tuple[str, float]],
    movies: list[MovieRecord],
) -> list[str]:
    warnings = []
    noisy_hits = [term for term, _score in terms if term in CLUSTER_STOP_WORDS]
    if noisy_hits:
        warnings.append(f"Noisy terms present: {', '.join(noisy_hits)}")
    if len(movies) < 3:
        warnings.append("Small cluster; evidence may be unstable.")
    return warnings
