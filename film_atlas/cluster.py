"""Cluster movie embeddings into local vibe neighborhoods."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

from film_atlas.embedding import load_embedding_records

CLUSTER_ASSIGNMENTS_FILENAME = "cluster_assignments.json"


@dataclass(frozen=True, slots=True)
class ClusterAssignment:
    tmdb_id: int
    title: str
    cluster_id: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def cluster_embeddings_file(
    *,
    embeddings_path: str | Path = "outputs/intermediate/embeddings.jsonl",
    output_dir: str | Path = "outputs",
    n_clusters: int | None = None,
) -> Path:
    """Cluster embeddings and write outputs/intermediate/cluster_assignments.json."""
    records = load_embedding_records(embeddings_path)
    assignments = cluster_embedding_records(records, n_clusters=n_clusters)
    path = Path(output_dir) / "intermediate" / CLUSTER_ASSIGNMENTS_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    cluster_count = len({assignment.cluster_id for assignment in assignments if assignment.cluster_id >= 0})
    path.write_text(
        json.dumps(
            {
                "clustering_method": "kmeans",
                "cluster_count": cluster_count,
                "outlier_count": 0,
                "assignments": [assignment.to_dict() for assignment in assignments],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def cluster_embedding_records(
    records: list[Any],
    *,
    n_clusters: int | None = None,
) -> list[ClusterAssignment]:
    """Cluster embedding records with normalized-vector k-means."""
    if not records:
        return []
    if len(records) == 1:
        return [ClusterAssignment(records[0].tmdb_id, records[0].title, 0)]

    matrix = normalize(np.array([record.embedding for record in records], dtype=float))
    cluster_count = n_clusters or _default_cluster_count(len(records))
    cluster_count = min(max(2, cluster_count), len(records))
    model = KMeans(n_clusters=cluster_count, random_state=42, n_init=10)
    labels = model.fit_predict(matrix)
    return [
        ClusterAssignment(
            tmdb_id=record.tmdb_id,
            title=record.title,
            cluster_id=int(labels[index]),
        )
        for index, record in enumerate(records)
    ]


def load_cluster_assignments(
    path: str | Path = "outputs/intermediate/cluster_assignments.json",
) -> list[ClusterAssignment]:
    """Load cluster assignments from disk."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        ClusterAssignment(
            tmdb_id=int(item["tmdb_id"]),
            title=str(item["title"]),
            cluster_id=int(item["cluster_id"]),
        )
        for item in payload.get("assignments", [])
    ]


def _default_cluster_count(count: int) -> int:
    return min(15, max(2, round(math.sqrt(count / 2))))
