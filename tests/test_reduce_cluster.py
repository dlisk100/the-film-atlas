from __future__ import annotations

from film_atlas.cluster import cluster_embedding_records
from film_atlas.embedding_cache import EmbeddingRecord
from film_atlas.reduce import reduce_embedding_records


def test_projection_output_shape() -> None:
    records = [
        EmbeddingRecord(1, "A", "model", "h1", [1.0, 0.0, 0.0], 1),
        EmbeddingRecord(2, "B", "model", "h2", [0.0, 1.0, 0.0], 1),
        EmbeddingRecord(3, "C", "model", "h3", [0.0, 0.0, 1.0], 1),
    ]

    coordinates = reduce_embedding_records(records)

    assert len(coordinates) == 3
    assert all(isinstance(coordinate.x, float) for coordinate in coordinates)
    assert all(isinstance(coordinate.y, float) for coordinate in coordinates)


def test_clustering_output_shape() -> None:
    records = [
        EmbeddingRecord(1, "A", "model", "h1", [1.0, 0.0], 1),
        EmbeddingRecord(2, "B", "model", "h2", [0.9, 0.1], 1),
        EmbeddingRecord(3, "C", "model", "h3", [0.0, 1.0], 1),
    ]

    assignments = cluster_embedding_records(records, n_clusters=2)

    assert len(assignments) == 3
    assert len({assignment.cluster_id for assignment in assignments}) == 2
