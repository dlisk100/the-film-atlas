from __future__ import annotations

from film_atlas.embedding_cache import EmbeddingRecord
from film_atlas.neighbors import compute_neighbors


def test_neighbors_exclude_self_matches() -> None:
    records = [
        EmbeddingRecord(1, "A", "model", "h1", [1.0, 0.0], 1),
        EmbeddingRecord(2, "B", "model", "h2", [0.9, 0.1], 1),
        EmbeddingRecord(3, "C", "model", "h3", [0.0, 1.0], 1),
    ]

    neighbors = compute_neighbors(records, top_n=2)

    assert all(entry.tmdb_id not in [neighbor.tmdb_id for neighbor in entry.neighbors] for entry in neighbors)
    assert neighbors[0].neighbors[0].title == "B"
