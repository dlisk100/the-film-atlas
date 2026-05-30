"""Compute cosine-similarity nearest neighbors from embeddings."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from film_atlas.embedding import load_embedding_records

NEIGHBORS_FILENAME = "neighbors.json"


@dataclass(frozen=True, slots=True)
class NeighborMatch:
    tmdb_id: int
    title: str
    similarity: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class MovieNeighbors:
    tmdb_id: int
    title: str
    neighbors: list[NeighborMatch]

    def to_dict(self) -> dict[str, Any]:
        return {
            "tmdb_id": self.tmdb_id,
            "title": self.title,
            "neighbors": [neighbor.to_dict() for neighbor in self.neighbors],
        }


def compute_neighbors_file(
    *,
    embeddings_path: str | Path = "outputs/intermediate/embeddings.jsonl",
    output_dir: str | Path = "outputs",
    top_n: int = 10,
) -> Path:
    """Compute nearest neighbors and write outputs/intermediate/neighbors.json."""
    records = load_embedding_records(embeddings_path)
    neighbors = compute_neighbors(records, top_n=top_n)
    path = Path(output_dir) / "intermediate" / NEIGHBORS_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([entry.to_dict() for entry in neighbors], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path


def compute_neighbors(records: list[Any], *, top_n: int = 10) -> list[MovieNeighbors]:
    """Compute cosine nearest neighbors and exclude self-matches."""
    if not records:
        return []
    matrix = np.array([record.embedding for record in records], dtype=float)
    similarities = cosine_similarity(matrix)
    output = []
    for source_index, source in enumerate(records):
        ranked_indexes = np.argsort(similarities[source_index])[::-1]
        matches = []
        for neighbor_index in ranked_indexes:
            if neighbor_index == source_index:
                continue
            neighbor = records[int(neighbor_index)]
            matches.append(
                NeighborMatch(
                    tmdb_id=neighbor.tmdb_id,
                    title=neighbor.title,
                    similarity=float(similarities[source_index, neighbor_index]),
                )
            )
            if len(matches) >= top_n:
                break
        output.append(MovieNeighbors(source.tmdb_id, source.title, matches))
    return output


def load_neighbors(path: str | Path = "outputs/intermediate/neighbors.json") -> list[MovieNeighbors]:
    """Load nearest-neighbor records from disk."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        MovieNeighbors(
            tmdb_id=int(item["tmdb_id"]),
            title=str(item["title"]),
            neighbors=[
                NeighborMatch(
                    tmdb_id=int(neighbor["tmdb_id"]),
                    title=str(neighbor["title"]),
                    similarity=float(neighbor["similarity"]),
                )
                for neighbor in item.get("neighbors", [])
            ],
        )
        for item in payload
    ]
