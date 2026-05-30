"""Project embedding vectors into 2D coordinates."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import PCA

from film_atlas.embedding import load_embedding_records

COORDINATES_FILENAME = "coordinates.json"


@dataclass(frozen=True, slots=True)
class CoordinateRecord:
    tmdb_id: int
    title: str
    x: float
    y: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def reduce_embeddings_file(
    *,
    embeddings_path: str | Path = "outputs/intermediate/embeddings.jsonl",
    output_dir: str | Path = "outputs",
) -> Path:
    """Reduce embeddings to 2D coordinates and write outputs/intermediate/coordinates.json."""
    records = load_embedding_records(embeddings_path)
    coordinates = reduce_embedding_records(records)
    path = Path(output_dir) / "intermediate" / COORDINATES_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "projection_method": "pca",
                "coordinates": [coordinate.to_dict() for coordinate in coordinates],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def reduce_embedding_records(records: list[Any]) -> list[CoordinateRecord]:
    """Project embedding records into 2D with PCA."""
    if not records:
        return []
    matrix = np.array([record.embedding for record in records], dtype=float)
    coords = _pca_2d(matrix)
    return [
        CoordinateRecord(
            tmdb_id=record.tmdb_id,
            title=record.title,
            x=float(coords[index, 0]),
            y=float(coords[index, 1]),
        )
        for index, record in enumerate(records)
    ]


def load_coordinates(path: str | Path = "outputs/intermediate/coordinates.json") -> list[CoordinateRecord]:
    """Load coordinate records from disk."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        CoordinateRecord(
            tmdb_id=int(item["tmdb_id"]),
            title=str(item["title"]),
            x=float(item["x"]),
            y=float(item["y"]),
        )
        for item in payload.get("coordinates", [])
    ]


def _pca_2d(matrix: np.ndarray) -> np.ndarray:
    rows, columns = matrix.shape
    if rows == 1:
        return np.array([[0.0, 0.0]])
    if columns == 1:
        return np.column_stack([matrix[:, 0], np.zeros(rows)])
    components = min(2, rows, columns)
    coords = PCA(n_components=components, random_state=42).fit_transform(matrix)
    if coords.shape[1] == 1:
        return np.column_stack([coords[:, 0], np.zeros(rows)])
    return coords
