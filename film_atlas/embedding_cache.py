"""Embedding cache helpers for Milestone 2."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from film_atlas.models import SemanticProfile


@dataclass(frozen=True, slots=True)
class EmbeddingRecord:
    """A cached embedding for one semantic profile."""

    tmdb_id: int
    title: str
    model: str
    profile_hash: str
    embedding: list[float]
    estimated_tokens: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> EmbeddingRecord:
        return cls(
            tmdb_id=int(value["tmdb_id"]),
            title=str(value.get("title") or ""),
            model=str(value["model"]),
            profile_hash=str(value["profile_hash"]),
            embedding=[float(item) for item in value["embedding"]],
            estimated_tokens=int(value.get("estimated_tokens") or 0),
        )


def profile_hash(profile: SemanticProfile) -> str:
    """Hash the embedding-relevant identity and profile text."""
    payload = {
        "tmdb_id": profile.tmdb_id,
        "title": profile.title,
        "profile_text": profile.profile_text,
    }
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def cache_key(tmdb_id: int, model: str, digest: str) -> str:
    """Return the stable cache key for a profile embedding."""
    return f"{model}:{tmdb_id}:{digest}"


def load_embedding_cache(path: str | Path) -> dict[str, EmbeddingRecord]:
    """Load an embeddings JSONL cache keyed by model, profile id, and profile hash."""
    cache_path = Path(path)
    if not cache_path.exists():
        return {}

    records: dict[str, EmbeddingRecord] = {}
    for line in cache_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = EmbeddingRecord.from_dict(json.loads(line))
        records[cache_key(record.tmdb_id, record.model, record.profile_hash)] = record
    return records


def write_embedding_cache(path: str | Path, records: list[EmbeddingRecord]) -> Path:
    """Write selected embedding records as JSONL."""
    cache_path = Path(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(record.to_dict(), sort_keys=True) for record in records]
    cache_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return cache_path
