from __future__ import annotations

import json
from pathlib import Path

from film_atlas.embedding import embed_profiles_file, parse_embedding_response
from film_atlas.embedding_cache import EmbeddingRecord, load_embedding_cache, profile_hash, write_embedding_cache
from film_atlas.models import SemanticProfile


class FakeEmbeddingClient:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed_texts(self, texts: list[str], *, model: str) -> tuple[list[list[float]], int]:
        self.calls.append(texts)
        return [[float(index), 0.1, 0.2] for index, _text in enumerate(texts)], 10

    def close(self) -> None:
        return None


def test_profile_hash_changes_when_profile_text_changes() -> None:
    original = SemanticProfile(1, "A", 2000, "same", [], [])
    changed = SemanticProfile(1, "A", 2000, "different", [], [])

    assert profile_hash(original) != profile_hash(changed)


def test_embedding_cache_loads_records_by_model_id_and_hash(tmp_path: Path) -> None:
    profile = SemanticProfile(1, "A", 2000, "profile text", [], [])
    record = EmbeddingRecord(1, "A", "model-a", profile_hash(profile), [0.1, 0.2], 5)
    path = tmp_path / "embeddings.jsonl"

    write_embedding_cache(path, [record])
    cache = load_embedding_cache(path)

    assert len(cache) == 1
    assert next(iter(cache.values())).embedding == [0.1, 0.2]


def test_embed_profiles_reuses_unchanged_cached_profiles(tmp_path: Path) -> None:
    profile = SemanticProfile(1, "A", 2000, "profile text", [], [])
    profiles_path = tmp_path / "profiles.json"
    profiles_path.write_text(json.dumps([profile.to_dict()]), encoding="utf-8")
    output_dir = tmp_path / "outputs"
    embedding_path = output_dir / "intermediate" / "embeddings.jsonl"
    record = EmbeddingRecord(1, "A", "model-a", profile_hash(profile), [0.1, 0.2], 5)
    write_embedding_cache(embedding_path, [record])
    client = FakeEmbeddingClient()

    result = embed_profiles_file(
        api_key="unused",
        profiles_path=profiles_path,
        output_dir=output_dir,
        model="model-a",
        client=client,  # type: ignore[arg-type]
    )

    assert result.cached_reused_count == 1
    assert result.new_embedding_count == 0
    assert client.calls == []


def test_parse_embedding_response_orders_vectors_by_index() -> None:
    payload = {
        "data": [
            {"index": 1, "embedding": [2, 2]},
            {"index": 0, "embedding": [1, 1]},
        ],
        "usage": {"prompt_tokens": 12},
    }

    vectors, tokens = parse_embedding_response(payload)

    assert vectors == [[1.0, 1.0], [2.0, 2.0]]
    assert tokens == 12
