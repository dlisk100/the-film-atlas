"""OpenAI embedding estimation and generation for Milestone 2."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx

from film_atlas.config import MissingCredentialsError
from film_atlas.embedding_cache import (
    EmbeddingRecord,
    cache_key,
    load_embedding_cache,
    profile_hash,
    write_embedding_cache,
)
from film_atlas.models import SemanticProfile
from film_atlas.profiles import load_profiles

EMBEDDINGS_FILENAME = "embeddings.jsonl"
EMBEDDING_MANIFEST_FILENAME = "embedding_manifest.json"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_PRICES_PER_1M_TOKENS = {
    "text-embedding-3-large": 0.13,
    "text-embedding-3-small": 0.02,
}


@dataclass(frozen=True, slots=True)
class EmbeddingEstimate:
    """Approximate embedding token and cost estimate."""

    profile_count: int
    selected_count: int
    model: str
    estimated_tokens: int
    estimated_cost_usd: float
    price_per_1m_tokens: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EmbeddingRunResult:
    """Embedding run paths and counts."""

    profiles_available: int
    embedded_count: int
    model: str
    estimated_tokens: int
    estimated_cost_usd: float
    cached_reused_count: int
    new_embedding_count: int
    embedding_path: Path
    manifest_path: Path
    api_prompt_tokens: int | None


class OpenAIEmbeddingClient:
    """Small HTTP client for OpenAI embeddings."""

    def __init__(
        self,
        api_key: str | None,
        *,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 60.0,
        max_retries: int = 3,
        backoff_seconds: float = 1.0,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.api_key = api_key
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self.sleep = sleep
        self.client = httpx.Client(base_url=base_url, timeout=timeout, transport=transport)

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> OpenAIEmbeddingClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def embed_texts(self, texts: list[str], *, model: str) -> tuple[list[list[float]], int | None]:
        """Embed a batch of texts and return vectors plus API token usage if present."""
        if not self.api_key:
            raise MissingCredentialsError(
                "OPENAI_API_KEY is required for live embedding calls. "
                "Add it to .env, then run the command again."
            )
        payload = {
            "model": model,
            "input": texts,
            "encoding_format": "float",
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        response_json = self._post_with_retries("/embeddings", payload=payload, headers=headers)
        return parse_embedding_response(response_json)

    def _post_with_retries(
        self,
        endpoint: str,
        *,
        payload: dict[str, Any],
        headers: dict[str, str],
    ) -> dict[str, Any]:
        retry_statuses = {429, 500, 502, 503, 504}
        last_response: httpx.Response | None = None
        for attempt in range(self.max_retries + 1):
            response = self.client.post(endpoint, json=payload, headers=headers)
            last_response = response
            if response.status_code not in retry_statuses:
                response.raise_for_status()
                return response.json()
            if attempt < self.max_retries:
                retry_after = response.headers.get("Retry-After")
                delay = float(retry_after) if retry_after else self.backoff_seconds * 2**attempt
                self.sleep(delay)
        if last_response is None:
            raise RuntimeError("OpenAI embedding request failed before receiving a response.")
        last_response.raise_for_status()
        return last_response.json()


def parse_embedding_response(payload: dict[str, Any]) -> tuple[list[list[float]], int | None]:
    """Parse an OpenAI embeddings response into ordered vectors and token usage."""
    data = sorted(payload.get("data") or [], key=lambda item: int(item["index"]))
    embeddings = [[float(value) for value in item["embedding"]] for item in data]
    usage = payload.get("usage") or {}
    prompt_tokens = usage.get("prompt_tokens") or usage.get("total_tokens")
    return embeddings, int(prompt_tokens) if prompt_tokens is not None else None


def estimate_profiles(
    profiles: list[SemanticProfile],
    *,
    model: str = DEFAULT_EMBEDDING_MODEL,
    limit: int | None = None,
) -> EmbeddingEstimate:
    """Estimate embedding tokens and cost for semantic profiles."""
    selected = profiles[:limit] if limit is not None else profiles
    estimated_tokens = sum(estimate_text_tokens(profile.profile_text) for profile in selected)
    price = EMBEDDING_PRICES_PER_1M_TOKENS.get(model, 0.13)
    return EmbeddingEstimate(
        profile_count=len(profiles),
        selected_count=len(selected),
        model=model,
        estimated_tokens=estimated_tokens,
        estimated_cost_usd=estimated_tokens / 1_000_000 * price,
        price_per_1m_tokens=price,
    )


def estimate_profiles_file(
    *,
    profiles_path: str | Path = "data/processed/profiles.json",
    model: str = DEFAULT_EMBEDDING_MODEL,
    limit: int | None = None,
) -> EmbeddingEstimate:
    """Estimate embedding cost for profiles loaded from disk."""
    return estimate_profiles(load_profiles(profiles_path), model=model, limit=limit)


def embed_profiles_file(
    *,
    api_key: str | None,
    profiles_path: str | Path = "data/processed/profiles.json",
    output_dir: str | Path = "outputs",
    model: str = DEFAULT_EMBEDDING_MODEL,
    limit: int | None = None,
    batch_size: int = 64,
    client: OpenAIEmbeddingClient | None = None,
) -> EmbeddingRunResult:
    """Embed selected profiles, reusing cached embeddings for unchanged profiles."""
    profiles = load_profiles(profiles_path)
    selected = profiles[:limit] if limit is not None else profiles
    estimate = estimate_profiles(profiles, model=model, limit=limit)

    intermediate_dir = Path(output_dir) / "intermediate"
    embedding_path = intermediate_dir / EMBEDDINGS_FILENAME
    manifest_path = intermediate_dir / EMBEDDING_MANIFEST_FILENAME

    existing_cache = load_embedding_cache(embedding_path)
    records_by_key: dict[str, EmbeddingRecord] = {}
    to_embed: list[SemanticProfile] = []
    cached_reused_count = 0

    for profile in selected:
        digest = profile_hash(profile)
        key = cache_key(profile.tmdb_id, model, digest)
        cached = existing_cache.get(key)
        if cached:
            records_by_key[key] = cached
            cached_reused_count += 1
        else:
            to_embed.append(profile)

    api_prompt_tokens = 0
    owns_client = client is None
    active_client = client or OpenAIEmbeddingClient(api_key)
    try:
        for batch in _chunks(to_embed, batch_size):
            texts = [profile.profile_text for profile in batch]
            vectors, prompt_tokens = active_client.embed_texts(texts, model=model)
            if prompt_tokens is not None:
                api_prompt_tokens += prompt_tokens
            if len(vectors) != len(batch):
                raise RuntimeError("OpenAI embedding response count did not match request count.")
            for profile, vector in zip(batch, vectors, strict=True):
                digest = profile_hash(profile)
                record = EmbeddingRecord(
                    tmdb_id=profile.tmdb_id,
                    title=profile.title,
                    model=model,
                    profile_hash=digest,
                    embedding=vector,
                    estimated_tokens=estimate_text_tokens(profile.profile_text),
                )
                key = cache_key(profile.tmdb_id, model, digest)
                records_by_key[key] = record
    finally:
        if owns_client:
            active_client.close()

    ordered_records = _order_records(selected, records_by_key, model)
    write_embedding_cache(embedding_path, ordered_records)
    manifest = {
        "profiles_available": len(profiles),
        "embedded_count": len(ordered_records),
        "model": model,
        "estimated_tokens": estimate.estimated_tokens,
        "estimated_cost_usd": estimate.estimated_cost_usd,
        "price_per_1m_tokens": estimate.price_per_1m_tokens,
        "cached_reused_count": cached_reused_count,
        "new_embedding_count": len(to_embed),
        "api_prompt_tokens": api_prompt_tokens or None,
        "profiles_path": str(profiles_path),
        "embeddings_path": str(embedding_path),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    return EmbeddingRunResult(
        profiles_available=len(profiles),
        embedded_count=len(ordered_records),
        model=model,
        estimated_tokens=estimate.estimated_tokens,
        estimated_cost_usd=estimate.estimated_cost_usd,
        cached_reused_count=cached_reused_count,
        new_embedding_count=len(to_embed),
        embedding_path=embedding_path,
        manifest_path=manifest_path,
        api_prompt_tokens=api_prompt_tokens or None,
    )


def load_embedding_records(path: str | Path = "outputs/intermediate/embeddings.jsonl") -> list[EmbeddingRecord]:
    """Load embedding records from JSONL in file order."""
    cache_path = Path(path)
    if not cache_path.exists():
        return []
    return [
        EmbeddingRecord.from_dict(json.loads(line))
        for line in cache_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def estimate_text_tokens(text: str) -> int:
    """Conservative token approximation for preflight cost checks."""
    if not text:
        return 0
    by_chars = math.ceil(len(text) / 4)
    by_words = math.ceil(len(text.split()) * 1.35)
    return max(1, by_chars, by_words)


def _chunks(items: list[SemanticProfile], size: int) -> list[list[SemanticProfile]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def _order_records(
    profiles: list[SemanticProfile],
    records_by_key: dict[str, EmbeddingRecord],
    model: str,
) -> list[EmbeddingRecord]:
    ordered = []
    for profile in profiles:
        digest = profile_hash(profile)
        ordered.append(records_by_key[cache_key(profile.tmdb_id, model, digest)])
    return ordered
