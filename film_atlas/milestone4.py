"""Milestone 4 scaled dataset, hierarchy, and public export pipeline."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from film_atlas.cluster import ClusterAssignment, cluster_embedding_records
from film_atlas.cluster_labels import (
    ClusterLabelCandidate,
    LabelingEstimate,
    OpenAIClusterLabelClient,
    build_label_messages,
    estimate_labeling,
    label_clusters,
    load_label_cache,
    render_human_editable_labels,
    write_label_cache,
)
from film_atlas.embedding import (
    DEFAULT_EMBEDDING_MODEL,
    EmbeddingRunResult,
    embed_profiles_file,
    estimate_profiles,
    estimate_text_tokens,
    load_embedding_records,
)
from film_atlas.embedding_cache import cache_key, load_embedding_cache, profile_hash
from film_atlas.fetch import DETAILS_FILENAME, DISCOVER_FILENAME, RAW_DIR_NAME, dedupe_movies, write_json
from film_atlas.inspect_clusters import ClusterEvidence, build_cluster_evidence
from film_atlas.milestone2_report import QUALITY_CHECK_MOVIES
from film_atlas.models import MovieRecord
from film_atlas.neighbors import compute_neighbors
from film_atlas.normalize import MOVIES_JSON_FILENAME, MOVIES_PARQUET_FILENAME, normalize_movie_detail
from film_atlas.profiles import build_semantic_profile, load_profiles
from film_atlas.reduce import CoordinateRecord
from film_atlas.tmdb_client import TMDbClient

MILESTONE4_DIRNAME = "milestone_4"
HIERARCHY_DIRNAME = "hierarchy"
PUBLIC_EXPORT_DIRNAME = "public_export"
MILESTONE4_REPORT_FILENAME = "milestone_4_report.md"
MILESTONE4_SUMMARY_FILENAME = "milestone_4_summary.json"
SCALE_DATASET_SUMMARY_FILENAME = "scale_dataset_summary.json"
LayerName = Literal["macro", "neighborhood", "micro"]


class Milestone4Error(RuntimeError):
    """Raised when Milestone 4 cannot proceed."""


@dataclass(frozen=True, slots=True)
class ScaleDatasetResult:
    target: int
    selected_count: int
    candidate_count: int
    detail_count: int
    since_year: int
    min_votes: int
    movies_path: Path
    profiles_path: Path
    discover_path: Path
    details_path: Path
    summary_path: Path

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ("movies_path", "profiles_path", "discover_path", "details_path", "summary_path"):
            data[key] = str(data[key])
        return data


@dataclass(frozen=True, slots=True)
class EmbeddingCostEstimate:
    profile_count: int
    cached_count: int
    new_count: int
    estimated_tokens: int
    estimated_full_cost_usd: float
    estimated_live_cost_usd: float
    model: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class LayerSummary:
    layer: LayerName
    k: int
    cluster_count: int
    labeled: bool
    cluster_sizes: list[int]
    tiny_cluster_count: int
    coherence_average: float | None
    coherence_min: float | None
    coherence_max: float | None
    label_estimate: LabelingEstimate
    cached_labels_reused: int
    new_labels_generated: int
    sample_labels: list[dict[str, Any]]
    assignments_path: Path
    evidence_path: Path
    labels_path: Path | None
    cache_path: Path

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["label_estimate"] = self.label_estimate.to_dict()
        for key in ("assignments_path", "evidence_path", "cache_path"):
            data[key] = str(data[key])
        data["labels_path"] = str(self.labels_path) if self.labels_path else None
        return data


@dataclass(frozen=True, slots=True)
class HierarchyResult:
    movie_count: int
    embedding_model: str
    embedding_estimate: EmbeddingCostEstimate | None
    projection_method: str
    coordinates_path: Path
    neighbors_path: Path
    summary_path: Path
    layers: list[LayerSummary]
    labels_total_estimated_cost_usd: float
    labels_generated_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "movie_count": self.movie_count,
            "embedding_model": self.embedding_model,
            "embedding_estimate": (
                self.embedding_estimate.to_dict() if self.embedding_estimate else None
            ),
            "projection_method": self.projection_method,
            "coordinates_path": str(self.coordinates_path),
            "neighbors_path": str(self.neighbors_path),
            "summary_path": str(self.summary_path),
            "layers": [layer.to_dict() for layer in self.layers],
            "labels_total_estimated_cost_usd": self.labels_total_estimated_cost_usd,
            "labels_generated_count": self.labels_generated_count,
        }


@dataclass(frozen=True, slots=True)
class PublicExportResult:
    export_dir: Path
    file_sizes: dict[str, int]
    manifest_path: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "export_dir": str(self.export_dir),
            "file_sizes": self.file_sizes,
            "manifest_path": str(self.manifest_path),
        }


@dataclass(frozen=True, slots=True)
class Milestone4Result:
    scale: ScaleDatasetResult
    embedding: EmbeddingRunResult
    hierarchy: HierarchyResult
    export: PublicExportResult
    report_path: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "scale": self.scale.to_dict(),
            "embedding": {
                "profiles_available": self.embedding.profiles_available,
                "embedded_count": self.embedding.embedded_count,
                "model": self.embedding.model,
                "estimated_tokens": self.embedding.estimated_tokens,
                "estimated_cost_usd": self.embedding.estimated_cost_usd,
                "cached_reused_count": self.embedding.cached_reused_count,
                "new_embedding_count": self.embedding.new_embedding_count,
                "embedding_path": str(self.embedding.embedding_path),
                "manifest_path": str(self.embedding.manifest_path),
                "api_prompt_tokens": self.embedding.api_prompt_tokens,
            },
            "hierarchy": self.hierarchy.to_dict(),
            "export": self.export.to_dict(),
            "report_path": str(self.report_path),
        }


def scale_dataset_file(
    client: TMDbClient,
    *,
    target: int = 2000,
    since_year: int = 1980,
    min_votes: int = 100,
    data_dir: str | Path = "data",
    output_dir: str | Path = "outputs",
    min_runtime: int = 60,
    candidate_limit: int | None = None,
    sort_by: str = "vote_count.desc",
    refresh: bool = False,
    max_review_chars: int = 180,
) -> ScaleDatasetResult:
    """Fetch, filter, normalize, and profile a scaled English-language movie sample."""
    if target < 1:
        raise Milestone4Error("--target must be at least 1.")
    if since_year < 1880:
        raise Milestone4Error("--since-year is unexpectedly early.")

    data_path = Path(data_dir)
    output_path = Path(output_dir)
    resolved_candidate_limit = candidate_limit or max(target, math.ceil(target * 1.4))
    release_date_gte = f"{since_year}-01-01"
    release_date_lte = date.today().isoformat()

    candidates = client.discover_movies(
        limit=resolved_candidate_limit,
        min_votes=min_votes,
        sort_by=sort_by,
        min_runtime=min_runtime,
        release_date_gte=release_date_gte,
        release_date_lte=release_date_lte,
        exclude_future=True,
        refresh=refresh,
    )
    candidates = dedupe_movies(candidates)
    discover_payload = {
        "source": "tmdb:/discover/movie",
        "sampling_strategy": "scaled_english_since_year",
        "target": target,
        "candidate_limit": resolved_candidate_limit,
        "since_year": since_year,
        "min_votes": min_votes,
        "sort_by": sort_by,
        "min_runtime": min_runtime,
        "release_date_gte": release_date_gte,
        "release_date_lte": release_date_lte,
        "exclude_future": True,
        "movie_count": len(candidates),
        "results": candidates,
    }
    discover_path = write_json(data_path / RAW_DIR_NAME / DISCOVER_FILENAME, discover_payload)

    details = [client.movie_details(int(movie["id"]), refresh=refresh) for movie in candidates]
    eligible = [
        detail
        for detail in details
        if is_scaled_eligible_detail(
            detail,
            since_year=since_year,
            min_votes=min_votes,
            min_runtime=min_runtime,
            today=date.today(),
        )
    ]
    selected_details = sorted(eligible, key=_scaled_detail_sort_key)[:target]
    if not selected_details:
        raise Milestone4Error("No eligible movies were found for the scaled dataset.")

    details_payload = {
        "source": "tmdb:/movie/{movie_id}",
        "discover_path": str(discover_path),
        "sampling_strategy": "scaled_english_since_year",
        "target": target,
        "candidate_count": len(candidates),
        "detail_count": len(selected_details),
        "details_fetched": len(details),
        "results": selected_details,
    }
    details_path = write_json(data_path / RAW_DIR_NAME / DETAILS_FILENAME, details_payload)

    movies = [normalize_movie_detail(detail) for detail in selected_details]
    movies_path = _write_movie_records(movies, data_path)
    profiles = [
        build_semantic_profile(
            movie,
            include_reviews=True,
            max_review_chars=max_review_chars,
            review_weight="light",
        )
        for movie in movies
    ]
    profiles_path = data_path / "processed" / "profiles.json"
    profiles_path.write_text(
        json.dumps([profile.to_dict() for profile in profiles], indent=2, sort_keys=True),
        encoding="utf-8",
    )

    summary_path = output_path / "intermediate" / MILESTONE4_DIRNAME / SCALE_DATASET_SUMMARY_FILENAME
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    result = ScaleDatasetResult(
        target=target,
        selected_count=len(movies),
        candidate_count=len(candidates),
        detail_count=len(details),
        since_year=since_year,
        min_votes=min_votes,
        movies_path=movies_path,
        profiles_path=profiles_path,
        discover_path=discover_path,
        details_path=details_path,
        summary_path=summary_path,
    )
    summary_path.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return result


def is_scaled_eligible_detail(
    detail: dict[str, Any],
    *,
    since_year: int,
    min_votes: int,
    min_runtime: int,
    today: date,
) -> bool:
    """Return whether a TMDb detail payload satisfies Milestone 4 dataset filters."""
    release_date = str(detail.get("release_date") or "")
    year = _year_from_release_date(release_date)
    if year is None or year < since_year:
        return False
    if release_date and release_date > today.isoformat():
        return False
    if detail.get("adult") is True or detail.get("video") is True:
        return False
    if str(detail.get("original_language") or "").lower() != "en":
        return False
    if int(detail.get("runtime") or 0) < min_runtime:
        return False
    if int(detail.get("vote_count") or 0) < min_votes:
        return False
    return bool(str(detail.get("overview") or "").strip())


def estimate_uncached_embedding_cost(
    *,
    profiles_path: str | Path,
    embeddings_path: str | Path,
    model: str = DEFAULT_EMBEDDING_MODEL,
    limit: int | None = None,
) -> EmbeddingCostEstimate:
    """Estimate the live embedding cost after reusable cache entries."""
    profiles = load_profiles(profiles_path)
    selected = profiles[:limit] if limit is not None else profiles
    estimate = estimate_profiles(profiles, model=model, limit=limit)
    cache = load_embedding_cache(embeddings_path)
    new_profiles = [
        profile
        for profile in selected
        if cache_key(profile.tmdb_id, model, profile_hash(profile)) not in cache
    ]
    new_tokens = sum(estimate_text_tokens(profile.profile_text) for profile in new_profiles)
    live_cost = new_tokens / 1_000_000 * estimate.price_per_1m_tokens
    return EmbeddingCostEstimate(
        profile_count=len(selected),
        cached_count=len(selected) - len(new_profiles),
        new_count=len(new_profiles),
        estimated_tokens=estimate.estimated_tokens,
        estimated_full_cost_usd=estimate.estimated_cost_usd,
        estimated_live_cost_usd=live_cost,
        model=model,
    )


def build_hierarchy_file(
    *,
    movies_path: str | Path = "data/processed/movies.json",
    profiles_path: str | Path = "data/processed/profiles.json",
    embeddings_path: str | Path = "outputs/intermediate/embeddings.jsonl",
    output_dir: str | Path = "outputs",
    macro_k: int = 12,
    neighborhood_k: int = 75,
    micro_k: int = 200,
    label_model: str,
    openai_api_key: str | None,
    embedding_estimate: EmbeddingCostEstimate | None = None,
    label_batch_size: int = 5,
    cost_gate_usd: float = 5.0,
    micro_label_cost_limit_usd: float = 2.50,
    label_client: OpenAIClusterLabelClient | None = None,
    projection_method: Literal["auto", "umap", "pca"] = "auto",
) -> HierarchyResult:
    """Build hierarchy layers, labels, neighbors, projection, and private summaries."""
    movies = _load_movies(movies_path)
    profiles = load_profiles(profiles_path)
    embeddings = load_embedding_records(embeddings_path)
    if not embeddings:
        raise Milestone4Error(f"No embeddings found at {embeddings_path}.")
    if len(embeddings) != len(profiles):
        raise Milestone4Error(
            "Embedding/profile counts do not match; rebuild embeddings for the active profiles."
        )

    output_path = Path(output_dir)
    hierarchy_dir = output_path / "intermediate" / HIERARCHY_DIRNAME
    hierarchy_dir.mkdir(parents=True, exist_ok=True)

    neighbors = compute_neighbors(embeddings, top_n=10)
    neighbors_path = hierarchy_dir / "neighbors.json"
    neighbors_path.write_text(
        json.dumps([entry.to_dict() for entry in neighbors], indent=2, sort_keys=True),
        encoding="utf-8",
    )

    coordinates, resolved_projection = project_embedding_records(
        embeddings,
        method=projection_method,
    )
    coordinates_path = hierarchy_dir / "coordinates.json"
    coordinates_path.write_text(
        json.dumps(
            {
                "projection_method": resolved_projection,
                "coordinates": [coordinate.to_dict() for coordinate in coordinates],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    assignment_sets = {
        "macro": cluster_embedding_records(embeddings, n_clusters=macro_k),
        "neighborhood": cluster_embedding_records(embeddings, n_clusters=neighborhood_k),
        "micro": cluster_embedding_records(embeddings, n_clusters=micro_k),
    }
    parent_maps = {
        "macro": {},
        "neighborhood": _majority_parent_map(
            child_assignments=assignment_sets["neighborhood"],
            parent_assignments=assignment_sets["macro"],
        ),
        "micro": _majority_parent_map(
            child_assignments=assignment_sets["micro"],
            parent_assignments=assignment_sets["neighborhood"],
        ),
    }

    prepared_layers = []
    for layer, assignments in assignment_sets.items():
        evidence = build_cluster_evidence(
            movies=movies,
            profiles=profiles,
            embeddings=embeddings,
            assignments=assignments,
            neighbors=neighbors,
        )
        cache_path = hierarchy_dir / f"{layer}_label_cache.json"
        estimate = estimate_labeling(
            evidence,
            cache=load_label_cache(cache_path),
            model=label_model,
            openai_api_key=openai_api_key,
            batch_size=label_batch_size,
        )
        prepared_layers.append((layer, assignments, evidence, cache_path, estimate))

    macro_neighborhood_cost = sum(
        estimate.estimated_cost_usd
        for layer, _assignments, _evidence, _cache_path, estimate in prepared_layers
        if layer != "micro"
    )
    micro_estimate = next(
        estimate for layer, _assignments, _evidence, _cache_path, estimate in prepared_layers if layer == "micro"
    )
    embedding_live_cost = embedding_estimate.estimated_live_cost_usd if embedding_estimate else 0.0
    if embedding_live_cost + macro_neighborhood_cost > cost_gate_usd:
        raise Milestone4Error(
            f"Estimated Milestone 4 OpenAI cost is "
            f"${embedding_live_cost + macro_neighborhood_cost:.4f}, which exceeds ${cost_gate_usd:.2f}. "
            "Pause here and get explicit approval before labeling."
        )
    label_micro = (
        embedding_live_cost + macro_neighborhood_cost + micro_estimate.estimated_cost_usd
        <= cost_gate_usd
        and micro_estimate.estimated_cost_usd <= micro_label_cost_limit_usd
    )

    if not openai_api_key:
        raise Milestone4Error(
            "OPENAI_API_KEY is missing. Milestone 4 hierarchy labeling requires the key in .env."
        )

    layer_summaries = []
    labels_generated_count = 0
    labels_total_cost = 0.0
    for layer, assignments, evidence, cache_path, estimate in prepared_layers:
        should_label = layer != "micro" or label_micro
        candidates: list[ClusterLabelCandidate] = []
        cached_reused = 0
        new_labels = 0
        labels_path: Path | None = None
        if should_label:
            candidates, cached_reused, new_labels = label_clusters(
                evidence,
                cache=load_label_cache(cache_path),
                model=label_model,
                api_key=openai_api_key,
                batch_size=label_batch_size,
                client=label_client,
            )
            write_label_cache(cache_path, candidates)
            labels_path = hierarchy_dir / f"{layer}_cluster_labels.json"
            labels_path.write_text(
                json.dumps([candidate.to_dict() for candidate in candidates], indent=2, sort_keys=True),
                encoding="utf-8",
            )
            editable_path = hierarchy_dir / f"{layer}_human_editable_labels.json"
            editable_path.write_text(
                json.dumps(render_human_editable_labels(candidates), indent=2, sort_keys=True),
                encoding="utf-8",
            )
            labels_generated_count += len(candidates)
            labels_total_cost += estimate.estimated_cost_usd

        assignments_path = hierarchy_dir / f"{layer}_cluster_assignments.json"
        assignments_path.write_text(
            json.dumps(
                {
                    "layer": layer,
                    "clustering_method": "kmeans",
                    "cluster_count": len({item.cluster_id for item in assignments}),
                    "parent_layer": _parent_layer(layer),
                    "parents": parent_maps[layer],
                    "assignments": [assignment.to_dict() for assignment in assignments],
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        evidence_path = hierarchy_dir / f"{layer}_cluster_evidence.json"
        evidence_path.write_text(
            json.dumps([entry.to_dict() for entry in evidence], indent=2, sort_keys=True),
            encoding="utf-8",
        )
        layer_summaries.append(
            _layer_summary(
                layer=layer,  # type: ignore[arg-type]
                k={"macro": macro_k, "neighborhood": neighborhood_k, "micro": micro_k}[layer],
                assignments=assignments,
                evidence=evidence,
                labeled=should_label,
                estimate=estimate,
                cached_labels_reused=cached_reused,
                new_labels_generated=new_labels,
                candidates=candidates,
                assignments_path=assignments_path,
                evidence_path=evidence_path,
                labels_path=labels_path,
                cache_path=cache_path,
            )
        )

    result = HierarchyResult(
        movie_count=len(embeddings),
        embedding_model=embeddings[0].model,
        embedding_estimate=embedding_estimate,
        projection_method=resolved_projection,
        coordinates_path=coordinates_path,
        neighbors_path=neighbors_path,
        summary_path=hierarchy_dir / MILESTONE4_SUMMARY_FILENAME,
        layers=layer_summaries,
        labels_total_estimated_cost_usd=labels_total_cost,
        labels_generated_count=labels_generated_count,
    )
    result.summary_path.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return result


def project_embedding_records(
    records: list[Any],
    *,
    method: Literal["auto", "umap", "pca"] = "auto",
) -> tuple[list[CoordinateRecord], str]:
    """Project embeddings to 2D with UMAP when available, otherwise PCA."""
    if not records:
        return [], "none"
    matrix = np.array([record.embedding for record in records], dtype=float)
    if method in {"auto", "umap"} and len(records) >= 4:
        try:
            import umap  # type: ignore
        except Exception:
            if method == "umap":
                raise Milestone4Error("UMAP projection requested, but umap-learn is unavailable.")
        else:
            neighbors = min(30, max(2, len(records) - 1))
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=neighbors,
                min_dist=0.05,
                metric="cosine",
                random_state=42,
            )
            coords = reducer.fit_transform(matrix)
            return _coordinate_records(records, coords), "umap"
    coords = _pca_2d(matrix)
    return _coordinate_records(records, coords), "pca"


def export_atlas_data_file(
    *,
    movies_path: str | Path = "data/processed/movies.json",
    hierarchy_dir: str | Path = "outputs/intermediate/hierarchy",
    output_dir: str | Path = "outputs",
) -> PublicExportResult:
    """Write sanitized static JSON files for the future Astro frontend."""
    movies = _load_movies(movies_path)
    hierarchy_path = Path(hierarchy_dir)
    export_dir = Path(output_dir) / PUBLIC_EXPORT_DIRNAME
    export_dir.mkdir(parents=True, exist_ok=True)

    coordinates_payload = _read_json(hierarchy_path / "coordinates.json")
    coordinates = {
        int(item["tmdb_id"]): item for item in coordinates_payload.get("coordinates") or []
    }
    neighbors = [
        _sanitize_neighbors(entry) for entry in _read_json(hierarchy_path / "neighbors.json")
    ]
    layers = {
        "macro": _load_layer_export(hierarchy_path, "macro"),
        "neighborhood": _load_layer_export(hierarchy_path, "neighborhood"),
        "micro": _load_layer_export(hierarchy_path, "micro"),
    }
    assignment_maps = {
        layer: {
            int(item["tmdb_id"]): int(item["cluster_id"])
            for item in payload["assignments"].get("assignments") or []
        }
        for layer, payload in layers.items()
    }
    labels = _export_labels(layers)
    movie_rows = [_sanitize_movie(movie) for movie in movies]
    point_rows = []
    for movie in movies:
        coord = coordinates.get(movie.tmdb_id, {})
        point_rows.append(
            {
                "tmdb_id": movie.tmdb_id,
                "x": float(coord.get("x") or 0.0),
                "y": float(coord.get("y") or 0.0),
                "macro_id": assignment_maps["macro"].get(movie.tmdb_id),
                "neighborhood_id": assignment_maps["neighborhood"].get(movie.tmdb_id),
                "micro_id": assignment_maps["micro"].get(movie.tmdb_id),
            }
        )

    files = {
        "movies.json": movie_rows,
        "points.json": point_rows,
        "macro_clusters.json": _export_clusters(layers["macro"]),
        "neighborhood_clusters.json": _export_clusters(layers["neighborhood"]),
        "micro_clusters.json": _export_clusters(layers["micro"]),
        "neighbors.json": neighbors,
        "labels.json": labels,
    }
    manifest = {
        "project": "The Film Atlas",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "movie_count": len(movie_rows),
        "projection_method": coordinates_payload.get("projection_method"),
        "files": sorted([*files, "manifest.json"]),
        "layers": {
            layer: {
                "cluster_count": len(payload["clusters"]),
                "labeled": bool(payload["labels"]),
            }
            for layer, payload in layers.items()
        },
        "privacy": {
            "contains_raw_reviews": False,
            "contains_embeddings": False,
            "contains_api_keys": False,
        },
    }
    files["manifest.json"] = manifest

    file_sizes = {}
    for filename, payload in files.items():
        path = export_dir / filename
        path.write_text(json.dumps(payload, separators=(",", ":"), sort_keys=True), encoding="utf-8")
        file_sizes[filename] = path.stat().st_size

    return PublicExportResult(
        export_dir=export_dir,
        file_sizes=file_sizes,
        manifest_path=export_dir / "manifest.json",
    )


def milestone_4_file(
    client: TMDbClient,
    *,
    target: int = 2000,
    since_year: int = 1980,
    min_votes: int = 100,
    data_dir: str | Path = "data",
    output_dir: str | Path = "outputs",
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    label_model: str,
    openai_api_key: str | None,
    macro_k: int = 12,
    neighborhood_k: int = 75,
    micro_k: int = 200,
    embedding_batch_size: int = 64,
    label_batch_size: int = 5,
    cost_gate_usd: float = 5.0,
    refresh: bool = False,
    label_client: OpenAIClusterLabelClient | None = None,
) -> Milestone4Result:
    """Run the full Milestone 4 data, hierarchy, label, export, and report flow."""
    scale = scale_dataset_file(
        client,
        target=target,
        since_year=since_year,
        min_votes=min_votes,
        data_dir=data_dir,
        output_dir=output_dir,
        refresh=refresh,
    )
    embedding_path = Path(output_dir) / "intermediate" / "embeddings.jsonl"
    embedding_estimate = estimate_uncached_embedding_cost(
        profiles_path=scale.profiles_path,
        embeddings_path=embedding_path,
        model=embedding_model,
        limit=target,
    )
    approximate_label_cost = approximate_labeling_cost(
        cluster_count=macro_k + neighborhood_k + micro_k,
        model=label_model,
    )
    if embedding_estimate.estimated_live_cost_usd + approximate_label_cost > cost_gate_usd:
        raise Milestone4Error(
            f"Estimated Milestone 4 OpenAI cost is "
            f"${embedding_estimate.estimated_live_cost_usd + approximate_label_cost:.4f}, "
            f"which exceeds ${cost_gate_usd:.2f}. Pause here and get explicit approval."
        )
    if not openai_api_key:
        raise Milestone4Error(
            "OPENAI_API_KEY is missing. Milestone 4 embedding and labeling require the key in .env."
        )

    embedding = embed_profiles_file(
        api_key=openai_api_key,
        profiles_path=scale.profiles_path,
        output_dir=output_dir,
        model=embedding_model,
        limit=target,
        batch_size=embedding_batch_size,
    )
    hierarchy = build_hierarchy_file(
        movies_path=scale.movies_path,
        profiles_path=scale.profiles_path,
        embeddings_path=embedding.embedding_path,
        output_dir=output_dir,
        macro_k=macro_k,
        neighborhood_k=neighborhood_k,
        micro_k=micro_k,
        label_model=label_model,
        openai_api_key=openai_api_key,
        embedding_estimate=embedding_estimate,
        label_batch_size=label_batch_size,
        cost_gate_usd=cost_gate_usd,
        label_client=label_client,
    )
    export = export_atlas_data_file(
        movies_path=scale.movies_path,
        hierarchy_dir=Path(output_dir) / "intermediate" / HIERARCHY_DIRNAME,
        output_dir=output_dir,
    )
    report_path = generate_milestone_4_report_file(
        movies_path=scale.movies_path,
        output_dir=output_dir,
        scale=scale,
        embedding=embedding,
        hierarchy=hierarchy,
        export=export,
    )
    return Milestone4Result(
        scale=scale,
        embedding=embedding,
        hierarchy=hierarchy,
        export=export,
        report_path=report_path,
    )


def generate_milestone_4_report_file(
    *,
    movies_path: str | Path = "data/processed/movies.json",
    output_dir: str | Path = "outputs",
    scale: ScaleDatasetResult | None = None,
    embedding: EmbeddingRunResult | None = None,
    hierarchy: HierarchyResult | None = None,
    export: PublicExportResult | None = None,
) -> Path:
    """Render the Milestone 4 report from current artifacts."""
    movies = _load_movies(movies_path)
    output_path = Path(output_dir)
    if scale is None:
        scale = _load_scale_summary(output_path / "intermediate" / MILESTONE4_DIRNAME)
    if hierarchy is None:
        hierarchy = _load_hierarchy_summary(output_path / "intermediate" / HIERARCHY_DIRNAME)
    if export is None:
        export_dir = output_path / PUBLIC_EXPORT_DIRNAME
        export = PublicExportResult(
            export_dir=export_dir,
            file_sizes={
                path.name: path.stat().st_size
                for path in sorted(export_dir.glob("*.json"))
                if path.is_file()
            },
            manifest_path=export_dir / "manifest.json",
        )
    report_path = output_path / "reports" / MILESTONE4_REPORT_FILENAME
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        render_milestone_4_report(
            movies=movies,
            scale=scale,
            embedding=embedding,
            hierarchy=hierarchy,
            export=export,
        ),
        encoding="utf-8",
    )
    return report_path


def render_milestone_4_report(
    *,
    movies: list[MovieRecord],
    scale: ScaleDatasetResult | None,
    embedding: EmbeddingRunResult | None,
    hierarchy: HierarchyResult,
    export: PublicExportResult,
) -> str:
    """Render a human-readable Milestone 4 status report."""
    genre_counts = Counter(genre for movie in movies for genre in movie.genres)
    years = Counter((movie.year // 10) * 10 for movie in movies if movie.year is not None)
    keyword_coverage = _percent(sum(1 for movie in movies if movie.keywords), len(movies))
    review_coverage = _percent(sum(1 for movie in movies if movie.reviews), len(movies))
    overview_coverage = _percent(sum(1 for movie in movies if movie.overview), len(movies))
    label_cost = hierarchy.labels_total_estimated_cost_usd
    embedding_cost = (
        embedding.estimated_cost_usd
        if embedding
        else (hierarchy.embedding_estimate.estimated_full_cost_usd if hierarchy.embedding_estimate else 0.0)
    )

    lines = [
        "# The Film Atlas - Milestone 4 Report",
        "",
        "Milestone 4 creates a scaled English-language film dataset, light-review semantic "
        "profiles, full-vector embeddings, hierarchical k-means atlas layers, draft labels, "
        "2D map coordinates, and sanitized static JSON for a future Astro frontend.",
        "",
        "## Summary",
        "",
        f"- Movie count: {len(movies)}",
        f"- Dataset target: {scale.target if scale else 'n/a'}",
        f"- Since year: {scale.since_year if scale else 'n/a'}",
        f"- Minimum vote count: {scale.min_votes if scale else 'n/a'}",
        f"- Projection method: {hierarchy.projection_method}",
        f"- Embedding model: {hierarchy.embedding_model}",
        f"- Embedding estimated full cost: ${embedding_cost:.4f}",
        f"- Label estimated live cost: ${label_cost:.4f}",
        f"- Labels generated: {hierarchy.labels_generated_count}",
        f"- Public export: {export.export_dir}",
        "",
        "## Coverage",
        "",
        f"- Overviews: {overview_coverage}",
        f"- Keywords: {keyword_coverage}",
        f"- Reviews: {review_coverage}",
        "",
        "## Year Distribution",
        "",
        _counts_line(years, suffix="s"),
        "",
        "## Genre Distribution",
        "",
        _counts_line(genre_counts.most_common(15)),
        "",
        "## Embedding Run",
        "",
        _embedding_summary(embedding, hierarchy.embedding_estimate),
        "",
        "## Hierarchy Layers",
        "",
    ]
    for layer in hierarchy.layers:
        lines.extend(
            [
                f"### {layer.layer}",
                "",
                f"- k target: {layer.k}",
                f"- clusters: {layer.cluster_count}",
                f"- labeled: {layer.labeled}",
                f"- label cost: ${layer.label_estimate.estimated_cost_usd:.4f}",
                f"- label cache reused/new: {layer.cached_labels_reused}/{layer.new_labels_generated}",
                f"- coherence average/range: {_fmt_float(layer.coherence_average)} "
                f"({_fmt_float(layer.coherence_min)}-{_fmt_float(layer.coherence_max)})",
                f"- tiny clusters (<5 movies): {layer.tiny_cluster_count}",
                f"- size distribution: {', '.join(map(str, layer.cluster_sizes))}",
                "",
                "Sample labels:",
                "",
                _sample_labels(layer),
                "",
            ]
        )

    lines.extend(
        [
            "## Quality-Check Movie Neighbors",
            "",
            _quality_check_neighbors(Path(hierarchy.neighbors_path)),
            "",
            "## Frontend Export File Sizes",
            "",
            _export_sizes(export),
            "",
            "## Recommendation",
            "",
            _frontend_recommendation(len(movies), hierarchy, export),
            "",
        ]
    )
    return "\n".join(lines)


def approximate_labeling_cost(*, cluster_count: int, model: str) -> float:
    """Approximate label cost before embeddings and cluster evidence exist."""
    sample_messages = build_label_messages(
        [
            ClusterEvidence(
                cluster_id=0,
                cluster_size=20,
                representative_movies=["Sample A", "Sample B", "Sample C"],
                top_official_genres=[("Drama", 12), ("Thriller", 8)],
                top_tmdb_keywords=[("identity", 5), ("memory", 3)],
                aggregated_profile_terms=[("haunted city", 0.4), ("moral pressure", 0.3)],
                in_cluster_neighbor_pairs=[],
                coherence_score=0.5,
                warnings=[],
            )
        ]
    )
    input_tokens = estimate_text_tokens(json.dumps(sample_messages, ensure_ascii=False)) * cluster_count
    output_tokens = 450 * cluster_count
    from film_atlas.cluster_labels import FALLBACK_PRICE_PER_1M_TOKENS, LABEL_MODEL_PRICES_PER_1M_TOKENS

    price = LABEL_MODEL_PRICES_PER_1M_TOKENS.get(model, FALLBACK_PRICE_PER_1M_TOKENS)
    return (
        input_tokens / 1_000_000 * price["input"]
        + output_tokens / 1_000_000 * price["output"]
    )


def _write_movie_records(movies: list[MovieRecord], data_dir: Path) -> Path:
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    json_path = processed_dir / MOVIES_JSON_FILENAME
    json_path.write_text(
        json.dumps([movie.to_dict() for movie in movies], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    frame = pd.DataFrame([movie.to_dict() for movie in movies])
    frame.to_parquet(processed_dir / MOVIES_PARQUET_FILENAME, index=False)
    return json_path


def _scaled_detail_sort_key(detail: dict[str, Any]) -> tuple[int, int, float, str]:
    has_keywords = 1 if _detail_keywords(detail) else 0
    return (
        -has_keywords,
        -int(detail.get("vote_count") or 0),
        -float(detail.get("popularity") or 0),
        str(detail.get("title") or ""),
    )


def _detail_keywords(detail: dict[str, Any]) -> list[str]:
    keywords = detail.get("keywords")
    if not isinstance(keywords, dict):
        return []
    return [
        str(item.get("name"))
        for item in keywords.get("keywords") or []
        if isinstance(item, dict) and item.get("name")
    ]


def _year_from_release_date(value: str) -> int | None:
    if len(value) < 4 or not value[:4].isdigit():
        return None
    return int(value[:4])


def _load_movies(path: str | Path) -> list[MovieRecord]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [MovieRecord.from_dict(item) for item in payload]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _coordinate_records(records: list[Any], coords: np.ndarray) -> list[CoordinateRecord]:
    return [
        CoordinateRecord(
            tmdb_id=record.tmdb_id,
            title=record.title,
            x=float(coords[index, 0]),
            y=float(coords[index, 1]),
        )
        for index, record in enumerate(records)
    ]


def _pca_2d(matrix: np.ndarray) -> np.ndarray:
    rows, columns = matrix.shape
    if rows == 1:
        return np.array([[0.0, 0.0]])
    components = min(2, rows, columns)
    coords = PCA(n_components=components, random_state=42).fit_transform(matrix)
    if coords.shape[1] == 1:
        return np.column_stack([coords[:, 0], np.zeros(rows)])
    return coords


def _majority_parent_map(
    *,
    child_assignments: list[ClusterAssignment],
    parent_assignments: list[ClusterAssignment],
) -> dict[int, int]:
    parent_by_movie = {assignment.tmdb_id: assignment.cluster_id for assignment in parent_assignments}
    votes: dict[int, Counter[int]] = defaultdict(Counter)
    for assignment in child_assignments:
        parent_id = parent_by_movie.get(assignment.tmdb_id)
        if parent_id is not None:
            votes[assignment.cluster_id][parent_id] += 1
    return {
        child_id: parent_counts.most_common(1)[0][0]
        for child_id, parent_counts in votes.items()
        if parent_counts
    }


def _parent_layer(layer: str) -> str | None:
    if layer == "neighborhood":
        return "macro"
    if layer == "micro":
        return "neighborhood"
    return None


def _layer_summary(
    *,
    layer: LayerName,
    k: int,
    assignments: list[ClusterAssignment],
    evidence: list[ClusterEvidence],
    labeled: bool,
    estimate: LabelingEstimate,
    cached_labels_reused: int,
    new_labels_generated: int,
    candidates: list[ClusterLabelCandidate],
    assignments_path: Path,
    evidence_path: Path,
    labels_path: Path | None,
    cache_path: Path,
) -> LayerSummary:
    sizes = Counter(assignment.cluster_id for assignment in assignments)
    coherence_values = [
        item.coherence_score for item in evidence if item.coherence_score is not None
    ]
    return LayerSummary(
        layer=layer,
        k=k,
        cluster_count=len(sizes),
        labeled=labeled,
        cluster_sizes=sorted(sizes.values(), reverse=True),
        tiny_cluster_count=sum(1 for size in sizes.values() if size < 5),
        coherence_average=float(np.mean(coherence_values)) if coherence_values else None,
        coherence_min=min(coherence_values, default=None),
        coherence_max=max(coherence_values, default=None),
        label_estimate=estimate,
        cached_labels_reused=cached_labels_reused,
        new_labels_generated=new_labels_generated,
        sample_labels=[
            {
                "cluster_id": candidate.cluster_id,
                "recommended_label": candidate.recommended_label,
                "confidence_score": candidate.confidence_score,
                "description": candidate.one_sentence_description,
            }
            for candidate in sorted(candidates, key=lambda item: item.cluster_id)[:8]
        ],
        assignments_path=assignments_path,
        evidence_path=evidence_path,
        labels_path=labels_path,
        cache_path=cache_path,
    )


def _load_layer_export(hierarchy_path: Path, layer: str) -> dict[str, Any]:
    assignments = _read_json(hierarchy_path / f"{layer}_cluster_assignments.json")
    evidence = _read_json(hierarchy_path / f"{layer}_cluster_evidence.json")
    labels_path = hierarchy_path / f"{layer}_cluster_labels.json"
    labels = _read_json(labels_path) if labels_path.exists() else []
    clusters = {
        int(item["cluster_id"]): item
        for item in evidence
    }
    return {
        "layer": layer,
        "assignments": assignments,
        "clusters": clusters,
        "labels": labels,
    }


def _export_labels(layers: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for layer, payload in layers.items():
        for label in payload["labels"]:
            output.append(
                {
                    "label_id": f"{layer}:{label['cluster_id']}",
                    "layer": layer,
                    "cluster_id": int(label["cluster_id"]),
                    "recommended_label": label.get("recommended_label"),
                    "plain_label": label.get("plain_label"),
                    "description": label.get("one_sentence_description"),
                    "confidence_score": label.get("confidence_score"),
                }
            )
    return output


def _export_clusters(payload: dict[str, Any]) -> list[dict[str, Any]]:
    labels_by_id = {int(label["cluster_id"]): label for label in payload["labels"]}
    parents = {int(key): value for key, value in (payload["assignments"].get("parents") or {}).items()}
    clusters = []
    for cluster_id, evidence in sorted(payload["clusters"].items()):
        label = labels_by_id.get(cluster_id)
        clusters.append(
            {
                "cluster_id": cluster_id,
                "parent_cluster_id": parents.get(cluster_id),
                "size": int(evidence["cluster_size"]),
                "label_id": f"{payload['layer']}:{cluster_id}" if label else None,
                "recommended_label": label.get("recommended_label") if label else None,
                "description": label.get("one_sentence_description") if label else None,
                "representative_movies": evidence.get("representative_movies") or [],
                "top_genres": evidence.get("top_official_genres") or [],
                "top_keywords": evidence.get("top_tmdb_keywords") or [],
                "terms": [term for term, _score in evidence.get("aggregated_profile_terms") or []],
                "coherence_score": evidence.get("coherence_score"),
            }
        )
    return clusters


def _sanitize_movie(movie: MovieRecord) -> dict[str, Any]:
    return {
        "tmdb_id": movie.tmdb_id,
        "title": movie.title,
        "original_title": movie.original_title,
        "year": movie.year,
        "runtime": movie.runtime,
        "overview": movie.overview,
        "genres": movie.genres,
        "vote_average": movie.vote_average,
    }


def _sanitize_neighbors(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "tmdb_id": int(entry["tmdb_id"]),
        "neighbors": [
            {
                "tmdb_id": int(neighbor["tmdb_id"]),
                "title": neighbor["title"],
                "similarity": float(neighbor["similarity"]),
            }
            for neighbor in entry.get("neighbors") or []
        ],
    }


def _load_hierarchy_summary(hierarchy_dir: Path) -> HierarchyResult:
    payload = _read_json(hierarchy_dir / MILESTONE4_SUMMARY_FILENAME)
    layers = []
    for item in payload["layers"]:
        layers.append(
            LayerSummary(
                layer=item["layer"],
                k=int(item["k"]),
                cluster_count=int(item["cluster_count"]),
                labeled=bool(item["labeled"]),
                cluster_sizes=[int(value) for value in item["cluster_sizes"]],
                tiny_cluster_count=int(item["tiny_cluster_count"]),
                coherence_average=item.get("coherence_average"),
                coherence_min=item.get("coherence_min"),
                coherence_max=item.get("coherence_max"),
                label_estimate=LabelingEstimate(**item["label_estimate"]),
                cached_labels_reused=int(item["cached_labels_reused"]),
                new_labels_generated=int(item["new_labels_generated"]),
                sample_labels=list(item.get("sample_labels") or []),
                assignments_path=Path(item["assignments_path"]),
                evidence_path=Path(item["evidence_path"]),
                labels_path=Path(item["labels_path"]) if item.get("labels_path") else None,
                cache_path=Path(item["cache_path"]),
            )
        )
    embedding_estimate = None
    if payload.get("embedding_estimate"):
        embedding_estimate = EmbeddingCostEstimate(**payload["embedding_estimate"])
    return HierarchyResult(
        movie_count=int(payload["movie_count"]),
        embedding_model=str(payload["embedding_model"]),
        embedding_estimate=embedding_estimate,
        projection_method=str(payload["projection_method"]),
        coordinates_path=Path(payload["coordinates_path"]),
        neighbors_path=Path(payload["neighbors_path"]),
        summary_path=Path(payload["summary_path"]),
        layers=layers,
        labels_total_estimated_cost_usd=float(payload["labels_total_estimated_cost_usd"]),
        labels_generated_count=int(payload["labels_generated_count"]),
    )


def _load_scale_summary(milestone_dir: Path) -> ScaleDatasetResult | None:
    path = milestone_dir / SCALE_DATASET_SUMMARY_FILENAME
    if not path.exists():
        return None
    payload = _read_json(path)
    return ScaleDatasetResult(
        target=int(payload["target"]),
        selected_count=int(payload["selected_count"]),
        candidate_count=int(payload["candidate_count"]),
        detail_count=int(payload["detail_count"]),
        since_year=int(payload["since_year"]),
        min_votes=int(payload["min_votes"]),
        movies_path=Path(payload["movies_path"]),
        profiles_path=Path(payload["profiles_path"]),
        discover_path=Path(payload["discover_path"]),
        details_path=Path(payload["details_path"]),
        summary_path=Path(payload["summary_path"]),
    )


def _percent(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "0.0%"
    return f"{numerator / denominator * 100:.1f}%"


def _counts_line(items: Any, *, suffix: str = "") -> str:
    if isinstance(items, Counter):
        pairs = sorted(items.items())
    else:
        pairs = list(items)
    return ", ".join(f"{key}{suffix}: {count}" for key, count in pairs) or "none"


def _embedding_summary(
    embedding: EmbeddingRunResult | None,
    estimate: EmbeddingCostEstimate | None,
) -> str:
    if embedding:
        return "\n".join(
            [
                f"- Embedded count: {embedding.embedded_count}",
                f"- Cached/new embeddings: {embedding.cached_reused_count}/{embedding.new_embedding_count}",
                f"- Estimated tokens: {embedding.estimated_tokens}",
                f"- Estimated full cost: ${embedding.estimated_cost_usd:.4f}",
                f"- API prompt tokens: {embedding.api_prompt_tokens or 'n/a'}",
            ]
        )
    if estimate:
        return "\n".join(
            [
                f"- Profiles: {estimate.profile_count}",
                f"- Cached/new embeddings: {estimate.cached_count}/{estimate.new_count}",
                f"- Estimated full cost: ${estimate.estimated_full_cost_usd:.4f}",
                f"- Estimated live cost: ${estimate.estimated_live_cost_usd:.4f}",
            ]
        )
    return "- Embedding summary unavailable."


def _sample_labels(layer: LayerSummary) -> str:
    if not layer.sample_labels:
        return "_No labels generated for this layer._"
    return "\n".join(
        f"- Cluster {item['cluster_id']}: {item['recommended_label']} "
        f"({float(item['confidence_score']):.2f})"
        for item in layer.sample_labels
    )


def _quality_check_neighbors(neighbors_path: Path) -> str:
    payload = _read_json(neighbors_path)
    by_title = {str(item.get("title") or "").lower(): item for item in payload}
    lines = []
    for title in QUALITY_CHECK_MOVIES:
        item = by_title.get(title.lower())
        if not item:
            continue
        lines.append(
            f"- {item['title']}: "
            + ", ".join(
                f"{neighbor['title']} ({float(neighbor['similarity']):.3f})"
                for neighbor in (item.get("neighbors") or [])[:5]
            )
        )
    return "\n".join(lines) or "_No quality-check movies present._"


def _export_sizes(export: PublicExportResult) -> str:
    return "\n".join(
        f"- {filename}: {size:,} bytes"
        for filename, size in sorted(export.file_sizes.items())
    )


def _frontend_recommendation(
    movie_count: int,
    hierarchy: HierarchyResult,
    export: PublicExportResult,
) -> str:
    required_files = {
        "manifest.json",
        "movies.json",
        "points.json",
        "macro_clusters.json",
        "neighborhood_clusters.json",
        "micro_clusters.json",
        "neighbors.json",
        "labels.json",
    }
    missing = sorted(required_files.difference(export.file_sizes))
    labeled_macro_neighborhood = all(
        layer.labeled for layer in hierarchy.layers if layer.layer in {"macro", "neighborhood"}
    )
    if movie_count >= 1500 and not missing and labeled_macro_neighborhood:
        return (
            "Proceed to Astro frontend planning: the scaled dataset, hierarchy, labels, "
            "neighbors, projection, and sanitized export files are present."
        )
    return (
        "Hold before Astro frontend: "
        + ("missing export files " + ", ".join(missing) if missing else "scale or labels need review")
        + "."
    )


def _fmt_float(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"
