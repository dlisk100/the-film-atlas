"""Large-scale semantic QA for the public Film Atlas export."""

from __future__ import annotations

import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha256
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from film_atlas.config import MissingCredentialsError

LARGE_AUDIT_JSON_FILENAME = "large_audit.json"
LARGE_AUDIT_REPORT_FILENAME = "milestone_5_large_audit.md"

AUDIT_MODEL_PRICES_PER_1M_TOKENS = {
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
}

LAYER_ORDER = ("macro", "neighborhood", "micro")


class LargeAuditError(RuntimeError):
    """Raised when the large audit cannot complete."""


@dataclass(frozen=True, slots=True)
class LargeAuditResult:
    json_path: Path
    report_path: Path
    reviewed_count: int
    verdict_counts: dict[str, int]
    estimated_cost_usd: float


class OpenAIAuditClient:
    """Small HTTP client for LLM-assisted audit batches."""

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
        timeout_config = httpx.Timeout(
            timeout,
            connect=min(10.0, timeout),
            read=timeout,
            write=min(20.0, timeout),
            pool=min(10.0, timeout),
        )
        self.client = httpx.Client(base_url=base_url, timeout=timeout_config, transport=transport)

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> OpenAIAuditClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def audit_batch(self, rows: list[dict[str, Any]], *, model: str) -> dict[str, Any]:
        if not self.api_key:
            raise MissingCredentialsError(
                "OPENAI_API_KEY is required for the LLM-assisted large audit. "
                "Add it to .env, then run the command again."
            )
        payload = {
            "model": model,
            "temperature": 0.0,
            "max_tokens": max(900, len(rows) * 140),
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are auditing a movie vibe atlas. Judge whether each movie's "
                        "macro, neighborhood, micro, and nearest-neighbor display are plausible "
                        "from the supplied TMDb overview, genres, keywords, and neighbors. "
                        "Keep poetic vibe labels acceptable. Do not demand generic genre labels. "
                        "Only mark fail for egregious contradictions like an adult live-action "
                        "film called family animation, a non-space film called space mission, "
                        "or obvious title-confusion neighbors. Do not mark fail merely because "
                        "one supplied keyword looks noisy when the genres, overview, labels, "
                        "and neighbors are otherwise plausible. Use mixed for defensible but "
                        "noticeably imprecise classifications. Return strict JSON only."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "schema": {
                                "results": [
                                    {
                                        "tmdb_id": "integer",
                                        "verdict": "pass|mixed|fail",
                                        "severity": "0 pass, 1 mild, 2 real issue, 3 egregious",
                                        "issue_type": "none|label|neighbor|both",
                                        "layer": "none|macro|neighborhood|micro|neighbors|path",
                                        "rationale": "short reason, <= 140 chars",
                                        "suggested_fix": "short optional cluster-label or neighbor fix",
                                    }
                                ]
                            },
                            "movies": rows,
                        },
                        ensure_ascii=True,
                    ),
                },
            ],
        }
        response_json = self._post_with_retries(
            "/chat/completions",
            payload=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        choice = (response_json.get("choices") or [{}])[0]
        content = ((choice.get("message") or {}).get("content") or "").strip()
        parsed = parse_json_content(content)
        return {
            "results": parsed if isinstance(parsed, list) else parsed.get("results") or [],
            "usage": response_json.get("usage") or {},
        }

    def _post_with_retries(
        self,
        endpoint: str,
        *,
        payload: dict[str, Any],
        headers: dict[str, str],
    ) -> dict[str, Any]:
        retry_statuses = {429, 500, 502, 503, 504}
        last_response: httpx.Response | None = None
        last_error: httpx.RequestError | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.post(endpoint, json=payload, headers=headers)
            except httpx.RequestError as exc:
                last_error = exc
                if attempt < self.max_retries:
                    self.sleep(self.backoff_seconds * 2**attempt)
                    continue
                raise LargeAuditError(
                    f"OpenAI audit request failed after retries: {type(exc).__name__}"
                ) from exc
            last_response = response
            if response.status_code not in retry_statuses:
                response.raise_for_status()
                return response.json()
            if attempt < self.max_retries:
                retry_after = response.headers.get("Retry-After")
                delay = float(retry_after) if retry_after else self.backoff_seconds * 2**attempt
                self.sleep(delay)
        if last_response is None:
            if last_error is not None:
                raise LargeAuditError(
                    f"OpenAI audit request failed after retries: {type(last_error).__name__}"
                ) from last_error
            raise LargeAuditError("OpenAI audit request failed before receiving a response.")
        last_response.raise_for_status()
        return last_response.json()


def large_audit_file(
    *,
    api_key: str | None,
    export_dir: str | Path = "outputs/public_export",
    output_dir: str | Path = "outputs",
    model: str = "gpt-4.1-mini",
    review_count: int = 1000,
    batch_size: int = 25,
    workers: int = 1,
    seed: int = 42,
    client: OpenAIAuditClient | None = None,
) -> LargeAuditResult:
    """Run a reproducible large semantic QA pass over the public export."""
    resolved_export_dir = Path(export_dir)
    export = load_public_export(resolved_export_dir)
    export_signature = public_export_signature(export)
    structural = structural_audit(export)
    selected_ids = select_audit_ids(export, structural, review_count=review_count, seed=seed)
    rows = [build_audit_row(export, tmdb_id) for tmdb_id in selected_ids]
    checkpoint_path = (
        Path(output_dir)
        / "experiments"
        / "classification_v2"
        / f"large_audit_batches_{export_signature[:12]}.jsonl"
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    completed_by_id, usage = load_audit_checkpoint(
        checkpoint_path,
        model=model,
        export_signature=export_signature,
    )

    llm_results: list[dict[str, Any]] = list(completed_by_id.values())
    worker_count = max(1, int(workers))
    if client is not None:
        worker_count = 1
    owns_client = client is None and worker_count == 1
    active_client = client or (OpenAIAuditClient(api_key) if worker_count == 1 else None)
    try:
        batches = _chunks(rows, batch_size)
        if worker_count == 1:
            if active_client is None:
                raise LargeAuditError("Large audit client was not initialized.")
            for index, batch in enumerate(batches, start=1):
                pending_batch = [row for row in batch if row["tmdb_id"] not in completed_by_id]
                if not pending_batch:
                    print(f"large-audit batch {index}/{len(batches)} already complete", flush=True)
                    continue
                print(
                    f"large-audit batch {index}/{len(batches)} reviewing "
                    f"{len(pending_batch)} movies",
                    flush=True,
                )
                payload = audit_rows_with_completion(active_client, pending_batch, model=model)
                normalized = normalize_llm_results(
                    payload["results"],
                    expected_ids={row["tmdb_id"] for row in pending_batch},
                )
                llm_results.extend(normalized)
                for row in normalized:
                    completed_by_id[int(row["tmdb_id"])] = row
                for key in usage:
                    usage[key] += int((payload.get("usage") or {}).get(key) or 0)
                append_audit_checkpoint(
                    checkpoint_path,
                    {
                        "model": model,
                        "export_signature": export_signature,
                        "batch_index": index,
                        "batch_count": len(batches),
                        "results": normalized,
                        "usage": payload.get("usage") or {},
                    },
                )
        else:
            completed_payloads = _audit_batches_parallel(
                api_key=api_key,
                batches=batches,
                completed_by_id=completed_by_id,
                model=model,
                batch_count=len(batches),
                workers=worker_count,
            )
            for payload in sorted(completed_payloads, key=lambda item: int(item["batch_index"])):
                normalized = payload["results"]
                llm_results.extend(normalized)
                for row in normalized:
                    completed_by_id[int(row["tmdb_id"])] = row
                for key in usage:
                    usage[key] += int((payload.get("usage") or {}).get(key) or 0)
                append_audit_checkpoint(
                    checkpoint_path,
                    {
                        "model": model,
                        "export_signature": export_signature,
                        "batch_index": payload["batch_index"],
                        "batch_count": len(batches),
                        "results": normalized,
                        "usage": payload.get("usage") or {},
                    },
                )
    finally:
        if owns_client and active_client is not None:
            active_client.close()

    result_by_id = {int(row["tmdb_id"]): row for row in llm_results}
    reviewed = []
    for row in rows:
        verdict = result_by_id.get(row["tmdb_id"])
        if verdict is None:
            verdict = {
                "tmdb_id": row["tmdb_id"],
                "verdict": "mixed",
                "severity": 2,
                "issue_type": "label",
                "layer": "path",
                "rationale": "missing LLM result for this row",
                "suggested_fix": "rerun this batch",
            }
        reviewed.append({**row, "llm_audit": verdict, "structural_flags": structural["flags_by_id"].get(row["tmdb_id"], [])})

    payload = {
        "model": model,
        "export_signature": export_signature,
        "reviewed_count": len(reviewed),
        "export_movie_count": len(export["movies"]),
        "selection": {
            "requested_review_count": review_count,
            "seed": seed,
            "selected_all_movies": len(reviewed) == len(export["movies"]),
        },
        "usage": usage,
        "estimated_cost_usd": estimate_audit_cost(model, usage),
        "verdict_counts": dict(Counter(item["llm_audit"]["verdict"] for item in reviewed)),
        "severity_counts": dict(Counter(str(item["llm_audit"]["severity"]) for item in reviewed)),
        "issue_type_counts": dict(Counter(item["llm_audit"]["issue_type"] for item in reviewed)),
        "layer_counts": dict(Counter(item["llm_audit"]["layer"] for item in reviewed)),
        "structural_summary": structural["summary"],
        "top_issues": top_issue_rows(reviewed, limit=80),
        "reviewed_movies": reviewed,
    }

    output_path = Path(output_dir)
    json_path = output_path / "experiments" / "classification_v2" / LARGE_AUDIT_JSON_FILENAME
    report_path = output_path / "reports" / LARGE_AUDIT_REPORT_FILENAME
    json_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    report_path.write_text(render_large_audit_report(payload), encoding="utf-8")
    return LargeAuditResult(
        json_path=json_path,
        report_path=report_path,
        reviewed_count=len(reviewed),
        verdict_counts=payload["verdict_counts"],
        estimated_cost_usd=payload["estimated_cost_usd"],
    )


def load_public_export(export_dir: Path) -> dict[str, Any]:
    movies = _read_json(export_dir / "movies.json")
    points = _read_json(export_dir / "points.json")
    neighbors = _read_json(export_dir / "neighbors.json")
    clusters = {
        "macro": _read_json(export_dir / "macro_clusters.json"),
        "neighborhood": _read_json(export_dir / "neighborhood_clusters.json"),
        "micro": _read_json(export_dir / "micro_clusters.json"),
    }
    return {
        "movies": movies,
        "points": points,
        "neighbors": neighbors,
        "clusters": clusters,
        "movies_by_id": {int(movie["tmdb_id"]): movie for movie in movies},
        "points_by_id": {int(point["tmdb_id"]): point for point in points},
        "neighbors_by_id": {int(entry["tmdb_id"]): entry for entry in neighbors},
        "clusters_by_layer": {
            layer: {int(cluster["cluster_id"]): cluster for cluster in layer_clusters}
            for layer, layer_clusters in clusters.items()
        },
    }


def public_export_signature(export: dict[str, Any]) -> str:
    """Hash the public classification state so checkpoints cannot cross exports."""
    payload = {
        "points": sorted(
            [
                {
                    "tmdb_id": point["tmdb_id"],
                    "macro_id": point["macro_id"],
                    "neighborhood_id": point["neighborhood_id"],
                    "micro_id": point["micro_id"],
                }
                for point in export["points"]
            ],
            key=lambda point: int(point["tmdb_id"]),
        ),
        "labels": [
            {
                "layer": layer,
                "cluster_id": cluster["cluster_id"],
                "parent_cluster_id": cluster.get("parent_cluster_id"),
                "recommended_label": cluster.get("recommended_label"),
            }
            for layer in LAYER_ORDER
            for cluster in sorted(
                export["clusters"][layer],
                key=lambda cluster: int(cluster["cluster_id"]),
            )
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(encoded).hexdigest()


def structural_audit(export: dict[str, Any]) -> dict[str, Any]:
    flags_by_id: dict[int, list[dict[str, Any]]] = defaultdict(list)
    duplicate_label_count = 0
    label_contradiction_count = 0
    title_confusion_count = 0
    low_neighbor_overlap_count = 0

    for movie in export["movies"]:
        tmdb_id = int(movie["tmdb_id"])
        point = export["points_by_id"].get(tmdb_id)
        if not point:
            flags_by_id[tmdb_id].append({"type": "missing_point", "severity": 3})
            continue
        labels = labels_for_point(export, point)
        if _has_duplicate_path_label(labels):
            duplicate_label_count += 1
            flags_by_id[tmdb_id].append({"type": "duplicate_path_label", "severity": 2})

        contradictions = label_contradictions(movie, labels)
        label_contradiction_count += len(contradictions)
        flags_by_id[tmdb_id].extend(contradictions)

        neighbor_flags = neighbor_flags_for_movie(export, movie)
        for flag in neighbor_flags:
            if flag["type"] == "possible_title_confusion":
                title_confusion_count += 1
            if flag["type"] == "low_neighbor_genre_overlap":
                low_neighbor_overlap_count += 1
        flags_by_id[tmdb_id].extend(neighbor_flags)

    cluster_summary = cluster_quality_summary(export)
    return {
        "flags_by_id": {tmdb_id: flags for tmdb_id, flags in flags_by_id.items() if flags},
        "summary": {
            "movies_with_flags": sum(1 for flags in flags_by_id.values() if flags),
            "duplicate_path_label_movies": duplicate_label_count,
            "label_contradiction_flags": label_contradiction_count,
            "possible_title_confusion_flags": title_confusion_count,
            "low_neighbor_genre_overlap_flags": low_neighbor_overlap_count,
            "cluster_quality": cluster_summary,
        },
    }


def select_audit_ids(
    export: dict[str, Any],
    structural: dict[str, Any],
    *,
    review_count: int,
    seed: int,
) -> list[int]:
    movies = export["movies"]
    target = min(max(review_count, 1), len(movies))
    if target == len(movies):
        return [int(movie["tmdb_id"]) for movie in movies]

    rng = random.Random(seed)
    selected: set[int] = set()
    by_micro: dict[int, list[int]] = defaultdict(list)
    for point in export["points"]:
        by_micro[int(point["micro_id"])].append(int(point["tmdb_id"]))
    micro_groups = list(by_micro.values())
    rng.shuffle(micro_groups)
    for tmdb_ids in micro_groups:
        if len(selected) >= target:
            break
        selected.add(rng.choice(sorted(tmdb_ids)))

    scored = []
    flags_by_id = structural["flags_by_id"]
    for movie in movies:
        tmdb_id = int(movie["tmdb_id"])
        flags = flags_by_id.get(tmdb_id, [])
        popularity = float(movie.get("popularity") or 0)
        vote_count = int(movie.get("vote_count") or 0)
        score = (
            sum(int(flag.get("severity") or 1) for flag in flags) * 10
            + min(10.0, popularity / 20)
            + min(8.0, vote_count / 2500)
            + rng.random()
        )
        scored.append((score, tmdb_id))
    for _score, tmdb_id in sorted(scored, reverse=True):
        selected.add(tmdb_id)
        if len(selected) >= target:
            break
    return sorted(selected, key=lambda tmdb_id: export["movies_by_id"][tmdb_id]["title"].lower())


def build_audit_row(export: dict[str, Any], tmdb_id: int) -> dict[str, Any]:
    movie = export["movies_by_id"][tmdb_id]
    point = export["points_by_id"][tmdb_id]
    labels = labels_for_point(export, point)
    neighbor_entry = export["neighbors_by_id"].get(tmdb_id, {})
    neighbors = []
    for neighbor in (neighbor_entry.get("neighbors") or [])[:5]:
        neighbor_movie = export["movies_by_id"].get(int(neighbor["tmdb_id"]), {})
        neighbors.append(
            {
                "title": neighbor.get("title"),
                "year": neighbor_movie.get("year"),
                "genres": neighbor_movie.get("genres") or [],
                "similarity": round(float(neighbor.get("similarity") or 0), 4),
            }
        )
    return {
        "tmdb_id": tmdb_id,
        "title": movie.get("title"),
        "year": movie.get("year"),
        "genres": movie.get("genres") or [],
        "keywords": (movie.get("keywords") or [])[:12],
        "overview": _shorten(movie.get("overview") or "", 300),
        "labels": labels,
        "neighbors": neighbors,
    }


def labels_for_point(export: dict[str, Any], point: dict[str, Any]) -> dict[str, str]:
    return {
        layer: str(
            export["clusters_by_layer"][layer]
            .get(int(point[f"{layer}_id"]), {})
            .get("recommended_label")
            or ""
        )
        for layer in LAYER_ORDER
    }


def label_contradictions(movie: dict[str, Any], labels: dict[str, str]) -> list[dict[str, Any]]:
    text_by_layer = {layer: _norm(label) for layer, label in labels.items()}
    genre_text = _norm(" ".join(movie.get("genres") or []))
    keyword_text = _norm(" ".join(movie.get("keywords") or []))
    combined = f"{genre_text} {keyword_text}"
    flags = []
    checks = [
        (("animation", "animated", "cartoon"), ("animation",), "animation_label_without_animation"),
        (("musical", "music"), ("music", "musical", "musician", "singer", "band"), "music_label_without_music_signal"),
        (("space", "astronaut", "cosmic"), ("science fiction", "space", "astronaut", "alien"), "space_label_without_space_signal"),
        (("alien",), ("science fiction", "alien"), "alien_label_without_sci_fi_signal"),
        (("horror", "haunting", "dread"), ("horror", "thriller", "supernatural"), "horror_label_without_horror_signal"),
        (("war", "battlefield", "soldier"), ("war", "history", "military", "soldier"), "war_label_without_war_signal"),
    ]
    for layer, label_text in text_by_layer.items():
        if layer == "macro":
            continue
        for label_terms, required_terms, flag_type in checks:
            if not any(term in label_text for term in label_terms):
                continue
            if any(term in combined for term in required_terms):
                continue
            flags.append(
                {
                    "type": flag_type,
                    "severity": 2 if layer == "micro" else 1,
                    "layer": layer,
                    "label": labels[layer],
                }
            )
    return flags


def neighbor_flags_for_movie(export: dict[str, Any], movie: dict[str, Any]) -> list[dict[str, Any]]:
    entry = export["neighbors_by_id"].get(int(movie["tmdb_id"]), {})
    neighbors = entry.get("neighbors") or []
    if not neighbors:
        return [{"type": "missing_neighbors", "severity": 2}]
    source_genres = set(movie.get("genres") or [])
    source_tokens = _title_tokens(movie.get("title") or "")
    top_neighbors = neighbors[:5]
    overlap_count = 0
    flags = []
    for neighbor in top_neighbors:
        neighbor_movie = export["movies_by_id"].get(int(neighbor["tmdb_id"]), {})
        neighbor_genres = set(neighbor_movie.get("genres") or [])
        if source_genres.intersection(neighbor_genres):
            overlap_count += 1
        neighbor_tokens = _title_tokens(neighbor.get("title") or "")
        shared_title_tokens = source_tokens.intersection(neighbor_tokens)
        if shared_title_tokens and not source_genres.intersection(neighbor_genres):
            flags.append(
                {
                    "type": "possible_title_confusion",
                    "severity": 2,
                    "neighbor": neighbor.get("title"),
                    "shared_tokens": sorted(shared_title_tokens),
                }
            )
    if source_genres and overlap_count <= 1:
        flags.append(
            {
                "type": "low_neighbor_genre_overlap",
                "severity": 1,
                "overlap_count_top5": overlap_count,
            }
        )
    return flags


def cluster_quality_summary(export: dict[str, Any]) -> dict[str, Any]:
    points_by_layer = {
        layer: defaultdict(list)
        for layer in LAYER_ORDER
    }
    for point in export["points"]:
        for layer in LAYER_ORDER:
            points_by_layer[layer][int(point[f"{layer}_id"])].append(int(point["tmdb_id"]))
    output = {}
    for layer in LAYER_ORDER:
        weak = []
        for cluster in export["clusters"][layer]:
            tmdb_ids = points_by_layer[layer][int(cluster["cluster_id"])]
            genre_counter = Counter(
                genre
                for tmdb_id in tmdb_ids
                for genre in export["movies_by_id"][tmdb_id].get("genres", [])
            )
            dominant_share = 0.0
            if tmdb_ids and genre_counter:
                dominant_share = genre_counter.most_common(1)[0][1] / len(tmdb_ids)
            coherence = cluster.get("coherence_score")
            if (coherence is not None and float(coherence) < 0.57) or dominant_share < 0.35:
                weak.append(
                    {
                        "cluster_id": cluster["cluster_id"],
                        "label": cluster.get("recommended_label"),
                        "size": len(tmdb_ids),
                        "coherence": coherence,
                        "dominant_genre_share": round(dominant_share, 3),
                    }
                )
        output[layer] = {
            "cluster_count": len(export["clusters"][layer]),
            "weak_cluster_count": len(weak),
            "weak_clusters": sorted(
                weak,
                key=lambda item: (
                    item["coherence"] if item["coherence"] is not None else 0,
                    item["dominant_genre_share"],
                ),
            )[:12],
        }
    return output


def normalize_llm_results(
    results: list[dict[str, Any]],
    *,
    expected_ids: set[int],
) -> list[dict[str, Any]]:
    output = []
    seen_ids: set[int] = set()
    for item in results:
        try:
            tmdb_id = int(item["tmdb_id"])
        except (KeyError, TypeError, ValueError):
            continue
        if tmdb_id not in expected_ids:
            continue
        if tmdb_id in seen_ids:
            continue
        seen_ids.add(tmdb_id)
        verdict = str(item.get("verdict") or "mixed").lower()
        if verdict not in {"pass", "mixed", "fail"}:
            verdict = "mixed"
        try:
            severity = int(item.get("severity") or 0)
        except (TypeError, ValueError):
            severity = 1
        output.append(
            {
                "tmdb_id": tmdb_id,
                "verdict": verdict,
                "severity": max(0, min(3, severity)),
                "issue_type": _choice(item.get("issue_type"), {"none", "label", "neighbor", "both"}, "label"),
                "layer": _choice(
                    item.get("layer"),
                    {"none", "macro", "neighborhood", "micro", "neighbors", "path"},
                    "path",
                ),
                "rationale": _shorten(str(item.get("rationale") or ""), 180),
                "suggested_fix": _shorten(str(item.get("suggested_fix") or ""), 180),
            }
        )
    return output


def audit_rows_with_fallback(
    client: OpenAIAuditClient,
    rows: list[dict[str, Any]],
    *,
    model: str,
) -> dict[str, Any]:
    try:
        return client.audit_batch(rows, model=model)
    except LargeAuditError:
        if len(rows) <= 1:
            row = rows[0]
            return {
                "results": [
                    {
                        "tmdb_id": row["tmdb_id"],
                        "verdict": "mixed",
                        "severity": 2,
                        "issue_type": "label",
                        "layer": "path",
                        "rationale": "audit response parse failed for this single row",
                        "suggested_fix": "rerun this row",
                    }
                ],
                "usage": {},
            }
        midpoint = len(rows) // 2
        left = audit_rows_with_fallback(client, rows[:midpoint], model=model)
        right = audit_rows_with_fallback(client, rows[midpoint:], model=model)
        usage = {
            key: int((left.get("usage") or {}).get(key) or 0)
            + int((right.get("usage") or {}).get(key) or 0)
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        }
        return {
            "results": [*(left.get("results") or []), *(right.get("results") or [])],
            "usage": usage,
        }


def audit_rows_with_completion(
    client: OpenAIAuditClient,
    rows: list[dict[str, Any]],
    *,
    model: str,
) -> dict[str, Any]:
    """Audit rows and retry any row omitted from an otherwise valid model response."""
    payload = audit_rows_with_fallback(client, rows, model=model)
    expected_ids = {int(row["tmdb_id"]) for row in rows}
    normalized = normalize_llm_results(payload.get("results") or [], expected_ids=expected_ids)
    present_ids = {int(row["tmdb_id"]) for row in normalized}
    missing_rows = [row for row in rows if int(row["tmdb_id"]) not in present_ids]
    usage = {
        key: int((payload.get("usage") or {}).get(key) or 0)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }
    if not missing_rows:
        return {"results": normalized, "usage": usage}

    if len(rows) <= 1:
        row = rows[0]
        normalized.append(
            {
                "tmdb_id": row["tmdb_id"],
                "verdict": "mixed",
                "severity": 2,
                "issue_type": "label",
                "layer": "path",
                "rationale": "audit response omitted this single row",
                "suggested_fix": "rerun this row",
            }
        )
        return {"results": normalized, "usage": usage}

    retry_payload = audit_rows_with_completion(client, missing_rows, model=model)
    retry_results = normalize_llm_results(
        retry_payload.get("results") or [],
        expected_ids={int(row["tmdb_id"]) for row in missing_rows},
    )
    retry_by_id = {int(row["tmdb_id"]): row for row in retry_results}
    normalized.extend(retry_by_id[tmdb_id] for tmdb_id in sorted(retry_by_id))
    for key in usage:
        usage[key] += int((retry_payload.get("usage") or {}).get(key) or 0)
    return {"results": normalized, "usage": usage}


def _audit_batches_parallel(
    *,
    api_key: str | None,
    batches: list[list[dict[str, Any]]],
    completed_by_id: dict[int, dict[str, Any]],
    model: str,
    batch_count: int,
    workers: int,
) -> list[dict[str, Any]]:
    pending: list[tuple[int, list[dict[str, Any]]]] = []
    for index, batch in enumerate(batches, start=1):
        pending_batch = [row for row in batch if row["tmdb_id"] not in completed_by_id]
        if not pending_batch:
            print(f"large-audit batch {index}/{batch_count} already complete", flush=True)
            continue
        pending.append((index, pending_batch))

    if not pending:
        return []

    print(
        f"large-audit running {len(pending)} pending batches with {workers} workers",
        flush=True,
    )
    completed: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_audit_single_batch, api_key, batch, model): (index, batch)
            for index, batch in pending
        }
        for future in as_completed(futures):
            index, batch = futures[future]
            payload = future.result()
            normalized = normalize_llm_results(
                payload["results"],
                expected_ids={row["tmdb_id"] for row in batch},
            )
            completed.append(
                {
                    "batch_index": index,
                    "results": normalized,
                    "usage": payload.get("usage") or {},
                }
            )
            print(
                f"large-audit batch {index}/{batch_count} complete "
                f"with {len(normalized)} results",
                flush=True,
            )
    return completed


def _audit_single_batch(
    api_key: str | None,
    rows: list[dict[str, Any]],
    model: str,
) -> dict[str, Any]:
    with OpenAIAuditClient(api_key) as client:
        return audit_rows_with_completion(client, rows, model=model)


def parse_json_content(content: str) -> Any:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(content[start : end + 1])
            except json.JSONDecodeError:
                pass
        raise LargeAuditError("OpenAI audit response was not valid JSON.")


def load_audit_checkpoint(
    path: Path,
    *,
    model: str,
    export_signature: str,
) -> tuple[dict[int, dict[str, Any]], dict[str, int]]:
    completed: dict[int, dict[str, Any]] = {}
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    if not path.exists():
        return completed, usage
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if payload.get("model") != model or payload.get("export_signature") != export_signature:
            continue
        for result in payload.get("results") or []:
            try:
                completed[int(result["tmdb_id"])] = result
            except (KeyError, TypeError, ValueError):
                continue
        for key in usage:
            usage[key] += int((payload.get("usage") or {}).get(key) or 0)
    return completed, usage


def append_audit_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True))
        handle.write("\n")


def top_issue_rows(reviewed: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    issue_rows = [
        row
        for row in reviewed
        if row["llm_audit"]["verdict"] != "pass" or row.get("structural_flags")
    ]
    return sorted(
        issue_rows,
        key=lambda row: (
            row["llm_audit"]["severity"],
            1 if row["llm_audit"]["verdict"] == "fail" else 0,
            len(row.get("structural_flags") or []),
        ),
        reverse=True,
    )[:limit]


def render_large_audit_report(payload: dict[str, Any]) -> str:
    lines = [
        "# The Film Atlas - Milestone 5 Large Audit",
        "",
        "This report is an LLM-assisted semantic QA pass over the public atlas export. "
        "It uses only the exported movie metadata, labels, and neighbor lists; raw private "
        "reviews and embeddings are not included here.",
        "",
        "## Summary",
        "",
        f"- Reviewed movies: {payload['reviewed_count']} / {payload['export_movie_count']}",
        f"- Model: {payload['model']}",
        f"- Estimated audit cost: ${payload['estimated_cost_usd']:.4f}",
        f"- Verdict counts: {_format_counts(payload['verdict_counts'])}",
        f"- Severity counts: {_format_counts(payload['severity_counts'])}",
        f"- Issue types: {_format_counts(payload['issue_type_counts'])}",
        f"- Layers: {_format_counts(payload['layer_counts'])}",
        "",
        "## Structural Scan",
        "",
    ]
    summary = payload["structural_summary"]
    lines.extend(
        [
            f"- Movies with heuristic flags: {summary['movies_with_flags']}",
            f"- Duplicate path-label movies: {summary['duplicate_path_label_movies']}",
            f"- Label contradiction flags: {summary['label_contradiction_flags']}",
            f"- Possible title-confusion flags: {summary['possible_title_confusion_flags']}",
            f"- Low top-5 neighbor genre-overlap flags: {summary['low_neighbor_genre_overlap_flags']}",
            "",
            "## Weak Cluster Signals",
            "",
        ]
    )
    for layer, layer_summary in summary["cluster_quality"].items():
        lines.append(
            f"- {layer}: {layer_summary['weak_cluster_count']} weak/coarse clusters "
            f"out of {layer_summary['cluster_count']}"
        )
    lines.extend(
        [
            "",
            "## Top Issue Rows",
            "",
            "| Movie | Verdict | Severity | Layer | Labels | Neighbors | Rationale | Suggested fix |",
            "| --- | --- | ---: | --- | --- | --- | --- | --- |",
        ]
    )
    for row in payload["top_issues"]:
        audit = row["llm_audit"]
        labels = " / ".join(row["labels"][layer] for layer in LAYER_ORDER)
        neighbors = ", ".join(
            f"{neighbor['title']} ({neighbor.get('year') or 'n/a'})"
            for neighbor in row.get("neighbors", [])[:5]
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    _table_escape(f"{row['title']} ({row.get('year') or 'n/a'})"),
                    audit["verdict"],
                    str(audit["severity"]),
                    audit["layer"],
                    _table_escape(labels),
                    _table_escape(neighbors),
                    _table_escape(audit["rationale"]),
                    _table_escape(audit["suggested_fix"]),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def estimate_audit_cost(model: str, usage: dict[str, int]) -> float:
    prices = AUDIT_MODEL_PRICES_PER_1M_TOKENS.get(
        model,
        AUDIT_MODEL_PRICES_PER_1M_TOKENS["gpt-4.1-mini"],
    )
    return (
        usage.get("prompt_tokens", 0) / 1_000_000 * prices["input"]
        + usage.get("completion_tokens", 0) / 1_000_000 * prices["output"]
    )


def _chunks(rows: list[Any], size: int) -> list[list[Any]]:
    return [rows[index : index + size] for index in range(0, len(rows), max(1, size))]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _has_duplicate_path_label(labels: dict[str, str]) -> bool:
    normalized = [_norm(labels[layer]) for layer in LAYER_ORDER]
    return len(set(normalized)) != len(normalized)


def _title_tokens(title: str) -> set[str]:
    stop = {"the", "a", "an", "of", "and", "or", "part", "chapter", "movie"}
    return {
        token
        for token in _norm(title).replace(":", " ").replace("-", " ").split()
        if len(token) > 3 and token not in stop
    }


def _norm(value: str) -> str:
    return " ".join(value.lower().replace("-", " ").replace("‑", " ").split())


def _shorten(value: str, limit: int) -> str:
    clean = " ".join(str(value).split())
    if len(clean) <= limit:
        return clean
    return clean[: max(0, limit - 1)].rstrip() + "..."


def _choice(value: Any, allowed: set[str], default: str) -> str:
    normalized = str(value or default).lower()
    return normalized if normalized in allowed else default


def _format_counts(counts: dict[str, int]) -> str:
    if not counts:
        return "none"
    return ", ".join(f"{key}: {value}" for key, value in sorted(counts.items()))


def _table_escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")
