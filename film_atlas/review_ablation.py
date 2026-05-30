"""Review-weight ablation pipeline for The Film Atlas."""

from __future__ import annotations

import json
import statistics
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

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
    EmbeddingEstimate,
    OpenAIEmbeddingClient,
    embed_profiles_file,
    estimate_profiles,
    estimate_text_tokens,
    load_embedding_records,
)
from film_atlas.embedding_cache import (
    cache_key,
    load_embedding_cache,
    profile_hash,
    write_embedding_cache,
)
from film_atlas.inspect_clusters import ClusterEvidence, build_cluster_evidence
from film_atlas.milestone2_report import QUALITY_CHECK_MOVIES
from film_atlas.models import MovieRecord, SemanticProfile
from film_atlas.neighbors import MovieNeighbors, compute_neighbors
from film_atlas.normalize import load_movie_records
from film_atlas.profiles import ReviewWeight, build_semantic_profile

REVIEW_ABLATION_REPORT_FILENAME = "review_ablation_report.md"
REVIEW_ABLATION_SUMMARY_FILENAME = "review_ablation_summary.json"
REVIEW_ABLATION_DIRNAME = "review_ablation"
ReviewVariantName = Literal["no_reviews", "light_reviews", "medium_reviews"]
NOISE_TERMS = {
    "based",
    "film",
    "movie",
    "new",
    "story",
    "review",
    "reviews",
    "em",
    "character",
    "characters",
}


class ReviewAblationError(RuntimeError):
    """Raised when the review ablation cannot proceed."""


@dataclass(frozen=True, slots=True)
class ReviewVariantConfig:
    name: ReviewVariantName
    include_reviews: bool
    review_weight: ReviewWeight
    max_review_chars: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class VariantCostEstimate:
    variant: str
    profile_count: int
    embedding_estimate: EmbeddingEstimate
    cached_embedding_count: int
    new_embedding_count: int
    labeling_estimate: LabelingEstimate | None
    estimated_live_cost_usd: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant": self.variant,
            "profile_count": self.profile_count,
            "embedding_estimate": self.embedding_estimate.to_dict(),
            "cached_embedding_count": self.cached_embedding_count,
            "new_embedding_count": self.new_embedding_count,
            "labeling_estimate": (
                self.labeling_estimate.to_dict() if self.labeling_estimate else None
            ),
            "estimated_live_cost_usd": self.estimated_live_cost_usd,
        }


@dataclass(frozen=True, slots=True)
class VariantSummary:
    variant: str
    profile_count: int
    profile_tokens: int
    embedding_model: str
    embedding_estimated_cost_usd: float
    cached_embeddings_reused: int
    new_embeddings_generated: int
    label_model: str
    label_estimated_cost_usd: float
    cached_labels_reused: int
    new_labels_generated: int
    cluster_count: int
    coherence_average: float | None
    coherence_min: float | None
    coherence_max: float | None
    cluster_sizes: list[int]
    tiny_cluster_count: int
    ari_vs_light: float | None
    nmi_vs_light: float | None
    label_confidence_average: float | None
    weakest_labels: list[dict[str, Any]]
    noisy_terms: list[tuple[str, int]]
    quality_check_neighbors: dict[str, list[str]]
    output_dir: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ReviewAblationSummary:
    variants: list[VariantSummary]
    total_estimated_cost_usd: float
    recommended_variant: str | None
    recommendation_note: str
    review_signal_note: str
    medium_noise_note: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_review_variants(value: str) -> list[ReviewVariantName]:
    """Parse comma-separated review ablation variants."""
    valid = {"no_reviews", "light_reviews", "medium_reviews"}
    variants = []
    for raw_part in value.split(","):
        variant = raw_part.strip()
        if not variant:
            continue
        if variant not in valid:
            raise ReviewAblationError(
                f"Unknown review ablation variant {variant!r}. "
                f"Choose from: {', '.join(sorted(valid))}."
            )
        variants.append(variant)
    if not variants:
        raise ReviewAblationError("At least one review ablation variant is required.")
    return list(dict.fromkeys(variants))  # type: ignore[return-value]


def review_variant_config(name: ReviewVariantName) -> ReviewVariantConfig:
    """Return profile settings for an ablation variant."""
    if name == "no_reviews":
        return ReviewVariantConfig(name, include_reviews=False, review_weight="light", max_review_chars=0)
    if name == "light_reviews":
        return ReviewVariantConfig(name, include_reviews=True, review_weight="light", max_review_chars=180)
    if name == "medium_reviews":
        return ReviewVariantConfig(name, include_reviews=True, review_weight="medium", max_review_chars=180)
    raise ReviewAblationError(f"Unsupported review variant: {name}")


def review_ablation_file(
    *,
    variants: list[ReviewVariantName],
    movies_path: str | Path = "data/processed/movies.json",
    output_dir: str | Path = "outputs",
    global_embeddings_path: str | Path = "outputs/intermediate/embeddings.jsonl",
    global_label_cache_path: str | Path = "outputs/intermediate/cluster_label_cache.json",
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    label_model: str,
    openai_api_key: str | None,
    k: int = 35,
    limit: int | None = 500,
    embedding_batch_size: int = 64,
    label_batch_size: int = 5,
    embedding_client: OpenAIEmbeddingClient | None = None,
    label_client: OpenAIClusterLabelClient | None = None,
) -> tuple[Path, Path, ReviewAblationSummary]:
    """Run the review ablation and write report plus summary JSON."""
    movies_file = Path(movies_path)
    if not movies_file.exists():
        raise ReviewAblationError(f"Movies file not found at {movies_file}.")
    movies = load_movie_records(movies_file)
    selected_movies = movies[:limit] if limit is not None else movies
    if not selected_movies:
        raise ReviewAblationError("No movies available for review ablation.")

    base_dir = Path(output_dir) / "intermediate" / REVIEW_ABLATION_DIRNAME
    variant_estimates = []
    prepared_profiles: dict[str, list[SemanticProfile]] = {}
    for variant in variants:
        variant_dir = base_dir / variant
        profiles = build_variant_profiles(
            selected_movies,
            config=review_variant_config(variant),
            output_path=variant_dir / "profiles.json",
        )
        prepared_profiles[variant] = profiles
        seed_embedding_cache(
            profiles=profiles,
            model=embedding_model,
            target_path=variant_dir / "intermediate" / "embeddings.jsonl",
            source_paths=[global_embeddings_path],
        )
        variant_estimates.append(
            estimate_variant_cost(
                variant=variant,
                profiles=profiles,
                variant_dir=variant_dir,
                embedding_model=embedding_model,
                label_model=label_model,
                openai_api_key=openai_api_key,
                global_label_cache_path=global_label_cache_path,
            )
        )

    preflight_cost = sum(item.estimated_live_cost_usd for item in variant_estimates)
    if preflight_cost > 1:
        raise ReviewAblationError(
            f"Estimated review ablation live cost is ${preflight_cost:.4f}, which exceeds $1. "
            "Pause here and get explicit approval before running live calls."
        )
    if not openai_api_key:
        raise ReviewAblationError(
            "OPENAI_API_KEY is missing. Review ablation can estimate locally, but live "
            "embedding and labeling require the key in .env."
        )

    variant_runtime: list[dict[str, Any]] = []
    for estimate in variant_estimates:
        variant_dir = base_dir / estimate.variant
        profiles_path = variant_dir / "profiles.json"
        embed_result = embed_profiles_file(
            api_key=openai_api_key,
            profiles_path=profiles_path,
            output_dir=variant_dir,
            model=embedding_model,
            limit=limit,
            batch_size=embedding_batch_size,
            client=embedding_client,
        )
        embeddings = load_embedding_records(embed_result.embedding_path)
        assignments = cluster_embedding_records(embeddings, n_clusters=k)
        neighbors = compute_neighbors(embeddings, top_n=10)
        evidence = build_cluster_evidence(
            movies=selected_movies,
            profiles=prepared_profiles[estimate.variant],
            embeddings=embeddings,
            assignments=assignments,
            neighbors=neighbors,
        )
        write_variant_artifacts(
            variant_dir=variant_dir,
            assignments=assignments,
            neighbors=neighbors,
            evidence=evidence,
        )

        combined_label_cache = load_label_cache(global_label_cache_path)
        combined_label_cache.update(load_label_cache(variant_dir / "cluster_label_cache.json"))
        label_estimate = estimate_labeling(
            evidence,
            cache=combined_label_cache,
            model=label_model,
            openai_api_key=openai_api_key,
            batch_size=label_batch_size,
        )
        total_so_far = (
            sum(item.estimated_live_cost_usd for item in variant_estimates)
            - estimate.estimated_live_cost_usd
            + label_estimate.estimated_cost_usd
            + estimate.embedding_estimate.estimated_cost_usd
        )
        if total_so_far > 1:
            raise ReviewAblationError(
                f"Estimated review ablation live cost is ${total_so_far:.4f}, which exceeds $1. "
                "Pause here and get explicit approval before labeling."
            )
        candidates, cached_labels, new_labels = label_clusters(
            evidence,
            cache=combined_label_cache,
            model=label_model,
            api_key=openai_api_key,
            batch_size=label_batch_size,
            client=label_client,
        )
        write_label_cache(variant_dir / "cluster_label_cache.json", candidates)
        (variant_dir / "cluster_label_candidates.json").write_text(
            json.dumps([candidate.to_dict() for candidate in candidates], indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (variant_dir / "human_editable_cluster_labels.json").write_text(
            json.dumps(render_human_editable_labels(candidates), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        variant_runtime.append(
            {
                "variant": estimate.variant,
                "profiles": prepared_profiles[estimate.variant],
                "embedding_result": embed_result,
                "assignments": assignments,
                "neighbors": neighbors,
                "evidence": evidence,
                "label_estimate": label_estimate,
                "candidates": candidates,
                "cached_labels": cached_labels,
                "new_labels": new_labels,
                "variant_dir": variant_dir,
            }
        )

    summary = compare_review_variants(variant_runtime)
    summary_path = base_dir / REVIEW_ABLATION_SUMMARY_FILENAME
    report_path = Path(output_dir) / "reports" / REVIEW_ABLATION_REPORT_FILENAME
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    report_path.write_text(render_review_ablation_report(summary), encoding="utf-8")
    return summary_path, report_path, summary


def build_variant_profiles(
    movies: list[MovieRecord],
    *,
    config: ReviewVariantConfig,
    output_path: str | Path,
) -> list[SemanticProfile]:
    """Build variant-specific profiles without touching the active profiles file."""
    profiles = [
        build_semantic_profile(
            movie,
            include_reviews=config.include_reviews,
            max_review_chars=config.max_review_chars,
            review_weight=config.review_weight,
        )
        for movie in movies
    ]
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([profile.to_dict() for profile in profiles], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return profiles


def seed_embedding_cache(
    *,
    profiles: list[SemanticProfile],
    model: str,
    target_path: str | Path,
    source_paths: list[str | Path],
) -> int:
    """Seed a variant embedding cache with matching records from existing caches."""
    target = Path(target_path)
    if target.exists():
        return len(load_embedding_cache(target))
    source_cache = {}
    for source_path in source_paths:
        source_cache.update(load_embedding_cache(source_path))
    seeded = []
    for profile in profiles:
        digest = profile_hash(profile)
        cached = source_cache.get(cache_key(profile.tmdb_id, model, digest))
        if cached:
            seeded.append(cached)
    if seeded:
        write_embedding_cache(target, seeded)
    return len(seeded)


def estimate_variant_cost(
    *,
    variant: str,
    profiles: list[SemanticProfile],
    variant_dir: Path,
    embedding_model: str,
    label_model: str,
    openai_api_key: str | None,
    global_label_cache_path: str | Path,
) -> VariantCostEstimate:
    """Estimate variant cost before live calls."""
    embedding_estimate = estimate_profiles(profiles, model=embedding_model)
    embedding_cache = load_embedding_cache(variant_dir / "intermediate" / "embeddings.jsonl")
    new_embeddings = 0
    for profile in profiles:
        key = cache_key(profile.tmdb_id, embedding_model, profile_hash(profile))
        if key not in embedding_cache:
            new_embeddings += 1
    cached_embeddings = len(profiles) - new_embeddings
    approximate_label_estimate = approximate_labeling_estimate(
        cluster_count=35,
        model=label_model,
        openai_api_key=openai_api_key,
        cached_count=len(load_label_cache(global_label_cache_path)) if variant == "light_reviews" else 0,
    )
    live_embedding_cost = embedding_estimate.estimated_cost_usd * (new_embeddings / len(profiles))
    return VariantCostEstimate(
        variant=variant,
        profile_count=len(profiles),
        embedding_estimate=embedding_estimate,
        cached_embedding_count=cached_embeddings,
        new_embedding_count=new_embeddings,
        labeling_estimate=approximate_label_estimate,
        estimated_live_cost_usd=live_embedding_cost + approximate_label_estimate.estimated_cost_usd,
    )


def approximate_labeling_estimate(
    *,
    cluster_count: int,
    model: str,
    openai_api_key: str | None,
    cached_count: int = 0,
) -> LabelingEstimate:
    """Conservatively estimate label cost before variant embeddings exist."""
    clusters_to_label = max(0, cluster_count - min(cached_count, cluster_count))
    sample_prompt_tokens = estimate_text_tokens(
        json.dumps(
            build_label_messages(
                [
                    ClusterEvidence(
                        cluster_id=0,
                        cluster_size=15,
                        representative_movies=["Sample A", "Sample B", "Sample C"],
                        top_official_genres=[("Drama", 10), ("Thriller", 5)],
                        top_tmdb_keywords=[("identity", 4), ("friendship", 3)],
                        aggregated_profile_terms=[("memory", 0.5), ("city", 0.3)],
                        in_cluster_neighbor_pairs=[],
                        coherence_score=0.5,
                        warnings=[],
                    )
                ]
            ),
            ensure_ascii=False,
        )
    )
    input_tokens = clusters_to_label * sample_prompt_tokens
    output_tokens = clusters_to_label * 450
    if clusters_to_label == 0:
        cost = 0.0
    else:
        cost = _label_cost(model, input_tokens, output_tokens)
    return LabelingEstimate(
        cluster_count=cluster_count,
        cached_count=cluster_count - clusters_to_label,
        clusters_to_label=clusters_to_label,
        model=model,
        estimated_input_tokens=input_tokens,
        estimated_output_tokens=output_tokens,
        estimated_cost_usd=cost,
        openai_api_key_status="set" if openai_api_key else "missing",
    )


def write_variant_artifacts(
    *,
    variant_dir: Path,
    assignments: list[ClusterAssignment],
    neighbors: list[MovieNeighbors],
    evidence: list[ClusterEvidence],
) -> None:
    """Write per-variant local artifacts."""
    variant_dir.mkdir(parents=True, exist_ok=True)
    cluster_count = len({assignment.cluster_id for assignment in assignments if assignment.cluster_id >= 0})
    (variant_dir / "cluster_assignments.json").write_text(
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
    (variant_dir / "neighbors.json").write_text(
        json.dumps([entry.to_dict() for entry in neighbors], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (variant_dir / "cluster_evidence.json").write_text(
        json.dumps([entry.to_dict() for entry in evidence], indent=2, sort_keys=True),
        encoding="utf-8",
    )


def compare_review_variants(variant_runtime: list[dict[str, Any]]) -> ReviewAblationSummary:
    """Compute comparison metrics across variants."""
    baseline = next(
        (item for item in variant_runtime if item["variant"] == "light_reviews"),
        variant_runtime[0] if variant_runtime else None,
    )
    baseline_assignments = baseline["assignments"] if baseline else []
    summaries = []
    for item in variant_runtime:
        summaries.append(
            summarize_variant(
                variant=item["variant"],
                profiles=item["profiles"],
                embedding_result=item["embedding_result"],
                assignments=item["assignments"],
                neighbors=item["neighbors"],
                evidence=item["evidence"],
                label_estimate=item["label_estimate"],
                candidates=item["candidates"],
                cached_labels=item["cached_labels"],
                new_labels=item["new_labels"],
                variant_dir=item["variant_dir"],
                baseline_assignments=baseline_assignments,
            )
        )
    recommended = recommend_variant(summaries)
    return ReviewAblationSummary(
        variants=summaries,
        total_estimated_cost_usd=sum(
            item.embedding_estimated_cost_usd + item.label_estimated_cost_usd
            for item in summaries
        ),
        recommended_variant=recommended.variant if recommended else None,
        recommendation_note=recommendation_note(recommended, summaries),
        review_signal_note=review_signal_note(summaries),
        medium_noise_note=medium_noise_note(summaries),
    )


def summarize_variant(
    *,
    variant: str,
    profiles: list[SemanticProfile],
    embedding_result: Any,
    assignments: list[ClusterAssignment],
    neighbors: list[MovieNeighbors],
    evidence: list[ClusterEvidence],
    label_estimate: LabelingEstimate,
    candidates: list[ClusterLabelCandidate],
    cached_labels: int,
    new_labels: int,
    variant_dir: Path,
    baseline_assignments: list[ClusterAssignment],
) -> VariantSummary:
    """Summarize one completed ablation variant."""
    sizes = Counter(assignment.cluster_id for assignment in assignments if assignment.cluster_id >= 0)
    coherence_values = [
        cluster.coherence_score for cluster in evidence if cluster.coherence_score is not None
    ]
    confidences = [candidate.confidence_score for candidate in candidates]
    return VariantSummary(
        variant=variant,
        profile_count=len(profiles),
        profile_tokens=sum(estimate_text_tokens(profile.profile_text) for profile in profiles),
        embedding_model=embedding_result.model,
        embedding_estimated_cost_usd=embedding_result.estimated_cost_usd,
        cached_embeddings_reused=embedding_result.cached_reused_count,
        new_embeddings_generated=embedding_result.new_embedding_count,
        label_model=label_estimate.model,
        label_estimated_cost_usd=label_estimate.estimated_cost_usd,
        cached_labels_reused=cached_labels,
        new_labels_generated=new_labels,
        cluster_count=len(sizes),
        coherence_average=statistics.mean(coherence_values) if coherence_values else None,
        coherence_min=min(coherence_values, default=None),
        coherence_max=max(coherence_values, default=None),
        cluster_sizes=sorted(sizes.values(), reverse=True),
        tiny_cluster_count=sum(1 for size in sizes.values() if size < 5),
        ari_vs_light=assignment_similarity(assignments, baseline_assignments, metric="ari"),
        nmi_vs_light=assignment_similarity(assignments, baseline_assignments, metric="nmi"),
        label_confidence_average=statistics.mean(confidences) if confidences else None,
        weakest_labels=weakest_labels(candidates),
        noisy_terms=noisy_terms(evidence),
        quality_check_neighbors=quality_check_neighbors(neighbors),
        output_dir=str(variant_dir),
    )


def assignment_similarity(
    assignments: list[ClusterAssignment],
    baseline_assignments: list[ClusterAssignment],
    *,
    metric: Literal["ari", "nmi"],
) -> float | None:
    """Compare cluster assignments on common movie IDs."""
    if not assignments or not baseline_assignments:
        return None
    by_id = {assignment.tmdb_id: assignment.cluster_id for assignment in assignments}
    baseline_by_id = {assignment.tmdb_id: assignment.cluster_id for assignment in baseline_assignments}
    common_ids = sorted(set(by_id).intersection(baseline_by_id))
    if len(common_ids) < 2:
        return None
    labels = [by_id[tmdb_id] for tmdb_id in common_ids]
    baseline_labels = [baseline_by_id[tmdb_id] for tmdb_id in common_ids]
    if metric == "ari":
        return float(adjusted_rand_score(baseline_labels, labels))
    return float(normalized_mutual_info_score(baseline_labels, labels))


def weakest_labels(candidates: list[ClusterLabelCandidate], *, limit: int = 5) -> list[dict[str, Any]]:
    """Return lowest-confidence labels for review."""
    return [
        {
            "cluster_id": candidate.cluster_id,
            "recommended_label": candidate.recommended_label,
            "confidence_score": candidate.confidence_score,
            "label_risk_notes": candidate.label_risk_notes,
            "possible_misfits": candidate.possible_misfits,
        }
        for candidate in sorted(candidates, key=lambda item: (item.confidence_score, item.cluster_id))[
            :limit
        ]
    ]


def noisy_terms(evidence: list[ClusterEvidence], *, limit: int = 12) -> list[tuple[str, int]]:
    """Count obvious generic/review-noise terms across cluster evidence."""
    counter: Counter[str] = Counter()
    for cluster in evidence:
        for term, _score in cluster.aggregated_profile_terms:
            if term in NOISE_TERMS:
                counter[term] += 1
    return counter.most_common(limit)


def quality_check_neighbors(neighbors: list[MovieNeighbors]) -> dict[str, list[str]]:
    """Return quality-check movie nearest neighbors."""
    by_title = {entry.title.lower(): entry for entry in neighbors}
    output = {}
    for title in QUALITY_CHECK_MOVIES:
        entry = by_title.get(title.lower())
        if entry:
            output[entry.title] = [
                f"{neighbor.title} ({neighbor.similarity:.3f})"
                for neighbor in entry.neighbors[:5]
            ]
    return output


def recommend_variant(summaries: list[VariantSummary]) -> VariantSummary | None:
    """Pick the most labelable review setting."""
    if not summaries:
        return None

    def score(summary: VariantSummary) -> float:
        confidence = summary.label_confidence_average or 0
        coherence = summary.coherence_average or 0
        tiny_penalty = summary.tiny_cluster_count * 0.02
        noise_penalty = sum(count for _term, count in summary.noisy_terms) * 0.002
        medium_penalty = 0.03 if summary.variant == "medium_reviews" else 0
        return confidence + coherence - tiny_penalty - noise_penalty - medium_penalty

    return max(summaries, key=score)


def recommendation_note(
    recommended: VariantSummary | None,
    summaries: list[VariantSummary],
) -> str:
    """Explain the recommendation."""
    if recommended is None:
        return "No variant completed, so no review-weight recommendation is available."
    return (
        f"Use {recommended.variant}: it has average label confidence "
        f"{_fmt_float(recommended.label_confidence_average)}, coherence "
        f"{_fmt_float(recommended.coherence_average)}, {recommended.tiny_cluster_count} tiny clusters, "
        f"and {sum(count for _term, count in recommended.noisy_terms)} obvious noise-term hits."
    )


def review_signal_note(summaries: list[VariantSummary]) -> str:
    """State whether reviews appear useful."""
    no_reviews = _find_variant(summaries, "no_reviews")
    light = _find_variant(summaries, "light_reviews")
    if not no_reviews or not light:
        return "Insufficient variants to compare no_reviews with light_reviews."
    confidence_delta = (light.label_confidence_average or 0) - (
        no_reviews.label_confidence_average or 0
    )
    coherence_delta = (light.coherence_average or 0) - (no_reviews.coherence_average or 0)
    if confidence_delta >= 0 and coherence_delta >= -0.01:
        return (
            "Light review snippets appear to help or preserve vibe discovery: label confidence "
            f"delta {_fmt_signed(confidence_delta)}, coherence delta {_fmt_signed(coherence_delta)}."
        )
    return (
        "Light review snippets may be hurting signal: label confidence delta "
        f"{_fmt_signed(confidence_delta)}, coherence delta {_fmt_signed(coherence_delta)}."
    )


def medium_noise_note(summaries: list[VariantSummary]) -> str:
    """State whether medium review weight adds noise."""
    light = _find_variant(summaries, "light_reviews")
    medium = _find_variant(summaries, "medium_reviews")
    if not light or not medium:
        return "Insufficient variants to compare light_reviews with medium_reviews."
    medium_noise = sum(count for _term, count in medium.noisy_terms)
    light_noise = sum(count for _term, count in light.noisy_terms)
    confidence_delta = (medium.label_confidence_average or 0) - (
        light.label_confidence_average or 0
    )
    if medium_noise > light_noise or confidence_delta < -0.02:
        return (
            "Medium review weight appears noisier than light_reviews: noise hits "
            f"{medium_noise} vs {light_noise}, confidence delta {_fmt_signed(confidence_delta)}."
        )
    return (
        "Medium review weight does not show obvious extra noise by these diagnostics: noise hits "
        f"{medium_noise} vs {light_noise}, confidence delta {_fmt_signed(confidence_delta)}."
    )


def render_review_ablation_report(summary: ReviewAblationSummary) -> str:
    """Render the review ablation comparison report."""
    lines = [
        "# The Film Atlas - Milestone 3.25 Review-Weight Ablation",
        "",
        "This report compares no-review, light-review, and medium-review profile variants on "
        "the same movie set using the same embedding model, k-means k=35, and the same draft "
        "labeling style. It does not fetch new TMDb data, scrape websites, export public JSON, "
        "or touch frontend code.",
        "",
        "## Summary",
        "",
        f"- Recommended variant: {summary.recommended_variant or 'n/a'}",
        f"- Recommendation: {summary.recommendation_note}",
        f"- Do reviews improve vibe discovery? {summary.review_signal_note}",
        f"- Does medium add noise? {summary.medium_noise_note}",
        f"- Total estimated live cost: ${summary.total_estimated_cost_usd:.4f}",
        "",
        "## Variant Metrics",
        "",
        "| Variant | Profiles | Tokens | Embed Cost | Cache Reused/New | Coherence Avg | Coherence Range | Tiny <5 | ARI vs Light | NMI vs Light | Label Confidence | Label Cache Reused/New | Noise Terms |",
        "| --- | ---: | ---: | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for variant in summary.variants:
        lines.append(
            f"| {variant.variant} | {variant.profile_count} | {variant.profile_tokens} | "
            f"${variant.embedding_estimated_cost_usd:.4f} | "
            f"{variant.cached_embeddings_reused}/{variant.new_embeddings_generated} | "
            f"{_fmt_float(variant.coherence_average)} | "
            f"{_fmt_range(variant.coherence_min, variant.coherence_max)} | "
            f"{variant.tiny_cluster_count} | {_fmt_float(variant.ari_vs_light)} | "
            f"{_fmt_float(variant.nmi_vs_light)} | "
            f"{_fmt_float(variant.label_confidence_average)} | "
            f"{variant.cached_labels_reused}/{variant.new_labels_generated} | "
            f"{_term_counts(variant.noisy_terms)} |"
        )

    lines.extend(["", "## Cluster Size Distributions", ""])
    for variant in summary.variants:
        lines.append(f"- {variant.variant}: {', '.join(map(str, variant.cluster_sizes))}")

    lines.extend(["", "## Quality-Check Neighbors", ""])
    for variant in summary.variants:
        lines.extend([f"### {variant.variant}", "", _quality_neighbors_markdown(variant), ""])

    lines.extend(["## Weakest Labels", ""])
    for variant in summary.variants:
        lines.extend([f"### {variant.variant}", "", _weakest_labels_markdown(variant), ""])

    lines.extend(
        [
            "## Review Signal Notes",
            "",
            "### Examples Where Reviews May Improve Vibe Signal",
            "",
            _review_change_examples(summary, improved=True),
            "",
            "### Examples Where Reviews May Hurt Or Add Noise",
            "",
            _review_change_examples(summary, improved=False),
            "",
            "## Recommendation Before Scaling",
            "",
            summary.recommendation_note,
            "",
        ]
    )
    return "\n".join(lines)


def _review_change_examples(summary: ReviewAblationSummary, *, improved: bool) -> str:
    no_reviews = _find_variant(summary.variants, "no_reviews")
    light = _find_variant(summary.variants, "light_reviews")
    medium = _find_variant(summary.variants, "medium_reviews")
    if not no_reviews or not light:
        return "_No no_reviews/light_reviews comparison available._"
    examples = []
    for title, light_neighbors in light.quality_check_neighbors.items():
        no_neighbors = no_reviews.quality_check_neighbors.get(title)
        if not no_neighbors or no_neighbors == light_neighbors:
            continue
        if improved and len(set(light_neighbors).intersection(no_neighbors)) < 4:
            examples.append(
                f"- {title}: light_reviews neighbors shift to {', '.join(light_neighbors[:3])} "
                f"from no_reviews {', '.join(no_neighbors[:3])}."
            )
        if not improved and medium and title in medium.quality_check_neighbors:
            medium_neighbors = medium.quality_check_neighbors[title]
            if len(set(medium_neighbors).intersection(light_neighbors)) < 3:
                examples.append(
                    f"- {title}: medium_reviews shifts away from light_reviews "
                    f"({', '.join(medium_neighbors[:3])})."
                )
        if len(examples) >= 4:
            break
    if examples:
        return "\n".join(examples)
    if improved:
        return "_No strong quality-check neighbor improvements were obvious from this heuristic._"
    return "_No strong review-hurt examples were obvious beyond the noise and weak-label diagnostics._"


def _quality_neighbors_markdown(variant: VariantSummary) -> str:
    if not variant.quality_check_neighbors:
        return "_No quality-check movies present._"
    return "\n".join(
        f"- {title}: {', '.join(neighbors)}"
        for title, neighbors in sorted(variant.quality_check_neighbors.items())
    )


def _weakest_labels_markdown(variant: VariantSummary) -> str:
    if not variant.weakest_labels:
        return "_No labels available._"
    return "\n".join(
        f"- Cluster {item['cluster_id']} ({item['recommended_label']}): "
        f"confidence {item['confidence_score']:.2f}; {item['label_risk_notes'] or 'review needed.'}"
        for item in variant.weakest_labels
    )


def _find_variant(summaries: list[VariantSummary], name: str) -> VariantSummary | None:
    return next((summary for summary in summaries if summary.variant == name), None)


def _label_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    from film_atlas.cluster_labels import FALLBACK_PRICE_PER_1M_TOKENS, LABEL_MODEL_PRICES_PER_1M_TOKENS

    prices = LABEL_MODEL_PRICES_PER_1M_TOKENS.get(model, FALLBACK_PRICE_PER_1M_TOKENS)
    return (
        input_tokens / 1_000_000 * prices["input"]
        + output_tokens / 1_000_000 * prices["output"]
    )


def _term_counts(items: list[tuple[str, int]]) -> str:
    return ", ".join(f"{term} ({count})" for term, count in items) or "none"


def _fmt_float(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _fmt_range(low: float | None, high: float | None) -> str:
    if low is None or high is None:
        return "n/a"
    return f"{low:.3f}-{high:.3f}"


def _fmt_signed(value: float) -> str:
    return f"{value:+.3f}"
