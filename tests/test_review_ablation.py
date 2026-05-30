from __future__ import annotations

import json
from pathlib import Path

import pytest

from film_atlas.models import MovieRecord
from film_atlas.review_ablation import (
    ReviewAblationError,
    ReviewAblationSummary,
    VariantSummary,
    assignment_similarity,
    build_variant_profiles,
    parse_review_variants,
    render_review_ablation_report,
    review_ablation_file,
    review_variant_config,
)


class FakeEmbeddingClient:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed_texts(self, texts: list[str], *, model: str) -> tuple[list[list[float]], int]:
        self.calls.append(texts)
        vectors = []
        for text in texts:
            lower = text.lower()
            vectors.append(
                [
                    1.0 if "space" in lower else 0.0,
                    1.0 if "crime" in lower else 0.0,
                    1.0 if "love" in lower else 0.0,
                ]
            )
        return vectors, 10

    def close(self) -> None:
        return None


class FakeLabelClient:
    def __init__(self) -> None:
        self.calls: list[list[int]] = []

    def label_batch(self, evidences: list[object], *, model: str) -> list[dict[str, object]]:
        self.calls.append([getattr(entry, "cluster_id") for entry in evidences])
        return [
            {
                "cluster_id": getattr(entry, "cluster_id"),
                "plain_label": "Fixture Vibe",
                "poetic_label": "Fixture Glow",
                "spotify_style_label": "Fixture Mix",
                "recommended_label": f"Fixture Cluster {getattr(entry, 'cluster_id')}",
                "one_sentence_description": "A fixture-only label.",
                "why_this_label_fits": "The evidence is stable in tests.",
                "representative_movies": getattr(entry, "representative_movies"),
                "edge_case_movies": [],
                "possible_misfits": [],
                "confidence_score": 0.8,
                "label_risk_notes": "",
            }
            for entry in evidences
        ]

    def close(self) -> None:
        return None


def _movie(tmdb_id: int, title: str, overview: str, reviews: list[str]) -> MovieRecord:
    return MovieRecord(
        tmdb_id=tmdb_id,
        imdb_id=None,
        title=title,
        original_title=title,
        release_date="2000-01-01",
        year=2000,
        runtime=100,
        overview=overview,
        genres=["Drama"],
        keywords=["space" if "space" in overview.lower() else "crime"],
        poster_path=None,
        backdrop_path=None,
        vote_average=7.0,
        vote_count=1000,
        popularity=10.0,
        reviews=reviews,
    )


def _movies() -> list[MovieRecord]:
    return [
        _movie(1, "Orbit One", "A space mission.", ["space awe love"]),
        _movie(2, "Orbit Two", "Another space mission.", ["space wonder"]),
        _movie(3, "City One", "A crime story.", ["crime tension"]),
        _movie(4, "City Two", "Another crime story.", ["crime night"]),
    ]


def test_variant_config_and_parse() -> None:
    assert parse_review_variants("no_reviews,light_reviews,no_reviews") == [
        "no_reviews",
        "light_reviews",
    ]
    assert review_variant_config("no_reviews").include_reviews is False
    assert review_variant_config("medium_reviews").review_weight == "medium"

    with pytest.raises(ReviewAblationError):
        parse_review_variants("loud_reviews")


def test_profile_output_isolation(tmp_path: Path) -> None:
    no_path = tmp_path / "no" / "profiles.json"
    medium_path = tmp_path / "medium" / "profiles.json"

    no_profiles = build_variant_profiles(
        _movies(),
        config=review_variant_config("no_reviews"),
        output_path=no_path,
    )
    medium_profiles = build_variant_profiles(
        _movies(),
        config=review_variant_config("medium_reviews"),
        output_path=medium_path,
    )

    assert no_path.exists()
    assert medium_path.exists()
    assert "Review language" not in no_profiles[0].profile_text
    assert "Review language" in medium_profiles[0].profile_text
    assert no_path.read_text(encoding="utf-8") != medium_path.read_text(encoding="utf-8")


def test_comparison_metrics() -> None:
    from film_atlas.cluster import ClusterAssignment

    baseline = [ClusterAssignment(1, "A", 0), ClusterAssignment(2, "B", 0)]
    same = [ClusterAssignment(1, "A", 1), ClusterAssignment(2, "B", 1)]

    assert assignment_similarity(same, baseline, metric="ari") == 1.0
    assert assignment_similarity(same, baseline, metric="nmi") == 1.0


def test_report_rendering() -> None:
    summary = ReviewAblationSummary(
        variants=[
            VariantSummary(
                variant="light_reviews",
                profile_count=2,
                profile_tokens=100,
                embedding_model="embed-model",
                embedding_estimated_cost_usd=0.001,
                cached_embeddings_reused=1,
                new_embeddings_generated=1,
                label_model="label-model",
                label_estimated_cost_usd=0.01,
                cached_labels_reused=0,
                new_labels_generated=2,
                cluster_count=2,
                coherence_average=0.5,
                coherence_min=0.4,
                coherence_max=0.6,
                cluster_sizes=[1, 1],
                tiny_cluster_count=2,
                ari_vs_light=1.0,
                nmi_vs_light=1.0,
                label_confidence_average=0.8,
                weakest_labels=[],
                noisy_terms=[("movie", 1)],
                quality_check_neighbors={"The Matrix": ["Blade Runner (0.900)"]},
                output_dir="outputs/intermediate/review_ablation/light_reviews",
            )
        ],
        total_estimated_cost_usd=0.011,
        recommended_variant="light_reviews",
        recommendation_note="Use light_reviews.",
        review_signal_note="Light helps.",
        medium_noise_note="No medium comparison.",
    )

    report = render_review_ablation_report(summary)

    assert "Milestone 3.25 Review-Weight Ablation" in report
    assert "Which" not in report
    assert "Recommended variant: light_reviews" in report
    assert "Quality-Check Neighbors" in report


def test_review_ablation_file_writes_isolated_outputs(tmp_path: Path) -> None:
    movies_path = tmp_path / "data" / "processed" / "movies.json"
    movies_path.parent.mkdir(parents=True)
    movies_path.write_text(
        json.dumps([movie.to_dict() for movie in _movies()]),
        encoding="utf-8",
    )

    summary_path, report_path, summary = review_ablation_file(
        variants=["no_reviews", "light_reviews"],
        movies_path=movies_path,
        output_dir=tmp_path / "outputs",
        embedding_model="embed-model",
        label_model="label-model",
        openai_api_key="unused",
        k=2,
        limit=4,
        embedding_client=FakeEmbeddingClient(),  # type: ignore[arg-type]
        label_client=FakeLabelClient(),  # type: ignore[arg-type]
    )

    assert summary_path.exists()
    assert report_path.exists()
    assert len(summary.variants) == 2
    assert (
        tmp_path / "outputs" / "intermediate" / "review_ablation" / "no_reviews" / "profiles.json"
    ).exists()
    assert (
        tmp_path
        / "outputs"
        / "intermediate"
        / "review_ablation"
        / "light_reviews"
        / "cluster_label_candidates.json"
    ).exists()


def test_missing_movies_handled_gracefully(tmp_path: Path) -> None:
    with pytest.raises(ReviewAblationError, match="Movies file not found"):
        review_ablation_file(
            variants=["no_reviews"],
            movies_path=tmp_path / "missing.json",
            output_dir=tmp_path / "outputs",
            embedding_model="embed-model",
            label_model="label-model",
            openai_api_key="unused",
            embedding_client=FakeEmbeddingClient(),  # type: ignore[arg-type]
            label_client=FakeLabelClient(),  # type: ignore[arg-type]
        )
