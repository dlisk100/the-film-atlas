from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from film_atlas.embedding_cache import EmbeddingRecord, profile_hash, write_embedding_cache
from film_atlas.milestone4 import (
    build_hierarchy_file,
    export_atlas_data_file,
    generate_milestone_4_report_file,
    is_scaled_eligible_detail,
    scale_dataset_file,
)
from film_atlas.models import MovieRecord, SemanticProfile


class FakeTMDbClient:
    def __init__(self, details: dict[int, dict[str, Any]]) -> None:
        self.details = details

    def discover_movies(self, **_kwargs: Any) -> list[dict[str, Any]]:
        return [{"id": tmdb_id, "title": detail["title"]} for tmdb_id, detail in self.details.items()]

    def movie_details(self, tmdb_id: int, *, refresh: bool = False) -> dict[str, Any]:
        return self.details[tmdb_id]


class FakeLabelClient:
    def label_batch(self, evidences: list[object], *, model: str) -> list[dict[str, object]]:
        return [
            {
                "cluster_id": getattr(evidence, "cluster_id"),
                "plain_label": "Fixture Label",
                "poetic_label": "Fixture Glow",
                "spotify_style_label": "Fixture Mix",
                "recommended_label": f"Fixture {getattr(evidence, 'cluster_id')}",
                "one_sentence_description": "A stable fixture label.",
                "why_this_label_fits": "The fixture evidence is grouped together.",
                "representative_movies": getattr(evidence, "representative_movies"),
                "edge_case_movies": [],
                "possible_misfits": [],
                "confidence_score": 0.8,
                "label_risk_notes": "",
            }
            for evidence in evidences
        ]

    def close(self) -> None:
        return None


def _detail(
    tmdb_id: int,
    title: str,
    *,
    year: int = 2001,
    runtime: int = 100,
    vote_count: int = 100,
    overview: str = "A film about memory and pressure.",
    keywords: list[str] | None = None,
    reviews: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "id": tmdb_id,
        "imdb_id": f"tt{tmdb_id:07d}",
        "title": title,
        "original_title": title,
        "release_date": f"{year}-01-01",
        "runtime": runtime,
        "overview": overview,
        "genres": [{"name": "Drama"}],
        "keywords": {"keywords": [{"name": keyword} for keyword in (keywords or [])]},
        "poster_path": "/poster.jpg",
        "backdrop_path": "/backdrop.jpg",
        "vote_average": 7.0,
        "vote_count": vote_count,
        "popularity": float(vote_count),
        "reviews": {
            "results": [{"content": review} for review in (reviews or ["careful tone"])]
        },
        "original_language": "en",
        "production_countries": [{"name": "United States of America"}],
        "production_companies": [{"name": "Fixture Studio"}],
        "credits": {"cast": [], "crew": []},
        "adult": False,
        "video": False,
    }


def _movie(tmdb_id: int, title: str, profile_text: str) -> MovieRecord:
    return MovieRecord(
        tmdb_id=tmdb_id,
        imdb_id=None,
        title=title,
        original_title=title,
        release_date="2001-01-01",
        year=2001,
        runtime=100,
        overview=profile_text,
        genres=["Drama"],
        keywords=["memory", "city"],
        poster_path=None,
        backdrop_path=None,
        vote_average=7.0,
        vote_count=100,
        popularity=10.0,
        reviews=["raw review should stay private"],
        original_language="en",
    )


def _write_fixture_dataset(tmp_path: Path) -> tuple[Path, Path, Path]:
    movies = [
        _movie(1, "Memory One", "memory city pressure"),
        _movie(2, "Memory Two", "memory city dream"),
        _movie(3, "Space One", "space lonely awe"),
        _movie(4, "Space Two", "space mission awe"),
        _movie(5, "Crime One", "crime night moral"),
        _movie(6, "Crime Two", "crime city moral"),
    ]
    profiles = [
        SemanticProfile(
            tmdb_id=movie.tmdb_id,
            title=movie.title,
            year=movie.year,
            profile_text=movie.overview or "",
            genres=movie.genres,
            keywords=movie.keywords,
        )
        for movie in movies
    ]
    movies_path = tmp_path / "data" / "processed" / "movies.json"
    profiles_path = tmp_path / "data" / "processed" / "profiles.json"
    embeddings_path = tmp_path / "outputs" / "intermediate" / "embeddings.jsonl"
    movies_path.parent.mkdir(parents=True)
    movies_path.write_text(
        json.dumps([movie.to_dict() for movie in movies], indent=2),
        encoding="utf-8",
    )
    profiles_path.write_text(
        json.dumps([profile.to_dict() for profile in profiles], indent=2),
        encoding="utf-8",
    )
    vectors = [
        [1.0, 0.0, 0.0],
        [0.9, 0.1, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.9, 0.1],
        [0.0, 0.0, 1.0],
        [0.1, 0.0, 0.9],
    ]
    write_embedding_cache(
        embeddings_path,
        [
            EmbeddingRecord(
                tmdb_id=profile.tmdb_id,
                title=profile.title,
                model="fixture-embed",
                profile_hash=profile_hash(profile),
                embedding=vectors[index],
                estimated_tokens=10,
            )
            for index, profile in enumerate(profiles)
        ],
    )
    return movies_path, profiles_path, embeddings_path


def test_scaled_filtering_and_keyword_preference(tmp_path: Path) -> None:
    details = {
        1: _detail(1, "No Keywords", vote_count=1000, keywords=[]),
        2: _detail(2, "Has Keywords", vote_count=200, keywords=["identity"]),
        3: _detail(3, "Too Short", runtime=20, keywords=["identity"]),
    }
    assert is_scaled_eligible_detail(
        details[2],
        since_year=1980,
        min_votes=100,
        min_runtime=60,
        today=date(2026, 1, 1),
    )
    assert not is_scaled_eligible_detail(
        details[3],
        since_year=1980,
        min_votes=100,
        min_runtime=60,
        today=date(2026, 1, 1),
    )

    result = scale_dataset_file(
        FakeTMDbClient(details),  # type: ignore[arg-type]
        target=1,
        since_year=1980,
        min_votes=100,
        data_dir=tmp_path / "data",
        output_dir=tmp_path / "outputs",
        candidate_limit=3,
    )
    movies = json.loads(result.movies_path.read_text(encoding="utf-8"))

    assert result.selected_count == 1
    assert movies[0]["title"] == "Has Keywords"
    assert result.profiles_path.exists()


def test_hierarchy_metrics_and_public_export_are_sanitized(tmp_path: Path) -> None:
    movies_path, profiles_path, embeddings_path = _write_fixture_dataset(tmp_path)
    output_dir = tmp_path / "outputs"

    hierarchy = build_hierarchy_file(
        movies_path=movies_path,
        profiles_path=profiles_path,
        embeddings_path=embeddings_path,
        output_dir=output_dir,
        macro_k=2,
        neighborhood_k=3,
        micro_k=4,
        label_model="fixture-label",
        openai_api_key="unused",
        label_client=FakeLabelClient(),  # type: ignore[arg-type]
        projection_method="pca",
    )
    export = export_atlas_data_file(
        movies_path=movies_path,
        hierarchy_dir=output_dir / "intermediate" / "hierarchy",
        output_dir=output_dir,
    )

    assert hierarchy.movie_count == 6
    assert [layer.cluster_count for layer in hierarchy.layers] == [2, 3, 4]
    assert (export.export_dir / "manifest.json").exists()
    movies_payload = json.loads((export.export_dir / "movies.json").read_text(encoding="utf-8"))
    points_payload = json.loads((export.export_dir / "points.json").read_text(encoding="utf-8"))
    manifest_payload = json.loads((export.export_dir / "manifest.json").read_text(encoding="utf-8"))

    assert "reviews" not in movies_payload[0]
    assert "embedding" not in movies_payload[0]
    assert {"macro_id", "neighborhood_id", "micro_id"}.issubset(points_payload[0])
    assert "neighbors.json" not in manifest_payload["files"]
    assert manifest_payload["neighbor_shards"]["count"] == 100
    assert (export.export_dir / "neighbor_shards" / "00.json").exists()


def test_milestone_4_report_rendering(tmp_path: Path) -> None:
    movies_path, profiles_path, embeddings_path = _write_fixture_dataset(tmp_path)
    output_dir = tmp_path / "outputs"
    hierarchy = build_hierarchy_file(
        movies_path=movies_path,
        profiles_path=profiles_path,
        embeddings_path=embeddings_path,
        output_dir=output_dir,
        macro_k=2,
        neighborhood_k=3,
        micro_k=4,
        label_model="fixture-label",
        openai_api_key="unused",
        label_client=FakeLabelClient(),  # type: ignore[arg-type]
        projection_method="pca",
    )
    export = export_atlas_data_file(
        movies_path=movies_path,
        hierarchy_dir=output_dir / "intermediate" / "hierarchy",
        output_dir=output_dir,
    )
    report_path = generate_milestone_4_report_file(
        movies_path=movies_path,
        output_dir=output_dir,
        hierarchy=hierarchy,
        export=export,
    )

    report = report_path.read_text(encoding="utf-8")
    assert "Milestone 4 Report" in report
    assert "Frontend Export File Sizes" in report
    assert "Recommendation" in report
