from __future__ import annotations

import json
from pathlib import Path

import pytest

from film_atlas.clustering_methods import (
    FULL_EMBEDDING_INPUT_SPACE,
    ClusteringMethodComparisonError,
    compare_clustering_methods,
    compare_clustering_methods_file,
    render_clustering_method_comparison,
)
from film_atlas.embedding_cache import EmbeddingRecord
from film_atlas.models import MovieRecord, SemanticProfile


def _movie(tmdb_id: int, title: str, genre: str, keyword: str) -> MovieRecord:
    return MovieRecord(
        tmdb_id=tmdb_id,
        imdb_id=None,
        title=title,
        original_title=title,
        release_date="2000-01-01",
        year=2000,
        runtime=100,
        overview=f"{title} overview.",
        genres=[genre],
        keywords=[keyword],
        poster_path=None,
        backdrop_path=None,
        vote_average=7.0,
        vote_count=1000,
        popularity=10.0,
    )


def _fixture_records() -> tuple[list[EmbeddingRecord], list[MovieRecord], list[SemanticProfile]]:
    embeddings = [
        EmbeddingRecord(1, "Moon Signal", "model", "h1", [1.0, 0.0, 0.0], 1),
        EmbeddingRecord(2, "Orbit Gate", "model", "h2", [0.95, 0.05, 0.0], 1),
        EmbeddingRecord(3, "City Heat", "model", "h3", [0.0, 1.0, 0.0], 1),
        EmbeddingRecord(4, "Neon Case", "model", "h4", [0.05, 0.95, 0.0], 1),
        EmbeddingRecord(5, "Kitchen Hearts", "model", "h5", [0.0, 0.0, 1.0], 1),
        EmbeddingRecord(6, "Summer Table", "model", "h6", [0.0, 0.05, 0.95], 1),
    ]
    movies = [
        _movie(1, "Moon Signal", "Science Fiction", "space"),
        _movie(2, "Orbit Gate", "Science Fiction", "space"),
        _movie(3, "City Heat", "Crime", "city"),
        _movie(4, "Neon Case", "Crime", "city"),
        _movie(5, "Kitchen Hearts", "Romance", "food"),
        _movie(6, "Summer Table", "Romance", "food"),
    ]
    profiles = [
        SemanticProfile(1, "Moon Signal", 2000, "space orbit signal", ["Science Fiction"], ["space"]),
        SemanticProfile(2, "Orbit Gate", 2000, "space orbit gate", ["Science Fiction"], ["space"]),
        SemanticProfile(3, "City Heat", 2000, "city crime chase", ["Crime"], ["city"]),
        SemanticProfile(4, "Neon Case", 2000, "city crime neon", ["Crime"], ["city"]),
        SemanticProfile(5, "Kitchen Hearts", 2000, "food romance kitchen", ["Romance"], ["food"]),
        SemanticProfile(6, "Summer Table", 2000, "food romance summer", ["Romance"], ["food"]),
    ]
    return embeddings, movies, profiles


def test_method_comparison_computes_metrics() -> None:
    embeddings, movies, profiles = _fixture_records()

    comparison = compare_clustering_methods(
        methods=["kmeans", "agglomerative"],
        embeddings=embeddings,
        movies=movies,
        profiles=profiles,
        kmeans_k=3,
        agglomerative_k=3,
    )

    assert comparison.input_space == FULL_EMBEDDING_INPUT_SPACE
    assert comparison.recommended_method in {"kmeans", "agglomerative"}
    assert len(comparison.results) == 2
    assert all(result.cluster_count == 3 for result in comparison.results)
    assert all(result.median_cluster_size == 2 for result in comparison.results)
    assert all(result.coherence_average is not None for result in comparison.results)
    assert all(result.sample_clusters for result in comparison.results)


def test_graph_clustering_handles_small_fixture_data() -> None:
    embeddings, movies, profiles = _fixture_records()

    comparison = compare_clustering_methods(
        methods=["graph"],
        embeddings=embeddings,
        movies=movies,
        profiles=profiles,
        graph_neighbors=1,
    )

    result = comparison.results[0]
    assert result.status == "completed"
    assert result.cluster_count >= 2
    assert result.assignments
    assert result.sample_clusters


def test_method_comparison_report_renders() -> None:
    embeddings, movies, profiles = _fixture_records()
    comparison = compare_clustering_methods(
        methods=["kmeans", "hdbscan"],
        embeddings=embeddings,
        movies=movies,
        profiles=profiles,
        kmeans_k=3,
    )

    report = render_clustering_method_comparison(comparison)

    assert "Milestone 2.6 Clustering Method Comparison" in report
    assert "Critical Input-Space Check" in report
    assert "full_embedding_vectors" in report
    assert "Direct Answers" in report
    assert "Quality-Check Movie Cluster Assignments" in report
    assert "hdbscan" in report
    assert any(result.method == "hdbscan" for result in comparison.results)
    assert {result.status for result in comparison.results}.issubset({"completed", "skipped"})


def test_missing_embeddings_handled_gracefully(tmp_path: Path) -> None:
    with pytest.raises(ClusteringMethodComparisonError, match="does not call OpenAI"):
        compare_clustering_methods_file(
            methods=["kmeans"],
            embeddings_path=tmp_path / "missing.jsonl",
            movies_path=tmp_path / "movies.json",
            profiles_path=tmp_path / "profiles.json",
            output_dir=tmp_path / "outputs",
        )


def test_clustering_uses_full_embeddings_not_2d_coordinates_by_default() -> None:
    embeddings = [
        EmbeddingRecord(1, "Bright A", "model", "h1", [1.0, 0.0, 1.0], 1),
        EmbeddingRecord(2, "Bright B", "model", "h2", [1.0, 0.0, 0.9], 1),
        EmbeddingRecord(3, "Dark A", "model", "h3", [1.0, 0.0, -1.0], 1),
        EmbeddingRecord(4, "Dark B", "model", "h4", [1.0, 0.0, -0.9], 1),
    ]
    movies = [
        _movie(1, "Bright A", "Fantasy", "bright"),
        _movie(2, "Bright B", "Fantasy", "bright"),
        _movie(3, "Dark A", "Horror", "dark"),
        _movie(4, "Dark B", "Horror", "dark"),
    ]
    profiles = [
        SemanticProfile(1, "Bright A", 2000, "bright wonder magic", ["Fantasy"], ["bright"]),
        SemanticProfile(2, "Bright B", 2000, "bright wonder spell", ["Fantasy"], ["bright"]),
        SemanticProfile(3, "Dark A", 2000, "dark fear shadow", ["Horror"], ["dark"]),
        SemanticProfile(4, "Dark B", 2000, "dark fear night", ["Horror"], ["dark"]),
    ]

    comparison = compare_clustering_methods(
        methods=["kmeans"],
        embeddings=embeddings,
        movies=movies,
        profiles=profiles,
        kmeans_k=2,
    )
    assignments = {item.title: item.cluster_id for item in comparison.results[0].assignments}

    assert comparison.input_space == FULL_EMBEDDING_INPUT_SPACE
    assert assignments["Bright A"] == assignments["Bright B"]
    assert assignments["Dark A"] == assignments["Dark B"]
    assert assignments["Bright A"] != assignments["Dark A"]


def test_method_comparison_file_writes_outputs(tmp_path: Path) -> None:
    embeddings, movies, profiles = _fixture_records()
    embeddings_path = tmp_path / "outputs" / "intermediate" / "embeddings.jsonl"
    movies_path = tmp_path / "data" / "processed" / "movies.json"
    profiles_path = tmp_path / "data" / "processed" / "profiles.json"
    embeddings_path.parent.mkdir(parents=True)
    movies_path.parent.mkdir(parents=True)
    embeddings_path.write_text(
        "\n".join(json.dumps(record.to_dict()) for record in embeddings) + "\n",
        encoding="utf-8",
    )
    movies_path.write_text(json.dumps([movie.to_dict() for movie in movies]), encoding="utf-8")
    profiles_path.write_text(
        json.dumps([profile.to_dict() for profile in profiles]),
        encoding="utf-8",
    )

    json_path, report_path = compare_clustering_methods_file(
        methods=["kmeans"],
        embeddings_path=embeddings_path,
        movies_path=movies_path,
        profiles_path=profiles_path,
        output_dir=tmp_path / "outputs",
        kmeans_k=3,
    )

    assert json_path.exists()
    assert report_path.exists()
