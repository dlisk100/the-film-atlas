from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from film_atlas.cli import app
from film_atlas.cluster_sweep import (
    ClusterSweepError,
    parse_ks,
    render_cluster_sweep_report,
    sweep_clusters,
    sweep_clusters_file,
)
from film_atlas.embedding_cache import EmbeddingRecord, write_embedding_cache
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


def _records() -> tuple[list[EmbeddingRecord], list[MovieRecord], list[SemanticProfile]]:
    embeddings = [
        EmbeddingRecord(1, "Moon Signal", "model", "h1", [1.0, 0.0, 0.0], 1),
        EmbeddingRecord(2, "Orbit Gate", "model", "h2", [0.9, 0.1, 0.0], 1),
        EmbeddingRecord(3, "City Heat", "model", "h3", [0.0, 1.0, 0.0], 1),
        EmbeddingRecord(4, "Neon Case", "model", "h4", [0.0, 0.9, 0.1], 1),
        EmbeddingRecord(5, "Kitchen Hearts", "model", "h5", [0.0, 0.0, 1.0], 1),
        EmbeddingRecord(6, "Summer Table", "model", "h6", [0.1, 0.0, 0.9], 1),
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


def test_parse_ks_sorts_dedupes_and_rejects_invalid_values() -> None:
    assert parse_ks("35, 15,25,15") == [15, 25, 35]

    with pytest.raises(ClusterSweepError):
        parse_ks("15,nope")

    with pytest.raises(ClusterSweepError):
        parse_ks("1")


def test_sweep_clusters_computes_metrics() -> None:
    embeddings, movies, profiles = _records()

    result = sweep_clusters(ks=[2, 3], embeddings=embeddings, movies=movies, profiles=profiles)

    assert result.embedding_count == 6
    assert result.ks == [2, 3]
    assert [item.cluster_count for item in result.results] == [2, 3]
    assert result.results[1].average_cluster_size == 2
    assert result.results[1].largest_cluster_size >= 2
    assert result.results[1].coherence_average is not None
    assert len(result.results[1].clusters) == 3
    assert result.results[1].sample_clusters
    assert result.recommended_k in {2, 3}


def test_missing_embeddings_path_is_graceful(tmp_path: Path) -> None:
    with pytest.raises(ClusterSweepError, match="does not call OpenAI"):
        sweep_clusters_file(
            ks=[2],
            embeddings_path=tmp_path / "missing.jsonl",
            movies_path=tmp_path / "movies.json",
            profiles_path=tmp_path / "profiles.json",
            output_dir=tmp_path / "outputs",
        )


def test_report_renders_required_sections() -> None:
    embeddings, movies, profiles = _records()
    result = sweep_clusters(ks=[2], embeddings=embeddings, movies=movies, profiles=profiles)

    report = render_cluster_sweep_report(result)

    assert "Milestone 2.5 Cluster Sweep Report" in report
    assert "Sweep Metrics" in report
    assert "Sample Clusters" in report
    assert "Recommended k" in report
    assert "Recommendation For Milestone 3" in report


def test_cli_accepts_ks_and_writes_sweep_outputs(tmp_path: Path) -> None:
    embeddings, movies, profiles = _records()
    embeddings_path = tmp_path / "outputs" / "intermediate" / "embeddings.jsonl"
    movies_path = tmp_path / "data" / "processed" / "movies.json"
    profiles_path = tmp_path / "data" / "processed" / "profiles.json"
    output_dir = tmp_path / "outputs"

    write_embedding_cache(embeddings_path, embeddings)
    movies_path.parent.mkdir(parents=True, exist_ok=True)
    movies_path.write_text(json.dumps([movie.to_dict() for movie in movies]), encoding="utf-8")
    profiles_path.write_text(
        json.dumps([profile.to_dict() for profile in profiles]),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "sweep-clusters",
            "--ks",
            "2,3",
            "--embeddings-path",
            str(embeddings_path),
            "--movies-path",
            str(movies_path),
            "--profiles-path",
            str(profiles_path),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0
    assert (output_dir / "intermediate" / "cluster_sweep.json").exists()
    assert (output_dir / "reports" / "cluster_sweep_report.md").exists()
