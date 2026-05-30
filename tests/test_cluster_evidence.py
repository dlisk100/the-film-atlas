from __future__ import annotations

from film_atlas.cluster import ClusterAssignment
from film_atlas.embedding_cache import EmbeddingRecord
from film_atlas.inspect_clusters import build_cluster_evidence
from film_atlas.models import MovieRecord, SemanticProfile
from film_atlas.neighbors import MovieNeighbors, NeighborMatch


def _movie(tmdb_id: int, title: str) -> MovieRecord:
    return MovieRecord(
        tmdb_id=tmdb_id,
        imdb_id=None,
        title=title,
        original_title=title,
        release_date="2000-01-01",
        year=2000,
        runtime=100,
        overview="A tense urban story.",
        genres=["Drama"],
        keywords=["friendship", "city"],
        poster_path=None,
        backdrop_path=None,
        vote_average=7.0,
        vote_count=1000,
        popularity=10.0,
    )


def test_cluster_evidence_includes_representatives_terms_and_pairs() -> None:
    evidence = build_cluster_evidence(
        movies=[_movie(1, "A"), _movie(2, "B")],
        profiles=[
            SemanticProfile(1, "A", 2000, "city friendship neon", ["Drama"], ["city"]),
            SemanticProfile(2, "B", 2000, "city friendship night", ["Drama"], ["city"]),
        ],
        embeddings=[
            EmbeddingRecord(1, "A", "model", "h1", [1.0, 0.0], 1),
            EmbeddingRecord(2, "B", "model", "h2", [0.9, 0.1], 1),
        ],
        assignments=[ClusterAssignment(1, "A", 0), ClusterAssignment(2, "B", 0)],
        neighbors=[
            MovieNeighbors(1, "A", [NeighborMatch(2, "B", 0.99)]),
            MovieNeighbors(2, "B", [NeighborMatch(1, "A", 0.99)]),
        ],
    )

    assert len(evidence) == 1
    assert evidence[0].cluster_size == 2
    assert evidence[0].representative_movies
    assert evidence[0].top_official_genres[0] == ("Drama", 2)
    assert evidence[0].in_cluster_neighbor_pairs[0]["source_title"] == "A"
