from __future__ import annotations

import json

from film_atlas.classification_v2 import (
    ProfileVariantSpec,
    _apply_label_repairs,
    _apply_public_audit_point_reassignments,
    _find_audit_movie,
    _hierarchical_layers,
    _hierarchy_mismatch_rate,
    _rerank_neighbors_for_display,
    allocate_child_counts,
    build_variant_profiles,
)
from film_atlas.cluster import ClusterAssignment
from film_atlas.cluster_labels import ClusterLabelCandidate, EvidenceSummary
from film_atlas.embedding_cache import EmbeddingRecord
from film_atlas.inspect_clusters import ClusterEvidence
from film_atlas.models import MovieRecord
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
        overview=f"{title} follows workplace anxiety and comic rebellion.",
        genres=["Comedy"],
        keywords=["workplace", "office", "rebellion"],
        poster_path=None,
        backdrop_path=None,
        vote_average=7.0,
        vote_count=100,
        popularity=10.0,
        reviews=[f"{title} turns cubicle frustration into deadpan comedy."],
        production_companies=["Example Studio"],
        cast=["Jane Actor"],
        directors=["Pat Director"],
    )


def _embedding(tmdb_id: int, title: str, values: list[float]) -> EmbeddingRecord:
    return EmbeddingRecord(
        tmdb_id=tmdb_id,
        title=title,
        model="fixture",
        profile_hash=str(tmdb_id),
        embedding=values,
        estimated_tokens=1,
    )


def test_allocate_child_counts_preserves_total_and_caps() -> None:
    counts = allocate_child_counts({10: 12, 20: 5, 30: 1}, 9)

    assert sum(counts.values()) == 9
    assert counts[30] == 1
    assert all(1 <= count <= {10: 12, 20: 5, 30: 1}[cluster_id] for cluster_id, count in counts.items())


def test_no_title_rich_profile_redacts_title_from_raw_review_text() -> None:
    movie = _movie(1, "Office Space")
    raw_details = {
        1: {
            "tagline": "Office Space makes cubicle revolt feel cathartic.",
            "reviews": {
                "results": [
                    {
                        "content": (
                            "Office Space is a workplace comedy about office malaise, "
                            "boss pressure, and quiet rebellion."
                        )
                    }
                ]
            },
        }
    }
    spec = ProfileVariantSpec(
        name="fixture_no_title",
        include_title=False,
        include_tagline=True,
        review_weight="heavy",
        max_review_chars=500,
        review_count=1,
    )

    profile = build_variant_profiles([movie], raw_details, spec)[0]

    assert "Office Space" not in profile.profile_text
    assert "workplace comedy" in profile.profile_text
    assert "cubicle" in profile.profile_text


def test_hierarchical_layers_have_zero_parent_mismatch() -> None:
    embeddings = [
        _embedding(1, "A1", [0.0, 0.0, 0.0]),
        _embedding(2, "A2", [0.0, 0.1, 0.0]),
        _embedding(3, "A3", [0.1, 0.0, 0.0]),
        _embedding(4, "A4", [0.1, 0.1, 0.0]),
        _embedding(5, "B1", [5.0, 5.0, 5.0]),
        _embedding(6, "B2", [5.0, 5.1, 5.0]),
        _embedding(7, "B3", [5.1, 5.0, 5.0]),
        _embedding(8, "B4", [5.1, 5.1, 5.0]),
    ]

    assignments, parents, _note = _hierarchical_layers(
        embeddings,
        macro_k=2,
        neighborhood_k=4,
        micro_k=6,
        method="kmeans",
    )

    assert _hierarchy_mismatch_rate(assignments, parents) == 0


def test_display_neighbor_rerank_prefers_contextual_matches() -> None:
    sound = _movie(1, "Sound of Metal")
    sound.genres = ["Drama", "Music"]
    sound.keywords = ["deaf", "drums", "addiction"]
    coda = _movie(2, "CODA")
    coda.genres = ["Drama", "Music", "Romance"]
    coda.keywords = ["deaf", "family", "singing"]
    quiet_place = _movie(3, "A Quiet Place")
    quiet_place.genres = ["Horror", "Drama", "Science Fiction"]
    quiet_place.keywords = ["deaf", "alien invasion", "survival horror"]
    neighbors = [
        MovieNeighbors(
            tmdb_id=1,
            title="Sound of Metal",
            neighbors=[
                NeighborMatch(tmdb_id=3, title="A Quiet Place", similarity=0.95),
                NeighborMatch(tmdb_id=2, title="CODA", similarity=0.94),
            ],
        )
    ]
    assignments = {
        "macro": [
            ClusterAssignment(1, "Sound of Metal", 10),
            ClusterAssignment(2, "CODA", 10),
            ClusterAssignment(3, "A Quiet Place", 20),
        ],
        "neighborhood": [
            ClusterAssignment(1, "Sound of Metal", 30),
            ClusterAssignment(2, "CODA", 30),
            ClusterAssignment(3, "A Quiet Place", 40),
        ],
        "micro": [
            ClusterAssignment(1, "Sound of Metal", 50),
            ClusterAssignment(2, "CODA", 50),
            ClusterAssignment(3, "A Quiet Place", 60),
        ],
    }

    reranked = _rerank_neighbors_for_display(
        movies=[sound, coda, quiet_place],
        neighbors=neighbors,
        assignments_by_layer=assignments,
        top_n=2,
    )

    assert reranked[0].neighbors[0].title == "CODA"


def test_label_repairs_prevent_duplicate_parent_child_names() -> None:
    parent = _candidate(cluster_id=1, label="Wizard School Dark YA Fantasy")
    child = _candidate(cluster_id=143, label="Wizard School Dark YA Fantasy")
    evidence = [
        ClusterEvidence(
            cluster_id=143,
            cluster_size=8,
            representative_movies=["Harry Potter and the Prisoner of Azkaban"],
            top_official_genres=[("Fantasy", 8)],
            top_tmdb_keywords=[("wizard", 8), ("magic", 8)],
            aggregated_profile_terms=[("wizard", 0.4), ("school of witchcraft", 0.3)],
            in_cluster_neighbor_pairs=[],
            coherence_score=0.75,
            warnings=[],
        )
    ]

    repaired, repairs = _apply_label_repairs(
        layer="micro",
        candidates=[child],
        evidence=evidence,
        parent_maps={"macro": {}, "neighborhood": {}, "micro": {143: 1}},
        labels_by_layer={"macro": [], "neighborhood": [parent], "micro": []},
    )

    assert repaired[0].recommended_label != parent.recommended_label
    assert repaired[0].recommended_label == "Wizard School Dark YA Fantasy: Hogwarts Branch"
    assert repairs


def test_public_point_reassignment_moves_known_outlier_and_refreshes_sizes(tmp_path) -> None:
    movies = [
        {
            "tmdb_id": 1,
            "title": "Madame Web",
            "year": 2024,
            "genres": ["Action", "Fantasy"],
            "keywords": ["superhero"],
            "vote_count": 10,
            "popularity": 10.0,
        },
        {
            "tmdb_id": 2,
            "title": "Daredevil",
            "year": 2003,
            "genres": ["Action", "Fantasy"],
            "keywords": ["superhero", "vigilante"],
            "vote_count": 20,
            "popularity": 20.0,
        },
    ]
    points = [
        {"tmdb_id": 1, "macro_id": 1, "neighborhood_id": 10, "micro_id": 29, "x": 0.0, "y": 0.0},
        {"tmdb_id": 2, "macro_id": 11, "neighborhood_id": 71, "micro_id": 189, "x": 1.0, "y": 1.0},
    ]
    clusters = {
        "macro": [
            _public_cluster(1, "Creature Sci-Fi", 1),
            _public_cluster(11, "Gothic Heroes", 1),
        ],
        "neighborhood": [
            _public_cluster(10, "Creature Zone", 1),
            _public_cluster(71, "Hero Zone", 1),
        ],
        "micro": [
            _public_cluster(29, "Creature Pocket", 1),
            _public_cluster(189, "Hero Pocket", 1),
        ],
    }
    (tmp_path / "movies.json").write_text(json.dumps(movies), encoding="utf-8")
    (tmp_path / "points.json").write_text(json.dumps(points), encoding="utf-8")
    (tmp_path / "neighbors.json").write_text(json.dumps([]), encoding="utf-8")
    for layer, layer_clusters in clusters.items():
        (tmp_path / f"{layer}_clusters.json").write_text(json.dumps(layer_clusters), encoding="utf-8")

    repairs = _apply_public_audit_point_reassignments(tmp_path)

    repaired_points = json.loads((tmp_path / "points.json").read_text(encoding="utf-8"))
    repaired_macro_clusters = json.loads((tmp_path / "macro_clusters.json").read_text(encoding="utf-8"))
    assert repairs[0]["title"] == "Madame Web"
    assert repaired_points[0]["macro_id"] == 11
    assert repaired_points[0]["neighborhood_id"] == 71
    assert repaired_points[0]["micro_id"] == 189
    assert {cluster["cluster_id"]: cluster["size"] for cluster in repaired_macro_clusters} == {1: 0, 11: 2}


def test_find_audit_movie_honors_explicit_parenthetical_year() -> None:
    movies = [
        {"tmdb_id": 1, "title": "The Karate Kid", "year": 1984},
        {"tmdb_id": 2, "title": "The Karate Kid", "year": 2010},
    ]

    assert _find_audit_movie(movies, "The Karate Kid")["tmdb_id"] == 1
    assert _find_audit_movie(movies, "The Karate Kid (2010)")["tmdb_id"] == 2


def _candidate(cluster_id: int, label: str) -> ClusterLabelCandidate:
    return ClusterLabelCandidate(
        cluster_id=cluster_id,
        cluster_size=8,
        evidence_hash=str(cluster_id),
        model="fixture",
        prompt_version="fixture",
        plain_label=label.lower(),
        poetic_label=label,
        spotify_style_label=label,
        recommended_label=label,
        one_sentence_description="Fixture label.",
        why_this_label_fits="Fixture evidence.",
        representative_movies=[],
        edge_case_movies=[],
        possible_misfits=[],
        confidence_score=0.7,
        label_risk_notes="",
        evidence_summary=EvidenceSummary(genres=[], tmdb_keywords=[], aggregated_terms=[]),
    )


def _public_cluster(cluster_id: int, label: str, size: int) -> dict:
    return {
        "cluster_id": cluster_id,
        "coherence_score": 0.5,
        "description": label,
        "label_id": f"fixture:{cluster_id}",
        "parent_cluster_id": None,
        "recommended_label": label,
        "representative_movies": [],
        "size": size,
        "terms": [],
        "top_genres": [],
        "top_keywords": [],
    }
