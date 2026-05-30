from __future__ import annotations

from film_atlas.cluster import ClusterAssignment
from film_atlas.inspect_clusters import ClusterEvidence
from film_atlas.milestone2_report import render_milestone_2_report
from film_atlas.neighbors import MovieNeighbors, NeighborMatch


def test_milestone_2_report_contains_required_sections() -> None:
    report = render_milestone_2_report(
        profile_count=2,
        embedded_count=2,
        manifest={
            "model": "text-embedding-3-large",
            "estimated_tokens": 100,
            "estimated_cost_usd": 0.000013,
            "cached_reused_count": 1,
            "new_embedding_count": 1,
        },
        assignments=[ClusterAssignment(1, "The Matrix", 0), ClusterAssignment(2, "Dark City", 0)],
        neighbors=[
            MovieNeighbors(1, "The Matrix", [NeighborMatch(2, "Dark City", 0.88)]),
            MovieNeighbors(2, "Dark City", [NeighborMatch(1, "The Matrix", 0.88)]),
        ],
        evidence=[
            ClusterEvidence(
                cluster_id=0,
                cluster_size=2,
                representative_movies=["The Matrix", "Dark City"],
                top_official_genres=[("Science Fiction", 2)],
                top_tmdb_keywords=[("dystopia", 2)],
                aggregated_profile_terms=[("simulated reality", 0.4)],
                in_cluster_neighbor_pairs=[],
                coherence_score=0.88,
                warnings=[],
            )
        ],
    )

    assert "Profiles available: 2" in report
    assert "Embedding model: text-embedding-3-large" in report
    assert "Quality-Check Movie Neighbors" in report
    assert "The Matrix" in report
    assert "Recommendation For Milestone 3" in report
