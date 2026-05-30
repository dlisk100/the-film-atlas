from __future__ import annotations

import json
from pathlib import Path

import pytest

from film_atlas.cluster_labels import (
    ClusterLabelCandidate,
    ClusterLabelError,
    build_label_messages,
    candidate_from_response,
    estimate_labeling,
    evidence_hash,
    label_cache_key,
    label_clusters,
    load_or_build_label_evidence,
    parse_label_response,
    render_human_editable_labels,
    render_label_review_markdown,
    write_label_cache,
)
from film_atlas.inspect_clusters import ClusterEvidence


class FakeLabelClient:
    def __init__(self) -> None:
        self.calls: list[list[int]] = []

    def label_batch(self, evidences: list[ClusterEvidence], *, model: str) -> list[dict[str, object]]:
        self.calls.append([entry.cluster_id for entry in evidences])
        return [
            {
                "cluster_id": entry.cluster_id,
                "plain_label": "Tech Noir Chase",
                "poetic_label": "Neon Machines at Midnight",
                "spotify_style_label": "Cyberpunk Pursuit Mode",
                "recommended_label": "Neon Techno-Paranoia",
                "one_sentence_description": "Sleek futures where identity and machines collide.",
                "why_this_label_fits": "The evidence centers robots, dystopia, action, and cyberpunk.",
                "representative_movies": entry.representative_movies,
                "edge_case_movies": entry.representative_movies[-1:],
                "possible_misfits": [],
                "confidence_score": 0.86,
                "label_risk_notes": "Watch for franchise language.",
            }
            for entry in evidences
        ]

    def close(self) -> None:
        return None


def _evidence(cluster_id: int = 1) -> ClusterEvidence:
    return ClusterEvidence(
        cluster_id=cluster_id,
        cluster_size=3,
        representative_movies=["The Matrix", "Blade Runner", "RoboCop"],
        top_official_genres=[("Science Fiction", 3), ("Action", 2)],
        top_tmdb_keywords=[("dystopia", 3), ("robot", 2), ("cyberpunk", 2)],
        aggregated_profile_terms=[("robot", 0.5), ("future", 0.4), ("identity", 0.3)],
        in_cluster_neighbor_pairs=[],
        coherence_score=0.72,
        warnings=[],
    )


def test_label_prompt_construction_includes_required_guidance() -> None:
    messages = build_label_messages([_evidence()])

    assert messages[0]["role"] == "system"
    assert "vivid but useful" in messages[0]["content"]
    assert "The Matrix" in messages[1]["content"]
    assert "recommended_label" in messages[1]["content"]


def test_label_response_parsing_and_candidate_normalization() -> None:
    payload = {
        "clusters": [
            {
                "cluster_id": 1,
                "plain_label": "Robot Dystopias",
                "poetic_label": "Chrome Under Siege",
                "spotify_style_label": "Neon Paranoia",
                "recommended_label": "Neon Techno-Paranoia",
                "one_sentence_description": "Futures where machines and identity blur.",
                "why_this_label_fits": "Robots, dystopia, and cyberpunk dominate.",
                "representative_movies": ["The Matrix", "Blade Runner"],
                "edge_case_movies": ["RoboCop"],
                "possible_misfits": [],
                "confidence_score": 1.2,
                "label_risk_notes": "May be too sci-fi broad.",
            }
        ]
    }

    parsed = parse_label_response(json.dumps(payload))
    candidate = candidate_from_response(_evidence(), parsed[0], model="model-a")

    assert candidate.recommended_label == "Neon Techno-Paranoia"
    assert candidate.confidence_score == 1.0
    assert candidate.evidence_summary.aggregated_terms[:2] == ["robot", "future"]


def test_label_cache_reuses_unchanged_evidence() -> None:
    evidence = _evidence()
    candidate = candidate_from_response(
        evidence,
        {
            "cluster_id": 1,
            "plain_label": "Robot Dystopias",
            "poetic_label": "Chrome Under Siege",
            "spotify_style_label": "Neon Paranoia",
            "recommended_label": "Neon Techno-Paranoia",
            "one_sentence_description": "Futures where machines and identity blur.",
            "why_this_label_fits": "Robots, dystopia, and cyberpunk dominate.",
            "confidence_score": 0.8,
        },
        model="model-a",
    )
    cache = {label_cache_key("model-a", evidence_hash(evidence)): candidate}
    client = FakeLabelClient()

    candidates, cached_count, new_count = label_clusters(
        [evidence],
        cache=cache,
        model="model-a",
        api_key="unused",
        batch_size=5,
        client=client,  # type: ignore[arg-type]
    )

    assert cached_count == 1
    assert new_count == 0
    assert candidates[0].cached is True
    assert client.calls == []


def test_review_markdown_and_human_editable_json_render() -> None:
    evidence = _evidence()
    candidate = candidate_from_response(
        evidence,
        FakeLabelClient().label_batch([evidence], model="model-a")[0],
        model="model-a",
    )

    review = render_label_review_markdown([candidate])
    editable = render_human_editable_labels([candidate])

    assert "Cluster 1: Neon Techno-Paranoia" in review
    assert "Review fields" in review
    assert editable[0]["human_review"]["final_label"] == "Neon Techno-Paranoia"
    assert editable[0]["evidence_summary"]["genres"][0] == ("Science Fiction", 3)


def test_missing_cluster_evidence_and_embeddings_handled_gracefully(tmp_path: Path) -> None:
    with pytest.raises(ClusterLabelError, match="Cluster evidence not found"):
        load_or_build_label_evidence(
            evidence_path=tmp_path / "missing.json",
            embeddings_path=tmp_path / "missing_embeddings.jsonl",
            movies_path=tmp_path / "movies.json",
            profiles_path=tmp_path / "profiles.json",
            k=35,
        )

    with pytest.raises(ClusterLabelError, match="No embeddings found"):
        load_or_build_label_evidence(
            evidence_path=None,
            embeddings_path=tmp_path / "missing_embeddings.jsonl",
            movies_path=tmp_path / "movies.json",
            profiles_path=tmp_path / "profiles.json",
            k=35,
        )


def test_label_cache_write_and_estimate_uses_hash(tmp_path: Path) -> None:
    evidence = _evidence()
    candidate = candidate_from_response(
        evidence,
        FakeLabelClient().label_batch([evidence], model="model-a")[0],
        model="model-a",
    )
    cache_path = tmp_path / "cluster_label_cache.json"

    write_label_cache(cache_path, [candidate])
    cache_payload = json.loads(cache_path.read_text(encoding="utf-8"))
    estimate = estimate_labeling(
        [evidence],
        cache={next(iter(cache_payload["entries"])): candidate},
        model="model-a",
        openai_api_key=None,
    )

    assert next(iter(cache_payload["entries"])).startswith("model-a:")
    assert estimate.cached_count == 1
    assert estimate.clusters_to_label == 0
    assert estimate.openai_api_key_status == "missing"


def test_label_candidate_from_dict_round_trip() -> None:
    evidence = _evidence()
    candidate = candidate_from_response(
        evidence,
        FakeLabelClient().label_batch([evidence], model="model-a")[0],
        model="model-a",
    )

    hydrated = ClusterLabelCandidate.from_dict(candidate.to_dict())

    assert hydrated.to_dict() == candidate.to_dict()
