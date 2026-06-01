import httpx

from film_atlas.large_audit import (
    OpenAIAuditClient,
    audit_rows_with_completion,
    label_contradictions,
    normalize_llm_results,
    public_export_signature,
    select_audit_ids,
)


def test_label_contradictions_flags_animation_micro_without_animation() -> None:
    movie = {
        "title": "Office Space",
        "genres": ["Comedy"],
        "keywords": ["workplace", "office"],
    }
    labels = {
        "macro": "Adult Comedy",
        "neighborhood": "Workplace Frustration Comedy",
        "micro": "Bright Family Animation Comedy",
    }

    flags = label_contradictions(movie, labels)

    assert flags
    assert flags[0]["type"] == "animation_label_without_animation"
    assert flags[0]["layer"] == "micro"


def test_normalize_llm_results_filters_unknown_ids_and_bounds_severity() -> None:
    rows = normalize_llm_results(
        [
            {
                "tmdb_id": 1,
                "verdict": "FAIL",
                "severity": 9,
                "issue_type": "both",
                "layer": "micro",
                "rationale": "bad label",
            },
            {"tmdb_id": 2, "verdict": "pass"},
        ],
        expected_ids={1},
    )

    assert rows == [
        {
            "tmdb_id": 1,
            "verdict": "fail",
            "severity": 3,
            "issue_type": "both",
            "layer": "micro",
            "rationale": "bad label",
            "suggested_fix": "",
        }
    ]


def test_select_audit_ids_covers_microclusters_before_filling() -> None:
    export = {
        "movies": [
            {"tmdb_id": 1, "title": "A", "popularity": 1, "vote_count": 1},
            {"tmdb_id": 2, "title": "B", "popularity": 1, "vote_count": 1},
            {"tmdb_id": 3, "title": "C", "popularity": 1, "vote_count": 1},
        ],
        "points": [
            {"tmdb_id": 1, "micro_id": 10},
            {"tmdb_id": 2, "micro_id": 11},
            {"tmdb_id": 3, "micro_id": 11},
        ],
        "movies_by_id": {
            1: {"tmdb_id": 1, "title": "A", "popularity": 1, "vote_count": 1},
            2: {"tmdb_id": 2, "title": "B", "popularity": 1, "vote_count": 1},
            3: {"tmdb_id": 3, "title": "C", "popularity": 1, "vote_count": 1},
        },
    }

    selected = select_audit_ids(
        export,
        {"flags_by_id": {}},
        review_count=2,
        seed=1,
    )

    assert len(selected) == 2
    assert 1 in selected


def test_public_export_signature_sorts_points_without_comparing_dicts() -> None:
    export = {
        "points": [
            {"tmdb_id": 2, "macro_id": 1, "neighborhood_id": 10, "micro_id": 20},
            {"tmdb_id": 1, "macro_id": 1, "neighborhood_id": 11, "micro_id": 21},
        ],
        "clusters": {
            "macro": [{"cluster_id": 1, "recommended_label": "A"}],
            "neighborhood": [
                {"cluster_id": 11, "parent_cluster_id": 1, "recommended_label": "C"},
                {"cluster_id": 10, "parent_cluster_id": 1, "recommended_label": "B"},
            ],
            "micro": [
                {"cluster_id": 21, "parent_cluster_id": 11, "recommended_label": "E"},
                {"cluster_id": 20, "parent_cluster_id": 10, "recommended_label": "D"},
            ],
        },
    }
    reordered_export = {
        **export,
        "points": list(reversed(export["points"])),
        "clusters": {
            layer: list(reversed(clusters))
            for layer, clusters in export["clusters"].items()
        },
    }

    assert public_export_signature(export) == public_export_signature(reordered_export)


def test_audit_rows_with_completion_retries_omitted_rows() -> None:
    class OmittingClient:
        def __init__(self) -> None:
            self.calls: list[list[int]] = []

        def audit_batch(self, rows: list[dict[str, int]], *, model: str) -> dict[str, object]:
            self.calls.append([row["tmdb_id"] for row in rows])
            if len(self.calls) == 1:
                results = [
                    {
                        "tmdb_id": rows[0]["tmdb_id"],
                        "verdict": "pass",
                        "severity": 0,
                        "issue_type": "none",
                        "layer": "none",
                        "rationale": "ok",
                    }
                ]
            else:
                results = [
                    {
                        "tmdb_id": row["tmdb_id"],
                        "verdict": "pass",
                        "severity": 0,
                        "issue_type": "none",
                        "layer": "none",
                        "rationale": "retried",
                    }
                    for row in rows
                ]
            return {
                "results": results,
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }

    client = OmittingClient()

    payload = audit_rows_with_completion(
        client, [{"tmdb_id": 1}, {"tmdb_id": 2}], model="test-model"
    )

    assert client.calls == [[1, 2], [2]]
    assert [row["tmdb_id"] for row in payload["results"]] == [1, 2]
    assert payload["usage"] == {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4}


def test_openai_audit_client_retries_read_timeout() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ReadTimeout("slow response", request=request)
        return httpx.Response(200, json={"ok": True})

    client = OpenAIAuditClient(
        "test-key",
        transport=httpx.MockTransport(handler),
        sleep=lambda _seconds: None,
    )
    try:
        payload = client._post_with_retries("/chat/completions", payload={}, headers={})
    finally:
        client.close()

    assert payload == {"ok": True}
    assert calls == 2
