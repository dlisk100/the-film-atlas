from __future__ import annotations

from film_atlas.embedding import estimate_profiles, estimate_text_tokens
from film_atlas.models import SemanticProfile


def test_embedding_cost_estimate_uses_limit_and_model_price() -> None:
    profiles = [
        SemanticProfile(1, "A", 2000, "one two three four", [], []),
        SemanticProfile(2, "B", 2001, "x" * 400, [], []),
    ]

    estimate = estimate_profiles(profiles, model="text-embedding-3-large", limit=1)

    assert estimate.profile_count == 2
    assert estimate.selected_count == 1
    assert estimate.estimated_tokens == estimate_text_tokens(profiles[0].profile_text)
    assert estimate.estimated_cost_usd == estimate.estimated_tokens / 1_000_000 * 0.13
