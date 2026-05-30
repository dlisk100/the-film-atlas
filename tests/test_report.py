from __future__ import annotations

from film_atlas.models import MovieRecord
from film_atlas.report import render_report


def test_report_includes_sampling_bias_and_review_noise_diagnostics() -> None:
    movies = [
        MovieRecord(
            tmdb_id=1,
            imdb_id=None,
            title="Future Sequel",
            original_title="Future Sequel",
            release_date="2999-01-01",
            year=2999,
            runtime=100,
            overview="A future sequel.",
            genres=["Action"],
            keywords=["sequel", "superhero"],
            poster_path=None,
            backdrop_path=None,
            vote_average=7.0,
            vote_count=1000,
            popularity=50.0,
            reviews=["FULL REVIEW: wow wow wow!!! https://example.com #tag"],
        )
    ]

    report = render_report(discovered_count=1, movies=movies, profiles=[], neighbor_pairs=[])

    assert "From 2024 or later: 100.0% (1/1)" in report
    assert "From future release years: 100.0% (1/1)" in report
    assert "Franchise/Sequel Keywords" in report
    assert "| sequel | 1 |" in report
    assert "Warning: Current-release concentration is high" in report
    assert "Warning: Franchise/sequel concentration is high" in report
    assert "Review Noise Diagnostics" in report
    assert "uv run film-atlas fetch-balanced --per-decade 100 --start-year 1980 --end-year 2026" in report
