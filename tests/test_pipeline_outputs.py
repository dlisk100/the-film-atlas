from __future__ import annotations

import json
from pathlib import Path

from film_atlas.normalize import normalize_details_file
from film_atlas.profiles import build_profiles_file
from film_atlas.report import generate_report_file
from film_atlas.sample_map import make_sample_map_file


def test_fixture_pipeline_writes_profile_map_and_report(tmp_path: Path) -> None:
    details_payload = json.loads(
        Path("tests/fixtures/tmdb_movie_details_payload.json").read_text()
    )
    discover_payload = json.loads(Path("tests/fixtures/tmdb_discover_payload.json").read_text())
    details_path = tmp_path / "data" / "raw" / "movie_details.json"
    discover_path = tmp_path / "data" / "raw" / "discover_movies.json"
    details_path.parent.mkdir(parents=True)
    details_path.write_text(json.dumps(details_payload), encoding="utf-8")
    discover_path.write_text(json.dumps(discover_payload), encoding="utf-8")

    movies_path = normalize_details_file(details_path=details_path, output_dir=tmp_path / "data")
    profiles_path = build_profiles_file(movies_path=movies_path, output_dir=tmp_path / "data")
    map_result = make_sample_map_file(profiles_path=profiles_path, output_dir=tmp_path / "outputs")
    report_path = generate_report_file(data_dir=tmp_path / "data", output_dir=tmp_path / "outputs")

    assert movies_path.exists()
    assert profiles_path.exists()
    assert map_result.csv_path.exists()
    assert map_result.html_path.exists()
    report_text = report_path.read_text(encoding="utf-8")
    assert "Discovered movies: 1" in report_text
    assert "Detail records fetched: 1" in report_text
    assert "OpenAI embeddings and cluster labeling are out of scope" in report_text
