from __future__ import annotations

import json
import math
from pathlib import Path

from film_atlas.territory_layout import build_territory_layouts_file


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_build_territory_layouts_preserves_nested_public_export(tmp_path: Path) -> None:
    export_dir = tmp_path / "public_export"
    export_dir.mkdir()
    points = [
        {"tmdb_id": 1, "macro_id": 10, "neighborhood_id": 100, "micro_id": 1000, "x": 0.0, "y": 0.0},
        {"tmdb_id": 2, "macro_id": 10, "neighborhood_id": 100, "micro_id": 1000, "x": 0.1, "y": 0.0},
        {"tmdb_id": 3, "macro_id": 10, "neighborhood_id": 101, "micro_id": 1001, "x": 0.3, "y": 0.2},
        {"tmdb_id": 4, "macro_id": 20, "neighborhood_id": 200, "micro_id": 2000, "x": 5.0, "y": 5.0},
        {"tmdb_id": 5, "macro_id": 20, "neighborhood_id": 200, "micro_id": 2001, "x": 5.2, "y": 5.1},
        {"tmdb_id": 6, "macro_id": 20, "neighborhood_id": 201, "micro_id": 2002, "x": 5.3, "y": 4.9},
    ]
    _write_json(export_dir / "points.json", points)
    _write_json(export_dir / "macro_clusters.json", [
        {"cluster_id": 10, "parent_cluster_id": None, "size": 3},
        {"cluster_id": 20, "parent_cluster_id": None, "size": 3},
    ])
    _write_json(export_dir / "neighborhood_clusters.json", [
        {"cluster_id": 100, "parent_cluster_id": 10, "size": 2},
        {"cluster_id": 101, "parent_cluster_id": 10, "size": 1},
        {"cluster_id": 200, "parent_cluster_id": 20, "size": 2},
        {"cluster_id": 201, "parent_cluster_id": 20, "size": 1},
    ])
    _write_json(export_dir / "micro_clusters.json", [
        {"cluster_id": 1000, "parent_cluster_id": 100, "size": 2},
        {"cluster_id": 1001, "parent_cluster_id": 101, "size": 1},
        {"cluster_id": 2000, "parent_cluster_id": 200, "size": 1},
        {"cluster_id": 2001, "parent_cluster_id": 200, "size": 1},
        {"cluster_id": 2002, "parent_cluster_id": 201, "size": 1},
    ])
    neighbor_shards = export_dir / "neighbor_shards"
    neighbor_shards.mkdir()
    _write_json(neighbor_shards / "01.json", [
        {"tmdb_id": 1, "neighbors": [{"tmdb_id": 2, "title": "Two", "similarity": 0.95}]},
    ])
    _write_json(export_dir / "manifest.json", {"files": ["points.json"], "movie_count": 6})

    result = build_territory_layouts_file(export_dir=export_dir)

    assert result.variant_count == 1
    payload = json.loads(result.layout_path.read_text(encoding="utf-8"))
    assert {variant["id"] for variant in payload["variants"]} == {"semantic_gmap_cells"}

    for variant in payload["variants"]:
        assert {point["tmdb_id"] for point in variant["points"]} == {1, 2, 3, 4, 5, 6}
        assert "macro_weighted_neighbor_distance" in variant["metrics"]
        regions = {
            (region["layer"], region["cluster_id"]): region
            for region in variant["regions"]
        }
        for region in variant["regions"]:
            if region["layer"] == "macro":
                continue
            parent_layer = "macro" if region["layer"] == "neighborhood" else "neighborhood"
            parent = regions[(parent_layer, region["parent_cluster_id"])]
            distance = math.hypot(region["x"] - parent["x"], region["y"] - parent["y"])
            assert distance + region["radius"] <= parent["radius"] + 1e-5

        for point in variant["points"]:
            original = next(item for item in points if item["tmdb_id"] == point["tmdb_id"])
            micro = regions[("micro", original["micro_id"])]
            distance = math.hypot(point["x"] - micro["x"], point["y"] - micro["y"])
            assert distance <= micro["radius"] + 1e-5

    gmap_variant = next(variant for variant in payload["variants"] if variant["id"] == "semantic_gmap_cells")
    assert len(gmap_variant["gmap_cells"]) == 6
    assert gmap_variant["metrics"]["gmap_cells"] == 6
    for cell in gmap_variant["gmap_cells"]:
        assert {"tmdb_id", "macro_id", "neighborhood_id", "micro_id", "polygon"} <= set(cell)
        assert len(cell["polygon"]) >= 3

    manifest = json.loads((export_dir / "manifest.json").read_text(encoding="utf-8"))
    assert "territory_layouts.json" in manifest["files"]
    assert manifest["territory_layouts"]["variant_count"] == 1
