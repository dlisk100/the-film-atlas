"""Build experimental territory-map layouts from the public Film Atlas export."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

TERRITORY_LAYOUTS_FILENAME = "territory_layouts.json"
LayerName = Literal["macro", "neighborhood", "micro"]


class TerritoryLayoutError(RuntimeError):
    """Raised when territory layout generation cannot proceed."""


@dataclass(frozen=True, slots=True)
class TerritoryLayoutResult:
    export_dir: Path
    layout_path: Path
    variant_count: int
    movie_count: int
    region_count: int

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["export_dir"] = str(self.export_dir)
        data["layout_path"] = str(self.layout_path)
        return data


@dataclass(frozen=True, slots=True)
class SourcePoint:
    tmdb_id: int
    macro_id: int
    neighborhood_id: int
    micro_id: int
    x: float
    y: float


@dataclass(frozen=True, slots=True)
class Circle:
    id: int
    x: float
    y: float
    radius: float


@dataclass(frozen=True, slots=True)
class VariantSpec:
    id: str
    label: str
    description: str
    macro_fill_ratio: float
    macro_semantic_weight: float
    child_semantic_weight: float
    movie_local_weight: float
    fill_ratio: float
    padding_ratio: float


VARIANT_SPECS = [
    VariantSpec(
        id="strict_pack",
        label="Strict Packed Atlas",
        description=(
            "Maximizes visible hierarchy: movies pack into microclusters, microclusters "
            "pack into neighborhoods, and neighborhoods pack into macro territories."
        ),
        macro_fill_ratio=0.92,
        macro_semantic_weight=0.0,
        child_semantic_weight=0.0,
        movie_local_weight=0.15,
        fill_ratio=0.82,
        padding_ratio=0.018,
    ),
    VariantSpec(
        id="semantic_territories",
        label="Semantic Territory Atlas",
        description=(
            "Keeps the hierarchy nested while anchoring continents and subregions toward "
            "their original UMAP semantic directions."
        ),
        macro_fill_ratio=0.76,
        macro_semantic_weight=1.0,
        child_semantic_weight=0.92,
        movie_local_weight=0.92,
        fill_ratio=0.54,
        padding_ratio=0.07,
    ),
    VariantSpec(
        id="hybrid_micro_islands",
        label="Hybrid Micro-Islands",
        description=(
            "Uses readable macro territories, semantically nudged neighborhoods, and "
            "tighter local islands for microcluster-level browsing."
        ),
        macro_fill_ratio=0.84,
        macro_semantic_weight=0.55,
        child_semantic_weight=0.66,
        movie_local_weight=0.78,
        fill_ratio=0.50,
        padding_ratio=0.08,
    ),
]


def build_territory_layouts_file(
    *,
    export_dir: str | Path = "outputs/public_export",
) -> TerritoryLayoutResult:
    """Write experimental nested territory-layout variants into the public export."""
    export_path = Path(export_dir)
    points = _load_source_points(export_path / "points.json")
    if not points:
        raise TerritoryLayoutError(f"No points found at {export_path / 'points.json'}.")

    macro_clusters = _load_clusters(export_path / "macro_clusters.json")
    neighborhood_clusters = _load_clusters(export_path / "neighborhood_clusters.json")
    micro_clusters = _load_clusters(export_path / "micro_clusters.json")

    variants = [
        _build_variant(
            spec=spec,
            points=points,
            macro_clusters=macro_clusters,
            neighborhood_clusters=neighborhood_clusters,
            micro_clusters=micro_clusters,
        )
        for spec in VARIANT_SPECS
    ]
    region_count = max(len(variant["regions"]) for variant in variants)
    layout_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "film-atlas-territory-layout-v1",
        "movie_count": len(points),
        "variants": variants,
    }

    layout_path = export_path / TERRITORY_LAYOUTS_FILENAME
    layout_path.write_text(json.dumps(layout_payload, indent=2, sort_keys=True), encoding="utf-8")
    _update_manifest(export_path / "manifest.json", layout_payload)
    return TerritoryLayoutResult(
        export_dir=export_path,
        layout_path=layout_path,
        variant_count=len(variants),
        movie_count=len(points),
        region_count=region_count,
    )


def _load_source_points(path: Path) -> list[SourcePoint]:
    if not path.exists():
        raise TerritoryLayoutError(f"Missing public export file: {path}")
    raw_points = json.loads(path.read_text(encoding="utf-8"))
    points = []
    for raw in raw_points:
        try:
            points.append(
                SourcePoint(
                    tmdb_id=int(raw["tmdb_id"]),
                    macro_id=int(raw["macro_id"]),
                    neighborhood_id=int(raw["neighborhood_id"]),
                    micro_id=int(raw["micro_id"]),
                    x=float(raw["x"]),
                    y=float(raw["y"]),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise TerritoryLayoutError(f"Invalid point record in {path}: {raw!r}") from exc
    return points


def _load_clusters(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists():
        raise TerritoryLayoutError(f"Missing public export file: {path}")
    clusters = json.loads(path.read_text(encoding="utf-8"))
    return {int(cluster["cluster_id"]): cluster for cluster in clusters}


def _build_variant(
    *,
    spec: VariantSpec,
    points: list[SourcePoint],
    macro_clusters: dict[int, dict[str, Any]],
    neighborhood_clusters: dict[int, dict[str, Any]],
    micro_clusters: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    macro_points = _group_points(points, "macro")
    neighborhood_points = _group_points(points, "neighborhood")
    micro_points = _group_points(points, "micro")
    macro_centroids = _centroids(macro_points)
    neighborhood_centroids = _centroids(neighborhood_points)
    micro_centroids = _centroids(micro_points)
    bounds = _point_bounds(points)

    macro_ids = sorted(macro_points, key=lambda cluster_id: (-len(macro_points[cluster_id]), cluster_id))
    root_radius = 100.0
    macro_radii = _scaled_radii(
        {cluster_id: len(macro_points[cluster_id]) for cluster_id in macro_ids},
        parent_radius=root_radius,
        fill_ratio=spec.macro_fill_ratio,
        min_radius=8.5,
    )
    macro_targets = {
        cluster_id: _semantic_target(
            macro_centroids[cluster_id],
            parent_centroid=_global_centroid(points),
            bounds=bounds,
            radius=root_radius * 0.74,
        )
        for cluster_id in macro_ids
    }
    macro_circles = _pack_circles(
        ids=macro_ids,
        radii=macro_radii,
        parent=Circle(id=-1, x=0.0, y=0.0, radius=root_radius),
        targets=macro_targets,
        semantic_weight=spec.macro_semantic_weight,
        padding=1.3,
    )

    neighborhood_circles: dict[int, Circle] = {}
    micro_circles: dict[int, Circle] = {}
    region_records = []

    for macro_id in macro_ids:
        macro_circle = macro_circles[macro_id]
        macro_region = _region_record(
            layer="macro",
            cluster_id=macro_id,
            parent_cluster_id=None,
            macro_id=macro_id,
            neighborhood_id=None,
            circle=macro_circle,
            size=len(macro_points[macro_id]),
        )
        region_records.append(macro_region)

        child_ids = sorted(
            (
                cluster_id
                for cluster_id, cluster in neighborhood_clusters.items()
                if int(cluster.get("parent_cluster_id", -1)) == macro_id
                and cluster_id in neighborhood_points
            ),
            key=lambda cluster_id: (-len(neighborhood_points[cluster_id]), cluster_id),
        )
        child_radii = _scaled_radii(
            {cluster_id: len(neighborhood_points[cluster_id]) for cluster_id in child_ids},
            parent_radius=macro_circle.radius,
            fill_ratio=spec.fill_ratio,
            min_radius=max(2.2, macro_circle.radius * 0.08),
        )
        child_targets = _child_targets(
            child_ids=child_ids,
            child_centroids=neighborhood_centroids,
            parent_centroid=macro_centroids[macro_id],
            parent_circle=macro_circle,
        )
        neighborhood_circles.update(
            _pack_circles(
                ids=child_ids,
                radii=child_radii,
                parent=macro_circle,
                targets=child_targets,
                semantic_weight=spec.child_semantic_weight,
                padding=max(0.28, macro_circle.radius * spec.padding_ratio),
            )
        )

    neighborhood_to_macro = {
        cluster_id: int(cluster["parent_cluster_id"])
        for cluster_id, cluster in neighborhood_clusters.items()
        if cluster.get("parent_cluster_id") is not None
    }
    micro_to_neighborhood = {
        cluster_id: int(cluster["parent_cluster_id"])
        for cluster_id, cluster in micro_clusters.items()
        if cluster.get("parent_cluster_id") is not None
    }

    for neighborhood_id, neighborhood_circle in sorted(neighborhood_circles.items()):
        macro_id = neighborhood_to_macro[neighborhood_id]
        region_records.append(
            _region_record(
                layer="neighborhood",
                cluster_id=neighborhood_id,
                parent_cluster_id=macro_id,
                macro_id=macro_id,
                neighborhood_id=neighborhood_id,
                circle=neighborhood_circle,
                size=len(neighborhood_points[neighborhood_id]),
            )
        )
        child_ids = sorted(
            (
                cluster_id
                for cluster_id, parent_id in micro_to_neighborhood.items()
                if parent_id == neighborhood_id and cluster_id in micro_points
            ),
            key=lambda cluster_id: (-len(micro_points[cluster_id]), cluster_id),
        )
        child_radii = _scaled_radii(
            {cluster_id: len(micro_points[cluster_id]) for cluster_id in child_ids},
            parent_radius=neighborhood_circle.radius,
            fill_ratio=spec.fill_ratio,
            min_radius=max(0.72, neighborhood_circle.radius * 0.12),
        )
        child_targets = _child_targets(
            child_ids=child_ids,
            child_centroids=micro_centroids,
            parent_centroid=neighborhood_centroids[neighborhood_id],
            parent_circle=neighborhood_circle,
        )
        micro_circles.update(
            _pack_circles(
                ids=child_ids,
                radii=child_radii,
                parent=neighborhood_circle,
                targets=child_targets,
                semantic_weight=spec.child_semantic_weight,
                padding=max(0.12, neighborhood_circle.radius * spec.padding_ratio),
            )
        )

    for micro_id, micro_circle in sorted(micro_circles.items()):
        neighborhood_id = micro_to_neighborhood[micro_id]
        macro_id = neighborhood_to_macro[neighborhood_id]
        region_records.append(
            _region_record(
                layer="micro",
                cluster_id=micro_id,
                parent_cluster_id=neighborhood_id,
                macro_id=macro_id,
                neighborhood_id=neighborhood_id,
                circle=micro_circle,
                size=len(micro_points[micro_id]),
            )
        )

    point_records = []
    for micro_id in sorted(micro_points):
        if micro_id not in micro_circles:
            continue
        point_records.extend(
            _place_movies_in_micro(
                points=micro_points[micro_id],
                circle=micro_circles[micro_id],
                local_weight=spec.movie_local_weight,
            )
        )

    return {
        "id": spec.id,
        "label": spec.label,
        "description": spec.description,
        "algorithm": "nested_circle_pack",
        "metrics": {
            "macro_regions": len(macro_circles),
            "neighborhood_regions": len(neighborhood_circles),
            "micro_regions": len(micro_circles),
            "movie_points": len(point_records),
        },
        "points": sorted(point_records, key=lambda item: item["tmdb_id"]),
        "regions": sorted(
            region_records,
            key=lambda item: (_layer_sort(item["layer"]), item["cluster_id"]),
        ),
    }


def _group_points(points: list[SourcePoint], layer: LayerName) -> dict[int, list[SourcePoint]]:
    groups: dict[int, list[SourcePoint]] = defaultdict(list)
    for point in points:
        if layer == "macro":
            groups[point.macro_id].append(point)
        elif layer == "neighborhood":
            groups[point.neighborhood_id].append(point)
        else:
            groups[point.micro_id].append(point)
    return dict(groups)


def _centroids(groups: dict[int, list[SourcePoint]]) -> dict[int, tuple[float, float]]:
    return {
        cluster_id: (
            sum(point.x for point in cluster_points) / len(cluster_points),
            sum(point.y for point in cluster_points) / len(cluster_points),
        )
        for cluster_id, cluster_points in groups.items()
        if cluster_points
    }


def _global_centroid(points: list[SourcePoint]) -> tuple[float, float]:
    return (
        sum(point.x for point in points) / len(points),
        sum(point.y for point in points) / len(points),
    )


def _point_bounds(points: list[SourcePoint]) -> tuple[float, float, float, float]:
    return (
        min(point.x for point in points),
        max(point.x for point in points),
        min(point.y for point in points),
        max(point.y for point in points),
    )


def _scaled_radii(
    sizes: dict[int, int],
    *,
    parent_radius: float,
    fill_ratio: float,
    min_radius: float,
) -> dict[int, float]:
    if not sizes:
        return {}
    total = sum(sizes.values())
    scale = parent_radius * math.sqrt(fill_ratio / max(1, total))
    return {
        cluster_id: max(min_radius, math.sqrt(max(1, size)) * scale)
        for cluster_id, size in sizes.items()
    }


def _semantic_target(
    centroid: tuple[float, float],
    *,
    parent_centroid: tuple[float, float],
    bounds: tuple[float, float, float, float],
    radius: float,
) -> tuple[float, float]:
    min_x, max_x, min_y, max_y = bounds
    scale = max(max_x - min_x, max_y - min_y, 1.0)
    dx = (centroid[0] - parent_centroid[0]) / scale * 2.0
    dy = (centroid[1] - parent_centroid[1]) / scale * 2.0
    distance = math.hypot(dx, dy)
    if distance > 1.0:
        dx /= distance
        dy /= distance
    return (dx * radius, dy * radius)


def _child_targets(
    *,
    child_ids: list[int],
    child_centroids: dict[int, tuple[float, float]],
    parent_centroid: tuple[float, float],
    parent_circle: Circle,
) -> dict[int, tuple[float, float]]:
    deltas = {
        cluster_id: (
            child_centroids[cluster_id][0] - parent_centroid[0],
            child_centroids[cluster_id][1] - parent_centroid[1],
        )
        for cluster_id in child_ids
        if cluster_id in child_centroids
    }
    max_distance = max((math.hypot(dx, dy) for dx, dy in deltas.values()), default=1.0)
    targets = {}
    for cluster_id in child_ids:
        dx, dy = deltas.get(cluster_id, (0.0, 0.0))
        if max_distance > 0:
            dx /= max_distance
            dy /= max_distance
        targets[cluster_id] = (
            parent_circle.x + dx * parent_circle.radius * 0.55,
            parent_circle.y + dy * parent_circle.radius * 0.55,
        )
    return targets


def _pack_circles(
    *,
    ids: list[int],
    radii: dict[int, float],
    parent: Circle,
    targets: dict[int, tuple[float, float]],
    semantic_weight: float,
    padding: float,
) -> dict[int, Circle]:
    placed: dict[int, Circle] = {}
    for index, cluster_id in enumerate(ids):
        radius = radii[cluster_id]
        target = targets.get(cluster_id, (parent.x, parent.y))
        placed[cluster_id] = _place_circle(
            cluster_id=cluster_id,
            index=index,
            radius=radius,
            parent=parent,
            target=target,
            semantic_weight=semantic_weight,
            siblings=list(placed.values()),
            padding=padding,
        )
    return placed


def _place_circle(
    *,
    cluster_id: int,
    index: int,
    radius: float,
    parent: Circle,
    target: tuple[float, float],
    semantic_weight: float,
    siblings: list[Circle],
    padding: float,
) -> Circle:
    if not siblings and semantic_weight <= 0:
        return Circle(id=cluster_id, x=parent.x, y=parent.y, radius=radius)

    best: tuple[float, float, float] | None = None
    max_distance = max(0.0, parent.radius - radius - padding)
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    base_angle = (cluster_id * 0.61803398875 + index * 0.173) * math.tau
    target_angle = math.atan2(target[1] - parent.y, target[0] - parent.x)
    angle_start = target_angle * semantic_weight + base_angle * (1.0 - semantic_weight)

    candidates = [(target[0], target[1]), (parent.x, parent.y)]
    ring_count = 18
    angle_count = 34
    for ring_index in range(ring_count + 1):
        ring_fraction = math.sqrt(ring_index / max(1, ring_count))
        distance = max_distance * ring_fraction
        for angle_index in range(angle_count):
            angle = angle_start + angle_index * golden_angle
            candidates.append((
                parent.x + math.cos(angle) * distance,
                parent.y + math.sin(angle) * distance,
            ))

    for x, y in candidates:
        dx = x - parent.x
        dy = y - parent.y
        distance_from_parent = math.hypot(dx, dy)
        if distance_from_parent > max_distance and distance_from_parent > 0:
            scale = max_distance / distance_from_parent
            x = parent.x + dx * scale
            y = parent.y + dy * scale
        boundary_penalty = max(0.0, math.hypot(x - parent.x, y - parent.y) + radius - parent.radius)
        overlap_penalty = sum(
            max(0.0, sibling.radius + radius + padding - math.hypot(x - sibling.x, y - sibling.y))
            for sibling in siblings
        )
        target_distance = math.hypot(x - target[0], y - target[1])
        center_distance = math.hypot(x - parent.x, y - parent.y)
        score = (
            boundary_penalty * 1500.0
            + overlap_penalty * 1100.0
            + target_distance * (0.2 + semantic_weight)
            + center_distance * max(0.0, 0.18 - semantic_weight * 0.08)
        )
        if best is None or score < best[0]:
            best = (score, x, y)

    if best is None:
        return Circle(id=cluster_id, x=parent.x, y=parent.y, radius=radius)
    return Circle(id=cluster_id, x=best[1], y=best[2], radius=radius)


def _place_movies_in_micro(
    *,
    points: list[SourcePoint],
    circle: Circle,
    local_weight: float,
) -> list[dict[str, float | int]]:
    if not points:
        return []
    centroid_x = sum(point.x for point in points) / len(points)
    centroid_y = sum(point.y for point in points) / len(points)
    max_distance = max(
        (math.hypot(point.x - centroid_x, point.y - centroid_y) for point in points),
        default=0.0,
    )
    local_scale = (circle.radius * 0.78 / max_distance) if max_distance > 0 else 0.0
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    ordered = sorted(points, key=lambda point: (math.atan2(point.y - centroid_y, point.x - centroid_x), point.tmdb_id))
    records = []
    for index, point in enumerate(ordered):
        local_x = (point.x - centroid_x) * local_scale
        local_y = (point.y - centroid_y) * local_scale
        spiral_radius = circle.radius * 0.78 * math.sqrt((index + 0.5) / len(ordered))
        spiral_angle = index * golden_angle + (point.tmdb_id % 29) * 0.07
        spiral_x = math.cos(spiral_angle) * spiral_radius
        spiral_y = math.sin(spiral_angle) * spiral_radius
        offset_x = local_x * local_weight + spiral_x * (1.0 - local_weight)
        offset_y = local_y * local_weight + spiral_y * (1.0 - local_weight)
        distance = math.hypot(offset_x, offset_y)
        max_offset = circle.radius * 0.86
        if distance > max_offset and distance > 0:
            scale = max_offset / distance
            offset_x *= scale
            offset_y *= scale
        records.append({
            "tmdb_id": point.tmdb_id,
            "x": round(circle.x + offset_x, 6),
            "y": round(circle.y + offset_y, 6),
        })
    return records


def _region_record(
    *,
    layer: LayerName,
    cluster_id: int,
    parent_cluster_id: int | None,
    macro_id: int,
    neighborhood_id: int | None,
    circle: Circle,
    size: int,
) -> dict[str, float | int | str | None]:
    return {
        "layer": layer,
        "cluster_id": cluster_id,
        "parent_cluster_id": parent_cluster_id,
        "macro_id": macro_id,
        "neighborhood_id": neighborhood_id,
        "x": round(circle.x, 6),
        "y": round(circle.y, 6),
        "radius": round(circle.radius, 6),
        "size": size,
    }


def _layer_sort(layer: str) -> int:
    return {"macro": 0, "neighborhood": 1, "micro": 2}.get(layer, 3)


def _update_manifest(path: Path, payload: dict[str, Any]) -> None:
    if not path.exists():
        return
    manifest = json.loads(path.read_text(encoding="utf-8"))
    files = list(manifest.get("files") or [])
    if TERRITORY_LAYOUTS_FILENAME not in files:
        files.append(TERRITORY_LAYOUTS_FILENAME)
    manifest["files"] = files
    manifest["territory_layouts"] = {
        "file": TERRITORY_LAYOUTS_FILENAME,
        "variant_count": len(payload["variants"]),
        "variants": [
            {
                "id": variant["id"],
                "label": variant["label"],
                "description": variant["description"],
            }
            for variant in payload["variants"]
        ],
    }
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
