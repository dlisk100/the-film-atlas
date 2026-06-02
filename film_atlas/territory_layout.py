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
LayoutMethod = Literal["packed_baseline", "graph_stress", "umap_anchored_graph"]


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
class NeighborLink:
    source_tmdb_id: int
    target_tmdb_id: int
    similarity: float


@dataclass(frozen=True, slots=True)
class VariantSpec:
    id: str
    label: str
    description: str
    layout_method: LayoutMethod
    macro_fill_ratio: float
    macro_semantic_weight: float
    macro_target_ratio: float
    child_semantic_weight: float
    child_target_ratio: float
    graph_weight: float
    anchor_weight: float
    graph_iterations: int
    movie_local_weight: float
    movie_spread_ratio: float
    fill_ratio: float
    padding_ratio: float
    gmap_cells: bool = False


PRODUCTION_VARIANT_SPECS = [
    VariantSpec(
        id="semantic_gmap_cells",
        label="Semantic Cells",
        description=(
            "Final GMap-inspired atlas layer: films keep graph-driven semantic positions, "
            "then movie-level cells are shaded by macro, neighborhood, and micro membership "
            "so map borders emerge from the underlying film positions."
        ),
        layout_method="graph_stress",
        macro_fill_ratio=0.60,
        macro_semantic_weight=1.0,
        macro_target_ratio=0.86,
        child_semantic_weight=1.0,
        child_target_ratio=0.70,
        graph_weight=1.35,
        anchor_weight=0.055,
        graph_iterations=190,
        movie_local_weight=0.97,
        movie_spread_ratio=0.94,
        fill_ratio=0.38,
        padding_ratio=0.108,
        gmap_cells=True,
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
    neighbor_links = _load_neighbor_links(export_path / "neighbors.json")

    variants = [
        _build_variant(
            spec=spec,
            points=points,
            macro_clusters=macro_clusters,
            neighborhood_clusters=neighborhood_clusters,
            micro_clusters=micro_clusters,
            neighbor_links=neighbor_links,
        )
        for spec in PRODUCTION_VARIANT_SPECS
    ]
    region_count = max(len(variant["regions"]) for variant in variants)
    layout_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "film-atlas-semantic-graph-territory-layout-v5",
        "movie_count": len(points),
        "variants": variants,
    }

    layout_path = export_path / TERRITORY_LAYOUTS_FILENAME
    layout_path.write_text(json.dumps(layout_payload, separators=(",", ":"), sort_keys=True), encoding="utf-8")
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


def _load_neighbor_links(path: Path) -> list[NeighborLink]:
    if not path.exists():
        return []
    raw_records = json.loads(path.read_text(encoding="utf-8"))
    links = []
    for raw in raw_records:
        try:
            source_tmdb_id = int(raw["tmdb_id"])
        except (KeyError, TypeError, ValueError):
            continue
        for neighbor in raw.get("neighbors") or []:
            try:
                links.append(
                    NeighborLink(
                        source_tmdb_id=source_tmdb_id,
                        target_tmdb_id=int(neighbor["tmdb_id"]),
                        similarity=float(neighbor.get("similarity", 0.0)),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
    return links


def _build_variant(
    *,
    spec: VariantSpec,
    points: list[SourcePoint],
    macro_clusters: dict[int, dict[str, Any]],
    neighborhood_clusters: dict[int, dict[str, Any]],
    micro_clusters: dict[int, dict[str, Any]],
    neighbor_links: list[NeighborLink],
) -> dict[str, Any]:
    macro_points = _group_points(points, "macro")
    neighborhood_points = _group_points(points, "neighborhood")
    micro_points = _group_points(points, "micro")
    macro_centroids = _centroids(macro_points)
    neighborhood_centroids = _centroids(neighborhood_points)
    micro_centroids = _centroids(micro_points)
    bounds = _point_bounds(points)
    cluster_lookup = _cluster_lookup(points)
    movie_lookup = {point.tmdb_id: point for point in points}

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
            radius=root_radius * spec.macro_target_ratio,
        )
        for cluster_id in macro_ids
    }
    macro_affinities = _layout_affinities(
        spec=spec,
        ids=macro_ids,
        layer="macro",
        centroids=macro_centroids,
        sizes={cluster_id: len(macro_points[cluster_id]) for cluster_id in macro_ids},
        cluster_lookup=cluster_lookup,
        neighbor_links=neighbor_links,
        max_neighbors=8,
    )
    macro_circles = _layout_circles(
        ids=macro_ids,
        radii=macro_radii,
        parent=Circle(id=-1, x=0.0, y=0.0, radius=root_radius),
        targets=macro_targets,
        neighbor_affinities=macro_affinities,
        semantic_weight=spec.macro_semantic_weight,
        padding=1.3,
        spec=spec,
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
            target_radius_ratio=spec.child_target_ratio,
        )
        child_affinities = _layout_affinities(
            spec=spec,
            ids=child_ids,
            layer="neighborhood",
            centroids=neighborhood_centroids,
            sizes={cluster_id: len(neighborhood_points[cluster_id]) for cluster_id in child_ids},
            cluster_lookup=cluster_lookup,
            neighbor_links=neighbor_links,
            max_neighbors=7,
        )
        neighborhood_circles.update(
            _layout_circles(
                ids=child_ids,
                radii=child_radii,
                parent=macro_circle,
                targets=child_targets,
                neighbor_affinities=child_affinities,
                semantic_weight=spec.child_semantic_weight,
                padding=max(0.28, macro_circle.radius * spec.padding_ratio),
                spec=spec,
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
            target_radius_ratio=spec.child_target_ratio,
        )
        child_affinities = _layout_affinities(
            spec=spec,
            ids=child_ids,
            layer="micro",
            centroids=micro_centroids,
            sizes={cluster_id: len(micro_points[cluster_id]) for cluster_id in child_ids},
            cluster_lookup=cluster_lookup,
            neighbor_links=neighbor_links,
            max_neighbors=6,
        )
        micro_circles.update(
            _layout_circles(
                ids=child_ids,
                radii=child_radii,
                parent=neighborhood_circle,
                targets=child_targets,
                neighbor_affinities=child_affinities,
                semantic_weight=spec.child_semantic_weight,
                padding=max(0.12, neighborhood_circle.radius * spec.padding_ratio),
                spec=spec,
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
                movie_lookup=movie_lookup,
                neighbor_links=neighbor_links,
                local_weight=spec.movie_local_weight,
                spread_ratio=spec.movie_spread_ratio,
            )
        )

    variant = {
        "id": spec.id,
        "label": spec.label,
        "description": spec.description,
        "algorithm": f"semantic_graph_{spec.layout_method}",
        "metrics": {
            "macro_regions": len(macro_circles),
            "macro_weighted_neighbor_distance": _weighted_neighbor_distance(macro_circles, macro_affinities),
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
    if spec.gmap_cells:
        gmap_cells = _build_gmap_cells(point_records, movie_lookup)
        variant["gmap_cells"] = gmap_cells
        variant["metrics"]["gmap_cells"] = len(gmap_cells)
        variant["metrics"]["gmap_outer_vertices"] = _gmap_outer_vertex_count(gmap_cells)
    return variant


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


def _cluster_lookup(points: list[SourcePoint]) -> dict[LayerName, dict[int, int]]:
    return {
        "macro": {point.tmdb_id: point.macro_id for point in points},
        "neighborhood": {point.tmdb_id: point.neighborhood_id for point in points},
        "micro": {point.tmdb_id: point.micro_id for point in points},
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
    target_radius_ratio: float,
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
            parent_circle.x + dx * parent_circle.radius * target_radius_ratio,
            parent_circle.y + dy * parent_circle.radius * target_radius_ratio,
        )
    return targets


def _semantic_affinities(
    ids: list[int],
    centroids: dict[int, tuple[float, float]],
    *,
    max_neighbors: int,
) -> dict[int, dict[int, float]]:
    affinities: dict[int, dict[int, float]] = {}
    for cluster_id in ids:
        centroid = centroids.get(cluster_id)
        if centroid is None:
            affinities[cluster_id] = {}
            continue
        neighbors = []
        for other_id in ids:
            if other_id == cluster_id:
                continue
            other_centroid = centroids.get(other_id)
            if other_centroid is None:
                continue
            distance = math.hypot(centroid[0] - other_centroid[0], centroid[1] - other_centroid[1])
            neighbors.append((distance, other_id))
        neighbors.sort(key=lambda item: (item[0], item[1]))
        scale = max((distance for distance, _ in neighbors[:max_neighbors]), default=1.0)
        affinities[cluster_id] = {
            other_id: 1.0 - (rank / max(1, max_neighbors))
            + max(0.0, 1.0 - distance / max(scale, 1e-9)) * 0.45
            for rank, (distance, other_id) in enumerate(neighbors[:max_neighbors])
        }
    return affinities


def _semantic_graph_affinities(
    *,
    ids: list[int],
    layer: LayerName,
    centroids: dict[int, tuple[float, float]],
    sizes: dict[int, int],
    cluster_lookup: dict[LayerName, dict[int, int]],
    neighbor_links: list[NeighborLink],
    max_neighbors: int,
) -> dict[int, dict[int, float]]:
    allowed = set(ids)
    affinities: dict[int, defaultdict[int, float]] = {
        cluster_id: defaultdict(float)
        for cluster_id in ids
    }

    centroid_affinities = _semantic_affinities(ids, centroids, max_neighbors=max_neighbors)
    for cluster_id, neighbors in centroid_affinities.items():
        for other_id, weight in neighbors.items():
            if other_id in allowed:
                affinities[cluster_id][other_id] += weight * 0.72

    cluster_by_movie = cluster_lookup[layer]
    pair_weights: defaultdict[tuple[int, int], float] = defaultdict(float)
    for link in neighbor_links:
        source_id = cluster_by_movie.get(link.source_tmdb_id)
        target_id = cluster_by_movie.get(link.target_tmdb_id)
        if source_id is None or target_id is None or source_id == target_id:
            continue
        if source_id not in allowed or target_id not in allowed:
            continue
        left, right = sorted((source_id, target_id))
        pair_weights[(left, right)] += max(0.0, link.similarity - 0.42)

    for (left, right), raw_weight in pair_weights.items():
        size_scale = math.sqrt(max(1, sizes.get(left, 1)) * max(1, sizes.get(right, 1)))
        link_weight = min(3.2, raw_weight / max(1.0, size_scale) * 9.0)
        affinities[left][right] += link_weight
        affinities[right][left] += link_weight

    trimmed: dict[int, dict[int, float]] = {}
    for cluster_id, neighbors in affinities.items():
        trimmed[cluster_id] = dict(
            sorted(
                neighbors.items(),
                key=lambda item: (-item[1], item[0]),
            )[:max_neighbors]
        )
    return trimmed


def _layout_affinities(
    *,
    spec: VariantSpec,
    ids: list[int],
    layer: LayerName,
    centroids: dict[int, tuple[float, float]],
    sizes: dict[int, int],
    cluster_lookup: dict[LayerName, dict[int, int]],
    neighbor_links: list[NeighborLink],
    max_neighbors: int,
) -> dict[int, dict[int, float]]:
    if spec.layout_method == "packed_baseline":
        return _semantic_affinities(ids, centroids, max_neighbors=max(3, max_neighbors - 2))
    return _semantic_graph_affinities(
        ids=ids,
        layer=layer,
        centroids=centroids,
        sizes=sizes,
        cluster_lookup=cluster_lookup,
        neighbor_links=neighbor_links,
        max_neighbors=max_neighbors,
    )


def _layout_circles(
    *,
    ids: list[int],
    radii: dict[int, float],
    parent: Circle,
    targets: dict[int, tuple[float, float]],
    neighbor_affinities: dict[int, dict[int, float]],
    semantic_weight: float,
    padding: float,
    spec: VariantSpec,
) -> dict[int, Circle]:
    if spec.layout_method == "packed_baseline":
        return _pack_circles(
            ids=ids,
            radii=radii,
            parent=parent,
            targets=targets,
            neighbor_affinities=neighbor_affinities,
            semantic_weight=semantic_weight,
            padding=padding,
        )
    return _graph_layout_circles(
        ids=ids,
        radii=radii,
        parent=parent,
        targets=targets,
        neighbor_affinities=neighbor_affinities,
        semantic_weight=semantic_weight,
        padding=padding,
        graph_weight=spec.graph_weight,
        anchor_weight=spec.anchor_weight,
        iterations=spec.graph_iterations,
    )


def _graph_layout_circles(
    *,
    ids: list[int],
    radii: dict[int, float],
    parent: Circle,
    targets: dict[int, tuple[float, float]],
    neighbor_affinities: dict[int, dict[int, float]],
    semantic_weight: float,
    padding: float,
    graph_weight: float,
    anchor_weight: float,
    iterations: int,
) -> dict[int, Circle]:
    if not ids:
        return {}

    positions: dict[int, list[float]] = {}
    for index, cluster_id in enumerate(ids):
        radius = radii[cluster_id]
        target_x, target_y = targets.get(cluster_id, (parent.x, parent.y))
        jitter = parent.radius * 0.004 * (1.0 - min(1.0, semantic_weight))
        angle = (cluster_id * 0.61803398875 + index * 0.173) * math.tau
        x, y = _project_inside_parent(
            x=target_x + math.cos(angle) * jitter,
            y=target_y + math.sin(angle) * jitter,
            radius=radius,
            parent=parent,
            padding=padding,
        )
        positions[cluster_id] = [x, y]

    if len(ids) == 1:
        cluster_id = ids[0]
        x, y = positions[cluster_id]
        return {cluster_id: Circle(id=cluster_id, x=x, y=y, radius=radii[cluster_id])}

    max_affinity = max(
        (
            max(neighbors.values(), default=0.0)
            for neighbors in neighbor_affinities.values()
        ),
        default=1.0,
    )
    max_affinity = max(1.0, max_affinity)

    for iteration in range(max(1, iterations)):
        progress = iteration / max(1, iterations - 1)
        cooling = 1.0 - progress * 0.82
        forces = {cluster_id: [0.0, 0.0] for cluster_id in ids}

        for cluster_id in ids:
            x, y = positions[cluster_id]
            target_x, target_y = targets.get(cluster_id, (parent.x, parent.y))
            forces[cluster_id][0] += (target_x - x) * anchor_weight * semantic_weight
            forces[cluster_id][1] += (target_y - y) * anchor_weight * semantic_weight

        for left_index, left_id in enumerate(ids):
            left_x, left_y = positions[left_id]
            left_radius = radii[left_id]
            for right_id in ids[left_index + 1:]:
                right_x, right_y = positions[right_id]
                right_radius = radii[right_id]
                dx = right_x - left_x
                dy = right_y - left_y
                distance = math.hypot(dx, dy)
                if distance < 1e-6:
                    angle = (left_id * 0.37 + right_id * 0.61) * math.tau
                    dx = math.cos(angle) * 1e-3
                    dy = math.sin(angle) * 1e-3
                    distance = 1e-3
                unit_x = dx / distance
                unit_y = dy / distance
                minimum_distance = left_radius + right_radius + padding
                affinity = (
                    neighbor_affinities.get(left_id, {}).get(right_id, 0.0)
                    + neighbor_affinities.get(right_id, {}).get(left_id, 0.0)
                ) / 2.0

                if affinity > 0:
                    normalized = min(1.0, affinity / max_affinity)
                    desired_distance = minimum_distance * (1.05 + (1.0 - normalized) * 1.08)
                    pull = (distance - desired_distance) * graph_weight * normalized * 0.022
                    forces[left_id][0] += unit_x * pull
                    forces[left_id][1] += unit_y * pull
                    forces[right_id][0] -= unit_x * pull
                    forces[right_id][1] -= unit_y * pull
                elif distance < minimum_distance * 1.45:
                    repel = (minimum_distance * 1.45 - distance) * 0.006
                    forces[left_id][0] -= unit_x * repel
                    forces[left_id][1] -= unit_y * repel
                    forces[right_id][0] += unit_x * repel
                    forces[right_id][1] += unit_y * repel

                overlap = minimum_distance - distance
                if overlap > 0:
                    push = overlap * 0.105
                    forces[left_id][0] -= unit_x * push
                    forces[left_id][1] -= unit_y * push
                    forces[right_id][0] += unit_x * push
                    forces[right_id][1] += unit_y * push

        for cluster_id in ids:
            x, y = positions[cluster_id]
            force_x, force_y = forces[cluster_id]
            max_step = parent.radius * (0.09 * cooling + 0.012)
            step_length = math.hypot(force_x, force_y)
            if step_length > max_step and step_length > 0:
                scale = max_step / step_length
                force_x *= scale
                force_y *= scale
            x, y = _project_inside_parent(
                x=x + force_x * cooling,
                y=y + force_y * cooling,
                radius=radii[cluster_id],
                parent=parent,
                padding=padding,
            )
            positions[cluster_id] = [x, y]

    _resolve_circle_overlaps(
        ids=ids,
        radii=radii,
        positions=positions,
        parent=parent,
        padding=padding,
        passes=72,
    )

    return {
        cluster_id: Circle(
            id=cluster_id,
            x=positions[cluster_id][0],
            y=positions[cluster_id][1],
            radius=radii[cluster_id],
        )
        for cluster_id in ids
    }


def _project_inside_parent(
    *,
    x: float,
    y: float,
    radius: float,
    parent: Circle,
    padding: float,
) -> tuple[float, float]:
    max_distance = max(0.0, parent.radius - radius - padding)
    dx = x - parent.x
    dy = y - parent.y
    distance = math.hypot(dx, dy)
    if distance > max_distance and distance > 0:
        scale = max_distance / distance
        return (parent.x + dx * scale, parent.y + dy * scale)
    return (x, y)


def _resolve_circle_overlaps(
    *,
    ids: list[int],
    radii: dict[int, float],
    positions: dict[int, list[float]],
    parent: Circle,
    padding: float,
    passes: int,
) -> None:
    for _ in range(passes):
        moved = False
        for left_index, left_id in enumerate(ids):
            for right_id in ids[left_index + 1:]:
                left_x, left_y = positions[left_id]
                right_x, right_y = positions[right_id]
                dx = right_x - left_x
                dy = right_y - left_y
                distance = math.hypot(dx, dy)
                minimum_distance = radii[left_id] + radii[right_id] + padding
                overlap = minimum_distance - distance
                if overlap <= 1e-5:
                    continue
                if distance < 1e-6:
                    angle = (left_id * 0.37 + right_id * 0.61) * math.tau
                    dx = math.cos(angle)
                    dy = math.sin(angle)
                    distance = 1.0
                unit_x = dx / distance
                unit_y = dy / distance
                shift = overlap * 0.52
                positions[left_id][0] -= unit_x * shift
                positions[left_id][1] -= unit_y * shift
                positions[right_id][0] += unit_x * shift
                positions[right_id][1] += unit_y * shift
                positions[left_id][:] = _project_inside_parent(
                    x=positions[left_id][0],
                    y=positions[left_id][1],
                    radius=radii[left_id],
                    parent=parent,
                    padding=padding,
                )
                positions[right_id][:] = _project_inside_parent(
                    x=positions[right_id][0],
                    y=positions[right_id][1],
                    radius=radii[right_id],
                    parent=parent,
                    padding=padding,
                )
                moved = True
        if not moved:
            return


def _pack_circles(
    *,
    ids: list[int],
    radii: dict[int, float],
    parent: Circle,
    targets: dict[int, tuple[float, float]],
    neighbor_affinities: dict[int, dict[int, float]],
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
            neighbor_affinities=neighbor_affinities.get(cluster_id, {}),
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
    neighbor_affinities: dict[int, float],
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
        neighbor_pull = 0.0
        for sibling in siblings:
            affinity = neighbor_affinities.get(sibling.id, 0.0)
            if affinity <= 0:
                continue
            ideal_distance = (sibling.radius + radius + padding) * 1.08
            sibling_distance = math.hypot(x - sibling.x, y - sibling.y)
            neighbor_pull += affinity * abs(sibling_distance - ideal_distance)
        target_distance = math.hypot(x - target[0], y - target[1])
        center_distance = math.hypot(x - parent.x, y - parent.y)
        score = (
            boundary_penalty * 1500.0
            + overlap_penalty * 1100.0
            + neighbor_pull * (0.34 + semantic_weight * 0.2)
            + target_distance * (0.2 + semantic_weight)
            + center_distance * max(0.0, 0.18 - semantic_weight * 0.08)
        )
        if best is None or score < best[0]:
            best = (score, x, y)

    if best is None:
        return Circle(id=cluster_id, x=parent.x, y=parent.y, radius=radius)
    return Circle(id=cluster_id, x=best[1], y=best[2], radius=radius)


def _weighted_neighbor_distance(
    circles: dict[int, Circle],
    affinities: dict[int, dict[int, float]],
) -> float:
    weighted_distance = 0.0
    total_weight = 0.0
    seen: set[tuple[int, int]] = set()
    for cluster_id, neighbors in affinities.items():
        source = circles.get(cluster_id)
        if source is None:
            continue
        for other_id, weight in neighbors.items():
            target = circles.get(other_id)
            if target is None or weight <= 0:
                continue
            key = tuple(sorted((cluster_id, other_id)))
            if key in seen:
                continue
            seen.add(key)
            edge_gap = max(
                0.0,
                math.hypot(source.x - target.x, source.y - target.y)
                - source.radius
                - target.radius,
            )
            weighted_distance += edge_gap * weight
            total_weight += weight
    if total_weight <= 0:
        return 0.0
    return round(weighted_distance / total_weight, 6)


def _place_movies_in_micro(
    *,
    points: list[SourcePoint],
    circle: Circle,
    movie_lookup: dict[int, SourcePoint],
    neighbor_links: list[NeighborLink],
    local_weight: float,
    spread_ratio: float,
) -> list[dict[str, float | int]]:
    if not points:
        return []
    centroid_x = sum(point.x for point in points) / len(points)
    centroid_y = sum(point.y for point in points) / len(points)
    max_distance = max(
        (math.hypot(point.x - centroid_x, point.y - centroid_y) for point in points),
        default=0.0,
    )
    local_scale = (circle.radius * spread_ratio / max_distance) if max_distance > 0 else 0.0
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    point_ids = {point.tmdb_id for point in points}
    local_neighbor_score: defaultdict[int, float] = defaultdict(float)
    for link in neighbor_links:
        if (
            link.source_tmdb_id in point_ids
            and link.target_tmdb_id in point_ids
            and link.target_tmdb_id in movie_lookup
        ):
            local_neighbor_score[link.source_tmdb_id] += max(0.0, link.similarity)
            local_neighbor_score[link.target_tmdb_id] += max(0.0, link.similarity) * 0.35
    ordered = sorted(
        points,
        key=lambda point: (
            math.atan2(point.y - centroid_y, point.x - centroid_x),
            -local_neighbor_score[point.tmdb_id],
            point.tmdb_id,
        ),
    )
    records = []
    for index, point in enumerate(ordered):
        local_x = (point.x - centroid_x) * local_scale
        local_y = (point.y - centroid_y) * local_scale
        spiral_radius = circle.radius * spread_ratio * math.sqrt((index + 0.5) / len(ordered))
        spiral_angle = index * golden_angle + (point.tmdb_id % 29) * 0.07
        spiral_x = math.cos(spiral_angle) * spiral_radius
        spiral_y = math.sin(spiral_angle) * spiral_radius
        offset_x = local_x * local_weight + spiral_x * (1.0 - local_weight)
        offset_y = local_y * local_weight + spiral_y * (1.0 - local_weight)
        distance = math.hypot(offset_x, offset_y)
        max_offset = circle.radius * min(0.93, spread_ratio + 0.08)
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


def _build_gmap_cells(
    point_records: list[dict[str, float | int]],
    movie_lookup: dict[int, SourcePoint],
) -> list[dict[str, Any]]:
    """Create movie-level Voronoi cells for the experimental GMap renderer."""
    if len(point_records) < 3:
        return _fallback_square_gmap_cells(point_records, movie_lookup)

    try:
        import numpy as np
        from scipy.spatial import ConvexHull, QhullError, Voronoi
    except ImportError as exc:  # pragma: no cover - dependency is present under uv.
        raise TerritoryLayoutError("GMap cells require scipy and numpy in the project env.") from exc

    sorted_records = sorted(point_records, key=lambda item: int(item["tmdb_id"]))
    coordinates = np.array(
        [[float(record["x"]), float(record["y"])] for record in sorted_records],
        dtype=float,
    )
    coordinates = _jitter_duplicate_coordinates(coordinates)
    try:
        voronoi = Voronoi(coordinates)
        regions, vertices = _finite_voronoi_regions(voronoi, radius=float(np.ptp(coordinates, axis=0).max() * 2.2))
        clip_polygon = _expanded_convex_hull(coordinates, ConvexHull, QhullError)
    except QhullError:
        return _fallback_square_gmap_cells(sorted_records, movie_lookup)

    cells = []
    for record, region_indices in zip(sorted_records, regions, strict=False):
        source = movie_lookup.get(int(record["tmdb_id"]))
        if source is None:
            continue
        polygon = [(float(vertices[index][0]), float(vertices[index][1])) for index in region_indices]
        polygon = _clip_polygon_to_convex_polygon(polygon, clip_polygon)
        polygon = _simplify_polygon(polygon)
        if len(polygon) < 3:
            continue
        cells.append({
            "tmdb_id": source.tmdb_id,
            "macro_id": source.macro_id,
            "neighborhood_id": source.neighborhood_id,
            "micro_id": source.micro_id,
            "polygon": [[round(x, 6), round(y, 6)] for x, y in polygon],
        })
    return cells


def _jitter_duplicate_coordinates(coordinates: Any) -> Any:
    seen: defaultdict[tuple[int, int], int] = defaultdict(int)
    adjusted = coordinates.copy()
    for index, (x, y) in enumerate(adjusted):
        key = (round(float(x) * 1_000_000), round(float(y) * 1_000_000))
        duplicate_index = seen[key]
        seen[key] += 1
        if duplicate_index == 0:
            continue
        angle = (index * 0.61803398875 + duplicate_index * 0.173) * math.tau
        radius = 1e-5 * duplicate_index
        adjusted[index][0] = float(x) + math.cos(angle) * radius
        adjusted[index][1] = float(y) + math.sin(angle) * radius
    return adjusted


def _finite_voronoi_regions(voronoi: Any, *, radius: float) -> tuple[list[list[int]], Any]:
    """Reconstruct infinite 2D Voronoi regions into finite polygons."""
    import numpy as np

    if voronoi.points.shape[1] != 2:
        raise TerritoryLayoutError("GMap Voronoi cells require 2D coordinates.")

    new_regions: list[list[int]] = []
    new_vertices = voronoi.vertices.tolist()
    center = voronoi.points.mean(axis=0)
    all_ridges: defaultdict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for (point_a, point_b), (vertex_a, vertex_b) in zip(
        voronoi.ridge_points,
        voronoi.ridge_vertices,
        strict=False,
    ):
        all_ridges[int(point_a)].append((int(point_b), int(vertex_a), int(vertex_b)))
        all_ridges[int(point_b)].append((int(point_a), int(vertex_a), int(vertex_b)))

    for point_index, region_index in enumerate(voronoi.point_region):
        vertices = voronoi.regions[region_index]
        if all(vertex >= 0 for vertex in vertices):
            new_regions.append([int(vertex) for vertex in vertices])
            continue

        new_region = [int(vertex) for vertex in vertices if vertex >= 0]
        for point_b, vertex_a, vertex_b in all_ridges[point_index]:
            if vertex_b < 0:
                vertex_a, vertex_b = vertex_b, vertex_a
            if vertex_a >= 0:
                continue

            tangent = voronoi.points[point_b] - voronoi.points[point_index]
            tangent /= np.linalg.norm(tangent)
            normal = np.array([-tangent[1], tangent[0]])
            midpoint = voronoi.points[[point_index, point_b]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, normal)) * normal
            far_point = voronoi.vertices[vertex_b] + direction * radius
            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        polygon_vertices = np.asarray([new_vertices[vertex] for vertex in new_region])
        polygon_center = polygon_vertices.mean(axis=0)
        angles = np.arctan2(
            polygon_vertices[:, 1] - polygon_center[1],
            polygon_vertices[:, 0] - polygon_center[0],
        )
        new_regions.append([
            vertex
            for _, vertex in sorted(zip(angles, new_region, strict=False), key=lambda item: item[0])
        ])

    return new_regions, np.asarray(new_vertices)


def _expanded_convex_hull(coordinates: Any, convex_hull: Any, qhull_error: type[Exception]) -> list[tuple[float, float]]:
    try:
        hull = convex_hull(coordinates)
    except qhull_error:
        min_x = float(coordinates[:, 0].min())
        max_x = float(coordinates[:, 0].max())
        min_y = float(coordinates[:, 1].min())
        max_y = float(coordinates[:, 1].max())
        span = max(max_x - min_x, max_y - min_y, 1.0)
        padding = span * 0.08
        return [
            (min_x - padding, min_y - padding),
            (max_x + padding, min_y - padding),
            (max_x + padding, max_y + padding),
            (min_x - padding, max_y + padding),
        ]

    center_x = float(coordinates[:, 0].mean())
    center_y = float(coordinates[:, 1].mean())
    span = float(max(coordinates[:, 0].max() - coordinates[:, 0].min(), coordinates[:, 1].max() - coordinates[:, 1].min(), 1.0))
    padding = span * 0.065
    expanded = []
    for point in coordinates[hull.vertices]:
        dx = float(point[0]) - center_x
        dy = float(point[1]) - center_y
        distance = math.hypot(dx, dy)
        if distance <= 1e-9:
            expanded.append((float(point[0]), float(point[1])))
        else:
            expanded.append((
                center_x + dx / distance * (distance + padding),
                center_y + dy / distance * (distance + padding),
            ))
    if _polygon_area(expanded) < 0:
        expanded.reverse()
    return _chaikin_polygon(expanded, iterations=1)


def _clip_polygon_to_convex_polygon(
    polygon: list[tuple[float, float]],
    clip_polygon: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    if len(polygon) < 3 or len(clip_polygon) < 3:
        return []
    if _polygon_area(clip_polygon) < 0:
        clip_polygon = list(reversed(clip_polygon))

    output = polygon
    for edge_index, edge_start in enumerate(clip_polygon):
        edge_end = clip_polygon[(edge_index + 1) % len(clip_polygon)]
        input_points = output
        output = []
        if not input_points:
            break
        previous = input_points[-1]
        previous_inside = _is_left_of_edge(previous, edge_start, edge_end)
        for current in input_points:
            current_inside = _is_left_of_edge(current, edge_start, edge_end)
            if current_inside != previous_inside:
                output.append(_line_intersection(previous, current, edge_start, edge_end))
            if current_inside:
                output.append(current)
            previous = current
            previous_inside = current_inside
    return output


def _is_left_of_edge(
    point: tuple[float, float],
    edge_start: tuple[float, float],
    edge_end: tuple[float, float],
) -> bool:
    return (
        (edge_end[0] - edge_start[0]) * (point[1] - edge_start[1])
        - (edge_end[1] - edge_start[1]) * (point[0] - edge_start[0])
    ) >= -1e-9


def _line_intersection(
    line_start: tuple[float, float],
    line_end: tuple[float, float],
    clip_start: tuple[float, float],
    clip_end: tuple[float, float],
) -> tuple[float, float]:
    x1, y1 = line_start
    x2, y2 = line_end
    x3, y3 = clip_start
    x4, y4 = clip_end
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denominator) < 1e-12:
        return line_end
    px = (
        (x1 * y2 - y1 * x2) * (x3 - x4)
        - (x1 - x2) * (x3 * y4 - y3 * x4)
    ) / denominator
    py = (
        (x1 * y2 - y1 * x2) * (y3 - y4)
        - (y1 - y2) * (x3 * y4 - y3 * x4)
    ) / denominator
    return (px, py)


def _polygon_area(polygon: list[tuple[float, float]]) -> float:
    return sum(
        polygon[index][0] * polygon[(index + 1) % len(polygon)][1]
        - polygon[(index + 1) % len(polygon)][0] * polygon[index][1]
        for index in range(len(polygon))
    ) / 2


def _chaikin_polygon(
    polygon: list[tuple[float, float]],
    *,
    iterations: int,
) -> list[tuple[float, float]]:
    points = polygon
    for _ in range(iterations):
        next_points = []
        for index, current in enumerate(points):
            next_point = points[(index + 1) % len(points)]
            next_points.append((
                current[0] * 0.75 + next_point[0] * 0.25,
                current[1] * 0.75 + next_point[1] * 0.25,
            ))
            next_points.append((
                current[0] * 0.25 + next_point[0] * 0.75,
                current[1] * 0.25 + next_point[1] * 0.75,
            ))
        points = next_points
    return points


def _simplify_polygon(polygon: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if len(polygon) < 4:
        return polygon
    simplified: list[tuple[float, float]] = []
    for point in polygon:
        if simplified and math.hypot(point[0] - simplified[-1][0], point[1] - simplified[-1][1]) < 1e-5:
            continue
        simplified.append(point)
    if len(simplified) > 1 and math.hypot(
        simplified[0][0] - simplified[-1][0],
        simplified[0][1] - simplified[-1][1],
    ) < 1e-5:
        simplified.pop()
    return simplified


def _fallback_square_gmap_cells(
    point_records: list[dict[str, float | int]],
    movie_lookup: dict[int, SourcePoint],
) -> list[dict[str, Any]]:
    if not point_records:
        return []
    span = max(
        max(float(point["x"]) for point in point_records) - min(float(point["x"]) for point in point_records),
        max(float(point["y"]) for point in point_records) - min(float(point["y"]) for point in point_records),
        1.0,
    )
    radius = span * 0.018
    cells = []
    for record in sorted(point_records, key=lambda item: int(item["tmdb_id"])):
        source = movie_lookup.get(int(record["tmdb_id"]))
        if source is None:
            continue
        x = float(record["x"])
        y = float(record["y"])
        cells.append({
            "tmdb_id": source.tmdb_id,
            "macro_id": source.macro_id,
            "neighborhood_id": source.neighborhood_id,
            "micro_id": source.micro_id,
            "polygon": [
                [round(x - radius, 6), round(y - radius, 6)],
                [round(x + radius, 6), round(y - radius, 6)],
                [round(x + radius, 6), round(y + radius, 6)],
                [round(x - radius, 6), round(y + radius, 6)],
            ],
        })
    return cells


def _gmap_outer_vertex_count(cells: list[dict[str, Any]]) -> int:
    edge_counts: defaultdict[tuple[tuple[int, int], tuple[int, int]], int] = defaultdict(int)
    for cell in cells:
        polygon = cell.get("polygon") or []
        for index, current in enumerate(polygon):
            next_point = polygon[(index + 1) % len(polygon)]
            current_key = (round(float(current[0]) * 10000), round(float(current[1]) * 10000))
            next_key = (round(float(next_point[0]) * 10000), round(float(next_point[1]) * 10000))
            edge_counts[tuple(sorted((current_key, next_key)))] += 1
    return sum(1 for count in edge_counts.values() if count == 1)


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
    path.write_text(json.dumps(manifest, separators=(",", ":"), sort_keys=True), encoding="utf-8")
