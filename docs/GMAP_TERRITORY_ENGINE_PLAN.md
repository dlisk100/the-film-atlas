# GMap-Inspired Territory Engine Plan

## Goal

Build an additive Film Atlas map layer where territory geometry is driven by the same semantic evidence that drives movie neighbors and clusters. The layer should be comparable against the existing territory renderers, not a replacement until it earns that role.

## Research Baseline

The target is closest to GMap-style graph visualization: start from a meaningful graph layout, build Voronoi-like cells around entities, and visually merge cells with the same cluster membership into geographic regions. GMap is relevant because the Film Atlas already has both graph-like neighbor evidence and categorical cluster membership.

Supporting references:

- [GMap: Visualizing Graphs and Clusters as Maps](https://yifanhu.net/PUB/pacvis2010.pdf)
- [MapSets: Visualizing Embedded and Clustered Graphs](https://www2.cs.arizona.edu/~alon/papers/mapsets.pdf)
- [Bubble Sets](https://vialabdev.science.ontariotechu.ca/research/bubble-sets)
- [Fast Dynamic Voronoi Treemaps](https://www.microsoft.com/en-us/research/publication/fast-dynamic-voronoi-treemaps-2/)

The key lesson is that map regions should not be decorative hulls around centroids. The representation needs to preserve the semantic embedding, avoid misleading overlap, and make cluster borders a consequence of point membership.

## Proposed Engine

1. Keep the current semantic graph layout as the source of truth for where films belong.
2. Generate one finite Voronoi cell per movie from those graph-driven movie positions.
3. Clip all movie cells to an expanded semantic outer hull so the map has a bounded coastline rather than an arbitrary rectangle.
4. Render cells by cluster membership:
   - Macro zoom: fill movie cells by macro membership and draw only macro boundary edges.
   - Neighborhood zoom: retain macro context, then draw neighborhood boundary edges.
   - Micro zoom: retain parent context, then draw micro boundary edges.
5. Hide same-cluster internal cell edges so adjacent cells visually merge into regions.
6. Keep the older power-cell, organic, coastal, dense-coast, and biological renderers as comparison controls.

## Additive Data Contract

Add a new territory variant:

```json
{
  "id": "semantic_gmap_cells",
  "label": "GMap Cells - Semantic",
  "points": [],
  "regions": [],
  "gmap_cells": [
    {
      "tmdb_id": 603,
      "macro_id": 12,
      "neighborhood_id": 104,
      "micro_id": 712,
      "polygon": [[x, y], [x, y], [x, y]]
    }
  ]
}
```

The existing `points` and `regions` arrays remain unchanged. `gmap_cells` is optional, so all older variants remain valid.

## Frontend Plan

Add a `GMap cells` render mode. When the selected layout has `gmap_cells`, the renderer:

1. Draws all movie Voronoi cells as low-alpha macro-colored fills.
2. Builds boundary-edge maps from shared cell edges.
3. Strokes only edges where the active layer's cluster id changes.
4. Uses original graph positions for dots in GMap mode so each dot stays inside its own cell.
5. Falls back to the clean territory renderer if a non-GMap layout is selected.

## Validation

Required checks:

1. `uv run pytest`
2. `uv run ruff check .`
3. Regenerate `outputs/public_export/territory_layouts.json`.
4. `pnpm build` in `frontend/`.
5. Browser QA at the local `/film-atlas/` page:
   - `semantic_gmap_cells` appears in the layout dropdown.
   - `GMap cells` appears in the render dropdown.
   - Macro, neighborhood, and micro zooms show different boundary levels.
   - Selected films still hover/select correctly.
   - No console errors.

## Known Tradeoffs

This first pass uses an expanded convex semantic hull as the outer coastline. That is more honest than a circle-based frame and keeps the Voronoi partition well-behaved, but it is not yet a full concave coastline. If the GMap cell layer feels promising, the next improvement should experiment with alpha-shape or k-nearest-neighbor concave hull clipping while preserving non-overlap.
