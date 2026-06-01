# Film Atlas Technical Writeup Reference Pack

This folder is a curated snapshot of the reports most useful for writing the
technical post for The Film Atlas. The files are numbered in the recommended
reading order.

The canonical originals still live in `docs/` and `outputs/reports/`. These
copies are gathered here so the writeup source material is easy to review in
one place.

For a single compact file to review with ChatGPT, start with
`00_compact_technical_writeup_reference.md`.

## Start Here

Use these three as the main spine of the post:

1. `02_data_implementation_report.md`
   - Best source for the full data/classification architecture.
   - Covers data sources, semantic profile construction, embeddings,
     clustering, labeling, audits, pivots, and the final 10k-film export.

2. `03_ui_ux_implementation_report.md`
   - Best source for the frontend and interaction story.
   - Covers the move from projection scatterplots to a semantic territory map,
     label placement, colors, render modes, browser QA, and final UI tradeoffs.

3. `06_milestone_5_final_validation.md`
   - Best source for the final validation numbers.
   - Captures 10,000 films, 16 / 180 / 750 hierarchy, 946 labels, privacy
     checks, audit pass rates, test status, and browser QA status.

## Product Context

- `01_project_brief.md`
  - Short project framing and original product goal.

- `04_milestone_5_exploratory_classification_upgrade.md`
  - The best source for the expanded experimental mandate.
  - Useful for explaining why the project became exploratory instead of simply
    scaling the first working pipeline.

## UI / Map Engine

- `05_gmap_territory_engine_plan.md`
  - Best source for the map-engine pivot.
  - Explains the GMap-inspired direction: semantic graph layout, Voronoi-like
    movie cells, non-overlapping territories, and cluster-derived boundaries.

- `03_ui_ux_implementation_report.md`
  - Main UI/UX narrative.
  - Use it for the final description of the shipped local frontend.

## Experiment Evidence

- `12_review_ablation_report.md`
  - Shows why review text was useful but needed to be kept light.

- `13_clustering_method_comparison.md`
  - Shows why simpler, labelable clustering beat approaches that looked more
    sophisticated but produced weaker product results.

- `14_cluster_sweep_report.md`
  - Early evidence for choosing cluster granularity.

- `11_milestone_5_bucket_scan.md`
  - Later hierarchy-size scan for the final macro / neighborhood / micro
    design.

## Audit Evidence

- `07_milestone_5_report.md`
  - Concise summary of the winning classification approach.

- `08_milestone_5_deep_audit.md`
  - Manual-style deep audit evidence.

- `09_milestone_5_large_audit_2k_final.md`
  - Large 2,000-film audit used to check for broad classification failures.

- `10_milestone_5_prior_fail_recheck.md`
  - Recheck showing targeted repairs improved earlier failures.

- `06_milestone_5_final_validation.md`
  - Final validation rollup and the best place to cite final numbers.

## Labeling Evidence

- `15_cluster_label_review.md`
  - Useful for showing the human-review style label gate.
  - Earlier 500-film stage, but valuable for explaining the labeling philosophy.

- `16_cluster_label_candidates.md`
  - Rawer label-candidate evidence from the earlier label review flow.
  - Useful only if the post needs to show how labels were inspected before
    approval.

## Suggested Screenshots For The Post

These images are not duplicated here, but they are the best current visual
references in `outputs/reports/`:

- `outputs/reports/frontend_portfolio_final_default.png`
  - General final UI hero screenshot.

- `outputs/reports/frontend_portfolio_final_office_space.png`
  - Selected-film panel example.

- `outputs/reports/frontend_cell_borders_dot_padding_macro.png`
  - Macro-level territory view after dot padding.

- `outputs/reports/frontend_cell_borders_dot_padding_micro.png`
  - Micro-level territory view after dot padding.

- `outputs/reports/frontend_color_mode_neighborhood_shades.png`
  - Neighborhood shade color mode.

- `outputs/reports/frontend_color_mode_micro_tiered_micro.png`
  - Micro shade color mode after color-mode experimentation.

- `outputs/reports/frontend_micro_labels_cell_placement_pass.png`
  - Later micro-label placement pass.

- `outputs/reports/gmap_cells_macro.png`
  - Early GMap-cell territory proof.

- `outputs/reports/semantic_atlas_balanced.png`
  - Earlier semantic atlas baseline, useful for comparison.

- `outputs/reports/broken_organic_borders_zoomed.png`
  - Useful only as a "what did not work" image if the post includes a design
    iteration section.

## Best Technical Storyline

The strongest post structure is probably:

1. Start with the product goal: map movies by experiential similarity, not just
   genre.
2. Explain why the first data version was not enough: thin profiles, title
   leakage, overly broad labels, and projection weirdness.
3. Show the data-system pivot: richer legitimate sources, synthesized semantic
   profiles, embeddings, hierarchical clustering, guarded labels, and large
   audits.
4. Show the UI-system pivot: raw scatterplot to semantic territory atlas to
   GMap-inspired cell territories.
5. Emphasize the validation loop: user audits, LLM-assisted audits, structural
   scans, browser QA, and final 10k export checks.
6. End with the portfolio angle: a static, inspectable, browser-only semantic
   map with no secrets or backend dependency.
