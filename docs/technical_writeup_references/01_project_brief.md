# The Film Atlas — Project Brief

The Film Atlas is a non-commercial portfolio project that maps English-language films into emergent “vibe genres.”

The final public-facing version will live on David’s personal website, which is an Astro + MDX + Tailwind static site deployed to GitHub Pages. This development repo is separate from that website repo. Its purpose is to build the offline data pipeline, experiments, clustering, labels, and static JSON outputs that can later be copied into the Astro site.

Core idea:
Official movie genres are broad and blunt: drama, comedy, thriller. The Film Atlas asks what genres emerge if films are clustered by semantic signals like plot summaries, keywords, and audience review language.

V1 product experience:
A beautiful, consumer-facing interactive map where each dot is a film. Nearby films should feel semantically similar. Users can search a movie, hover/click a film, see its discovered microgenre, read why it belongs there, and view nearest neighbors.

Important product decision:
The semantic “vibe map” should avoid clustering primarily by decade, country, cast, director, or production company. Those fields may be displayed or used in later “industry topology” modes, but they should not be included in the v1 semantic embedding/profile text.

Data sources:
- Use TMDb official API for Milestone 1.
- Do not scrape Letterboxd, IMDb, or any website.
- IMDb non-commercial datasets may be used in later milestones, but not required for Milestone 1.
- OpenAI embeddings and cluster labeling will happen in later milestones, not Milestone 1.

Milestone 1 goal:
Build a robust offline data-pipeline scaffold and produce a small TMDb-based data-quality proof. No paid API calls. No final website. No OpenAI.
