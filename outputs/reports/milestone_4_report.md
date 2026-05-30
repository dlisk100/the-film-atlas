# The Film Atlas - Milestone 4 Report

Milestone 4 creates a scaled English-language film dataset, light-review semantic profiles, full-vector embeddings, hierarchical k-means atlas layers, draft labels, 2D map coordinates, and sanitized static JSON for a future Astro frontend.

## Summary

- Movie count: 2000
- Dataset target: 2000
- Since year: 1980
- Minimum vote count: 100
- Projection method: umap
- Embedding model: text-embedding-3-large
- Embedding estimated full cost: $0.0481
- Label estimated live cost: $1.2038
- Labels generated: 287
- Public export: outputs/public_export

## Coverage

- Overviews: 100.0%
- Keywords: 100.0%
- Reviews: 93.0%

## Year Distribution

1980s: 94, 1990s: 231, 2000s: 535, 2010s: 875, 2020s: 265

## Genre Distribution

Action: 714, Drama: 689, Comedy: 618, Adventure: 582, Thriller: 532, Science Fiction: 407, Fantasy: 323, Crime: 318, Family: 279, Romance: 258, Horror: 258, Animation: 187, Mystery: 175, History: 89, War: 60

## Embedding Run

- Profiles: 2000
- Cached/new embeddings: 493/1507
- Estimated full cost: $0.0481
- Estimated live cost: $0.0354

## Hierarchy Layers

### macro

- k target: 12
- clusters: 12
- labeled: True
- label cost: $0.0511
- label cache reused/new: 0/12
- coherence average/range: 0.493 (0.460-0.524)
- tiny clusters (<5 movies): 0
- size distribution: 239, 228, 212, 183, 183, 180, 139, 136, 131, 126, 124, 119

Sample labels:

- Cluster 0: Folk Horror Survival Dread (0.73)
- Cluster 1: Adult Romance, Soft Edges (0.68)
- Cluster 2: Spy-Thriller: Revenge & Extraction (0.71)
- Cluster 3: Parody Crime & Buddy Cop Chaos (0.64)
- Cluster 4: Family Animation: Heart + Mischief (0.66)
- Cluster 5: Doomed Horizons: Sci‑Fi Action & Alien Menace (0.74)
- Cluster 6: Wizards, Witches & Wardrobes (0.69)
- Cluster 7: Battlefield Survival Dramas (0.71)

### neighborhood

- k target: 75
- clusters: 75
- labeled: True
- label cost: $0.3158
- label cache reused/new: 0/75
- coherence average/range: 0.560 (0.465-0.795)
- tiny clusters (<5 movies): 0
- size distribution: 73, 60, 59, 58, 56, 49, 49, 48, 46, 45, 43, 39, 39, 39, 38, 38, 38, 37, 37, 36, 36, 35, 35, 35, 34, 34, 33, 33, 33, 32, 31, 31, 27, 26, 26, 25, 24, 23, 23, 22, 22, 21, 19, 19, 19, 19, 19, 18, 18, 17, 17, 17, 17, 16, 16, 15, 14, 13, 13, 12, 12, 12, 11, 11, 10, 10, 9, 9, 8, 8, 8, 8, 7, 6, 5

Sample labels:

- Cluster 0: Teen-to-adult dramedy romance (0.58)
- Cluster 1: True-story historical survival epics (0.63)
- Cluster 2: Rom-com rivalry and messy hookups (0.66)
- Cluster 3: Web-slinging superhero universe stories (0.91)
- Cluster 4: Street-racing crime thrillers (0.86)
- Cluster 5: Moral-rot gangster dramas (0.72)
- Cluster 6: Domestic haunting horror (0.74)
- Cluster 7: Heart-on-silicon sci-fi (0.77)

### micro

- k target: 200
- clusters: 200
- labeled: True
- label cost: $0.8369
- label cache reused/new: 0/200
- coherence average/range: 0.596 (0.483-0.872)
- tiny clusters (<5 movies): 24
- size distribution: 31, 30, 29, 27, 26, 26, 25, 25, 24, 22, 21, 20, 19, 19, 18, 18, 17, 17, 17, 17, 17, 17, 16, 16, 16, 16, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 14, 14, 14, 14, 14, 14, 14, 13, 13, 13, 13, 13, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 1

Sample labels:

- Cluster 0: Holiday Skyscraper Cop Chaos (0.68)
- Cluster 1: Clown-Seeded Small-Town Haunt (0.63)
- Cluster 2: Kung-Fu Fantasy Mayhem (0.56)
- Cluster 3: Storm-Island Mind-Break Thrillers (0.56)
- Cluster 4: Creative Genius Biopic Drama (0.51)
- Cluster 5: Survival + Meaning-Seeking (0.52)
- Cluster 6: Dystopia Space-Action (0.57)
- Cluster 7: Dark Comedy Phone-Run Thrillers (0.54)

## Quality-Check Movie Neighbors

- No Country for Old Men: True Grit (0.644), Hell or High Water (0.642), Unforgiven (0.623), Sicario (0.618), O Brother, Where Art Thou? (0.615)
- The Social Network: Steve Jobs (0.600), Jobs (0.592), The Circle (0.592), The Wolf of Wall Street (0.580), Ex Machina (0.579)
- Mean Girls: Clueless (0.695), 10 Things I Hate About You (0.644), Pitch Perfect (0.639), Easy A (0.618), The Edge of Seventeen (0.617)
- Her: Ex Machina (0.677), A.I. Artificial Intelligence (0.636), Eternal Sunshine of the Spotless Mind (0.632), Transcendence (0.629), Ready Player One (0.618)
- Get Out: Us (0.694), The Gift (0.667), The Invitation (0.653), The Visit (0.645), You're Next (0.643)
- The Matrix: The Matrix Revolutions (0.793), The Matrix Resurrections (0.779), The Matrix Reloaded (0.765), The Terminator (0.688), Tron (0.662)
- Before Sunrise: Before Sunset (0.753), Before Midnight (0.690), Midnight in Paris (0.607), Lost in Translation (0.605), (500) Days of Summer (0.589)
- The Big Short: The Wolf of Wall Street (0.692), Hustlers (0.653), Money Monster (0.635), Tower Heist (0.614), The Disaster Artist (0.602)
- Mad Max: Fury Road: Mad Max Beyond Thunderdome (0.813), Mad Max 2 (0.795), Furiosa: A Mad Max Saga (0.793), Death Race (0.639), Dune (0.637)
- Lost in Translation: Sleepless in Seattle (0.624), Up in the Air (0.622), The Terminal (0.621), Eyes Wide Shut (0.618), Eternal Sunshine of the Spotless Mind (0.614)
- The Devil Wears Prada: Confessions of a Shopaholic (0.618), Bridget Jones's Diary (0.610), Mean Girls (0.603), Phantom Thread (0.599), The Proposal (0.584)
- Whiplash: Black Swan (0.648), La La Land (0.634), Punch-Drunk Love (0.610), Requiem for a Dream (0.610), The Master (0.607)
- Nightcrawler: Natural Born Killers (0.636), Drive (0.629), The Neon Demon (0.625), Nightmare Alley (0.624), You Were Never Really Here (0.620)
- Paddington 2: Paddington (0.875), Sing 2 (0.649), Rio 2 (0.645), Incredibles 2 (0.639), Christopher Robin (0.637)
- Pulp Fiction: Reservoir Dogs (0.733), True Romance (0.673), Jackie Brown (0.670), Once Upon a Time... in Hollywood (0.647), Kiss Kiss Bang Bang (0.642)
- The Shawshank Redemption: The Green Mile (0.609), The Fugitive (0.576), The Town (0.575), The Next Three Days (0.565), Mystic River (0.560)
- Interstellar: Ad Astra (0.738), The Martian (0.708), Gravity (0.697), Passengers (0.697), Star Trek Into Darkness (0.679)
- The Dark Knight: The Dark Knight Rises (0.798), Batman (0.753), Batman Begins (0.747), The Batman (0.736), Batman Returns (0.680)

## Frontend Export File Sizes

- labels.json: 106,882 bytes
- macro_clusters.json: 24,004 bytes
- manifest.json: 705 bytes
- micro_clusters.json: 353,276 bytes
- movies.json: 2,261,445 bytes
- neighborhood_clusters.json: 141,965 bytes
- neighbors.json: 2,510,839 bytes
- points.json: 306,977 bytes

## Recommendation

Proceed to Astro frontend planning: the scaled dataset, hierarchy, labels, neighbors, projection, and sanitized export files are present.
