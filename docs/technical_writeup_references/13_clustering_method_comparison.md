# The Film Atlas - Milestone 2.6 Clustering Method Comparison

This report compares local clustering methods over existing embeddings. It does not call OpenAI, re-embed profiles, generate final AI labels, export public website JSON, scrape websites, or touch frontend code.

## Critical Input-Space Check

- Method comparison input space: full_embedding_vectors
- Current clustering check: Current clustering uses normalized full embedding vectors from embeddings.jsonl. PCA/2D coordinates are produced separately for visualization only.
- Best-practice alignment: nearest neighbors use full embeddings; clustering uses full embeddings; PCA/2D projection is visualization only.

## Summary

- Embedded movies inspected: 500
- Methods requested: kmeans, agglomerative, graph, hdbscan
- Recommended method: kmeans
- Recommendation note: kmeans appears most labelable: 35 clusters, average size 14.286, largest 36, 1 tiny clusters, and coherence average 0.560.
- Milestone 3 readiness: Sufficient for a Milestone 3 labeling pass, with human review of edge clusters.
- Profile ablation recommendation: Recommended after method selection: run a small profile ablation or review-weight experiment to verify that reviews add useful vibe signal without adding noise.

## Method Metrics

| Method | Status | Clusters | Avg | Median | Largest | Smallest | Tiny <5 | Outliers | Coherence Avg | Coherence Range | Silhouette | Score | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |
| kmeans | completed | 35 | 14.286 | 13.000 | 36 | 4 | 1 | 0 | 0.560 | 0.484-0.691 | 0.064 | 0.529 | Fixed-k baseline on normalized full embedding vectors. |
| agglomerative | completed | 35 | 14.286 | 6.000 | 84 | 1 | 15 | 0 | 0.547 | 0.510-0.613 | 0.061 | -0.952 | Average-linkage agglomerative clustering with cosine distance. |
| graph | completed | 9 | 55.556 | 59.000 | 83 | 12 | 0 | 0 | 0.504 | 0.458-0.546 | 0.068 | -3.275 | k-nearest-neighbor graph over cosine similarities with NetworkX community detection. |
| hdbscan | completed | 2 | 85.000 | 85.000 | 163 | 7 | 0 | 330 | 0.544 | 0.480-0.607 | 0.063 | -8.438 | Optional density clustering over normalized full embedding vectors. |

## Direct Answers

- Most labelable neighborhoods: kmeans.
- Best at avoiding overly broad clusters: kmeans.
- Best at avoiding excessive tiny/franchise-only fragmentation: kmeans.
- Best preservation signal for sensible neighborhoods: kmeans.
- Data sufficiency for Milestone 3: Sufficient for a Milestone 3 labeling pass, with human review of edge clusters.
- Later ablation/review-weight experiment: Recommended after method selection: run a small profile ablation or review-weight experiment to verify that reviews add useful vibe signal without adding noise.

## Quality-Check Movie Cluster Assignments

### kmeans

| Movie | Present | Cluster | Cluster Size | Cluster Representatives |
| --- | --- | ---: | ---: | --- |
| No Country for Old Men | yes | 4 | 36 | Reservoir Dogs, Pulp Fiction, Casino, The Usual Suspects, Once Upon a Time in America, GoodFellas |
| The Social Network | no | n/a | n/a |  |
| Mean Girls | no | n/a | n/a |  |
| Her | yes | 12 | 20 | Interstellar, Gravity, Avatar, The Martian, Arrival, Passengers |
| Get Out | yes | 31 | 18 | Nope, Old, The Invisible Man, The Thing, Get Out, The Fly |
| The Matrix | yes | 23 | 25 | The Matrix, The Terminator, Total Recall, RoboCop, Blade Runner, The Matrix Resurrections |
| Before Sunrise | no | n/a | n/a |  |
| The Big Short | no | n/a | n/a |  |
| Mad Max: Fury Road | yes | 5 | 26 | A Quiet Place Part II, Mad Max: Fury Road, Mad Max Beyond Thunderdome, Aliens, Mad Max 2, Alien Resurrection |
| Lost in Translation | no | n/a | n/a |  |
| The Devil Wears Prada | yes | 6 | 24 | Black Swan, Requiem for a Dream, Shutter Island, The Prestige, Memento, Whiplash |
| Whiplash | yes | 6 | 24 | Black Swan, Requiem for a Dream, Shutter Island, The Prestige, Memento, Whiplash |
| Nightcrawler | no | n/a | n/a |  |
| Paddington 2 | no | n/a | n/a |  |
| The Godfather | no | n/a | n/a |  |
| Pulp Fiction | yes | 4 | 36 | Reservoir Dogs, Pulp Fiction, Casino, The Usual Suspects, Once Upon a Time in America, GoodFellas |
| The Shawshank Redemption | yes | 4 | 36 | Reservoir Dogs, Pulp Fiction, Casino, The Usual Suspects, Once Upon a Time in America, GoodFellas |
| Interstellar | yes | 12 | 20 | Interstellar, Gravity, Avatar, The Martian, Arrival, Passengers |
| The Dark Knight | yes | 21 | 13 | Batman, The Batman, The Dark Knight, Batman Begins, Batman Returns, The Dark Knight Rises |

### agglomerative

| Movie | Present | Cluster | Cluster Size | Cluster Representatives |
| --- | --- | ---: | ---: | --- |
| No Country for Old Men | yes | 14 | 32 | Reservoir Dogs, Pulp Fiction, Casino, The Usual Suspects, Once Upon a Time in America, GoodFellas |
| The Social Network | no | n/a | n/a |  |
| Mean Girls | no | n/a | n/a |  |
| Her | yes | 3 | 55 | Blade Runner 2049, Interstellar, The Matrix, Aliens, Rogue One: A Star Wars Story, Avatar |
| Get Out | yes | 0 | 36 | A Nightmare on Elm Street, Poltergeist, Scream, It, Beetlejuice, Nope |
| The Matrix | yes | 3 | 55 | Blade Runner 2049, Interstellar, The Matrix, Aliens, Rogue One: A Star Wars Story, Avatar |
| Before Sunrise | no | n/a | n/a |  |
| The Big Short | no | n/a | n/a |  |
| Mad Max: Fury Road | yes | 3 | 55 | Blade Runner 2049, Interstellar, The Matrix, Aliens, Rogue One: A Star Wars Story, Avatar |
| Lost in Translation | no | n/a | n/a |  |
| The Devil Wears Prada | yes | 17 | 2 | The Devil Wears Prada, Cruella |
| Whiplash | yes | 12 | 21 | Requiem for a Dream, Memento, American Psycho, Donnie Darko, Black Swan, Fight Club |
| Nightcrawler | no | n/a | n/a |  |
| Paddington 2 | no | n/a | n/a |  |
| The Godfather | no | n/a | n/a |  |
| Pulp Fiction | yes | 14 | 32 | Reservoir Dogs, Pulp Fiction, Casino, The Usual Suspects, Once Upon a Time in America, GoodFellas |
| The Shawshank Redemption | yes | 32 | 2 | The Green Mile, The Shawshank Redemption |
| Interstellar | yes | 3 | 55 | Blade Runner 2049, Interstellar, The Matrix, Aliens, Rogue One: A Star Wars Story, Avatar |
| The Dark Knight | yes | 1 | 84 | Spider-Man: No Way Home, The Avengers, Spider-Man, Man of Steel, Thor: The Dark World, Iron Man |

### graph

| Movie | Present | Cluster | Cluster Size | Cluster Representatives |
| --- | --- | ---: | ---: | --- |
| No Country for Old Men | yes | 4 | 59 | Reservoir Dogs, The Usual Suspects, Pulp Fiction, Once Upon a Time in America, Casino, The Departed |
| The Social Network | no | n/a | n/a |  |
| Mean Girls | no | n/a | n/a |  |
| Her | yes | 6 | 49 | Dirty Dancing, Forrest Gump, Top Gun, Eternal Sunshine of the Spotless Mind, La La Land, Edward Scissorhands |
| Get Out | yes | 5 | 50 | A Nightmare on Elm Street, It, Beetlejuice, Poltergeist, Scream, Gremlins |
| The Matrix | yes | 0 | 83 | Interstellar, The Matrix, Mad Max: Fury Road, Aliens, The Terminator, Blade Runner 2049 |
| Before Sunrise | no | n/a | n/a |  |
| The Big Short | no | n/a | n/a |  |
| Mad Max: Fury Road | yes | 0 | 83 | Interstellar, The Matrix, Mad Max: Fury Road, Aliens, The Terminator, Blade Runner 2049 |
| Lost in Translation | no | n/a | n/a |  |
| The Devil Wears Prada | yes | 6 | 49 | Dirty Dancing, Forrest Gump, Top Gun, Eternal Sunshine of the Spotless Mind, La La Land, Edward Scissorhands |
| Whiplash | yes | 4 | 59 | Reservoir Dogs, The Usual Suspects, Pulp Fiction, Once Upon a Time in America, Casino, The Departed |
| Nightcrawler | no | n/a | n/a |  |
| Paddington 2 | no | n/a | n/a |  |
| The Godfather | no | n/a | n/a |  |
| Pulp Fiction | yes | 4 | 59 | Reservoir Dogs, The Usual Suspects, Pulp Fiction, Once Upon a Time in America, Casino, The Departed |
| The Shawshank Redemption | yes | 4 | 59 | Reservoir Dogs, The Usual Suspects, Pulp Fiction, Once Upon a Time in America, Casino, The Departed |
| Interstellar | yes | 0 | 83 | Interstellar, The Matrix, Mad Max: Fury Road, Aliens, The Terminator, Blade Runner 2049 |
| The Dark Knight | yes | 1 | 79 | Spider-Man: No Way Home, The Avengers, Spider-Man, Man of Steel, Thor: The Dark World, Iron Man |

### hdbscan

| Movie | Present | Cluster | Cluster Size | Cluster Representatives |
| --- | --- | ---: | ---: | --- |
| No Country for Old Men | yes | -1 | n/a |  |
| The Social Network | no | n/a | n/a |  |
| Mean Girls | no | n/a | n/a |  |
| Her | yes | -1 | n/a |  |
| Get Out | yes | -1 | n/a |  |
| The Matrix | yes | 0 | 163 | Spider-Man: No Way Home, Guardians of the Galaxy, The Avengers, Iron Man, Spider-Man, Man of Steel |
| Before Sunrise | no | n/a | n/a |  |
| The Big Short | no | n/a | n/a |  |
| Mad Max: Fury Road | yes | 0 | 163 | Spider-Man: No Way Home, Guardians of the Galaxy, The Avengers, Iron Man, Spider-Man, Man of Steel |
| Lost in Translation | no | n/a | n/a |  |
| The Devil Wears Prada | yes | -1 | n/a |  |
| Whiplash | yes | -1 | n/a |  |
| Nightcrawler | no | n/a | n/a |  |
| Paddington 2 | no | n/a | n/a |  |
| The Godfather | no | n/a | n/a |  |
| Pulp Fiction | yes | 0 | 163 | Spider-Man: No Way Home, Guardians of the Galaxy, The Avengers, Iron Man, Spider-Man, Man of Steel |
| The Shawshank Redemption | yes | -1 | n/a |  |
| Interstellar | yes | 0 | 163 | Spider-Man: No Way Home, Guardians of the Galaxy, The Avengers, Iron Man, Spider-Man, Man of Steel |
| The Dark Knight | yes | 0 | 163 | Spider-Man: No Way Home, Guardians of the Galaxy, The Avengers, Iron Man, Spider-Man, Man of Steel |

## Sample Clusters

### kmeans

| Cluster | Size | Coherence | Representatives | Genres | Keywords | Terms |
| ---: | ---: | ---: | --- | --- | --- | --- |
| 4 | 36 | 0.524 | Reservoir Dogs, Pulp Fiction, Casino, The Usual Suspects, Once Upon a Time in America, GoodFellas, The Departed, Lock, Stock and Two Smoking Barrels | Crime (28), Drama (20), Thriller (16), Comedy (10), Action (6), Mystery (4) | based on novel or book (11), neo-noir (9), gangster (8), excited (7), murder (7), los angeles, california (7), corruption (5), robbery (5) | crime, drug, drama, prison, gangster, police, comedy, based, thriller, movie |
| 5 | 26 | 0.513 | A Quiet Place Part II, Mad Max: Fury Road, Mad Max Beyond Thunderdome, Aliens, Mad Max 2, Alien Resurrection, Godzilla vs. Kong, Furiosa: A Mad Max Saga | Science Fiction (19), Action (18), Adventure (11), Thriller (8), Horror (7), Crime (4) | sequel (10), post-apocalyptic future (9), alien (6), creature (6), dystopia (6), sci-fi horror (5), alien life-form (5), based on novel or book (4) | alien, horror, action, world, zombie, family, new, kong, godzilla, space |
| 23 | 25 | 0.562 | The Matrix, The Terminator, Total Recall, RoboCop, Blade Runner, The Matrix Resurrections, Blade Runner 2049, Terminator 2: Judgment Day | Science Fiction (25), Action (19), Thriller (12), Adventure (8), Drama (3), Comedy (2) | dystopia (22), artificial intelligence (a.i.) (10), cyberpunk (10), dystopian (10), based on novel or book (9), man vs machine (8), future (7), suspenseful (6) | robot, matrix, action, world, future, time, runner, science, blade runner, blade |
| 6 | 24 | 0.498 | Black Swan, Requiem for a Dream, Shutter Island, The Prestige, Memento, Whiplash, The Shining, American Psycho | Drama (21), Thriller (12), Mystery (8), Crime (4), Horror (3), Fantasy (3) | based on novel or book (10), psychological thriller (9), psychological horror (6), new york city (6), marriage crisis (4), based on true story (4), suspenseful (4), child abuse (3) | thriller, psychological, drama, horror, man, based, mystery, new, story, child |
| 32 | 22 | 0.488 | Big, Back to the Future, Home Alone, Groundhog Day, Honey, I Shrunk the Kids, Stand by Me, Forrest Gump, E.T. the Extra-Terrestrial | Comedy (16), Fantasy (10), Adventure (9), Family (9), Drama (8), Science Fiction (4) | coming of age (5), hilarious (5), based on novel or book (4), parent child relationship (4), 1950s (3), 1980s (3), friendship (3), duringcreditsstinger (3) | comedy, school, fantasy, time, relationship, day, world, adventure, single, home |
| 12 | 20 | 0.539 | Interstellar, Gravity, Avatar, The Martian, Arrival, Passengers, Armageddon, Independence Day | Science Fiction (17), Adventure (10), Drama (10), Action (7), Thriller (5), Romance (3) | space (8), disaster (6), astronaut (6), spacecraft (5), dystopia (5), dramatic (4), race against time (4), nasa (4) | space, alien, world, mission, science, science fiction, fiction, time, planet, disaster |
| 9 | 18 | 0.527 | Mission: Impossible, Mission: Impossible - Dead Reckoning Part One, The Bourne Identity, Inception, Skyfall, No Time to Die, Face/Off, Tenet | Action (13), Thriller (12), Adventure (9), Science Fiction (4), Crime (4), Drama (3) | spy (8), suspenseful (6), paris, france (5), based on novel or book (5), excited (4), undercover (3), secret identity (3), fictional government agency (3) | bond, secret, action, agent, time, mission, thriller, new, fbi, government |
| 31 | 18 | 0.484 | Nope, Old, The Invisible Man, The Thing, Get Out, The Fly, The Menu, The Substance | Horror (14), Science Fiction (8), Thriller (8), Mystery (5), Drama (2), Comedy (2) | based on novel or book (5), body horror (5), remake (4), frightened (4), survival (3), scientist (3), revenge (3), murder (3) | alien, horror, relationship, thriller, doll, based, survival, space, comedy, life |

### agglomerative

| Cluster | Size | Coherence | Representatives | Genres | Keywords | Terms |
| ---: | ---: | ---: | --- | --- | --- | --- |
| 1 | 84 | 0.532 | Spider-Man: No Way Home, The Avengers, Spider-Man, Man of Steel, Thor: The Dark World, Iron Man, Avengers: Endgame, Batman v Superman: Dawn of Justice | Action (81), Adventure (64), Science Fiction (56), Fantasy (18), Crime (9), Comedy (8) | based on comic (76), superhero (71), aftercreditsstinger (40), duringcreditsstinger (37), marvel cinematic universe (mcu) (32), super power (22), sequel (21), secret identity (13) | superhero, spider, man, action, universe, comic, adventure, marvel, based, new |
| 3 | 55 | 0.524 | Blade Runner 2049, Interstellar, The Matrix, Aliens, Rogue One: A Star Wars Story, Avatar, Mad Max: Fury Road, Dune | Science Fiction (55), Action (36), Adventure (33), Thriller (16), Drama (10), Comedy (3) | dystopia (28), based on novel or book (15), sequel (14), space opera (12), spacecraft (11), artificial intelligence (a.i.) (11), dystopian (10), future (10) | space, alien, action, future, science fiction, science, fiction, adventure, world, time |
| 4 | 50 | 0.531 | Frozen, Up, Monsters, Inc., How to Train Your Dragon, Shrek, Mulan, Onward, Cars | Family (49), Animation (47), Adventure (31), Comedy (27), Fantasy (22), Romance (8) | villain (34), aftercreditsstinger (16), duringcreditsstinger (15), cartoon (13), musical (12), friendship (10), cheerful (10), hopeful (10) | family, animation, villain, adventure, fantasy, comedy, cartoon, life, em, father |
| 9 | 45 | 0.510 | Die Hard, Bad Boys, Face/Off, Mission: Impossible - Dead Reckoning Part One, Mission: Impossible, Wrath of Man, Escape from New York, John Wick | Action (42), Thriller (33), Crime (17), Adventure (13), Science Fiction (9), Comedy (9) | los angeles, california (12), action hero (12), sequel (12), shootout (9), buddy cop (9), spy (8), suspenseful (7), excited (7) | action, cop, new, crime, police, thriller, time, movie, mission, secret |
| 0 | 36 | 0.524 | A Nightmare on Elm Street, Poltergeist, Scream, It, Beetlejuice, Nope, Friday the 13th, Gremlins | Horror (29), Thriller (15), Mystery (9), Comedy (8), Fantasy (7), Science Fiction (6) | frightened (10), supernatural horror (8), supernatural (8), anxious (8), based on novel or book (7), halloween (6), ghost (6), creature (6) | horror, alien, dead, thriller, supernatural, child, new, ghostbusters, paranormal, evil |
| 14 | 32 | 0.527 | Reservoir Dogs, Pulp Fiction, Casino, The Usual Suspects, Once Upon a Time in America, GoodFellas, The Departed, Lock, Stock and Two Smoking Barrels | Crime (26), Drama (18), Thriller (14), Comedy (10), Action (5), Western (4) | gangster (9), based on novel or book (8), neo-noir (8), excited (6), murder (6), robbery (6), heist (6), los angeles, california (6) | crime, drug, drama, gangster, police, comedy, based, mafia, thriller, story |
| 6 | 21 | 0.519 | Eternal Sunshine of the Spotless Mind, Notting Hill, La La Land, (500) Days of Summer, Forrest Gump, Dirty Dancing, The Notebook, Pretty Woman | Romance (18), Drama (17), Comedy (9), Fantasy (3), Music (1), Science Fiction (1) | valentine's day (5), duringcreditsstinger (4), based on novel or book (4), coming of age (3), new york city (3), romcom (3), los angeles, california (3), based on play or musical (3) | love, romance, life, day, drama, man, relationship, story, death, summer |
| 8 | 21 | 0.572 | The Lord of the Rings: The Fellowship of the Ring, Harry Potter and the Deathly Hallows: Part 2, The Hobbit: An Unexpected Journey, Harry Potter and the Deathly Hallows: Part 1, Harry Potter and the Philosopher's Stone, Harry Potter and the Prisoner of Azkaban, Harry Potter and the Chamber of Secrets, Harry Potter and the Order of the Phoenix | Adventure (21), Fantasy (21), Action (6), Family (4), Comedy (2) | wizard (16), magic (14), based on young adult novel (12), fantasy world (11), witch (11), based on novel or book (11), good versus evil (9), ghost (8) | harry, fantasy, school, ring, harry potter, potter, wizard, magic, world, based |

### graph

| Cluster | Size | Coherence | Representatives | Genres | Keywords | Terms |
| ---: | ---: | ---: | --- | --- | --- | --- |
| 0 | 83 | 0.504 | Interstellar, The Matrix, Mad Max: Fury Road, Aliens, The Terminator, Blade Runner 2049, Avatar, Rogue One: A Star Wars Story | Science Fiction (79), Action (50), Adventure (46), Thriller (26), Comedy (13), Drama (13) | dystopia (36), sequel (18), based on novel or book (18), artificial intelligence (a.i.) (13), alien (13), spacecraft (12), space opera (12), dystopian (12) | space, alien, action, time, world, future, science, science fiction, fiction, adventure |
| 1 | 79 | 0.539 | Spider-Man: No Way Home, The Avengers, Spider-Man, Man of Steel, Thor: The Dark World, Iron Man, Avengers: Endgame, Guardians of the Galaxy | Action (76), Adventure (57), Science Fiction (53), Fantasy (19), Crime (9), Comedy (8) | based on comic (75), superhero (73), aftercreditsstinger (40), duringcreditsstinger (35), marvel cinematic universe (mcu) (32), super power (22), sequel (20), secret identity (13) | superhero, spider, man, action, universe, comic, super, marvel, adventure, new |
| 2 | 65 | 0.479 | Die Hard, Bad Boys, Lock, Stock and Two Smoking Barrels, Kingsman: The Secret Service, Beverly Hills Cop, Face/Off, F9, Mission: Impossible | Action (41), Crime (33), Thriller (31), Comedy (30), Adventure (12), Mystery (7) | sequel (14), los angeles, california (13), based on novel or book (10), excited (10), buddy cop (10), shootout (9), action hero (9), amused (9) | action, comedy, new, crime, detective, time, cop, police, secret, thriller |
| 3 | 64 | 0.507 | Frozen, Monsters, Inc., Up, Shrek, Onward, How to Train Your Dragon, Mulan, Cars | Family (60), Animation (52), Adventure (39), Comedy (39), Fantasy (29), Drama (9) | villain (39), aftercreditsstinger (18), duringcreditsstinger (17), cartoon (15), musical (12), cheerful (12), anthropomorphism (11), friendship (10) | family, animation, villain, adventure, comedy, fantasy, cartoon, life, based, father |
| 4 | 59 | 0.485 | Reservoir Dogs, The Usual Suspects, Pulp Fiction, Once Upon a Time in America, Casino, The Departed, Requiem for a Dream, The Wolf of Wall Street | Drama (48), Thriller (26), Crime (25), Mystery (11), History (9), War (7) | based on novel or book (18), based on true story (16), biography (12), neo-noir (11), new york city (10), murder (10), world war ii (10), suspenseful (9) | crime, drama, war, based, drug, thriller, story, true, new, life |
| 5 | 50 | 0.499 | A Nightmare on Elm Street, It, Beetlejuice, Poltergeist, Scream, Gremlins, The Shining, Old | Horror (37), Thriller (18), Fantasy (14), Mystery (12), Comedy (10), Drama (8) | frightened (11), based on novel or book (9), supernatural horror (9), halloween (8), supernatural (8), ghost (8), murder (8), anxious (8) | horror, child, dead, thriller, supernatural, alien, relationship, new, killer, comedy |
| 6 | 49 | 0.458 | Dirty Dancing, Forrest Gump, Top Gun, Eternal Sunshine of the Spotless Mind, La La Land, Edward Scissorhands, 10 Things I Hate About You, Notting Hill | Drama (36), Romance (25), Comedy (18), Action (8), Fantasy (8), War (6) | coming of age (9), based on novel or book (8), friendship (7), dying and death (7), vietnam war (6), los angeles, california (6), teenager (5), high school (5) | war, love, school, romance, vietnam, death, drama, comedy, story, life |
| 7 | 39 | 0.516 | The Lord of the Rings: The Fellowship of the Ring, The Hobbit: An Unexpected Journey, The Hobbit: The Desolation of Smaug, The Lord of the Rings: The Return of the King, Harry Potter and the Deathly Hallows: Part 2, Pirates of the Caribbean: Dead Man's Chest, Harry Potter and the Prisoner of Azkaban, Harry Potter and the Chamber of Secrets | Adventure (38), Fantasy (27), Action (20), Science Fiction (7), Family (5), Comedy (3) | wizard (16), magic (14), based on young adult novel (12), witch (12), fantasy world (11), good versus evil (10), sequel (10), based on novel or book (9) | fantasy, harry, park, school, adventure, world, action, based, treasure, wizard |

### hdbscan

| Cluster | Size | Coherence | Representatives | Genres | Keywords | Terms |
| ---: | ---: | ---: | --- | --- | --- | --- |
| 0 | 163 | 0.480 | Spider-Man: No Way Home, Guardians of the Galaxy, The Avengers, Iron Man, Spider-Man, Man of Steel, Avengers: Endgame, Interstellar | Adventure (111), Action (109), Science Fiction (83), Fantasy (40), Comedy (27), Crime (24) | based on comic (63), superhero (60), aftercreditsstinger (47), sequel (36), duringcreditsstinger (36), marvel cinematic universe (mcu) (31), villain (28), based on novel or book (24) | action, superhero, space, adventure, based, world, fiction, science, man, science fiction |
| 1 | 7 | 0.607 | Poltergeist, Ghostbusters, Beetlejuice, Ghostbusters: Afterlife, A Nightmare on Elm Street, It, Friday the 13th | Horror (4), Comedy (3), Fantasy (3), Thriller (1), Drama (1), Adventure (1) | supernatural (3), ghost (3), haunted house (3), murder (3), frightened (3), paranormal phenomena (2), possession (2), nostalgic (2) | ghostbusters, town, paranormal, camp, small, horror, supernatural, afterlife, new, house |

## Recommendation For Milestone 3

kmeans appears most labelable: 35 clusters, average size 14.286, largest 36, 1 tiny clusters, and coherence average 0.560.
