# The Film Atlas - Milestone 2 Semantic Neighborhood Report

Milestone 2 uses OpenAI embeddings, local projection, local clustering, and local inspection only. It does not generate final AI microgenre labels, public website JSON, or frontend integration.

## Summary

- Profiles available: 500
- Movies embedded: 500
- Embedding model: text-embedding-3-large
- Estimated tokens: 99288
- Estimated cost: $0.0129
- Cached embeddings reused: 100
- New embeddings generated: 400
- Projection method: pca
- Clustering method: kmeans
- Cluster count: 15
- Outliers: 0 (0.0%)

## Cluster Size Distribution

| Cluster ID | Count |
| --- | ---: |
| 4 | 53 |
| 5 | 50 |
| 14 | 50 |
| 13 | 46 |
| 3 | 38 |
| 8 | 38 |
| 9 | 35 |
| 1 | 33 |
| 7 | 32 |
| 6 | 31 |
| 12 | 29 |
| 11 | 26 |
| 0 | 16 |
| 2 | 13 |
| 10 | 10 |

## Sample Nearest Neighbors

| Movie | Neighbor | Similarity |
| --- | --- | ---: |
| Back to the Future | Back to the Future Part II | 0.804 |
| The Shining | It | 0.636 |
| The Empire Strikes Back | Return of the Jedi | 0.793 |
| Return of the Jedi | The Empire Strikes Back | 0.793 |
| Blade Runner | Blade Runner 2049 | 0.814 |
| The Terminator | Terminator 2: Judgment Day | 0.821 |
| Back to the Future Part II | Back to the Future | 0.804 |
| Raiders of the Lost Ark | Indiana Jones and the Temple of Doom | 0.786 |
| Scarface | Casino | 0.661 |
| Dead Poets Society | The Breakfast Club | 0.602 |
| Die Hard | Die Hard 2 | 0.769 |
| E.T. the Extra-Terrestrial | Poltergeist | 0.626 |
| Full Metal Jacket | Platoon | 0.698 |
| Indiana Jones and the Last Crusade | Raiders of the Lost Ark | 0.783 |
| Aliens | Alien Resurrection | 0.816 |
| Indiana Jones and the Temple of Doom | Raiders of the Lost Ark | 0.786 |
| Ghostbusters | Ghostbusters II | 0.802 |
| Top Gun | Top Gun: Maverick | 0.799 |
| Predator | Prey | 0.718 |
| Batman | Batman Begins | 0.795 |

## Quality-Check Movie Neighbors

- No Country for Old Men: Fargo (0.593), Reservoir Dogs (0.587), The Big Lebowski (0.583), Pulp Fiction (0.582), Nope (0.579)
- Her: Ex Machina (0.677), Eternal Sunshine of the Spotless Mind (0.632), Ready Player One (0.618), (500) Days of Summer (0.608), Passengers (0.608)
- Get Out: Old (0.623), The Invisible Man (0.619), Gone Girl (0.613), Nope (0.605), Split (0.603)
- The Matrix: The Matrix Revolutions (0.793), The Matrix Resurrections (0.779), The Matrix Reloaded (0.765), The Terminator (0.688), Tron (0.662)
- Mad Max: Fury Road: Mad Max Beyond Thunderdome (0.813), Mad Max 2 (0.795), Furiosa: A Mad Max Saga (0.793), Dune (0.637), Escape from New York (0.628)
- The Devil Wears Prada: Cruella (0.569), Notting Hill (0.554), The Devil's Advocate (0.552), American Psycho (0.550), 10 Things I Hate About You (0.544)
- Whiplash: Black Swan (0.648), La La Land (0.634), Requiem for a Dream (0.610), The Wolf of Wall Street (0.578), Joker (0.559)
- Pulp Fiction: Reservoir Dogs (0.733), Jackie Brown (0.670), Once Upon a Time... in Hollywood (0.647), The Big Lebowski (0.638), Lock, Stock and Two Smoking Barrels (0.634)
- The Shawshank Redemption: The Green Mile (0.609), Reservoir Dogs (0.557), The Usual Suspects (0.549), GoodFellas (0.545), Once Upon a Time in America (0.540)
- Interstellar: The Martian (0.708), Gravity (0.697), Passengers (0.697), Arrival (0.675), Dune (0.657)
- The Dark Knight: The Dark Knight Rises (0.798), Batman (0.753), Batman Begins (0.747), The Batman (0.736), Batman Returns (0.680)

## Sample Clusters

### Cluster 4 (53 movies)

- Representative movies: Frozen, Monsters, Inc., Up, Mulan, Onward, How to Train Your Dragon, Encanto, Shrek
- Top official genres: Family (49), Animation (45), Adventure (34), Fantasy (28), Comedy (28), Romance (8)
- Top TMDb keywords: villain (34), musical (15), cartoon (14), aftercreditsstinger (14), duringcreditsstinger (13), anthropomorphism (9), hopeful (9), coming of age (8)
- Aggregated profile terms: family, animation, fantasy, villain, adventure, cartoon, comedy, world, em, based
- Coherence score: 0.519
- Warnings: none

### Cluster 5 (50 movies)

- Representative movies: A Nightmare on Elm Street, Scream, It, Poltergeist, Nope, The Invisible Man, Beetlejuice, Old
- Top official genres: Horror (40), Thriller (20), Mystery (12), Science Fiction (11), Fantasy (10), Comedy (9)
- Top TMDb keywords: based on novel or book (11), frightened (11), supernatural horror (9), anxious (9), murder (8), psychological horror (7), supernatural (7), halloween (6)
- Aggregated profile terms: horror, alien, thriller, child, dead, supernatural, new, vampire, killer, evil
- Coherence score: 0.503
- Warnings: none

### Cluster 14 (50 movies)

- Representative movies: Requiem for a Dream, Eternal Sunshine of the Spotless Mind, Forrest Gump, Whiplash, La La Land, The Prestige, The Curious Case of Benjamin Button, Good Will Hunting
- Top official genres: Drama (49), Romance (14), Thriller (8), History (7), Mystery (6), Crime (5)
- Top TMDb keywords: based on true story (14), based on novel or book (12), biography (9), friendship (6), tragedy (6), world war ii (6), suspenseful (5), parent child relationship (5)
- Aggregated profile terms: war, story, drama, love, based, life, man, true, film, death
- Coherence score: 0.473
- Warnings: none

### Cluster 13 (46 movies)

- Representative movies: Mad Max: Fury Road, Mad Max Beyond Thunderdome, Escape from New York, The Hunger Games, Predator, Furiosa: A Mad Max Saga, Mad Max 2, Dune
- Top official genres: Action (34), Science Fiction (24), Adventure (24), Thriller (16), Drama (13), War (11)
- Top TMDb keywords: dystopia (13), based on novel or book (11), post-apocalyptic future (9), revenge (7), suspenseful (6), based on true story (6), dramatic (6), army (5)
- Aggregated profile terms: war, future, action, world, based, adventure, army, fiction, movie, science
- Coherence score: 0.486
- Warnings: none

### Cluster 3 (38 movies)

- Representative movies: Interstellar, Avatar, The Matrix, RoboCop, The Terminator, Total Recall, Independence Day, Ready Player One
- Top official genres: Science Fiction (38), Adventure (19), Action (16), Drama (11), Comedy (9), Thriller (8)
- Top TMDb keywords: dystopia (15), artificial intelligence (a.i.) (9), alien (9), based on novel or book (7), dystopian (7), spacecraft (7), space (7), robot (7)
- Aggregated profile terms: space, robot, alien, time, world, science, fiction, science fiction, future, adventure
- Coherence score: 0.543
- Warnings: none

### Cluster 8 (38 movies)

- Representative movies: Dumb and Dumber, Beverly Hills Cop, Bad Boys, A Fish Called Wanda, The Naked Gun: From the Files of Police Squad!, Coming to America, Police Academy, Lock, Stock and Two Smoking Barrels
- Top official genres: Comedy (37), Crime (14), Romance (10), Drama (7), Fantasy (6), Action (6)
- Top TMDb keywords: amused (10), los angeles, california (9), hilarious (7), coming of age (5), absurd (5), buddy cop (5), high school (4), cheerful (4)
- Aggregated profile terms: comedy, school, new, time, romance, cop, police, day, christmas, detective
- Coherence score: 0.496
- Warnings: none

### Cluster 9 (35 movies)

- Representative movies: Mission: Impossible - Dead Reckoning Part One, Die Hard, Wrath of Man, John Wick, F9, No Time to Die, Face/Off, Mission: Impossible
- Top official genres: Action (33), Thriller (27), Crime (16), Adventure (8), Comedy (5), Drama (4)
- Top TMDb keywords: sequel (11), spy (9), shootout (8), revenge (8), los angeles, california (7), action hero (7), excited (7), duringcreditsstinger (7)
- Aggregated profile terms: action, secret, crime, thriller, time, bond, mission, wick, movie, revenge
- Coherence score: 0.515
- Warnings: none

### Cluster 1 (33 movies)

- Representative movies: Reservoir Dogs, Casino, Pulp Fiction, The Usual Suspects, Once Upon a Time in America, The Departed, GoodFellas, Heat
- Top official genres: Crime (26), Drama (21), Thriller (20), Comedy (8), Mystery (6), Action (3)
- Top TMDb keywords: based on novel or book (11), gangster (9), new york city (9), neo-noir (9), murder (8), heist (6), mafia (5), excited (5)
- Aggregated profile terms: crime, new, drug, thriller, drama, city, york, new york, gangster, based
- Coherence score: 0.521
- Warnings: none

### Cluster 7 (32 movies)

- Representative movies: The Avengers, Avengers: Endgame, Avengers: Infinity War, Captain America: Civil War, Thor: The Dark World, Avengers: Age of Ultron, Thor: Ragnarok, Guardians of the Galaxy
- Top official genres: Action (32), Adventure (28), Science Fiction (24), Fantasy (7), Comedy (5)
- Top TMDb keywords: based on comic (31), superhero (30), marvel cinematic universe (mcu) (29), aftercreditsstinger (27), duringcreditsstinger (21), sequel (11), hero (8), superhero team (5)
- Aggregated profile terms: marvel, universe, superhero, action, cinematic, marvel cinematic, cinematic universe, mcu, adventure, new
- Coherence score: 0.601
- Warnings: none

### Cluster 6 (31 movies)

- Representative movies: Batman v Superman: Dawn of Justice, Man of Steel, Batman, Superman, The Batman, The Suicide Squad, Wonder Woman 1984, Suicide Squad
- Top official genres: Action (30), Adventure (17), Fantasy (12), Science Fiction (12), Crime (7), Thriller (4)
- Top TMDb keywords: superhero (27), based on comic (27), super power (14), dc extended universe (dceu) (11), vigilante (8), crime fighter (7), duringcreditsstinger (7), secret identity (6)
- Aggregated profile terms: batman, crime, super, superhero, action, comic, dc, based, hero, world
- Coherence score: 0.549
- Warnings: none

## Warnings

_No major warnings._

## Recommendation For Milestone 3

Review the sample clusters and quality-check neighbors. If the neighborhoods feel semantically coherent, proceed to Milestone 3 with human-guided cluster naming or mocked label prompts before any final public export.
