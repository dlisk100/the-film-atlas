# The Film Atlas - Milestone 2.5 Cluster Sweep Report

This report compares local k-means granularities over existing embeddings. It does not call OpenAI, generate final AI labels, export public JSON, or touch frontend code.

## Summary

- Embedded movies inspected: 500
- k values tested: 15, 25, 35, 50
- Recommended k: 35
- Recommendation note: Use k=35 for Milestone 3 labeling: Promising granularity for Milestone 3 microgenre labeling.

## Sweep Metrics

| k | Clusters | Avg Size | Largest | Smallest | Coherence Avg | Coherence Range | Tiny <5 | Notes |
| ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |
| 15 | 15 | 33.3 | 53 | 10 | 0.540 | 0.473-0.667 | 0 | Too broad for final microgenre labels; useful as a baseline. |
| 25 | 25 | 20.0 | 34 | 6 | 0.554 | 0.491-0.735 | 0 | Promising granularity for Milestone 3 microgenre labeling. |
| 35 | 35 | 14.3 | 36 | 4 | 0.560 | 0.484-0.691 | 1 | Promising granularity for Milestone 3 microgenre labeling. |
| 50 | 50 | 10.0 | 22 | 3 | 0.574 | 0.485-0.777 | 4 | Promising granularity for Milestone 3 microgenre labeling. |

## Sample Clusters

### k=15

| Cluster | Size | Coherence | Representatives | Evidence terms |
| ---: | ---: | ---: | --- | --- |
| 4 | 53 | 0.519 | Frozen, Monsters, Inc., Up, Mulan, Onward, How to Train Your Dragon | family, animation, fantasy, villain, adventure, cartoon, comedy, world |
| 5 | 50 | 0.503 | A Nightmare on Elm Street, Scream, It, Poltergeist, Nope, The Invisible Man | horror, alien, thriller, child, dead, supernatural, new, vampire |
| 14 | 50 | 0.473 | Requiem for a Dream, Eternal Sunshine of the Spotless Mind, Forrest Gump, Whiplash, La La Land, The Prestige | war, story, drama, love, based, life, man, true |
| 13 | 46 | 0.486 | Mad Max: Fury Road, Mad Max Beyond Thunderdome, Escape from New York, The Hunger Games, Predator, Furiosa: A Mad Max Saga | war, future, action, world, based, adventure, army, fiction |
| 3 | 38 | 0.543 | Interstellar, Avatar, The Matrix, RoboCop, The Terminator, Total Recall | space, robot, alien, time, world, science, fiction, science fiction |
| 8 | 38 | 0.496 | Dumb and Dumber, Beverly Hills Cop, Bad Boys, A Fish Called Wanda, The Naked Gun: From the Files of Police Squad!, Coming to America | comedy, school, new, time, romance, cop, police, day |

### k=25

| Cluster | Size | Coherence | Representatives | Evidence terms |
| ---: | ---: | ---: | --- | --- |
| 5 | 34 | 0.531 | Monsters, Inc., Up, How to Train Your Dragon, Toy Story 3, Toy Story, Shrek | family, animation, villain, adventure, comedy, life, toy, fantasy |
| 4 | 32 | 0.511 | Inception, The Prestige, The Usual Suspects, Memento, Interstellar, Eternal Sunshine of the Spotless Mind | space, thriller, time, mystery, drama, mission, based, fiction |
| 20 | 32 | 0.495 | Ghostbusters, Back to the Future, Gremlins, Ghostbusters II, Beetlejuice, Ghostbusters: Afterlife | new, christmas, comedy, family, ghostbusters, school, time, new york |
| 17 | 30 | 0.499 | Once Upon a Time in America, Casino, GoodFellas, Scarface, The Departed, Requiem for a Dream | crime, drama, drug, based, new, prison, city, new york |
| 2 | 28 | 0.511 | Die Hard, Bad Boys, Rambo: First Blood Part II, Commando, Lethal Weapon, Lethal Weapon 2 | action, cop, vietnam, war, rocky, police, detective, daughter |
| 0 | 27 | 0.491 | Meet Joe Black, Edward Scissorhands, Notting Hill, Dirty Dancing, La La Land, Titanic | love, romance, death, life, vampire, story, drama, new |

### k=35

| Cluster | Size | Coherence | Representatives | Evidence terms |
| ---: | ---: | ---: | --- | --- |
| 4 | 36 | 0.524 | Reservoir Dogs, Pulp Fiction, Casino, The Usual Suspects, Once Upon a Time in America, GoodFellas | crime, drug, drama, prison, gangster, police, comedy, based |
| 5 | 26 | 0.513 | A Quiet Place Part II, Mad Max: Fury Road, Mad Max Beyond Thunderdome, Aliens, Mad Max 2, Alien Resurrection | alien, horror, action, world, zombie, family, new, kong |
| 23 | 25 | 0.562 | The Matrix, The Terminator, Total Recall, RoboCop, Blade Runner, The Matrix Resurrections | robot, matrix, action, world, future, time, runner, science |
| 6 | 24 | 0.498 | Black Swan, Requiem for a Dream, Shutter Island, The Prestige, Memento, Whiplash | thriller, psychological, drama, horror, man, based, mystery, new |
| 32 | 22 | 0.488 | Big, Back to the Future, Home Alone, Groundhog Day, Honey, I Shrunk the Kids, Stand by Me | comedy, school, fantasy, time, relationship, day, world, adventure |
| 12 | 20 | 0.539 | Interstellar, Gravity, Avatar, The Martian, Arrival, Passengers | space, alien, world, mission, science, science fiction, fiction, time |

### k=50

| Cluster | Size | Coherence | Representatives | Evidence terms |
| ---: | ---: | ---: | --- | --- |
| 35 | 22 | 0.547 | Up, Frozen, Inside Out, Onward, Big Hero 6, Zootopia | family, animation, villain, adventure, cartoon, life, father, comedy |
| 49 | 22 | 0.612 | The Avengers, Spider-Man: No Way Home, Spider-Man, Spider-Man: Homecoming, Captain America: Civil War, Doctor Strange | spider, superhero, man, new, spider man, marvel, captain, america |
| 13 | 19 | 0.525 | Inception, The Prestige, Memento, The Bourne Identity, Eternal Sunshine of the Spotless Mind, The Truman Show | thriller, mystery, identity, drama, secret, memory, time, new |
| 6 | 18 | 0.525 | Dumb and Dumber, A Fish Called Wanda, Lock, Stock and Two Smoking Barrels, The Naked Gun: From the Files of Police Squad!, The Big Lebowski, Ace Ventura: Pet Detective | comedy, police, time, crime, blues, comedy crime, ex, liar |
| 16 | 18 | 0.515 | A Quiet Place Part II, A Quiet Place, Jurassic World, Godzilla vs. Kong, The Thing, Love and Monsters | alien, horror, world, kong, godzilla, family, new, science |
| 28 | 18 | 0.536 | A Nightmare on Elm Street, Scream, Friday the 13th, Hellraiser, Five Nights at Freddy's, It | horror, child, killer, supernatural, dead, new, evil, thriller |

## Recommendation For Milestone 3

Use k=35 for Milestone 3 labeling: Promising granularity for Milestone 3 microgenre labeling.
