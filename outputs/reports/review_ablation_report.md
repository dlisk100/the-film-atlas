# The Film Atlas - Milestone 3.25 Review-Weight Ablation

This report compares no-review, light-review, and medium-review profile variants on the same movie set using the same embedding model, k-means k=35, and the same draft labeling style. It does not fetch new TMDb data, scrape websites, export public JSON, or touch frontend code.

## Summary

- Recommended variant: light_reviews
- Recommendation: Use light_reviews: it has average label confidence 0.679, coherence 0.560, 1 tiny clusters, and 36 obvious noise-term hits.
- Do reviews improve vibe discovery? Light review snippets appear to help or preserve vibe discovery: label confidence delta +0.036, coherence delta +0.007.
- Does medium add noise? Medium review weight appears noisier than light_reviews: noise hits 43 vs 36, confidence delta -0.029.
- Total estimated live cost: $0.3305

## Variant Metrics

| Variant | Profiles | Tokens | Embed Cost | Cache Reused/New | Coherence Avg | Coherence Range | Tiny <5 | ARI vs Light | NMI vs Light | Label Confidence | Label Cache Reused/New | Noise Terms |
| --- | ---: | ---: | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | --- | --- |
| no_reviews | 500 | 76178 | $0.0099 | 9/491 | 0.553 | 0.473-0.749 | 0 | 0.288 | 0.639 | 0.643 | 0/35 | based (14), new (13), story (6) |
| light_reviews | 500 | 99288 | $0.0129 | 500/0 | 0.560 | 0.484-0.691 | 1 | 1.000 | 1.000 | 0.679 | 35/0 | new (12), based (11), movie (6), story (4), em (2), film (1) |
| medium_reviews | 500 | 100397 | $0.0131 | 380/120 | 0.560 | 0.482-0.693 | 0 | 0.337 | 0.653 | 0.650 | 0/35 | based (14), new (13), story (7), movie (4), film (3), em (2) |

## Cluster Size Distributions

- no_reviews: 28, 28, 25, 25, 24, 23, 20, 20, 19, 17, 16, 15, 14, 14, 13, 13, 12, 12, 11, 11, 11, 11, 11, 11, 10, 10, 10, 9, 9, 9, 9, 8, 8, 8, 6
- light_reviews: 36, 26, 25, 24, 22, 20, 18, 18, 17, 17, 17, 16, 15, 15, 15, 14, 13, 13, 12, 12, 12, 12, 12, 11, 11, 11, 10, 10, 9, 9, 7, 6, 6, 5, 4
- medium_reviews: 33, 25, 25, 23, 22, 21, 20, 19, 18, 18, 18, 17, 17, 16, 15, 14, 14, 13, 12, 11, 11, 11, 11, 10, 10, 9, 9, 9, 9, 8, 8, 8, 6, 5, 5

## Quality-Check Neighbors

### no_reviews

- Get Out: Old (0.612), Gone Girl (0.595), Split (0.593), The Invisible Man (0.585), Nope (0.570)
- Her: Ex Machina (0.667), Eternal Sunshine of the Spotless Mind (0.605), (500) Days of Summer (0.591), I, Robot (0.579), Passengers (0.570)
- Interstellar: Gravity (0.709), The Martian (0.692), Star Trek (0.691), Passengers (0.688), Arrival (0.656)
- Mad Max: Fury Road: Mad Max Beyond Thunderdome (0.818), Mad Max 2 (0.798), Furiosa: A Mad Max Saga (0.771), Dune (0.644), Dune (0.611)
- No Country for Old Men: Fargo (0.580), Memento (0.573), The Hateful Eight (0.571), Nope (0.569), Wrath of Man (0.568)
- Pulp Fiction: Reservoir Dogs (0.679), L.A. Confidential (0.643), The Big Lebowski (0.631), Lock, Stock and Two Smoking Barrels (0.628), Once Upon a Time... in Hollywood (0.616)
- The Dark Knight: The Dark Knight Rises (0.825), Batman (0.812), Batman Begins (0.779), The Batman (0.742), Joker (0.709)
- The Devil Wears Prada: The Devil's Advocate (0.572), Cruella (0.564), American Psycho (0.555), Whiplash (0.534), Pretty Woman (0.525)
- The Matrix: The Matrix Revolutions (0.816), The Matrix Resurrections (0.797), The Matrix Reloaded (0.794), The Terminator (0.699), Tron (0.664)
- The Shawshank Redemption: The Green Mile (0.599), Reservoir Dogs (0.564), GoodFellas (0.552), The Usual Suspects (0.550), The Departed (0.549)
- Whiplash: Black Swan (0.647), La La Land (0.621), Requiem for a Dream (0.597), Joker (0.581), Amadeus (0.569)

### light_reviews

- Get Out: Old (0.623), The Invisible Man (0.619), Gone Girl (0.613), Nope (0.605), Split (0.603)
- Her: Ex Machina (0.677), Eternal Sunshine of the Spotless Mind (0.632), Ready Player One (0.618), (500) Days of Summer (0.608), Passengers (0.608)
- Interstellar: The Martian (0.708), Gravity (0.697), Passengers (0.697), Arrival (0.675), Dune (0.657)
- Mad Max: Fury Road: Mad Max Beyond Thunderdome (0.813), Mad Max 2 (0.795), Furiosa: A Mad Max Saga (0.793), Dune (0.637), Escape from New York (0.628)
- No Country for Old Men: Fargo (0.593), Reservoir Dogs (0.587), The Big Lebowski (0.583), Pulp Fiction (0.582), Nope (0.579)
- Pulp Fiction: Reservoir Dogs (0.733), Jackie Brown (0.670), Once Upon a Time... in Hollywood (0.647), The Big Lebowski (0.638), Lock, Stock and Two Smoking Barrels (0.634)
- The Dark Knight: The Dark Knight Rises (0.798), Batman (0.753), Batman Begins (0.747), The Batman (0.736), Batman Returns (0.680)
- The Devil Wears Prada: Cruella (0.569), Notting Hill (0.554), The Devil's Advocate (0.552), American Psycho (0.550), 10 Things I Hate About You (0.544)
- The Matrix: The Matrix Revolutions (0.793), The Matrix Resurrections (0.779), The Matrix Reloaded (0.765), The Terminator (0.688), Tron (0.662)
- The Shawshank Redemption: The Green Mile (0.609), Reservoir Dogs (0.557), The Usual Suspects (0.549), GoodFellas (0.545), Once Upon a Time in America (0.540)
- Whiplash: Black Swan (0.648), La La Land (0.634), Requiem for a Dream (0.610), The Wolf of Wall Street (0.578), Joker (0.559)

### medium_reviews

- Get Out: Old (0.623), The Invisible Man (0.619), Gone Girl (0.613), Nope (0.605), Split (0.603)
- Her: Ex Machina (0.694), Eternal Sunshine of the Spotless Mind (0.646), (500) Days of Summer (0.627), Passengers (0.619), Ready Player One (0.618)
- Interstellar: The Martian (0.708), Gravity (0.697), Passengers (0.697), Arrival (0.675), Star Trek (0.667)
- Mad Max: Fury Road: Mad Max Beyond Thunderdome (0.813), Furiosa: A Mad Max Saga (0.793), Mad Max 2 (0.792), Dune (0.637), Escape from New York (0.628)
- No Country for Old Men: Fargo (0.595), The Big Lebowski (0.589), Reservoir Dogs (0.589), Nope (0.582), Pulp Fiction (0.580)
- Pulp Fiction: Reservoir Dogs (0.733), Jackie Brown (0.670), Once Upon a Time... in Hollywood (0.647), The Big Lebowski (0.643), Kill Bill: Vol. 1 (0.639)
- The Dark Knight: The Dark Knight Rises (0.797), Batman (0.763), The Batman (0.745), Batman Begins (0.741), Batman Returns (0.687)
- The Devil Wears Prada: Cruella (0.569), The Devil's Advocate (0.556), Notting Hill (0.554), American Psycho (0.550), The Grand Budapest Hotel (0.543)
- The Matrix: The Matrix Revolutions (0.796), The Matrix Resurrections (0.779), The Matrix Reloaded (0.765), The Terminator (0.688), Tron (0.662)
- The Shawshank Redemption: The Green Mile (0.615), Reservoir Dogs (0.553), GoodFellas (0.540), Once Upon a Time in America (0.531), The Usual Suspects (0.529)
- Whiplash: Black Swan (0.642), La La Land (0.633), Requiem for a Dream (0.603), The Wolf of Wall Street (0.571), Joker (0.562)

## Weakest Labels

### no_reviews

- Cluster 17 (Spy thriller (secret agencies & double lives)): confidence 0.55; John Wick is more 'assassin underworld' than classic spy-agency; the label leans on 'double lives/secret identities' to bridge the gap, but overlap risk remains.
- Cluster 21 (War-Training Coming of Age): confidence 0.56; The cluster’s coherence is moderate; several titles (e.g., Cast Away, Dead Poets Society) fit the 'coming-of-age under hardship' vibe more than the military core.
- Cluster 3 (Family Superhero Comedy): confidence 0.57; “Superhero” may feel stretched for films that are more general family/identity comedy (e.g., Mrs. Doubtfire), and the cluster’s live-action inclusions can blur the “animated superhero” expectation.
- Cluster 23 (Dark Comedy Gangster Crime): confidence 0.57; Action-forward entries (Extraction, Wrath of Man) may dilute the comedy-crime emphasis, but the broader 'crooked deal' and crime-world overlap still matches the cluster’s center.
- Cluster 29 (Alien Creature Survival Horror): confidence 0.57; Rambo entries are more human-war survival than alien creature horror; they likely connect via “survival under threat” and action-horror adjacency, so the label may overreach unless the clustering evidence supports it strongly.

### light_reviews

- Cluster 17 (Reality Bends Mind-Game Dramedies): confidence 0.58; Cast Away is more survival-isolation than reality-mind-game, so the psychological “system trap” theme may be less obvious there.
- Cluster 34 (Gothic vampire romance horror): confidence 0.58; Twilight is more teen romance than classic gothic vamp horror, and An American Werewolf in London is werewolf-centered rather than vampire-centered; the label is vampire-heavy and may mislead for those expecting strict vamp-only content.
- Cluster 4 (Neo-noir gangster crime stories): confidence 0.60; The cluster mixes tone ranges (from dark neo-noir to more stylized/comedic crime); “neo-noir gangster” keeps the throughline but may overstate noir consistency for the lighter entries.
- Cluster 9 (Spy-Thriller Puzzle Missions): confidence 0.60; Face/Off is more identity-crime/vengeance than classic spy tradecraft, and Tenet/Inception tilt toward sci-fi mind-bending more than traditional espionage—so the “spy” emphasis may not fit every title equally.
- Cluster 14 (Dinosaur-and-theme-park adventure): confidence 0.60; Not all titles are dinosaur/park-centered (Madagascar and Land Before Time are more animal-adventure), so the label may be judged too broad unless 'contained creature wonder' is considered sufficient.

### medium_reviews

- Cluster 26 (Tearjerker Romance & Survival): confidence 0.48; Includes non-romance-forward dramas (Apollo 13, The Whale, Into the Wild), so the label leans on emotional romance-adjacent themes rather than strict relationship plots.
- Cluster 3 (Superhero spectacle with anti-hero energy): confidence 0.52; Sonic the Hedgehog is the clearest outlier: it’s more video-game adventure than superhero comic spectacle, even if it shares blockbuster tone and “powered hero” energy.
- Cluster 5 (Superhero mercenary action): confidence 0.54; This cluster leans heavily on superhero/after-credits patterns, but it also includes non-superhero picks (Top Gun: Maverick, Back to the Future Part III) and a family animation (The Iron Giant), which may dilute the label’s accuracy.
- Cluster 9 (Neo-noir crime heists): confidence 0.56; Some picks lean more toward ensemble crime drama or cult comedy than strict heist structure (especially The Big Lebowski and Once Upon a Time... in Hollywood), so “heists” should be treated as a dominant vibe rather than a required plot element.
- Cluster 27 (Hopeful Animated Comedy-Adventure): confidence 0.56; Some entries (e.g., Cars) are more action-adventure than comedy-of-feelings, though they still fit the broad animated heart-and-humor profile.

## Review Signal Notes

### Examples Where Reviews May Improve Vibe Signal

- Get Out: light_reviews neighbors shift to Old (0.623), The Invisible Man (0.619), Gone Girl (0.613) from no_reviews Old (0.612), Gone Girl (0.595), Split (0.593).
- Her: light_reviews neighbors shift to Ex Machina (0.677), Eternal Sunshine of the Spotless Mind (0.632), Ready Player One (0.618) from no_reviews Ex Machina (0.667), Eternal Sunshine of the Spotless Mind (0.605), (500) Days of Summer (0.591).
- Interstellar: light_reviews neighbors shift to The Martian (0.708), Gravity (0.697), Passengers (0.697) from no_reviews Gravity (0.709), The Martian (0.692), Star Trek (0.691).
- Mad Max: Fury Road: light_reviews neighbors shift to Mad Max Beyond Thunderdome (0.813), Mad Max 2 (0.795), Furiosa: A Mad Max Saga (0.793) from no_reviews Mad Max Beyond Thunderdome (0.818), Mad Max 2 (0.798), Furiosa: A Mad Max Saga (0.771).

### Examples Where Reviews May Hurt Or Add Noise

- Her: medium_reviews shifts away from light_reviews (Ex Machina (0.694), Eternal Sunshine of the Spotless Mind (0.646), (500) Days of Summer (0.627)).
- No Country for Old Men: medium_reviews shifts away from light_reviews (Fargo (0.595), The Big Lebowski (0.589), Reservoir Dogs (0.589)).
- The Dark Knight: medium_reviews shifts away from light_reviews (The Dark Knight Rises (0.797), Batman (0.763), The Batman (0.745)).
- The Shawshank Redemption: medium_reviews shifts away from light_reviews (The Green Mile (0.615), Reservoir Dogs (0.553), GoodFellas (0.540)).

## Recommendation Before Scaling

Use light_reviews: it has average label confidence 0.679, coherence 0.560, 1 tiny clusters, and 36 obvious noise-term hits.
