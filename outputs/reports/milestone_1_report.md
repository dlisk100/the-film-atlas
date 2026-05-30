# The Film Atlas - Milestone 1 Data Quality Report

Milestone 1 uses only TMDb API data, local processing, and a TF-IDF sample map. Final semantic embeddings, final clustering, cluster labels, and OpenAI calls are intentionally left for later milestones.

## Summary

- Discovered movies: 500
- Detail records fetched: 500
- With overview: 100.0% (500/500)
- With keywords: 99.8% (499/500)
- With reviews: 98.2% (491/500)
- From 2024 or later: 2.2% (11/500)
- From future release years: 0.0% (0/500)

## Year Distribution

| Year | Count |
| --- | ---: |
| 2021 | 30 |
| 2022 | 26 |
| 2020 | 18 |
| 1999 | 16 |
| 1997 | 16 |
| 2014 | 16 |
| 2017 | 16 |
| 2023 | 15 |
| 1987 | 14 |
| 1989 | 13 |
| 1988 | 13 |
| 1995 | 13 |
| 2009 | 13 |
| 1986 | 12 |
| 1998 | 12 |
| 2006 | 12 |
| 2016 | 12 |
| 1984 | 11 |
| 2008 | 11 |
| 2004 | 11 |
| 2007 | 11 |
| 2013 | 11 |
| 2015 | 11 |
| 1982 | 10 |
| 1996 | 10 |

## Top Official Genres

| Genre | Count |
| --- | ---: |
| Action | 220 |
| Adventure | 209 |
| Science Fiction | 160 |
| Comedy | 127 |
| Drama | 127 |
| Thriller | 111 |
| Fantasy | 107 |
| Crime | 77 |
| Family | 71 |
| Animation | 56 |
| Horror | 46 |
| Romance | 43 |
| Mystery | 37 |
| War | 16 |
| History | 13 |
| Music | 8 |
| Western | 4 |

## Top Keywords

| Keyword | Count |
| --- | ---: |
| based on comic | 85 |
| based on novel or book | 82 |
| superhero | 80 |
| duringcreditsstinger | 79 |
| sequel | 77 |
| aftercreditsstinger | 77 |
| villain | 57 |
| excited | 46 |
| dystopia | 41 |
| new york city | 38 |
| los angeles, california | 34 |
| suspenseful | 33 |
| revenge | 33 |
| marvel cinematic universe (mcu) | 32 |
| magic | 31 |
| good versus evil | 30 |
| amused | 30 |
| super power | 29 |
| murder | 28 |
| hilarious | 27 |
| intense | 26 |
| friendship | 26 |
| based on true story | 25 |
| alien | 25 |
| hero | 24 |

## Sampling Bias Diagnostics

- Movies from 2024 or later: 2.2% (11/500)
- Movies from future release years: 0.0% (0/500)
- Movies with franchise/sequel keywords: 43.6% (218/500)

### Franchise/Sequel Keywords

| Keyword | Count |
| --- | ---: |
| based on comic | 85 |
| superhero | 80 |
| duringcreditsstinger | 79 |
| sequel | 77 |
| aftercreditsstinger | 77 |
| marvel cinematic universe (mcu) | 32 |
| remake | 13 |
| superhero team | 9 |
| teen superhero | 7 |
| reboot | 6 |
| masked superhero | 6 |
| based on video game | 6 |
| superhero teamup | 5 |
| live action remake | 5 |
| aging superhero | 1 |

### Warnings

- Warning: Franchise/sequel concentration is high: 43.6% of movies have franchise-related keywords.

### Recommended Balanced Command

```bash
uv run film-atlas fetch-balanced --per-decade 100 --start-year 1980 --end-year 2026
uv run film-atlas fetch-details
uv run film-atlas normalize
uv run film-atlas build-profiles --review-weight light --max-review-chars 180
uv run film-atlas make-sample-map
uv run film-atlas report
```

## Review Noise Diagnostics

### Suspicious Review Terms

| Term | Count |
| --- | ---: |
| full | 340 |
| review | 306 |
| spoiler | 156 |
| http | 155 |
| https | 145 |
| rating | 138 |
| www | 117 |
| stars | 93 |
| letterboxd | 9 |

### Review Noise Examples

- Back to the Future: **Entertaining** A man goes back in time to save his mother - or something like that anyway - I was too entertained to fully grasp what was going on. Seriously, this film moves so
- The Shining: If you enjoy reading my Spoiler-Free reviews, please follow my blog :) With Doctor Sleep, an almost 40-year sequel to The Shining, being released this week, now it’s the perfect ti
- The Shining: "Darling, I'm not gonna hurt you. I'm just gonna bash your brains in." <i>The Shining</i> is a great example of how musical score and cinematography can elevate a movie to the best
- The Shining: "Darling, I'm not gonna hurt you. I'm just gonna bash your brains in." The Shining is a great example of how musical score and cinematography can elevate a movie to the best-of-the
- The Empire Strikes Back: I have reviewed this film before but I thought that it deserved an extra special mention. Yesterday, this was the first film I saw in a cinema since we were all confined to barrack

## Movies Missing Important Fields

| TMDb ID | Title | Missing Fields |
| ---: | --- | --- |
| 583083 | The Kissing Booth 2 | keywords |

## Sample Movie Text Profiles

### Back to the Future

```text
Title: Back to the Future
Overview: Eighties teenager Marty McFly is accidentally sent back in time to 1955, inadvertently disrupting his parents' first meeting and attracting his mother's romantic interest. Marty must repair the damage to history by rekindling his parents' romance and - with the help of his eccentric inventor friend Doc Brown - return to .
Genres: Adventure; Comedy; Science Fiction
Keywords: flying car; race against time; clock tower; car race; lightning; guitar; inventor; journey in the past; invention; time travel; bullying; mad scientist; love; fish out of water; terrorism; teenage love; destiny; burlesque; hidden identity; teenage life; changing the past or future; 1950s
Review language: **Entertaining** A man goes back in time to save his mother - or something like that anyway - I was too entertained to fully grasp what was going on. Seriously, this film moves so
```

### The Shining

```text
Title: The Shining
Overview: Jack Torrance accepts a caretaker job at the Overlook Hotel, where he, along with his wife Wendy and their son Danny, must live isolated from the rest of the world for the winter. But they aren't prepared for the madness that lurks within.
Genres: Horror; Thriller
Keywords: hotel; new year's eve; child abuse; based on novel or book; isolation; telepathy; delusion; halloween; snowstorm; colorado; seclusion; surrealism; writer's block; alcoholism; premonition; psychic power; caretaker; loneliness; vision; domestic violence; postmodern; psychological thriller; writer; twins; labyrinth; alcoholic; blizzard; mutilation; bloody body of child; extrasensory perception; uxoricide; motherhood; hypothermia; haunted hotel; psychological disintegration; disturbed; pediatrician; repetition; mother son relationship; supernatural power; new year; supernatural horror; ghost c
```

### The Empire Strikes Back

```text
Title: The Empire Strikes Back
Overview: The epic saga continues as Luke Skywalker, in hopes of defeating the evil Galactic Empire, learns the ways of the Jedi from aging master Yoda. But Darth Vader is more determined than ever to capture Luke. Meanwhile, rebel leader Princess Leia, cocky Han Solo, Chewbacca, and droids C-3PO and R2-D2 are thrown into various stages of capture, betrayal and despair.
Genres: Adventure; Action; Science Fiction
Keywords: rebellion; android; spacecraft; asteroid; rebel; space battle; snowstorm; space colony; swamp; sequel; space opera; arctic; intense; tuwaderalit
Review language: **Overrated ** An enjoyable film - just not as engaging as parts IV and VI. The argument that Jedi was ruined by little furry creatures is laughable as this instalment features a _
```

### Return of the Jedi

```text
Title: Return of the Jedi
Overview: Luke Skywalker leads a mission to rescue his friend Han Solo from the clutches of Jabba the Hutt, the Emperor prepares to crush the Rebellion with a more powerful Death Star, and the Rebel fleet mounts a massive attack on the space station. Luke Skywalker confronts Darth Vader in a final climactic duel before the evil Emperor.
Genres: Adventure; Action; Science Fiction
Keywords: spacecraft; sibling relationship; rebel; emperor; space battle; matter of life and death; forest; sequel; desert; space opera; bold; tuwaderalit
Review language: This is not quite Bantha fodder. Following on from the freshness of Star Wars (1977) and the all round greatness of craft and story telling that was The Empire Strikes Back (1980),
```

### Blade Runner

```text
Title: Blade Runner
Overview: In the smog-choked dystopian Los Angeles of 2019, blade runner Rick Deckard is called out of retirement to terminate a quartet of replicants who have escaped to Earth seeking their creator for a way to extend their short life spans.
Genres: Science Fiction; Drama; Thriller
Keywords: android; flying car; bounty hunter; artificial intelligence (a.i.); genetics; based on novel or book; dystopia; melancholy; futuristic; fugitive; cyberpunk; los angeles, california; alcoholic; origami; unicorn; tech noir; humanity; neo-noir; grim; human clone; humanoid robot; blade runner; 2010s; suspenseful; excited; dystopian
Review language: **Planet Noir** I declare _Blade Runner_ the best sci-fi movie of all time. Arguments? No? Okay. So long. Please upvote the guest book on your way out. WAIT! There's more. At the r
```

### The Terminator

```text
Title: The Terminator
Overview: In the post-apocalyptic future, reigning tyrannical supercomputers teleport a cyborg assassin known as the "Terminator" back to to kill Sarah Connor, whose unborn son is destined to lead insurgents against 21st century mechanical hegemony. Meanwhile, the human-resistance movement dispatches a lone warrior to safeguard Sarah. Can he stop the virtually indestructible killing machine?
Genres: Action; Thriller; Science Fiction
Keywords: man vs machine; artificial intelligence (a.i.); saving the world; laser gun; cyborg; killer robot; shotgun; rebel; dystopia; villain; time travel; los angeles, california; urban setting; future war; savior; tech noir; time paradox; action hero; griffith observatory; good versus evil; dystopian
Review language: I can't. Nobody goes home. Nobody else comes through. It's just him - and me. It's funny really, writing a review for T
```

### Back to the Future Part II

```text
Title: Back to the Future Part II
Overview: Marty and Doc are at it again as the time-traveling duo head to 2015 to nip some McFly family woes in the bud. But things go awry thanks to bully Biff Tannen and a pesky sports almanac. In a last-ditch attempt to set things straight, Marty finds himself bound for 1955 and face to face with his teenage parents -- again.
Genres: Adventure; Comedy; Science Fiction
Keywords: flying car; skateboarding; car race; lightning; guitar; inventor; time travel; diner; car crash; sequel; alternate history; thunderstorm; tunnel; high school dance; angry; hoverboard; 2010s; lighthearted; enthusiastic
Review language: You gotta go forward to save the past and back to alter the future. Yikes! Back to the Future Part II sees Marty & Jennifer coerced by Doc into travelling forward in time to correc
```

### Raiders of the Lost Ark

```text
Title: Raiders of the Lost Ark
Overview: When Dr. Indiana Jones – the tweed-suited professor who just happens to be a celebrated archaeologist – is hired by the government to locate the legendary Ark of the Covenant, he finds himself up against the entire Nazi regime.
Genres: Adventure; Action
Keywords: egypt; treasure; medallion; swastika; saving the world; nepal; himalaya mountain range; cairo; moses; hat; whip; leather jacket; mediterranean; ark of the covenant; ten commandments; nazi; excavation; riddle; treasure hunt; archaeologist; adventurer; archeology; globetrotting; religious history; 1930s
Review language: **Trailblazers of a Lost Art** Little wonder James Cameron and Joss Whelon movies are the biggest box-office earners. They are masters of cinematic rhetoric. The unfolding dramatic
```

### Scarface

```text
Title: Scarface
Overview: After getting a green card in exchange for assassinating a Cuban government official, Tony Montana stakes a claim on the drug trade in Miami. Viciously murdering anyone who stands in his way, Tony eventually becomes the biggest drug lord in the state, controlling nearly all the cocaine that comes through Miami. But increased pressure from the police, wars with Colombian drug cartels and his own drug-fueled paranoia serve to fuel the flames of his eventual downfall.
Genres: Action; Crime; Drama
Keywords: corruption; sibling relationship; miami, florida; cuba; loss of loved one; gangster; cocaine; rise and fall; remake; tragedy; drug cartel; mafia; drug lord; bitterness; rise to power; miami beach; cuban refugees; drug war; excited; tragic
Review language: Immensely great crime-drama that features some great performances and excellent writing from Oliver Stone (an
```

### Dead Poets Society

```text
Title: Dead Poets Society
Overview: At an elite, old-fashioned boarding school in New England, a passionate English teacher inspires his students to rebel against convention and seize the potential of every day, courting the disdain of the stern headmaster.
Genres: Drama
Keywords: individual; friendship; philosophy; poetry; literature; professor; based on true story; coming of age; teacher; school play; new england; vermont; schoolteacher; preparatory school; 1950s; teenager
Review language: Carpe Diem & The Punk Rock Movie. Dead Poets Society is directed by (Picnic At Hanging Rock/Gallipoli) and stars , , , , & . The script is written by Tom Schulman, based on his lif
```

### Die Hard

```text
Title: Die Hard
Overview: High above the city of L.A. a team of terrorists has seized a building, taken hostages, and declared war. One man has manages to escape... An off-duty cop hiding somewhere inside. He's alone, tired... and the only chance anyone has got.
Genres: Action; Thriller
Keywords: husband wife relationship; based on novel or book; s.w.a.t.; fbi; christmas party; vault; heist; murder; shootout; los angeles, california; terrorism; one man army; explosion; police officer; hostage negotiator; one night; lapd; christmas; action hero; hostages; patrol officer; absurd; high octane; suspenseful; amused; excited
Review language: **This is one of the definitive 80s Action Films.** There is no nonsense whatsoever, the plot moves along with such a pace that the viewer is not disturbed by implausabilities. pla
```

### E.T. the Extra-Terrestrial

```text
Title: E.T. the Extra-Terrestrial
Overview: An alien is left behind on Earth and saved by the 10-year-old Elliott who decides to keep him hidden in his home. While a task force hunts for the extra-terrestrial, Elliott, his brother, and his little sister Gertie form an emotional bond with their new friend, and try to help him find his way home.
Genres: Adventure; Science Fiction; Family
Keywords: farewell; space marine; operation; flying saucer; nasa; homesickness; loss of loved one; extraterrestrial technology; prosecution; riding a bicycle; halloween; finger; flowerpot; alien; single; single mother; alien contact; homesick; space sci-fi
Review language: Watched with my wife, the 7th grader, and the kindergartner. I only kind of half watched. It's been a long week. I really wanted to check out my youngest's reactions. Watching this
```

### Full Metal Jacket

```text
Title: Full Metal Jacket
Overview: A pragmatic U.S. Marine observes the dehumanizing effects the U.S.-Vietnam War has on his fellow recruits from their brutal boot camp training to the bloody street fighting in Hue.
Genres: Drama; War
Keywords: rescue; sniper; vietnam war; suicide; vietnam; helicopter; army; prostitute; based on novel or book; propaganda; war correspondent; journalism; recruit; infantry; war photographer; boot camp; jungle; sergeant; racism; genocide; fighting; platoon; combat; discipline; u.s. marine; obstacle course; military; anti war; mass grave; blanket party; soldiers; war; critical; frustrated; harsh; scathing
Review language: **The second half is better than the first half.** A film of two halves. The first half of the fiim focuses on the training of raw recruits and features shenanigans we have seen co
```

### Indiana Jones and the Last Crusade

```text
Title: Indiana Jones and the Last Crusade
Overview: In 1938, an art collector appeals to eminent archaeologist Dr. Indiana Jones to embark on a search for the Holy Grail. Indy learns that a medieval historian has vanished while searching for it, and the missing man is his own father, Dr. Henry Jones Sr.. He sets out to rescue his father by following clues in the old man's notebook, which his father had mailed to him before he went missing. Indy arrives in Venice, where he enlists the help of a beautiful academic, Dr. Elsa Schneider, along with Marcus Brody and Sallah. Together they must stop the Nazis from recovering the power of eternal life and taking over the world!
Genres: Adventure; Action
Keywords: saving the world; nazi; holy grail; venice, italy; entrapment; crusader; germany; riddle; brotherhood; zeppelin; tank; book burning; nazi officer; boat chase; gestapo; single father; tra
```

### Aliens

```text
Title: Aliens
Overview: Ripley, the sole survivor of the Nostromo's deadly encounter with the monstrous Alien, returns to Earth after drifting through space in hypersleep for 57 years. Although her story is initially met with skepticism, she agrees to accompany a team of Colonial Marines back to LV-426.
Genres: Action; Thriller; Science Fiction
Keywords: android; space marine; extraterrestrial technology; spaceman; space travel; settler; colony; cryogenics; vacuum; space colony; warrior woman; alien; space; female protagonist; creature; desolate; female hero; aggressive; military sci-fi; desolate planet; critical; sinister; sci-fi horror; assertive; commanding; empathetic; exhilarated; action horror; sci-fi action
Review language: One of my all time favorites. It still contains some of the drama and suspense of the first but with far more action leading to what I find a far more appealin
```

### Indiana Jones and the Temple of Doom

```text
Title: Indiana Jones and the Temple of Doom
Overview: After arriving in India, Indiana Jones is asked by a desperate village to find a mystical stone. He agrees – and stumbles upon a secret cult plotting a terrible plan in the catacombs of an ancient palace.
Genres: Adventure; Action
Keywords: treasure; skeleton; wind; elephant; heart; riddle; crocodile; bridge; treasure hunt; torture; india; monkey; archaeologist; conveyor belt; child driving car; mine car; rope bridge; splits; adventurer; archeology; 1930s; excited; exhilarated
Review language: **The best film in the series** _Raiders_ was great but suffered patches of slowness where the momentum was damaged - I know people who actually fast forward Raiders when Indy and
```

### Ghostbusters

```text
Title: Ghostbusters
Overview: After losing their university jobs, three parapsychologists start a ghost-catching business in New York City and uncover a supernatural threat that could destroy the world.
Genres: Comedy; Fantasy
Keywords: new york city; environmental protection agency; library; supernatural; paranormal phenomena; loser; slime; gatekeeper; nerd; giant monster; haunting; hybrid; possession; mythology; horror spoof; paranormal investigation; urban setting; super power; receptionist; world trade center; ghost; nostalgic; duringcreditsstinger; fighting supernatural; satirical; ghostbusters; witty; amused; enchant
Review language: They came, they saw, they briefly conquered the 80s. A trio of misfit parapsychologists set up business as Ghostbusters. Ideal really because although slow at first, their business
```

### Top Gun

```text
Title: Top Gun
Overview: For Lieutenant Pete 'Maverick' Mitchell and his friend and co-pilot Nick 'Goose' Bradshaw, being accepted into an elite training school for fighter pilots is a dream come true. But a tragedy, as well as personal demons, will threaten Pete's dreams of becoming an ace pilot.
Genres: Action; Drama; Romance
Keywords: dying and death; secret love; lovesickness; airplane; loss of loved one; self-discovery; hostility; fighter pilot; pilot; training camp; battle assignment; u.s. navy; cowardliness; homoeroticism; pilot school; based on magazine, newspaper or article; admiring; celebratory; commanding; exuberant; gay men
Review language: You'll struggle to find a more cheesy film, even so <em>'Top Gun'</em> is a super film. I need to check out more of 's work because he has yet to let me down from the flicks of his
```

### Predator

```text
Title: Predator
Overview: A team of elite commandos on a secret mission in a Central jungle come to find themselves hunted by an extraterrestrial warrior.
Genres: Science Fiction; Action; Adventure; Thriller
Keywords: guerrilla warfare; central and south america; predator; trap; alien; survival; stalking; creature; alien invasion; invisible; commando; aggressive; prey; violence
Review language: From about 1996 to about 2009 (roughly ages 4 til 17 for those playing at home), this was my all-time favourite movie. It was the first non-pirated VHS I ever owned, and it probabl
```

### Batman

```text
Title: Batman
Overview: Having witnessed his parents' brutal murder as a child, millionaire philanthropist Bruce Wayne fights crime in Gotham City disguised as Batman, a costumed hero who strikes fear into the hearts of villains. But when a deformed madman known as 'The Joker' seizes control of Gotham's criminal underworld, Batman must face his most ruthless nemesis ever while protecting both his identity and his love interest, reporter Vicki Vale.
Genres: Fantasy; Action; Crime
Keywords: dual identity; double life; chemical; crime fighter; superhero; villain; based on comic; vigilante; mobster; organized crime; criminal; super power; madness; vigilantism; cautionary; good versus evil
Review language: Vision not fully realised, but still a template of sorts. It could never have lived up to the hype back in , it was hailed as the film to rival the impact of "Jaws" & "Star Wars" a
```

## Example Nearest-Neighbor Pairs

| Source | Neighbor | Similarity |
| --- | --- | ---: |
| The Matrix Reloaded | The Matrix Revolutions | 0.657 |
| A Quiet Place | A Quiet Place Part II | 0.650 |
| Godzilla vs. Kong | Godzilla x Kong: The New Empire | 0.635 |
| Blade Runner | Blade Runner 2049 | 0.632 |
| Harry Potter and the Philosopher's Stone | Harry Potter and the Prisoner of Azkaban | 0.580 |
| Dune | Dune: Part Two | 0.578 |
| Pirates of the Caribbean: Dead Man's Chest | Pirates of the Caribbean: At World's End | 0.570 |
| Guardians of the Galaxy Vol. 2 | Guardians of the Galaxy Vol. 3 | 0.568 |
| Harry Potter and the Philosopher's Stone | Harry Potter and the Goblet of Fire | 0.563 |
| Jurassic Park | Jurassic World | 0.555 |

## Notes

- This report is a data-quality proof, not the final public website output.
- Raw TMDb responses live under data/cache/ and data/raw/, which are gitignored.
- Review language in profiles is truncated for semantic experimentation.
- Recommended balanced sampling command before Milestone 2: `uv run film-atlas fetch-balanced --per-decade 100 --start-year 1980 --end-year 2026`.
- OpenAI embeddings and cluster labeling are out of scope for Milestone 1.
