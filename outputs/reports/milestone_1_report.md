# The Film Atlas - Milestone 1 Data Quality Report

Milestone 1 uses only TMDb API data, local processing, and a TF-IDF sample map. Final semantic embeddings, final clustering, cluster labels, and OpenAI calls are intentionally left for later milestones.

## Summary

- Discovered movies: 100
- Detail records fetched: 100
- With overview: 100.0% (100/100)
- With keywords: 100.0% (100/100)
- With reviews: 98.0% (98/100)

## Year Distribution

| Year | Count |
| --- | ---: |
| 2026 | 23 |
| 2025 | 16 |
| 2024 | 6 |
| 1999 | 5 |
| 2021 | 4 |
| 2023 | 4 |
| 1994 | 3 |
| 2001 | 3 |
| 2010 | 3 |
| 2002 | 3 |
| 2006 | 2 |
| 2018 | 2 |
| 2008 | 2 |
| 2003 | 2 |
| 2019 | 2 |
| 2022 | 2 |
| 2009 | 2 |
| 2015 | 2 |
| 2013 | 1 |
| 1992 | 1 |
| 2014 | 1 |
| 2011 | 1 |
| 2012 | 1 |
| 1972 | 1 |
| 1997 | 1 |

## Top Official Genres

| Genre | Count |
| --- | ---: |
| Adventure | 40 |
| Action | 39 |
| Drama | 32 |
| Science Fiction | 30 |
| Comedy | 24 |
| Thriller | 23 |
| Fantasy | 17 |
| Crime | 17 |
| Animation | 15 |
| Family | 14 |
| Romance | 12 |
| Horror | 9 |
| Mystery | 7 |
| Music | 3 |
| History | 3 |
| War | 1 |

## Top Keywords

| Keyword | Count |
| --- | ---: |
| based on novel or book | 25 |
| sequel | 22 |
| aftercreditsstinger | 19 |
| duringcreditsstinger | 17 |
| superhero | 12 |
| based on comic | 11 |
| friendship | 9 |
| magic | 9 |
| new york city | 9 |
| villain | 8 |
| marvel cinematic universe (mcu) | 7 |
| space | 6 |
| love | 6 |
| secret identity | 6 |
| space travel | 5 |
| 3d animation | 5 |
| reboot | 5 |
| artificial intelligence (a.i.) | 5 |
| dystopia | 5 |
| anthropomorphism | 5 |
| playful | 5 |
| loss of loved one | 5 |
| woman director | 5 |
| suspenseful | 5 |
| superhero team | 5 |

## Movies Missing Important Fields

_No important fields are missing from the normalized sample._

## Sample Movie Text Profiles

### Lee Cronin's The Mummy

```text
Title: Lee Cronin's The Mummy
Overview: The young daughter of a journalist disappears into the desert without a trace—eight years later, the broken family is shocked when she is returned to them, as what should be a joyful reunion turns into a living nightmare.
Genres: Horror; Mystery
Keywords: journalist; egypt; monster; ritual; kidnapping; pyramid; investigation; supernatural; mummy; possession; disappearance; curse; demon; tomb; missing person; supernatural horror; body horror; horror
Review language: ’s The Mummy isn’t scary or memorable; it’s raunchy exploitation and over-orchestrated expired cheese. It is a horror film that reeks of nothing but ridiculousness. The sad part is there’s a decent enough concept buried somewhere within this vomit-drenched monstrosity and a killer ambiance; I had high hopes for this, but boy was I disappointed... Instead of getting Christopher Lee, Boris
```

### Project Hail Mary

```text
Title: Project Hail Mary
Overview: Science teacher Ryland Grace wakes up on a spaceship light years from home with no recollection of who he is or how he got there. As his memory returns, he begins to uncover his mission: solve the riddle of the mysterious substance causing the sun to die out. He must call on his scientific knowledge and unorthodox ideas to save everything on Earth from extinction.
Genres: Science Fiction; Adventure
Keywords: friendship; coma; based on novel or book; bravery; sun; language barrier; space travel; space mission; alien; space; memory loss; suicide mission; astronaut; scientist; curious; science teacher; save the planet; wonder; spaceship; interspecies friendship
Review language: When times are tough and world-weary souls have looked for an avenue of escapism to retreat from their woes, worries and weltschmerz, they’ve often flocked to the movies to relieve 
```

### The Super Mario Galaxy Movie

```text
Title: The Super Mario Galaxy Movie
Overview: Having thwarted Bowser's previous plot to marry Princess Peach, Mario and Luigi now face a fresh threat in Bowser Jr., who is determined to liberate his father from captivity and restore the family legacy. Alongside companions new and old, the brothers travel across the stars to stop the young heir's crusade.
Genres: Family; Comedy; Adventure; Fantasy; Animation
Keywords: galaxy; friendship; sibling relationship; space travel; turtle; sequel; slapstick comedy; space; robot; based on video game; buddy comedy; aftercreditsstinger; duringcreditsstinger; globetrotting; space adventure; children's adventure; father son relationship; parallel universe; brother brother relationship; talking animal; magic land; fictional country
Review language: Full review: Rating: B+ "The Super Mario Galaxy Movie is a sequel that, while losing some of the narrative
```

### Swapped

```text
Title: Swapped
Overview: A small woodland creature and a majestic bird, two natural sworn enemies of the Valley, magically trade places and set off on an adventure of a lifetime to switch back. Their journey soon uncovers a greater threat—one that could endanger not only their species, but the entire valley they call home.
Genres: Adventure; Animation; Family; Fantasy
Keywords: wolf; buddy; forest fire; woodlands; forest lore; bird; 3d animation; empathetic; vibrant; body swap; animal adventure
Review language: I recently watched this movie and found it to be an enjoyable experience overall. The performances were solid, the visuals were impressive, and the story kept me interested from beginning to end. While some parts felt predictable, the emotional moments and pacing made it worth watching.
```

### Mortal Kombat

```text
Title: Mortal Kombat
Overview: Washed-up MMA fighter Cole Young, unaware of his heritage, and hunted by Emperor Shang Tsung's best warrior, Sub-Zero, seeks out and trains with Earth's greatest champions as he prepares to stand against the enemies of Outworld in a high stakes battle for the universe.
Genres: Action; Fantasy; Adventure
Keywords: saving the world; magic; ninja fighter; gore; god; alternate dimension; shaolin monk; fighting; based on video game; martial arts tournament; reboot; hand to hand combat; casual
Review language: I will be short. You should understand how hard to make movies based on such a legendary universe (expectation is too high!), with a lot of characters that need screen time, and with limited budget, PLUS in a pandemic situation, - the director made a great job. This is the best adaptation of such a; Mortal Kombat Gives Fans Of The Franchise The Brutal Live-
```

### The Devil Wears Prada

```text
Title: The Devil Wears Prada
Overview: A young woman from the Midwest gets more than she bargained for when she moves to New York to become a writer and ends up as the assistant to the tyrannical, larger-than-life editor-in-chief of a major fashion magazine.
Genres: Drama; Comedy
Keywords: new york city; journalist; paris, france; based on novel or book; journalism; fashion journal; assistant; job entrant; job interview; editor-in-chief; fashion; fashion magazine; bullied; city life; fashion industry; wonder; absurd; conceited; mean spirited; ridiculous; workplace drama
Review language: Normally this sort of film wouldn't interest me, but I was fascinated by the cast (, , and ARE four of my favourite contemporary actors) AND I liked the three previous films I've seen about the fashion industry ('Ready to Wear', 'Zoolander'; The cast elevate this film up a lot. Everything else to do with 
```

### Hoppers

```text
Title: Hoppers
Overview: Scientists have discovered how to 'hop' human consciousness into lifelike robotic animals, allowing people to communicate with animals as animals. Animal lover Mabel seizes an opportunity to use the technology, uncovering mysteries within the animal world beyond anything she could have imagined.
Genres: Adventure; Animation; Comedy; Family; Science Fiction
Keywords: human vs nature; spy; beaver; oregon, usa; transhumanism; consciousness; aftercreditsstinger; duringcreditsstinger; human becoming an animal; robotic animal; 3d animation
Review language: I REALLY liked this movie, the best movie in years for me. To be fair, maybe that's just because of me being an environmentalist myself, but hey, in the end, reviews here are supposed to be subjective, I guess! Mabel, the main character, is a lovely main character. An environmentalist since ch; The town’s mayor wants
```

### Apex

```text
Title: Apex
Overview: A grieving woman pushing her limits on a solo adventure in the wild is ensnared in a twisted game with a cunning killer who thinks she's prey.
Genres: Action; Thriller
Keywords: canoe trip; rock climbing; nutcase; survival instinct; wilderness; missing persons case; cat‑and‑mouse chase; dangerous threat; trollveggen mountains, norway; extreme challenges; deadly fall; grieving woman; wandarra national park,; crossbow hunting; investigation resolution
Review language: Predictible. A silly and boring film, unfortunately. It could have been interesting but the landscape was the most interesting thing in the film.; This might not be the most original thing in the world, but it is executed well and the characters are relatable enough to build tension. That makes the moments of fright more engaging and leaves you ultimately quite satisfied. The casting is fantastic, and of
```

### Avatar: Fire and Ash

```text
Title: Avatar: Fire and Ash
Overview: In the wake of the devastating war against the RDA and the loss of their eldest son, Jake Sully and Neytiri face a new threat on Pandora: the Ash People, a violent and power-hungry Na'vi tribe led by the ruthless Varang. Jake's family must fight for their survival and the future of Pandora in a conflict that pushes them to their emotional and physical limits.
Genres: Science Fiction; Adventure; Fantasy
Keywords: witch; clone; space war; tribe; sequel; alien; transhumanism; family; space adventure; motion capture; family dynamics; rival; dreary; oscar winner
Review language: FULL SPOILER-FREE REVIEW @ "Avatar: Fire and Ash leaves me with mixed feelings of technical admiration and creative exhaustion. It's a film that lives of; One of the first things that stood out to me was how feminine this film felt in the best possible way. Almost every major fema
```

### Nymphomaniac: Vol. II

```text
Title: Nymphomaniac: Vol. II
Overview: The continuation of Joe's sexually dictated life delves into the darker aspects of her adult life and what led to her being in Seligman's care.
Genres: Drama; Mystery
Keywords: bondage; whip; sexuality; masochism; sex therapy; chapter; sadomasochism; sequel; sexual violence; love; nymphomaniac; tragedy; loneliness; masturbation; therapy; softcore; bdsm; addict; sexual pleasure; sexually aggressive woman; pleasure; virginity; abortion; sex; cruel
```

### Michael

```text
Title: Michael
Overview: The story of Michael Jackson, one of the most influential artists the world has ever known, and his life beyond the music. His journey from the discovery of his extraordinary talent as the lead of the Jackson Five, to the visionary artist whose creative ambition fueled a relentless pursuit to become the biggest entertainer in the world, highlighting both his life off-stage and some of the most iconic performances from his early solo career.
Genres: Music; Drama
Keywords: child abuse; sibling relationship; 1970s; abusive father; ambition; biography; singer; period drama; music history; price of fame; singer-songwriter; rise to fame; 1980s; 1960s; music; sentimental; approving; biographical drama; music drama; biopic
Review language: Given the whole slew of Jackson's involved in this stylish production, I couldn't help but feel a bit disappointed by the hollowness 
```

### The Matrix

```text
Title: The Matrix
Overview: Set in the 22nd century, The Matrix tells the story of a computer hacker who joins a group of underground insurgents fighting the vast and powerful computers who now rule the earth.
Genres: Action; Science Fiction
Keywords: man vs machine; martial arts; kung fu; dreams; fortune teller; artificial intelligence (a.i.); saving the world; hacker; self sacrifice; virtual reality; fight; prophecy; truth; philosophy; dystopia; allegory; insurgence; chosen one; pill; simulated reality; cyberpunk; dream world; action hero; gnosticism; oracle; soothsayer; commanding; allegory of the cave; dystopian
Review language: The Martix is a great example of a movie that will live for ever or a very log time. The story and concept are out of this world. plays his role with utter brilliance, the cast was very well put together and the graphics are still to this day amazing. All in 
```

### GOAT

```text
Title: GOAT
Overview: A small goat with big dreams gets a once-in-a-lifetime shot to join the pros and play roarball, a high-intensity, co-ed, full-contact sport dominated by the fastest, fiercest animals in the world.
Genres: Animation; Comedy; Family
Keywords: underdog; friendship; sports; allies; ambition; bullying; challenge; basketball; rivalry; coming of age; anthropomorphism; animals; intimidation; stereotype; basketball team; basketball player; big dreams; imaginary world; sneakers; vulture; playful; talking animal; anthropomorphic animal; athlete; inspirational; enemy; courage; feel good; heartwarming; exciting; fantasy sports; admiring; amused; joyful; sports comedy
Review language: The “Thorns” aren’t doing so well in the “Roarball” league despite the presence of the legendary leopard “Jet” so when owner “Flo” sees a video of the feisty young goat “Will” giving one of the spor
```

### Damage

```text
Title: Damage
Overview: The life of a respected politician at the height of his career crumbles when he becomes obsessed with his son's lover.
Genres: Drama; Romance
Keywords: london, england; sibling relationship; loss of loved one; longing; politician; scandal; in flagranti; femme fatale; older man younger woman relationship; extramarital affair; voyeur; father son relationship; tory politician
Review language: Harrowing movie, and have great chemistry. It’s a very ‘arty’ film . But it leaves you feeling queasy when you realize the consequences. Good show!; is a happily married (to ) government minister who meets his usually quite rakish son's () latest girlfriend (). The two click immediately - and soon they are doing a lot more than just clicking. That's about it - they carry out the
```

### After We Fell

```text
Title: After We Fell
Overview: Just as Tessa's life begins to become unglued, nothing is what she thought it would be. Not her friends nor her family. The only person that she should be able to rely on is Hardin, who is furious when he discovers the massive secret that she's been keeping. Before Tessa makes the biggest decision of her life, everything changes because of revelations about her family.
Genres: Drama; Romance
Keywords: based on novel or book; family history; love; teenage crush; woman director; family tension; zealous; mischievous; enraged
Review language: I seem to recall seeing the previous episode of this trilogy in the cinema - a beneficiary of the lockdown dearth that propelled some serious dross onto the big screen. This, mercifully, never found a home there and so could be watched, half-heartedly, from the comfort of my own living room. The rat; This is the point in t
```

### Zootopia 2

```text
Title: Zootopia 2
Overview: After cracking the biggest case in Zootopia's history, rookie cops Judy Hopps and Nick Wilde find themselves on the twisting trail of a great mystery when Gary De'Snake arrives and turns the animal metropolis upside down. To crack the case, Judy and Nick must go undercover to unexpected new parts of town, where their growing partnership is tested like never before.
Genres: Action; Adventure; Animation; Comedy; Family; Mystery
Keywords: snake; bunny; fox; cop; sequel; anthropomorphism; animals; displacement; buddy cop; buddy comedy; complex; playful; talking animal; 3d animation; embarrassed; excited
Review language: FULL SPOILER-FREE REVIEW @ Rating: A- "Zootopia 2 asserts itself as a worthy and even necessary sequel, overcoming the natural loss of the novelty factor with a narrative that dares; Very short: Zootopia 2 is lovely and has as many little details a
```

### Ready or Not: Here I Come

```text
Title: Ready or Not: Here I Come
Overview: Moments after surviving an all-out attack from the Le Domas family, Grace discovers she’s reached the next level of the nightmarish game — and this time with her estranged sister Faith at her side. Grace has one chance to survive, keep her sister alive, and claim the High Seat of the Council that controls the world. Four rival families are hunting her for the throne, and whoever wins rules it all.
Genres: Thriller; Horror; Comedy
Keywords: escape; ritual; dark comedy; satanism; gore; pact with the devil; sequel; game; exploding body; satanic ritual; satanic cult; estranged sister; sister sister relationship; satanic; sisters; death game; horror
Review language: Having barely escaped from her nuptials alive, poor old “Grace” () finds her recovery in hospital brought to quite a violent end. It seems that the fraternity that had assumed she would b
```

### Interstellar

```text
Title: Interstellar
Overview: The adventures of a group of explorers who make use of a newly discovered wormhole to surpass the limitations on human space travel and conquer the vast distances involved in an interstellar voyage.
Genres: Adventure; Drama; Science Fiction
Keywords: rescue; future; spacecraft; race against time; artificial intelligence (a.i.); nasa; time warp; dystopia; expedition; time travel; space travel; wormhole; famine; hibernation; black hole; quantum mechanics; family relationships; love; space; apocalypse; planet; robot; astronaut; scientist; single father; farmer; space station; gravity; space adventure; quest; time paradox; time-manipulation; cryonics; father daughter relationship; 2060s
Review language: Well, one off from two of this year's most expected movies alongside 'The Battle of Five Armies'. Like all the Chris Nolan fans, I was equally excited to see the
```

### The Shawshank Redemption

```text
Title: The Shawshank Redemption
Overview: Imprisoned in the 1940s for the double murder of his wife and her lover, upstanding banker Andy Dufresne begins a new life at the Shawshank prison, where he puts his accounting skills to work for an amoral warden. During his long stretch in prison, Dufresne comes to be admired by the other inmates -- including an older prisoner named Red -- for his integrity and unquenchable sense of hope.
Genres: Drama; Crime
Keywords: prison; friendship; police brutality; corruption; based on novel or book; freedom; hope; prison cell; delinquent; redemption; parole board; prison escape; wrongful imprisonment; interracial friendship; framed for murder; 1940s; voiceover
Review language: very good movie 9.5/10 محمد الشعراوى; Some birds aren't meant to be caged. The Shawshank Redemption is written and directed by . It is an adaptation of the Stephen King novella Ri
```

### Shelter

```text
Title: Shelter
Overview: A man living in self-imposed exile on a remote island rescues a young girl from a violent storm, setting off a chain of events that forces him out of seclusion to protect her from enemies tied to his past.
Genres: Action; Crime; Thriller
Keywords: home invasion; mysterious girl; ghosts of the past; isolated island; child protection; suspenseful; solitary life; hidden secret; close‑quarters combat; dangerous escalation; storm rescue; unexpected threat; violent assault; protective instinct; lone protector; intense action thriller; siege situation
Review language: I’ve always loved the idea of living in a remote Scottish lighthouse where the weather could close in and cut me off from everything and everyone - with, of course, wifi and all the conveniences of home. “Mason” () has had the same idea - only minus the mod cons, and survives frugally t; "Shelter" is, with
```

## Example Nearest-Neighbor Pairs

| Source | Neighbor | Similarity |
| --- | --- | ---: |
| Spider-Man: No Way Home | Spider-Man: Homecoming | 0.690 |
| Spider-Man | Spider-Man: Homecoming | 0.616 |
| Spider-Man: No Way Home | Spider-Man | 0.613 |
| Spider-Man: Homecoming | Spider-Man: Across the Spider-Verse | 0.545 |
| Spider-Man: No Way Home | Spider-Man: Across the Spider-Verse | 0.542 |
| The Lord of the Rings: The Return of the King | The Lord of the Rings: The Fellowship of the Ring | 0.505 |
| The Dark Knight | The Batman | 0.504 |
| Spider-Man | Spider-Man: Across the Spider-Verse | 0.488 |
| Harry Potter and the Philosopher's Stone | Harry Potter and the Chamber of Secrets | 0.479 |
| The Devil Wears Prada | The Devil Wears Prada 2 | 0.446 |

## Notes

- This report is a data-quality proof, not the final public website output.
- Raw TMDb responses live under data/cache/ and data/raw/, which are gitignored.
- Review language in profiles is truncated for semantic experimentation.
- OpenAI embeddings and cluster labeling are out of scope for Milestone 1.
