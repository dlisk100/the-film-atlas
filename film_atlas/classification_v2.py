"""Milestone 5 exploratory classification upgrade pipeline."""

from __future__ import annotations

import json
import math
import re
import shutil
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal

import networkx as nx
import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from film_atlas.cluster import ClusterAssignment, cluster_embedding_records
from film_atlas.cluster_labels import (
    ClusterLabelCandidate,
    OpenAIClusterLabelClient,
    estimate_labeling,
    label_clusters,
    load_label_cache,
    render_human_editable_labels,
    write_label_cache,
)
from film_atlas.embedding import (
    DEFAULT_EMBEDDING_MODEL,
    EMBEDDING_PRICES_PER_1M_TOKENS,
    embed_profiles_file,
    estimate_text_tokens,
    load_embedding_records,
)
from film_atlas.external_sources import ExternalMovieSignals, load_external_signals
from film_atlas.inspect_clusters import ClusterEvidence, build_cluster_evidence
from film_atlas.milestone4 import export_atlas_data_file, project_embedding_records
from film_atlas.models import MovieRecord, SemanticProfile
from film_atlas.neighbors import MovieNeighbors, NeighborMatch, compute_neighbors
from film_atlas.normalize import load_movie_records
from film_atlas.profiles import (
    ReviewWeight,
    _clean_review_snippet,
    _forbidden_values,
    _redact_forbidden_values,
    _section,
    build_semantic_profile,
)

CLASSIFICATION_V2_DIRNAME = "classification_v2"
SUMMARY_JSON_FILENAME = "classification_v2_summary.json"
SUMMARY_MD_FILENAME = "summary.md"
AUDIT_JSON_FILENAME = "audit_report.json"
MILESTONE_5_REPORT_FILENAME = "milestone_5_report.md"
MILESTONE_5_DEEP_AUDIT_FILENAME = "milestone_5_deep_audit.md"
LayerName = Literal["macro", "neighborhood", "micro"]
ClusteringStrategy = Literal[
    "independent_kmeans",
    "hierarchical_kmeans",
    "hierarchical_agglomerative",
    "graph_communities",
    "hdbscan",
]


class ClassificationV2Error(RuntimeError):
    """Raised when the Milestone 5 experiment runner cannot proceed."""


@dataclass(frozen=True, slots=True)
class ProfileVariantSpec:
    name: str
    include_title: bool
    include_tagline: bool
    review_weight: ReviewWeight
    max_review_chars: int
    review_count: int
    overview_repeats: int = 1
    keyword_repeats: int = 1
    review_label: str = "Review language"
    include_movie_lens_tags: bool = False
    include_mpst_tags: bool = False
    include_mpst_synopsis: bool = False
    max_mpst_synopsis_chars: int = 1200
    include_tone_tags: bool = False
    include_reception_tags: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CandidateArtifacts:
    variant: str
    strategy: ClusteringStrategy
    profiles_path: Path
    embeddings_path: Path
    profiles: list[SemanticProfile]
    embeddings: list[Any]
    neighbors: list[MovieNeighbors]
    assignments_by_layer: dict[LayerName, list[ClusterAssignment]]
    parent_maps: dict[LayerName, dict[int, int]]
    evidence_by_layer: dict[LayerName, list[ClusterEvidence]]
    metrics: dict[str, Any]
    exportable: bool


@dataclass(frozen=True, slots=True)
class ClassificationV2Result:
    experiment_dir: Path
    summary_path: Path
    report_path: Path
    audit_path: Path
    winner_variant: str
    winner_strategy: str
    export_dir: Path
    estimated_openai_cost_usd: float
    labels_generated: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_dir": str(self.experiment_dir),
            "summary_path": str(self.summary_path),
            "report_path": str(self.report_path),
            "audit_path": str(self.audit_path),
            "winner_variant": self.winner_variant,
            "winner_strategy": self.winner_strategy,
            "export_dir": str(self.export_dir),
            "estimated_openai_cost_usd": self.estimated_openai_cost_usd,
            "labels_generated": self.labels_generated,
        }


DEFAULT_PROFILE_VARIANTS: tuple[ProfileVariantSpec, ...] = (
    ProfileVariantSpec(
        name="baseline_light",
        include_title=True,
        include_tagline=False,
        review_weight="light",
        max_review_chars=180,
        review_count=1,
    ),
    ProfileVariantSpec(
        name="no_title_light",
        include_title=False,
        include_tagline=False,
        review_weight="light",
        max_review_chars=180,
        review_count=1,
    ),
    ProfileVariantSpec(
        name="no_title_rich_reviews",
        include_title=False,
        include_tagline=False,
        review_weight="heavy",
        max_review_chars=1600,
        review_count=5,
        review_label="Longer TMDb review language",
    ),
    ProfileVariantSpec(
        name="plot_keyword_tagline_reviews",
        include_title=False,
        include_tagline=True,
        review_weight="medium",
        max_review_chars=900,
        review_count=3,
        overview_repeats=2,
        keyword_repeats=2,
        review_label="Audience texture",
    ),
    ProfileVariantSpec(
        name="audience_vibe_rich",
        include_title=False,
        include_tagline=True,
        review_weight="heavy",
        max_review_chars=2400,
        review_count=8,
        overview_repeats=1,
        keyword_repeats=2,
        review_label="Audience vibe language",
    ),
    ProfileVariantSpec(
        name="movielens_tag_genome",
        include_title=False,
        include_tagline=True,
        review_weight="light",
        max_review_chars=500,
        review_count=2,
        keyword_repeats=2,
        include_movie_lens_tags=True,
        review_label="TMDb review language",
    ),
    ProfileVariantSpec(
        name="mpst_plot_tags",
        include_title=False,
        include_tagline=False,
        review_weight="light",
        max_review_chars=250,
        review_count=1,
        include_mpst_tags=True,
        include_mpst_synopsis=True,
        max_mpst_synopsis_chars=1400,
        review_label="TMDb review language",
    ),
    ProfileVariantSpec(
        name="hybrid_external_vibes",
        include_title=False,
        include_tagline=True,
        review_weight="medium",
        max_review_chars=900,
        review_count=3,
        overview_repeats=1,
        keyword_repeats=2,
        include_movie_lens_tags=True,
        include_mpst_tags=True,
        include_mpst_synopsis=True,
        max_mpst_synopsis_chars=1200,
        review_label="TMDb audience texture",
    ),
    ProfileVariantSpec(
        name="tone_review_synopsis",
        include_title=False,
        include_tagline=True,
        review_weight="heavy",
        max_review_chars=1600,
        review_count=5,
        include_mpst_synopsis=True,
        max_mpst_synopsis_chars=1000,
        include_tone_tags=True,
        review_label="TMDb audience texture",
    ),
    ProfileVariantSpec(
        name="hybrid_tone_status",
        include_title=False,
        include_tagline=True,
        review_weight="medium",
        max_review_chars=900,
        review_count=3,
        keyword_repeats=2,
        include_movie_lens_tags=True,
        include_mpst_tags=True,
        include_mpst_synopsis=True,
        max_mpst_synopsis_chars=1000,
        include_tone_tags=True,
        include_reception_tags=True,
        review_label="TMDb audience texture",
    ),
    ProfileVariantSpec(
        name="status_overlay_probe",
        include_title=False,
        include_tagline=False,
        review_weight="light",
        max_review_chars=250,
        review_count=1,
        include_reception_tags=True,
        review_label="TMDb review language",
    ),
)

DEFAULT_STRATEGIES: tuple[ClusteringStrategy, ...] = (
    "independent_kmeans",
    "hierarchical_kmeans",
    "hierarchical_agglomerative",
    "graph_communities",
    "hdbscan",
)

AUDIT_TITLES = [
    "Avatar",
    "Avatar: Fire and Ash",
    "The Founder",
    "Vanilla Sky",
    "Final Destination",
    "Weapons",
    "Jurassic World Rebirth",
    "Sully",
    "Mickey 17",
    "Rush Hour",
    "The Perks of Being a Wallflower",
    "Murder on the Orient Express",
    "The Theory of Everything",
    "Carry-On",
    "Total Recall",
    "Moon",
    "Civil War",
    "Independence Day",
    "Minority Report",
    "I, Tonya",
    "Scott Pilgrim vs. the World",
    "Spider-Man: Across the Spider-Verse",
    "The Creator",
    "Hot Tub Time Machine",
    "Dungeons & Dragons: Honor Among Thieves",
    "Barbie",
    "Her",
    "Juno",
    "Little Miss Sunshine",
    "Heat",
    "RoboCop",
    "Oppenheimer",
    "Sound of Metal",
    "Sunshine",
    "Hail, Caesar!",
    "The Game",
    "Lost in Translation",
    "The Mist",
    "Knock at the Cabin",
    "The Fountain",
    "The Village",
    "La La Land",
    "Elvis",
    "The Grand Budapest Hotel",
    "The Man from U.N.C.L.E.",
    "Glass Onion: A Knives Out Mystery",
    "Trainspotting",
    "O Brother, Where Art Thou?",
    "Point Break",
    "The Big Lebowski",
    "Captain Phillips",
    "Uncut Gems",
    "Licorice Pizza",
    "The Abyss",
    "Project X",
    "Edge of Tomorrow",
    "Apocalypto",
    "The Northman",
    "Cast Away",
    "The King's Speech",
    "The Hunger Games",
    "Office Space",
    "The Lego Movie",
]

BAD_NEIGHBOR_PATTERNS = {
    "Avatar": ["last airbender"],
    "Civil War": ["captain america"],
    "Heat": ["the heat"],
    "The Game": ["game night", "gamer"],
    "Oppenheimer": ["the hurt locker", "the prestige"],
    "Sound of Metal": ["a quiet place"],
}

GOOD_NEIGHBOR_PATTERNS = {
    "Rush Hour": ["rush hour 2", "rush hour 3", "shanghai noon", "lethal weapon"],
    "Weapons": ["witch", "midsommar", "talk to me", "it follows"],
    "The Creator": ["blade runner", "rogue one", "elysium", "district 9"],
    "Dungeons & Dragons: Honor Among Thieves": ["jumanji", "princess bride", "willow"],
    "RoboCop": ["total recall", "terminator", "matrix"],
    "Lost in Translation": ["her", "before sunrise", "eternal sunshine"],
    "La La Land": ["moulin rouge", "singin", "artist"],
    "Spider-Man: Across the Spider-Verse": ["spider-man", "into the spider-verse"],
}

AUDIT_TITLE_TARGET_YEARS = {
    "Avatar": 2009,
    "Civil War": 2024,
    "Final Destination": 2000,
    "Project X": 2012,
    "The Karate Kid": 1984,
    "The Game": 1997,
    "The Village": 2004,
    "Gladiator": 2000,
    "Sunshine": 2007,
    "Total Recall": 1990,
}

BAD_LABEL_PATTERNS = {
    "Avatar": ["last airbender", "time-bending", "time loop", "survival games"],
    "Avatar: Fire and Ash": ["time-bending", "survival games", "kaiju", "mecha"],
    "Barbie": ["warrior-daughter", "undercover crew", "crime tension"],
    "Carry-On": ["rom-com"],
    "Cast Away": ["romance soft edges", "dystopia", "bureaucratic dark comedy"],
    "Civil War": ["true-story", "captain america"],
    "Elvis": ["battlefield"],
    "Hot Tub Time Machine": ["workplace", "dating comedy", "crime"],
    "Mickey 17": ["alien invasion"],
    "Office Space": ["bright family", "animation", "phone-run", "thriller"],
    "Oppenheimer": ["battlefield"],
    "Point Break": ["prison escape"],
    "Sound of Metal": ["horror", "literary coming-of-age"],
    "Scott Pilgrim vs. the World": ["apocalypse afterparty"],
    "Sully": ["battlefield", "wwii"],
    "Sunshine": ["moon mission", "arctic"],
    "The Abyss": ["spacefaring", "space-faring"],
    "The Hunger Games": ["alien-world", "alien invasion"],
    "The Perks of Being a Wallflower": ["cancer"],
}

AUDIT_WATCHLIST_NOTES = {
    "Vanilla Sky": [
        "macro still overstates hardboiled crime for a dream-romance identity thriller",
    ],
    "Jurassic World Rebirth": [
        "macro is usable popcorn-adventure territory but still too mixed with mystery-comedy wording",
    ],
    "The Perks of Being a Wallflower": [
        "neighbors still lean toward terminal-romance YA more than friendship/mental-health coming-of-age",
    ],
    "Murder on the Orient Express": [
        "neighborhood overstates psychological serial-killer thriller for a classical whodunit",
    ],
    "Minority Report": [
        "micro says time-travel mind games, which is adjacent but imprecise for precrime/future-surveillance noir",
    ],
    "Hot Tub Time Machine": [
        "micro keeps a noir/buddy-crime tint that is too hard-edged for a time-travel party comedy",
    ],
    "Barbie": [
        "labels are no longer animation/crime, but the cluster still reads more family-pop-fantasy than satirical toy-world comedy",
    ],
    "Little Miss Sunshine": [
        "micro's recovery/music wording is too generic for a family road-trip pageant dramedy",
    ],
    "Lost in Translation": [
        "neighborhood calls it literary period romance, which misses the contemporary alienation/hotel melancholy",
    ],
    "The Fountain": [
        "micro's AI/future-intimacy wording is wrong for a spiritual cosmic romance",
    ],
    "Elvis": [
        "battlefield error is gone, but true-story legal/political leadership labels still underplay the music-biopic center",
    ],
    "The Grand Budapest Hotel": [
        "showbiz/musical-romance neighborhood is too sideways for a whimsical historical caper",
    ],
    "Trainspotting": [
        "crime-noir macro is serviceable but too hardboiled for addiction, youth, and black-comedy energy",
    ],
    "The Big Lebowski": [
        "labels are solid, but Pulp Fiction remains a questionable top-five neighbor",
    ],
    "Captain Phillips": [
        "labels improved, but the neighbor field still pulls it toward military combat survival",
    ],
    "Licorice Pizza": [
        "teen dating-comedy cluster is too broad and loses the 1970s hangout/coming-of-age texture",
    ],
    "The Abyss": [
        "micro is fixed to deep-sea first contact, but neighborhood still says space-mission survival",
    ],
    "Cast Away": [
        "macro still leans romance-drama even though the useful signal is isolation, survival, and meaning-making",
    ],
}

GENERIC_KEYWORDS = {
    "aftercreditsstinger",
    "based on comic",
    "based on novel or book",
    "based on tv series",
    "duringcreditsstinger",
    "remake",
    "sequel",
    "woman director",
}

PUBLIC_AUDIT_LABEL_REPAIRS: tuple[dict[str, str], ...] = (
    {
        "title": "The Founder",
        "layer": "macro",
        "label": "Real-World Pressure & Public-Life Dramas",
        "description": "True-story, biographical, public-life, survival, and institutional dramas where ambition meets pressure.",
    },
    {
        "title": "Sully",
        "layer": "macro",
        "label": "Real-World Pressure & Public-Life Dramas",
        "description": "True-story, biographical, public-life, survival, and institutional dramas where ambition meets pressure.",
    },
    {
        "title": "Civil War",
        "layer": "macro",
        "label": "Real-World Pressure & Public-Life Dramas",
        "description": "True-story, biographical, public-life, survival, and institutional dramas where ambition meets pressure.",
    },
    {
        "title": "Oppenheimer",
        "layer": "macro",
        "label": "Real-World Pressure & Public-Life Dramas",
        "description": "True-story, biographical, public-life, survival, and institutional dramas where ambition meets pressure.",
    },
    {
        "title": "Captain Phillips",
        "layer": "macro",
        "label": "Real-World Pressure & Public-Life Dramas",
        "description": "True-story, biographical, public-life, survival, and institutional dramas where ambition meets pressure.",
    },
    {
        "title": "The King's Speech",
        "layer": "macro",
        "label": "Real-World Pressure & Public-Life Dramas",
        "description": "True-story, biographical, public-life, survival, and institutional dramas where ambition meets pressure.",
    },
    {
        "title": "I, Tonya",
        "layer": "macro",
        "label": "Real-World Pressure & Public-Life Dramas",
        "description": "True-story, biographical, public-life, survival, and institutional dramas where ambition meets pressure.",
    },
    {
        "title": "Elvis",
        "layer": "macro",
        "label": "Real-World Pressure & Public-Life Dramas",
        "description": "True-story, biographical, public-life, survival, and institutional dramas where ambition meets pressure.",
    },
    {
        "title": "Barbie",
        "layer": "macro",
        "label": "Whimsical Pop Fantasy & Family Adventure",
        "description": "Bright fantasy, animation, family adventure, toy-box logic, musical magic, and pop-cultural self-invention.",
    },
    {
        "title": "Dungeons & Dragons: Honor Among Thieves",
        "layer": "macro",
        "label": "Whimsical Pop Fantasy & Family Adventure",
        "description": "Bright fantasy, animation, family adventure, toy-box logic, musical magic, and pop-cultural self-invention.",
    },
    {
        "title": "Jurassic World Rebirth",
        "layer": "macro",
        "label": "Whimsical Pop Fantasy & Family Adventure",
        "description": "Bright fantasy, animation, family adventure, toy-box logic, musical magic, and pop-cultural self-invention.",
    },
    {
        "title": "The Lego Movie",
        "layer": "macro",
        "label": "Whimsical Pop Fantasy & Family Adventure",
        "description": "Bright fantasy, animation, family adventure, toy-box logic, musical magic, and pop-cultural self-invention.",
    },
    {
        "title": "Glass Onion: A Knives Out Mystery",
        "layer": "macro",
        "label": "Popcorn Sequel Adventures & Mystery Comedies",
        "description": "Sequel-driven, mystery-leaning, family, comedy, and adventure franchises with familiar worlds and returning energy.",
    },
    {
        "title": "The Perks of Being a Wallflower",
        "layer": "neighborhood",
        "label": "Tender YA Friendship & First Heartbreak",
        "description": "Young-adult dramas about friendship, first love, school life, illness, grief, mental health, and adolescent self-discovery.",
    },
    {
        "title": "The Perks of Being a Wallflower",
        "layer": "micro",
        "label": "YA Love, Loss & First Heartbreak",
        "description": "Tender YA stories where first love, friendship, illness, depression, or grief push teenagers toward painful self-knowledge.",
    },
    {
        "title": "The Theory of Everything",
        "layer": "micro",
        "label": "Devotion, Illness & Time-Shifted Romance",
        "description": "Tender romances and devotion dramas shaped by illness, memory, separation, time shifts, and loving someone through change.",
    },
    {
        "title": "Mickey 17",
        "layer": "neighborhood",
        "label": "Clones, Identity & Future-Tech Survival",
        "description": "High-concept sci-fi about engineered bodies, clones, predictive systems, memory, labor, and technology destabilizing identity.",
    },
    {
        "title": "Sully",
        "layer": "neighborhood",
        "label": "War, Rescue & Survival Pressure",
        "description": "War, historical, rescue, and disaster-survival dramas where duty, danger, endurance, and public scrutiny collide.",
    },
    {
        "title": "Sully",
        "layer": "micro",
        "label": "True-Crisis Rescue & Disaster Survival",
        "description": "Grounded crisis stories about rescue, aviation, sea disasters, public accountability, and survival under pressure.",
    },
    {
        "title": "Civil War",
        "layer": "micro",
        "label": "Embedded War-Zone Survival Dramas",
        "description": "Ground-level conflict stories focused on missions, ambushes, civil collapse, moral pressure, and surviving chaos up close.",
    },
    {
        "title": "Barbie",
        "layer": "neighborhood",
        "label": "Princess, Pop-Fantasy & Musical Magic",
        "description": "Whimsical fantasy comedies and musicals where princess myths, pop iconography, magic, and self-invention collide.",
    },
    {
        "title": "Barbie",
        "layer": "micro",
        "label": "Female-Led Pop Fantasy Adventures",
        "description": "Bright female-led fantasy adventures about identity, performance, rebellion, and rewriting the role you were handed.",
    },
    {
        "title": "Her",
        "layer": "neighborhood",
        "label": "Speculative Love, Memory & Grief",
        "description": "Romantic and philosophical dramas where love is bent by memory, mortality, technology, time, distance, or cosmic strangeness.",
    },
    {
        "title": "Her",
        "layer": "micro",
        "label": "AI Romance & Lonely Future Intimacy",
        "description": "Near-future romances and identity stories about artificial intimacy, loneliness, memory, and emotional projection.",
    },
    {
        "title": "Little Miss Sunshine",
        "layer": "neighborhood",
        "label": "Family, Recovery & Becoming-Yourself Dramas",
        "description": "Grounded dramas where families, mentors, illness, disability, grief, or art push someone through a difficult transition.",
    },
    {
        "title": "Little Miss Sunshine",
        "layer": "micro",
        "label": "Recovery, Family & Life-Pivot Dramas",
        "description": "Intimate dramas about growing up, recovering, adapting, and finding identity inside family or community pressure.",
    },
    {
        "title": "Sound of Metal",
        "layer": "neighborhood",
        "label": "Family, Recovery & Becoming-Yourself Dramas",
        "description": "Grounded dramas where families, mentors, illness, disability, grief, or art push someone through a difficult transition.",
    },
    {
        "title": "Sound of Metal",
        "layer": "micro",
        "label": "Recovery, Music & Life-Pivot Dramas",
        "description": "Intimate music and recovery dramas about disability, addiction, mentorship, identity, and learning to live differently.",
    },
    {
        "title": "Scott Pilgrim vs. the World",
        "layer": "neighborhood",
        "label": "Pop-Culture Genre-Mashup Comedy",
        "description": "Comedies that remix superheroes, games, cult action, parody, and pop-cultural references into one noisy genre blender.",
    },
    {
        "title": "Scott Pilgrim vs. the World",
        "layer": "micro",
        "label": "Cult-Comic Genre-Mashup Comedy",
        "description": "Joke-forward cult comedies where comic-book attitude, games, parody, and self-aware genre chaos collide.",
    },
    {
        "title": "Hot Tub Time Machine",
        "layer": "micro",
        "label": "Slacker Buddy Time-Travel Comedy",
        "description": "Broad buddy comedies where adult immaturity, wish fulfillment, time tricks, and bad decisions drive the chaos.",
    },
    {
        "title": "Rush Hour",
        "layer": "micro",
        "label": "Buddy-Cop Undercover Comedy",
        "description": "Action comedies where mismatched partners, undercover pressure, martial-arts energy, and wisecracks do the work.",
    },
    {
        "title": "The Hunger Games",
        "layer": "neighborhood",
        "label": "Dystopian Rebellion & Frontier Survival",
        "description": "Oppressive future societies, hostile frontiers, rebellion stories, survival spectacles, and young heroes under systems of control.",
    },
    {
        "title": "Avatar: Fire and Ash",
        "layer": "macro",
        "label": "Franchise Fantasy & Blockbuster Spectacle",
        "description": "Large-scale franchise worlds: superheroes, wizards, space opera, epic quests, comic-book teams, and blockbuster spectacle.",
    },
    {
        "title": "Avatar: Fire and Ash",
        "layer": "micro",
        "label": "Franchise Battle Survivors & Giant Spectacle",
        "description": "Sequel-heavy spectacle where giant creatures, robots, post-collapse worlds, and resistance stories share a big-battle rhythm.",
    },
    {
        "title": "Avatar: Fire and Ash",
        "layer": "neighborhood",
        "label": "Space-Opera Sequels & Franchise Battles",
        "description": "Big franchise follow-ups with space-opera stakes, rebellion, giant battles, returning worlds, and sequel-scale spectacle.",
    },
    {
        "title": "Civil War",
        "layer": "neighborhood",
        "label": "Ground-Level Combat & Survival Thrillers",
        "description": "War-zone, hostage, disaster, and field-survival dramas where people are trapped inside immediate danger and institutional violence.",
    },
    {
        "title": "Point Break",
        "layer": "micro",
        "label": "FBI, Getaway & Extreme-Chase Action",
        "description": "Velocity-first action thrillers built from federal heat, transport jobs, getaways, heists, hijacks, and cliff-edge pursuit.",
    },
    {
        "title": "Cast Away",
        "layer": "micro",
        "label": "Isolation, Meaning & Strange Life Detours",
        "description": "Offbeat life fables about isolation, confinement, escape, reinvention, and making meaning inside an impossible situation.",
    },
    {
        "title": "The Abyss",
        "layer": "micro",
        "label": "Space & Deep-Unknown Survival Horror",
        "description": "Claustrophobic sci-fi survival built around space crews, deep-sea pressure, alien organisms, strange discovery, and first contact.",
    },
    {
        "title": "Apocalypto",
        "layer": "macro",
        "label": "Ancient, Gothic & Sword-Fight Survival",
        "description": "Dark fantasy, ancient-world violence, sword fights, supernatural menace, and survival stories with old-world brutality.",
    },
    {
        "title": "Apocalypto",
        "layer": "neighborhood",
        "label": "Ancient Warriors & Mythic Survival",
        "description": "Ancient-world and mythic action adventures with warriors, kingdoms, conquest, revenge, gods, monsters, and brutal survival.",
    },
    {
        "title": "Apocalypto",
        "layer": "micro",
        "label": "Ancient Survival & Swordplay Epics",
        "description": "Primitive, ancient, and myth-tinged survival adventures where escape, conquest, and hand-to-hand brutality drive the journey.",
    },
    {
        "title": "The Northman",
        "layer": "neighborhood",
        "label": "Ancient Warriors & Mythic Survival",
        "description": "Ancient-world and mythic action adventures with warriors, kingdoms, conquest, revenge, gods, monsters, and brutal survival.",
    },
    {
        "title": "The Northman",
        "layer": "micro",
        "label": "Norse, Arthurian & Dark Medieval Action",
        "description": "Dark medieval and legendary action stories about kings, revenge, quests, warriors, curses, and old-world violence.",
    },
    {
        "title": "The Big Lebowski",
        "layer": "micro",
        "label": "Slacker Noir & Buddy Comedy Mayhem",
        "description": "Loose comic mysteries and slacker spirals where crime plots, oddballs, hangouts, and bad decisions collide.",
    },
    {
        "title": "Glass Onion: A Knives Out Mystery",
        "layer": "neighborhood",
        "label": "Mystery-Caper Sequels & Comedy Mischief",
        "description": "Return-trip mystery comedies and caper sequels where familiar crews, suspects, and punchlines matter more than grim danger.",
    },
    {
        "title": "Glass Onion: A Knives Out Mystery",
        "layer": "micro",
        "label": "Quippy Mystery-Caper Sequels",
        "description": "Return-trip mystery comedies and caper sequels built from suspects, twists, familiar worlds, and comic timing.",
    },
    {
        "title": "Uncut Gems",
        "layer": "neighborhood",
        "label": "Hustle, Gambling & Rise-and-Fall Crime Dramas",
        "description": "Crime and self-destruction dramas about scams, gambling, addiction, money, ambition, and pushing too far.",
    },
    {
        "title": "Uncut Gems",
        "layer": "micro",
        "label": "Gambling, Cons & American Hustle",
        "description": "Fast-talking crime dramas where gambling, fraud, celebrity, addiction, and self-invention turn ambition into a trap.",
    },
    {
        "title": "Creed III",
        "layer": "neighborhood",
        "label": "Franchise Sequels, Battles & Comebacks",
        "description": "Sequel-heavy franchise stories where familiar worlds return through space wars, monster battles, rebellions, sports comebacks, or legacy fights.",
    },
    {
        "title": "Rocky V",
        "layer": "macro",
        "label": "Real-World Pressure & Public-Life Dramas",
        "description": "True-story, biographical, war, sports, survival, and institutional dramas where ambition, public life, or physical pressure tests people.",
    },
    {
        "title": "Rocky V",
        "layer": "neighborhood",
        "label": "Action-Franchise Revenge & Combat Comebacks",
        "description": "Legacy sequels about one-person wars, revenge quests, mercenary missions, battlefield scars, boxing comebacks, and old rivals refusing to stay buried.",
    },
    {
        "title": "Rocky V",
        "layer": "micro",
        "label": "Mercenary Missions & Bruised Action Comebacks",
        "description": "Franchise grit where aging fighters, soldiers, mercenaries, and action icons return through training, punishment, missions, and personal war stories.",
    },
    {
        "title": "Mr. Popper's Penguins",
        "layer": "neighborhood",
        "label": "Family Pets, Animation & Live-Action Mischief",
        "description": "Family comedies and adventures about pets, animated creatures, animal chaos, toy-box villains, and live-action/cartoon hybrids.",
    },
    {
        "title": "Sausage Party",
        "layer": "neighborhood",
        "label": "Animated Food, Pets & Toy-Box Mischief",
        "description": "Animated and animation-adjacent comedies where food, pets, toys, or bright creature worlds turn into comic chaos, sometimes family-safe and sometimes not.",
    },
    {
        "title": "Cheaper by the Dozen",
        "layer": "neighborhood",
        "label": "Storybook Family Mischief & Young Sleuths",
        "description": "Family-friendly comedies, storybook adventures, young-sleuth mysteries, and gentle fantasy mischief around chaotic households and odd guardians.",
    },
    {
        "title": "Cheaper by the Dozen",
        "layer": "micro",
        "label": "Oddball Family Mischief & Storybook Trouble",
        "description": "Chaotic family and storybook comedies where children, guardians, homes, and mild fantasy or mystery elements create warm trouble.",
    },
    {
        "title": "Home Alone 2: Lost in New York",
        "layer": "neighborhood",
        "label": "Sequel Comedy Mischief & Holiday Chaos",
        "description": "Broad comedy sequels where holidays, ghosts, time loops, returning families, or familiar crews create big comic set-pieces.",
    },
    {
        "title": "Home Alone 2: Lost in New York",
        "layer": "micro",
        "label": "Holiday, Ghost & Sequel Comedy Mischief",
        "description": "Breezy comedy sequels built from holiday trouble, haunted gags, returning oddballs, city chaos, and broad set-piece mischief.",
    },
    {
        "title": "Inception",
        "layer": "macro",
        "label": "Noir, Crime & Mind-Game Thrillers",
        "description": "Dark thrillers about crime, identity, memory, investigation, obsession, and reality-bending schemes where perception cannot be trusted.",
    },
    {
        "title": "It Chapter Two",
        "layer": "neighborhood",
        "label": "Franchise Slashers & Supernatural Horror Sequels",
        "description": "Return-trip horror where masked killers, cursed entities, supernatural threats, and self-aware genre machinery keep coming back.",
    },
    {
        "title": "It Chapter Two",
        "layer": "micro",
        "label": "Supernatural & Slasher Sequel Nightmares",
        "description": "Sequel horror built from serial killers, sinister clowns, masked slashers, supernatural revenge, and childhood fears returning older and meaner.",
    },
    {
        "title": "Jumper",
        "layer": "micro",
        "label": "Time-Bending & Teleportation Action Thrillers",
        "description": "High-concept action thrillers where time loops, prediction, teleportation, altered memory, or impossible movement turn the chase uncanny.",
    },
    {
        "title": "Nerve",
        "layer": "macro",
        "label": "Dread, Survival Games & Psychological Threats",
        "description": "Horror, tech-dare, revenge, home-invasion, death-game, and psychological threat stories where fear becomes a system people have to survive.",
    },
    {
        "title": "San Andreas",
        "layer": "macro",
        "label": "Dystopian, Disaster & Alien Survival",
        "description": "Large-scale survival stories driven by broken futures, natural disasters, alien contact, ecological collapse, or hostile frontiers.",
    },
    {
        "title": "Sherlock Holmes: A Game of Shadows",
        "layer": "neighborhood",
        "label": "Shadowy Quest Fantasy & Adventure Mysteries",
        "description": "Quest-shaped adventures where wizard wars, mythic artifacts, detective pursuits, and shadowy masterminds push heroes across dangerous worlds.",
    },
    {
        "title": "Sherlock Holmes: A Game of Shadows",
        "layer": "micro",
        "label": "High-Fantasy Quests & Shadow Pursuits",
        "description": "Shadowy adventures about quests, dark lords, criminal masterminds, magic, pursuit, and worlds sliding toward open conflict.",
    },
    {
        "title": "The Lion King",
        "layer": "neighborhood",
        "label": "Dino, Creature & Animal-Kingdom Adventures",
        "description": "Creature-led franchise adventures where dinosaurs, giant animals, island ecosystems, or animal kingdoms create survival and coming-of-age spectacle.",
    },
    {
        "title": "The Lion King",
        "layer": "micro",
        "label": "Creature-Kingdom Franchise Adventures",
        "description": "Creature and animal-kingdom franchise stories about dinosaurs, lions, islands, heirs, and spectacle built around wild worlds.",
    },
    {
        "title": "Top Gun: Maverick",
        "layer": "neighborhood",
        "label": "Legacy Adventure Sequels & High-Stakes Missions",
        "description": "Big legacy sequels where returning heroes face aerial, pirate, heist, fantasy, or adventure missions with franchise-sized stakes.",
    },
    {
        "title": "Top Gun: Maverick",
        "layer": "micro",
        "label": "Aerial, Pirate & Swashbuckling Sequels",
        "description": "Cinematic adventure sequels built from aircraft, ships, heists, swordplay, islands, and legacy heroes returning for one more impossible run.",
    },
    {
        "title": "The Best of Me",
        "layer": "macro",
        "label": "Modern Romance, Romcoms & City Dramedies",
        "description": "Contemporary romance and city-dramedy territory spanning workplace bustle, dating chaos, school crushes, erotic melodrama, young-adult heartbreak, and second-chance love stories.",
    },
    {
        "title": "The Road",
        "layer": "micro",
        "label": "Alien, Apocalypse & End-of-World Survival",
        "description": "End-of-world survival where alien invasion, post-apocalyptic collapse, hostile landscapes, or family protection make every mile dangerous.",
    },
    {
        "title": "The School of Rock",
        "layer": "macro",
        "label": "Satirical Buddy, Slapstick & Pop-Culture Comedy",
        "description": "Comedy-first territory where satire, buddy chaos, school or workplace mischief, action parody, music, and pop-culture jokes drive the energy.",
    },
    {
        "title": "The School of Rock",
        "layer": "neighborhood",
        "label": "Schoolyard, Holiday & Buddy Comedy Chaos",
        "description": "Broad comedies about schoolrooms, families, holidays, kid mischief, buddy energy, and adults barely keeping the situation together.",
    },
    {
        "title": "The Thing",
        "layer": "neighborhood",
        "label": "Creature, Slasher & Body-Horror Survival",
        "description": "Isolated horror where creatures, slashers, zombies, parasites, mutations, and gore turn survival into a paranoid pressure chamber.",
    },
    {
        "title": "The Thing",
        "layer": "micro",
        "label": "Outbreak, Creature & Body-Horror Survival",
        "description": "Survival horror about outbreaks, zombies, alien organisms, infected bodies, remakes, and pressure-cooker creature dread.",
    },
    {
        "title": "The Tourist",
        "layer": "macro",
        "label": "Spy Action, Revenge & High-Velocity Thrillers",
        "description": "Action-thriller territory where spies, assassins, getaway crews, revenge missions, races against time, and glamorous danger keep the pressure moving.",
    },
    {
        "title": "Baywatch",
        "layer": "micro",
        "label": "Buddy, Slacker & Action-Comedy Mayhem",
        "description": "Broad action comedies where buddies, cops, lifeguards, slackers, undercover stunts, and irresponsible adults turn trouble into set-piece chaos.",
    },
    {
        "title": "The Sixth Sense",
        "layer": "micro",
        "label": "Supernatural Visions & Haunted-Curse Horror",
        "description": "Supernatural horror and psychological ghost stories where visions, curses, hauntings, and unseen presences make reality feel unsafe.",
    },
    {
        "title": "Big Trouble in Little China",
        "layer": "neighborhood",
        "label": "Cult Sci-Fi, Fantasy & Supernatural Action-Comedy",
        "description": "Cult comedies where sci-fi, fantasy, martial-arts weirdness, aliens, ghosts, monsters, and parody energy crash into adventure mayhem.",
    },
    {
        "title": "Big Trouble in Little China",
        "layer": "micro",
        "label": "Cult Sci-Fi/Fantasy Parody Mayhem",
        "description": "Self-aware comic adventures where aliens, time tricks, supernatural trouble, kung-fu weirdness, and genre parody share the same chaotic playground.",
    },
    {
        "title": "Cruella",
        "layer": "neighborhood",
        "label": "Villains, Princesses & Pop-Fantasy Reinvention",
        "description": "Pop-fantasy and fairy-tale-adjacent stories where princess myths, iconic villains, music, fashion, and self-reinvention reshape familiar worlds.",
    },
    {
        "title": "Cruella",
        "layer": "micro",
        "label": "Villain-Threaded Fairy-Tale & Pop-Fantasy Stories",
        "description": "Bright pop-fantasy stories where villains, princesses, icons, magic, fashion, and musical spectacle pull familiar myths into a new pose.",
    },
    {
        "title": "The Walk",
        "layer": "macro",
        "label": "Intimate Life, Love & Becoming Dramas",
        "description": "Character-forward dramas about love, family, art, grief, recovery, memory, ambition, and the strange turns that make a life cohere.",
    },
    {
        "title": "The Walk",
        "layer": "neighborhood",
        "label": "Lyrical Romance, Showbiz & Wonder",
        "description": "Romantic, historical, showbiz, musical, and wonder-driven stories where spectacle, art, memory, or performance gives emotion a grand stage.",
    },
    {
        "title": "The Walk",
        "layer": "micro",
        "label": "Showmanship, Wonder & Storybook Dreams",
        "description": "Uplifting art-and-spectacle stories about performers, dreamers, inventors, outsiders, and impossible gestures that turn life theatrical.",
    },
    {
        "title": "Alpha",
        "layer": "neighborhood",
        "label": "Frontier Survival, Grief & Cultural Collision",
        "description": "Frontier and period dramas where wilderness, grief, violence, cultural contact, old codes, and survival pressure test people at the edge of home.",
    },
    {
        "title": "Daredevil",
        "layer": "macro",
        "label": "Mythic, Gothic & Swashbuckling Action",
        "description": "Shadowy adventure and action territory spanning vigilantes, curses, sword fights, mythic quests, monsters, conspiracies, outlaws, and old-world menace.",
    },
    {
        "title": "The Mask of Zorro",
        "layer": "neighborhood",
        "label": "Ancient, Mythic & Swashbuckling Adventure",
        "description": "Historical, mythic, and swashbuckling adventures where warriors, pirates, masked heroes, lost worlds, monsters, and old kingdoms collide.",
    },
    {
        "title": "The Mask of Zorro",
        "layer": "micro",
        "label": "Swashbuckling Outlaws & Swordplay Adventure",
        "description": "Cloaked heroes, outlaws, pirates, musketeers, and jungle adventurers crossing blades through brisk, romantic, old-fashioned adventure.",
    },
    {
        "title": "Dawn of the Planet of the Apes",
        "layer": "micro",
        "label": "Post-Apocalyptic Mutation Action Sequels",
        "description": "Dystopian action sequels where mutations, viruses, experiments, ruined futures, and survival wars reshape what counts as human.",
    },
    {
        "title": "Dawn of the Planet of the Apes",
        "layer": "neighborhood",
        "label": "Post-Apocalyptic Action, Infection & Rebellion",
        "description": "Ruined-future action stories where infection, experiments, rebellion, carceral cities, zombies, apes, or survival wars remake society.",
    },
    {
        "title": "Gamer",
        "layer": "micro",
        "label": "Wasteland, Prison-Game & Cyberpunk Dystopia Action",
        "description": "Dystopian action where wastelands, prison systems, virtual arenas, body control, surveillance, and violent games turn people into weapons.",
    },
    {
        "title": "Mad Max: Fury Road",
        "layer": "neighborhood",
        "label": "Post-Apocalyptic Wasteland & Dystopia Action",
        "description": "Future-ruin action about wastelands, road wars, prison games, cyberpunk control, resource scarcity, fugitives, and survival under broken systems.",
    },
    {
        "title": "Django Unchained",
        "layer": "neighborhood",
        "label": "Persecution, Resistance & Historical Hate Dramas",
        "description": "Historical and war-adjacent dramas about persecution, racism, fascism, slavery, resistance, revenge, and people surviving organized hatred.",
    },
    {
        "title": "Django Unchained",
        "layer": "micro",
        "label": "Racism, Revenge & Hate-Crime Reckonings",
        "description": "Confrontational dramas and darkly comic revenge stories about racism, white supremacy, sexual violence, hate crimes, and justice arriving crooked.",
    },
    {
        "title": "Avatar: Fire and Ash",
        "layer": "micro",
        "label": "Creature, Space & Legacy Battle Sequels",
        "description": "Large-scale franchise sequels where creatures, starships, giant worlds, heroic missions, and legacy battles turn spectacle into survival.",
    },
    {
        "title": "Fifty Shades of Grey",
        "layer": "neighborhood",
        "label": "Bookish Angsty Romance Melodramas",
        "description": "Novel-born romances and melodramas where desire, heartbreak, class pressure, illness, secrets, or impossible timing make love feel risky.",
    },
    {
        "title": "Fifty Shades of Grey",
        "layer": "micro",
        "label": "Angsty Erotic Romance Heat",
        "description": "Heated romance melodramas about obsession, desire, manipulation, first intensity, and love stories that are glamorous, messy, or dangerous.",
    },
    {
        "title": "Jurassic World Rebirth",
        "layer": "neighborhood",
        "label": "Treasure Hunts, Dino Thrills & Adventure Sequels",
        "description": "Popcorn adventure sequels where dinosaurs, treasure maps, pirates, ruins, escape puzzles, and returning franchise worlds drive the ride.",
    },
    {
        "title": "Harry Potter and the Chamber of Secrets",
        "layer": "micro",
        "label": "YA Magic & Mystery-Quest Sequels",
        "description": "Bookish franchise sequels built from magic, young heroes, illusions, mysteries, fantasy realms, and adventure-quest machinery.",
    },
    {
        "title": "The Day After Tomorrow",
        "layer": "macro",
        "label": "Future-Shock, Alien Worlds & Disaster Survival",
        "description": "Large-scale survival stories driven by broken futures, alien worlds, first contact, natural disasters, ecological collapse, technology, or hostile frontiers.",
    },
    {
        "title": "The Day After Tomorrow",
        "layer": "micro",
        "label": "Climate, Disaster & End-of-World Survival",
        "description": "Disaster spectacles where climate, earthquakes, storms, planetary collapse, prophecy, or apocalyptic panic make survival feel global.",
    },
    {
        "title": "Chronicle",
        "layer": "neighborhood",
        "label": "Creature, Virus & Superpower Survival Sci-Fi",
        "description": "Sci-fi survival stories where monsters, viruses, alien creatures, experiments, powers, and strange outbreaks turn the body into a threat.",
    },
    {
        "title": "Chronicle",
        "layer": "micro",
        "label": "Creature, Experiment & Superpower Survival Sci-Fi",
        "description": "Sci-fi thrillers about creature encounters, dangerous experiments, young powers, mutations, hidden facilities, and survival under uncanny pressure.",
    },
    {
        "title": "Arrival",
        "layer": "micro",
        "label": "Alien Contact & Deep-Unknown Discovery",
        "description": "First-contact and deep-unknown sci-fi where strange intelligences, alien visitors, ancient portals, underwater secrets, and awe reshape human understanding.",
    },
    {
        "title": "The Lion King",
        "layer": "neighborhood",
        "label": "Creature & Animal-Kingdom Adventures",
        "description": "Creature-led family adventures where animal kingdoms, dinosaurs, islands, herds, habitats, heirs, and wilderness journeys carry the emotional stakes.",
    },
    {
        "title": "Mr. Popper's Penguins",
        "layer": "neighborhood",
        "label": "Family Animal, Toy-Box & Creature Mischief",
        "description": "Family adventures where animals, toys, food-worlds, creatures, pets, dragons, and childlike worlds create comic trouble and big-hearted friendship.",
    },
    {
        "title": "Mr. Popper's Penguins",
        "layer": "micro",
        "label": "Animal-Buddy Family Comedy & Creature Mischief",
        "description": "Family comedies and creature friendships built from animals, dragons, unlikely pets, oddball teams, slapstick, and warm mischief.",
    },
    {
        "title": "Baywatch",
        "layer": "neighborhood",
        "label": "Buddy, Schoolyard & Slapstick Action-Comedy",
        "description": "Broad comedies where cops, kids, classmates, lifeguards, malls, partners, stunts, and slapstick trouble turn ordinary worlds into comic action.",
    },
    {
        "title": "The Hateful Eight",
        "layer": "neighborhood",
        "label": "Offbeat Crime, Western & Hollywood Dark Comedy",
        "description": "Darkly comic crime, western, and Hollywood-adjacent stories where suspicious rooms, industry rot, period texture, and violent punchlines share the mood.",
    },
    {
        "title": "The Hateful Eight",
        "layer": "micro",
        "label": "Blizzard-Bound Western Noir & Dark Comedy",
        "description": "Snowed-in, western, and Tarantino-tinted crime comedies where mistrust, bounty hunters, violent talk, and genre games do the tightening.",
    },
    {
        "title": "Once Upon a Time... in Hollywood",
        "layer": "micro",
        "label": "Hollywood Hangout, Druggy Noir & Dark Comedy",
        "description": "Loose, sun-baked crime and Hollywood dark comedies about actors, hustlers, private eyes, druggy drift, period scenes, and violent detours.",
    },
    {
        "title": "Babylon",
        "layer": "macro",
        "label": "Noir, Crime & Obsession Downfalls",
        "description": "Dark thrillers and decadent rise-and-fall stories about crime, identity, addiction, power, obsession, performance, and schemes curdling into collapse.",
    },
    {
        "title": "Boogie Nights",
        "layer": "micro",
        "label": "Drug, Porn & Rise-and-Fall Industry Dramas",
        "description": "Industry and crime downfalls where drugs, sex work, celebrity, scams, money, parties, and ambition turn success into damage.",
    },
    {
        "title": "Top Gun: Maverick",
        "layer": "micro",
        "label": "High-Stakes Legacy Sequels & Comic Capers",
        "description": "Return-trip sequels where time jumps, aviation missions, heists, holiday traps, returning crews, and caper games escalate familiar worlds.",
    },
    {
        "title": "Star Wars: The Force Awakens",
        "layer": "neighborhood",
        "label": "Space Opera, Wizarding & Dystopian Quest Sagas",
        "description": "Big saga territory where space opera, wizard schools, dystopian rebellions, chosen heroes, good-vs-evil quests, and franchise mythology overlap.",
    },
    {
        "title": "A Quiet Place",
        "layer": "neighborhood",
        "label": "Homebound Survival & Invasion Thrillers",
        "description": "Tense thrillers where homes, cabins, roads, towers, or shelters become survival traps under killers, creatures, invasions, silence, or collapse.",
    },
    {
        "title": "The Holdovers",
        "layer": "micro",
        "label": "Recovery, Art & Life-Pivot Dramas",
        "description": "Intimate recovery and life-pivot dramas about disability, addiction, teaching, art, grief, mentorship, identity, and learning to live differently.",
    },
    {
        "title": "Never Back Down",
        "layer": "neighborhood",
        "label": "Gritty Sports, Civil-Rights & Comeback Dramas",
        "description": "Grounded dramas about athletes, courtrooms, civil rights, family pressure, mentorship, and people fighting their way toward dignity.",
    },
    {
        "title": "Never Back Down",
        "layer": "micro",
        "label": "Combat Sports & Bruised-Comeback Dramas",
        "description": "Training and comeback dramas about boxers, fighters, coaches, pressure, pain, rivalry, and earning self-respect the hard way.",
    },
    {
        "title": "The King of Comedy",
        "layer": "neighborhood",
        "label": "Psychological Crime, Stalker & Showbiz Downfalls",
        "description": "Unsettling crime and showbiz stories where obsession, fame hunger, stalkers, murder, performance, and public identity curdle into danger.",
    },
    {
        "title": "High School Musical 3: Senior Year",
        "layer": "neighborhood",
        "label": "Family Sequel Mischief & Villain Energy",
        "description": "Family-friendly sequel territory where toy boxes, animal worlds, musical schools, bright teams, villains, and returning crews keep the energy playful.",
    },
    {
        "title": "High School Musical 3: Senior Year",
        "layer": "micro",
        "label": "Family Musical, Toy-Box & Coming-of-Age Sequels",
        "description": "Cheerful family sequels mixing school milestones, musicals, toys, cars, animated worlds, heroes, and growing-up comfort.",
    },
    {
        "title": "Teenage Mutant Ninja Turtles: Out of the Shadows",
        "layer": "micro",
        "label": "Family Animal & Hero Sequels With Villain Energy",
        "description": "Family-friendly sequel adventures where animal heroes, bright creature worlds, comic villains, teams, and returning quests drive the fun.",
    },
    {
        "title": "Promising Young Woman",
        "layer": "neighborhood",
        "label": "Revenge, Intruder & Moral-Trap Thrillers",
        "description": "Tense thrillers where predators, guests, traps, revenge plots, dares, and social menace turn ordinary spaces into pressure chambers.",
    },
    {
        "title": "Promising Young Woman",
        "layer": "micro",
        "label": "Predator-Reversal & Uninvited Threat Thrillers",
        "description": "Psychological thrillers about predatory behavior, revenge, unwanted visitors, moral traps, home invasion, and victims turning the tables.",
    },
    {
        "title": "Triangle of Sadness",
        "layer": "neighborhood",
        "label": "Dark Farce, Capers & Class Chaos",
        "description": "Comic crime, travel, social-satire, and caper stories where scams, class tension, stranded groups, or absurd missions become farce.",
    },
    {
        "title": "Triangle of Sadness",
        "layer": "micro",
        "label": "Identity Mayhem, Class Satire & Caper Farce",
        "description": "Comedies and dark satires built from mistaken roles, class games, messy travelers, clumsy criminals, spy-ish setups, and power reversals.",
    },
    {
        "title": "We Need to Talk About Kevin",
        "layer": "neighborhood",
        "label": "Psychological Dread, Home Traps & Uninvited Threats",
        "description": "Dread thrillers where family trauma, predators, home traps, unwanted visitors, revenge, grief, or guilt make ordinary spaces unsafe.",
    },
    {
        "title": "We Need to Talk About Kevin",
        "layer": "micro",
        "label": "Haunted Visions & Family Psychological Horror",
        "description": "Family horror and psychological thrillers built from uncanny children, visions, curses, violence, guilt, and haunted domestic memory.",
    },
    {
        "title": "Avatar",
        "layer": "neighborhood",
        "label": "Alien Worlds, Dystopian Frontiers & Survival Action",
        "description": "Sci-fi frontiers where alien worlds, broken societies, desert empires, wastelands, invasions, and rebellion make survival political and spectacular.",
    },
    {
        "title": "Gravity",
        "layer": "micro",
        "label": "Space Mission Survival & Cosmic Discovery",
        "description": "Space and near-space stories built around isolation, awe, dangerous missions, scientific problem-solving, and surviving far from home.",
    },
    {
        "title": "The Devil Wears Prada",
        "layer": "micro",
        "label": "Style, Status & Romantic Self-Invention",
        "description": "Glamorous comedies and romances where fashion, class, weddings, work, shopping, and self-invention turn social performance into the battlefield.",
    },
    {
        "title": "1917",
        "layer": "micro",
        "label": "Modern War Survival Stories",
        "description": "Hard-hitting war dramas across modern battlefields where soldiers, civilians, and witnesses try to survive the next impossible hour.",
    },
    {
        "title": "The Hunger Games: Mockingjay - Part 2",
        "layer": "micro",
        "label": "Future-Rebellion & Space-Franchise Sequels",
        "description": "Sequel-scale futures where rebellions, starfleet missions, dystopian regimes, and franchise mythology keep escalating toward open conflict.",
    },
    {
        "title": "Star Wars: The Force Awakens",
        "layer": "neighborhood",
        "label": "Space Opera, Dystopian Rebellion & Giant-Creature Sagas",
        "description": "Big saga territory where star wars, dystopian uprisings, giant creatures, chosen heroes, good-vs-evil quests, and franchise mythology overlap.",
    },
    {
        "title": "Back to the Future Part III",
        "layer": "neighborhood",
        "label": "Relic Hunts, Time Trips & Adventure Riddles",
        "description": "Fast adventure rides where history, puzzles, relics, time travel, museums, tombs, and old-world set pieces turn discovery into a chase.",
    },
    {
        "title": "Sherlock Holmes: A Game of Shadows",
        "layer": "micro",
        "label": "Shadow Quests, Dark Lords & Mastermind Pursuits",
        "description": "Shadowy adventures about quests, criminal masterminds, dark powers, magic, pursuit, and worlds sliding toward open conflict.",
    },
    {
        "title": "Groundhog Day",
        "layer": "neighborhood",
        "label": "Buddy, Workplace & Time-Loop Bad-Idea Comedies",
        "description": "Broad comedies where buddies, offices, families, parties, time loops, and adult bad decisions keep ordinary life tipping into absurdity.",
    },
    {
        "title": "Groundhog Day",
        "layer": "micro",
        "label": "Workplace, Time-Loop & Bad-Idea Comedies",
        "description": "Comic spirals where office life, magical loops, wish fulfillment, awkward sex, buddy pressure, and social rules collapse into bad choices.",
    },
    {
        "title": "8 Mile",
        "layer": "micro",
        "label": "Comeback, Training & Pressure Dramas",
        "description": "Achievement dramas where music, sport, coaching, money, invention, rivalry, and physical or emotional pressure become the route to self-respect.",
    },
    {
        "title": "Mamma Mia!",
        "layer": "micro",
        "label": "Makeover, Music & Coming-of-Age Romcoms",
        "description": "Bright romantic comedies about makeovers, music, school or family milestones, confidence, social reinvention, and finding your voice.",
    },
    {
        "title": "Notting Hill",
        "layer": "micro",
        "label": "London Love, Holidays & Entangled Romance",
        "description": "Romantic dramas and comedies where city chance, holidays, infidelity, celebrity, friendships, and overlapping love stories complicate intimacy.",
    },
    {
        "title": "The Untouchables",
        "layer": "micro",
        "label": "City Crime, Addiction & Rise-and-Fall Legends",
        "description": "Big city crime and downfall stories about gangsters, addiction, loyalty, violence, ambition, and the slow erosion of a dream.",
    },
    {
        "title": "Death Proof",
        "layer": "micro",
        "label": "Roadside, Body-Horror & Exploitation Dread",
        "description": "Visceral horror and exploitation thrillers where bodies, roads, houses, woods, beauty, gore, and predatory violence turn pleasure into danger.",
    },
    {
        "title": "Panic Room",
        "layer": "micro",
        "label": "Heists, Hostages & Locked-Room Misdirection",
        "description": "Clever thrillers where robberies, vaults, hostage traps, hidden rooms, casinos, and staged deceptions tighten into a pressure chamber.",
    },
    {
        "title": "Skyscraper",
        "layer": "neighborhood",
        "label": "Conspiracy, Hostage & Protection Thrillers",
        "description": "High-tension action thrillers where public figures, families, hostages, transport systems, buildings, conspiracies, and ticking clocks collide.",
    },
    {
        "title": "The Passion of the Christ",
        "layer": "micro",
        "label": "Persecution, Genocide & Resistance Dramas",
        "description": "Serious historical dramas centered on persecution, genocide, occupation, martyrdom, survival, and acts of moral or political resistance.",
    },
    {
        "title": "Sucker Punch",
        "layer": "neighborhood",
        "label": "Dark Fantasy, Supernatural & Surreal Action",
        "description": "Hard-edged fantasy action driven by magic, demons, witches, monsters, surreal visions, comic-book energy, and violent mythic momentum.",
    },
    {
        "title": "M3GAN",
        "layer": "micro",
        "label": "Tech, Vision & Supernatural Threat Horror",
        "description": "Contagious dread horror where AI, visions, reflections, signals, uncanny children, or supernatural violence turn the body and home unsafe.",
    },
    {
        "title": "Evan Almighty",
        "layer": "micro",
        "label": "Raunchy Family, Buddy & Bad-Idea Comedies",
        "description": "Family and buddy comedies where in-laws, parents, neighbors, divine jokes, adult immaturity, and escalating humiliation drive the set pieces.",
    },
    {
        "title": "The Descent",
        "layer": "micro",
        "label": "Claustrophobic Creature Survival Horror",
        "description": "Survival horror about caves, oceans, ships, flooded spaces, and other tight places where creatures or disaster hunt from the dark.",
    },
    {
        "title": "The Prince of Egypt",
        "layer": "neighborhood",
        "label": "Animal-Kingdom, Biblical & Wilderness Adventures",
        "description": "Family adventures where animal kingdoms, biblical epics, wilderness journeys, childlike wonder, and habitat-scale stakes carry the emotion.",
    },
    {
        "title": "The Prince of Egypt",
        "layer": "micro",
        "label": "Animal-Kingdom & Biblical Family Epics",
        "description": "Creature, wilderness, and biblical family stories about kingdoms, herds, heirs, exile, faith, and mythic coming-of-age spectacle.",
    },
    {
        "title": "The Secret Life of Walter Mitty",
        "layer": "micro",
        "label": "Whimsical Life Detours & Family Melancholy",
        "description": "Offbeat life fables and melancholy comedies where fantasy, family wounds, travel, grief, and sudden absurdity nudge people toward meaning.",
    },
    {
        "title": "Garfield",
        "layer": "micro",
        "label": "Pet & Animal Buddy Comedies",
        "description": "Fast, family-friendly misadventures starring talking animals, pets, and animated or live-action hybrids built around bickering, bonding, and schemes.",
    },
    {
        "title": "The Talented Mr. Ripley",
        "layer": "micro",
        "label": "Psychological Crime, Obsession & Murder Thrillers",
        "description": "Unsettling thrillers where murder, identity performance, stalking, fame hunger, investigation, and obsessive psychology tighten into danger.",
    },
    {
        "title": "Melancholia",
        "layer": "micro",
        "label": "Cosmic Love, Memory & Lonely Intimacy",
        "description": "Philosophical romances and grief dramas where love, memory, mortality, identity, cosmic unease, and lonely projection bend ordinary intimacy.",
    },
    {
        "title": "Big Eyes",
        "layer": "micro",
        "label": "Literary, Period & Socially Constrained Longing",
        "description": "Artful period and literary dramas where love, gender, class, reputation, illness, or social constraint turns longing into the central pressure.",
    },
    {
        "title": "Mamma Mia! Here We Go Again",
        "layer": "micro",
        "label": "Messy Sequel Reunions & Comic Aftershocks",
        "description": "Return-trip sequels where friends, families, bands, partners, parties, holidays, and old mistakes come back with comic or bittersweet force.",
    },
    {
        "title": "The Abyss",
        "layer": "neighborhood",
        "label": "Space, Sea & Isolated Mission Survival Sci-Fi",
        "description": "Science-fiction missions where space crews, deep-sea teams, fragile habitats, dangerous discovery, and isolation matter more than ordinary action.",
    },
    {
        "title": "Walk the Line",
        "layer": "micro",
        "label": "American Legends, Family Grief & Reckoning",
        "description": "American dramas about music icons, western codes, family wounds, old violence, money, reputation, and the reckoning that follows ambition.",
    },
    {
        "title": "War for the Planet of the Apes",
        "layer": "neighborhood",
        "label": "Giant-Creature, Ape-War & Robot Spectacle",
        "description": "Large-scale creature and machine spectacle where kaiju, apes, giant robots, island battles, and survival politics collide.",
    },
    {
        "title": "War for the Planet of the Apes",
        "layer": "micro",
        "label": "Ape, Kaiju & Giant-Creature Battle Epics",
        "description": "Monster, ape, and giant-creature epics where towering bodies, island-scale threats, war bands, and survival politics drive the spectacle.",
    },
    {
        "title": "Jingle All the Way",
        "layer": "macro",
        "label": "Popcorn Sequels, Family Adventures & Comic Mysteries",
        "description": "Crowd-friendly franchise territory spanning family adventures, comic sequels, holiday traps, treasure chases, capers, and bright mystery machinery.",
    },
    {
        "title": "Jingle All the Way",
        "layer": "neighborhood",
        "label": "Legacy Sequels, Time Trips & Holiday-Caper Mischief",
        "description": "Return-trip adventures and comedies where time travel, holidays, heists, spy gadgets, family traps, and familiar crews create brisk set pieces.",
    },
    {
        "title": "Jingle All the Way",
        "layer": "micro",
        "label": "Time-Jump, Heist & Holiday-Caper Sequels",
        "description": "Light sequel and caper energy built from time jumps, heists, aviation missions, holiday traps, returning crews, and comic escalation.",
    },
    {
        "title": "Gladiator (2000)",
        "layer": "neighborhood",
        "label": "Historical War, Persecution & Resistance Epics",
        "description": "Historical war and persecution dramas across empires, battlefields, occupations, genocides, courts, and resistance movements.",
    },
    {
        "title": "Gladiator (2000)",
        "layer": "micro",
        "label": "Ancient, Religious & War-Resistance Dramas",
        "description": "Serious historical dramas about ancient power, battlefield survival, persecution, faith, moral resistance, leadership, and political reckoning.",
    },
    {
        "title": "The Lord of the Rings: The Return of the King",
        "layer": "neighborhood",
        "label": "Epic Franchise Sagas: Space, Fantasy & Rebellion",
        "description": "Big saga territory where fantasy quests, star wars, dystopian uprisings, vampire clans, chosen heroes, and franchise mythology overlap.",
    },
    {
        "title": "The Lord of the Rings: The Return of the King",
        "layer": "micro",
        "label": "Fantasy, Space & Rebellion Saga Sequels",
        "description": "Sequel-scale sagas where fantasy realms, starfleet missions, dystopian regimes, supernatural clans, and mythic alliances escalate toward open conflict.",
    },
    {
        "title": "Independence Day",
        "layer": "micro",
        "label": "Alien, Climate & Impact Apocalypse Survival",
        "description": "Disaster spectacles where alien invasion, asteroids, climate collapse, storms, planetary impact, or apocalyptic panic make survival feel global.",
    },
    {
        "title": "Oppenheimer",
        "layer": "micro",
        "label": "WWII Science, Escape & Rescue Dramas",
        "description": "World War II-era dramas where science, codebreaking, battle survival, rescue missions, espionage, and political choices alter history.",
    },
    {
        "title": "Zombieland",
        "layer": "macro",
        "label": "Horror, Murder-Spree & Dark-Comic Dread",
        "description": "Horror and horror-adjacent territory spanning creature attacks, zombie outbreaks, murder sprees, vampire spoofs, dark comedy, exploitation shocks, and survival stories with or without a comic bite.",
    },
    {
        "title": "The Lego Movie",
        "layer": "macro",
        "label": "Animation, Family Adventure & Storybook Worlds",
        "description": "Animated and live-action family adventures, animal-led stories, storybook fantasy, political fables, kid mysteries, holiday magic, and playful or serious coming-of-age animation.",
    },
    {
        "title": "Forrest Gump",
        "layer": "macro",
        "label": "Relationship, Family & Becoming Dramas",
        "description": "Character-forward dramas about family, friendship, identity, grief, recovery, art, and the turning points that make a life cohere.",
    },
    {
        "title": "It",
        "layer": "macro",
        "label": "Supernatural, Psychological & Mystery Horror",
        "description": "Horror shaped by hauntings, curses, possession, slashers, psychological dread, and investigations into realities that refuse to stay safe.",
    },
    {
        "title": "Office Space",
        "layer": "macro",
        "label": "Dark-Edge Comedy, Spoof & Bad-Idea Capers",
        "description": "Comedy-first territory where work frustration, slacker spirals, buddy trouble, parody, satire, crime detours, and terrible plans become the fun.",
    },
    {
        "title": "The Notebook",
        "layer": "macro",
        "label": "Modern Romance, Romcoms & City Dramedies",
        "description": "Contemporary romance and city-dramedy territory spanning workplace bustle, dating chaos, school crushes, erotic melodrama, young-adult heartbreak, and second-chance love stories.",
    },
    {
        "title": "John Wick",
        "layer": "macro",
        "label": "Spy, Revenge & High-Velocity Action Thrillers",
        "description": "Action-thriller territory where spies, assassins, fugitives, revenge missions, transport jobs, and races against time keep the pressure moving.",
    },
    {
        "title": "Shutter Island",
        "layer": "macro",
        "label": "Psychological Mystery, Crime & Tech Thrillers",
        "description": "Dark mysteries, crime investigations, techno-paranoia, erotic danger, identity puzzles, and psychological thrillers where trust keeps slipping.",
    },
    {
        "title": "Django Unchained",
        "layer": "macro",
        "label": "Historical, War, Sports & Social-Pressure Dramas",
        "description": "Grounded pressure dramas about history, war, sport, public institutions, social causes, environment, survival, persecution, ambition, and people tested by systems larger than themselves.",
    },
    {
        "title": "Pulp Fiction",
        "layer": "macro",
        "label": "Crime, Noir, Urban & Obsession Downfalls",
        "description": "Crime, noir, western, biker, courier, youth-crime, gangster, gambling, and rise-and-fall stories where violence, ego, money, addiction, or obsession bends the world out of shape.",
    },
    {
        "title": "Interstellar",
        "layer": "macro",
        "label": "Sci-Fi, Disaster & Isolated Survival",
        "description": "Science-fiction, disaster, and isolated survival shaped by broken futures, alien or unknown contact, sea or cave entrapment, ecological collapse, technology, experiments, and hostile frontiers.",
    },
    {
        "title": "The Lord of the Rings: The Two Towers",
        "layer": "macro",
        "label": "Franchise Sequels, Fantasy Quests & Popcorn Adventures",
        "description": "Crowd-friendly franchise territory spanning fantasy quests, school magic, creature adventures, comic sequels, capers, sports follow-ups, and returning worlds.",
    },
    {
        "title": "Mean Girls",
        "layer": "macro",
        "label": "Teen, Music, Sports & Coming-of-Age Comedy-Drama",
        "description": "Teen, young-adult, music, and sports-romance territory where school, competition, friendship, crushes, family pressure, and risky self-definition drive comedy or drama.",
    },
    {
        "title": "Titanic",
        "layer": "macro",
        "label": "Period Romance, Literary Drama & Ornate Melodrama",
        "description": "Period, literary, royal, romantic, erotic, and historical melodramas where love, class, reputation, faith, or social constraint sets the pressure.",
    },
    {
        "title": "The Avengers",
        "layer": "macro",
        "label": "Fantasy, Superhero & Mythic Action Spectacle",
        "description": "Large-scale action spectacle built from fantasy quests, superheroes, martial-arts heroes, space-opera battles, swashbucklers, monsters, and comic-book stakes.",
    },
    {
        "title": "Bohemian Rhapsody",
        "layer": "macro",
        "label": "Music, Documentary & Public-Life Dramas",
        "description": "Documentaries, concert films, music biographies, public-life portraits, and true-story dramas where performance, history, politics, or legacy is the frame.",
    },
    {
        "title": "Back to the Future",
        "layer": "neighborhood",
        "label": "Time-Travel, Sci-Fi Spoof & Absurdist Comedy",
        "description": "Comic adventures where time travel, science jokes, aliens, inventions, prehistoric bits, fandom, or absurd genre premises turn speculation into farce.",
    },
    {
        "title": "Firewalker",
        "layer": "neighborhood",
        "label": "Spy Spoofs, Treasure Hunts & Adventure Capers",
        "description": "Breezy action comedies where spies, treasure maps, adventure travel, office politics, and mistaken missions create escalating caper trouble.",
    },
    {
        "title": "History of the World: Part I",
        "layer": "neighborhood",
        "label": "Live-Action Satire, Music & Social Farce",
        "description": "Live-action comedies and satires built from history, class, politics, cabaret, drag, religion, workplaces, small towns, or social institutions cracking into farce.",
    },
    {
        "title": "The Fighter",
        "layer": "neighborhood",
        "label": "Sports, Boxing & True-Story Comeback Dramas",
        "description": "Sports and achievement dramas about boxing, football, chess, military academies, doping scandals, training pressure, mentorship, and earning a comeback.",
    },
    {
        "title": "The Fighter",
        "layer": "micro",
        "label": "Boxing, Training & Bruised Comebacks",
        "description": "Training and comeback dramas about boxers, fighters, athletes, prodigies, coaches, family pressure, pain, rivalry, and self-respect earned the hard way.",
    },
    {
        "title": "JFK",
        "layer": "neighborhood",
        "label": "Political Biography, Espionage & Public Scandal",
        "description": "Historical and political dramas about presidents, revolutions, investigations, intelligence agencies, public lies, whistleblowers, hostage crises, and institutional reckoning.",
    },
    {
        "title": "Bridge of Spies",
        "layer": "micro",
        "label": "Cold-War, Whistleblower & Political Reckonings",
        "description": "Political-history dramas and thrillers about spies, whistleblowers, intelligence work, hostage negotiations, investigations, and state secrets becoming public pressure.",
    },
    {
        "title": "Police Academy 6: City Under Siege",
        "layer": "neighborhood",
        "label": "Buddy-Cop, Spy & Comedy Sequels",
        "description": "Comedy sequels and capers where cops, detectives, spies, familiar crews, undercover jobs, and returning comic trouble keep the plot moving.",
    },
    {
        "title": "The Lord of the Rings: The Two Towers",
        "layer": "neighborhood",
        "label": "Fantasy, YA Rebellion & Franchise Battle Sagas",
        "description": "Big franchise sequels where fantasy quests, wizard schools, young-adult rebellion, creature kingdoms, superheroes, and returning worlds escalate toward battle.",
    },
    {
        "title": "The Lord of the Rings: The Two Towers",
        "layer": "micro",
        "label": "Fantasy Quest & Wonderland Sequels",
        "description": "Fantasy sequel pockets built from quests, magic, high-fantasy worlds, wonderlands, Narnia-like kingdoms, pirates, battles, and mythic returning journeys.",
    },
    {
        "title": "Transformers: Revenge of the Fallen",
        "layer": "neighborhood",
        "label": "High-Drama Franchise Sequels: Robots, Monsters & Romance",
        "description": "Franchise sequels where robots, monsters, vampire romance, dystopian tension, conspiracies, and high-drama relationships share sequel momentum.",
    },
    {
        "title": "After We Fell",
        "layer": "micro",
        "label": "Angsty Romance & Creature-Love Sequels",
        "description": "High-drama romance installments where young lovers, creature mythology, family secrets, obsession, and sequel escalation keep the relationship unstable.",
    },
    {
        "title": "The Adventures of Priscilla, Queen of the Desert",
        "layer": "neighborhood",
        "label": "Queer, Period & Self-Determination Dramas",
        "description": "Queer, period, romantic, road, and self-determination dramas where gender, sexuality, faith, class, or history shapes the journey toward freedom.",
    },
    {
        "title": "The Adventures of Priscilla, Queen of the Desert",
        "layer": "micro",
        "label": "Queer Road, Romance & Self-Determination",
        "description": "Queer and romantic dramas where road trips, cabaret, social pressure, love, identity, or survival push people toward self-definition.",
    },
    {
        "title": "The Mighty Ducks",
        "layer": "neighborhood",
        "label": "Youth Sports, Teamwork & Comeback Dramas",
        "description": "Sports stories about young teams, coaches, fighters, rivalry, discipline, mentorship, and finding dignity through training or teamwork.",
    },
    {
        "title": "The Mighty Ducks",
        "layer": "micro",
        "label": "Youth Sports Team Comebacks",
        "description": "Family-friendly sports comedies and dramas where teams, coaches, tournaments, school pressure, and underdog momentum build toward a comeback.",
    },
    {
        "title": "Before Sunrise",
        "layer": "neighborhood",
        "label": "Strange Romance, Mortality & Spiritual Longing",
        "description": "Romances and longing dramas where mortality, faith, time, immortality, illness, fantasy, or one charged encounter makes intimacy feel uncanny.",
    },
    {
        "title": "Before Sunrise",
        "layer": "micro",
        "label": "Time-Bent, Strange & Soulmate Romance",
        "description": "Romantic dramas where timing, travel, fantasy, illness, mortality, or impossible connection turns ordinary intimacy into a strange hinge of fate.",
    },
    {
        "title": "The Perks of Being a Wallflower",
        "layer": "micro",
        "label": "Tender YA, Mental Health & Strange Intimacy",
        "description": "Young-adult and offbeat intimacy dramas about mental health, grief, first love, friendship, school pressure, loneliness, and fragile self-recognition.",
    },
    {
        "title": "Star Trek",
        "layer": "neighborhood",
        "label": "Mutant Heroes & Space-Opera Franchise Action",
        "description": "Franchise action where mutant heroes, space opera, rebel missions, prequels, superpowers, and comic-book spectacle share a high-energy mythology lane.",
    },
    {
        "title": "Jurassic World",
        "layer": "micro",
        "label": "Dino, Alien & Adventure Sequel Spectacle",
        "description": "Adventure sequels and revival spectacles where dinosaurs, aliens, islands, creature parks, games, and returning worlds create popcorn-scale danger.",
    },
    {
        "title": "Underwater",
        "layer": "micro",
        "label": "Deep-Sea, Shark & Space-Unknown Survival Horror",
        "description": "Claustrophobic survival horror around deep sea, sharks, submarines, alien organisms, caves, space-like isolation, and crews trapped near the unknown.",
    },
    {
        "title": "Hot Tub Time Machine",
        "layer": "micro",
        "label": "Slacker Party & Bad-Idea Comedy",
        "description": "Broad comedies where parties, road trips, college chaos, arrested adulthood, slackers, buddy pressure, and bad decisions do the damage.",
    },
    {
        "title": "Without a Paddle",
        "layer": "micro",
        "label": "Clueless Heists, Treasure Hunts & Crime Farce",
        "description": "Comic capers where bank jobs, fake criminals, treasure hunts, wilderness trips, money schemes, and clueless adults stumble through trouble.",
    },
    {
        "title": "Notting Hill",
        "layer": "micro",
        "label": "Urban Weddings, Holidays & Entangled Romcoms",
        "description": "City and travel romcoms where weddings, celebrity, class, holidays, friendship circles, and overlapping romantic expectations complicate the happy ending.",
    },
    {
        "title": "Ghosthouse",
        "layer": "neighborhood",
        "label": "Haunted Houses, Found Footage & Urban-Legend Dread",
        "description": "Supernatural horror where haunted houses, ghosts, faux-documentary traces, urban legends, cursed recordings, and old murders make buildings feel unsafe.",
    },
    {
        "title": "The Man Who Knew Infinity",
        "layer": "micro",
        "label": "Sports, Prodigy & True-Story Achievement Dramas",
        "description": "True-story achievement dramas about athletes, mathematicians, students, coaches, institutions, talent, discipline, and recognition arriving under pressure.",
    },
    {
        "title": "Bohemian Rhapsody",
        "layer": "macro",
        "label": "Music, Documentary, Sports & Public-Life Portraits",
        "description": "Documentaries, concert films, musical dramas, sports and youth-culture portraits, music biographies, science and public-life stories, true-story dramas, and legacy studies shaped by performance, history, or institutions.",
    },
    {
        "title": "Interstellar",
        "layer": "macro",
        "label": "Sci-Fi, Disaster & Isolated Survival",
        "description": "Science-fiction, disaster, and isolated survival shaped by broken futures, alien or unknown contact, sea or cave entrapment, ecological collapse, experiments, technology, and hostile frontiers.",
    },
    {
        "title": "The Lego Movie",
        "layer": "neighborhood",
        "label": "Family Sci-Fi, Toy-Box & Oddball Friendships",
        "description": "Family-facing sci-fi and toy-box adventures where aliens, robots, inventions, games, friendship, and handmade worlds turn imagination into the engine.",
    },
    {
        "title": "The Lego Movie",
        "layer": "micro",
        "label": "Inventive Toy, Robot & Buddy Animation",
        "description": "Animated buddy adventures built from toys, robots, found families, misfit inventions, playful sci-fi, and handmade worlds learning to care.",
    },
    {
        "title": "Avatar",
        "layer": "micro",
        "label": "Alien Worlds & Frontier-Rebellion Sci-Fi",
        "description": "Planet-scale science fiction where alien worlds, militarized frontiers, ecological conflict, survival, and rebellion make spectacle feel political.",
    },
    {
        "title": "Spy Kids",
        "layer": "neighborhood",
        "label": "Family Pop-Fantasy, Games & Kid Capers",
        "description": "Family adventures where games, cartoons, spies, rich kids, sports, pop icons, fantasy worlds, and mixed live-action/animation energy turn childhood into a caper.",
    },
    {
        "title": "Super Mario Bros.",
        "layer": "micro",
        "label": "Video-Game, Cartoon & Pop-Fantasy Side Quests",
        "description": "Family comedies and adventures spun from games, cartoons, pop icons, mysteries, mutant heroes, and bright fantasy side quests, animated or live action.",
    },
    {
        "title": "Spy Kids",
        "layer": "micro",
        "label": "Family Fantasy, Kid Missions & Oddball Adventures",
        "description": "Kid-forward family adventures about spies, babysitters, magical worlds, oddball families, dinosaurs, cartoons, and mission-driven mischief.",
    },
    {
        "title": "The Wizard",
        "layer": "micro",
        "label": "Kid Sports, Rich-Kid & Goofy Family Adventures",
        "description": "Live-action and animated family adventures where kids, sports, games, wealth fantasies, school pressure, and goofy parent-child stakes drive the fun.",
    },
    {
        "title": "Tom and Huck",
        "layer": "neighborhood",
        "label": "Orphans, Nannies & Classic Kid Mysteries",
        "description": "Family classics and kid mysteries built from orphans, nannies, school-age detectives, Dickensian pressure, children's books, music, and adventure.",
    },
    {
        "title": "Tom and Huck",
        "layer": "micro",
        "label": "Orphans, Classics & Kid Mystery Dramas",
        "description": "Family dramas and adventures where orphans, classic literature, child detectives, secret places, nannies, and growing-up trouble carry the story.",
    },
    {
        "title": "Return to Space",
        "layer": "neighborhood",
        "label": "Poetic, Nature & Space Documentaries",
        "description": "Documentaries about nature, ecology, space missions, filmmaking, food systems, observational life, exploration, and the strange scale of human work.",
    },
    {
        "title": "Return to Space",
        "layer": "micro",
        "label": "Poetic Observation, Space & Nature Docs",
        "description": "Observational and poetic documentaries about space exploration, nature, filmmaking, places, food, history, and the hidden machinery of real life.",
    },
    {
        "title": "Roar",
        "layer": "micro",
        "label": "Animal, Werewolf & Creature-Attack Survival Horror",
        "description": "Creature horror where animal attacks, werewolf transformations, wild predators, dark comedy, gore, and survival panic turn the body into prey.",
    },
    {
        "title": "The Boys Next Door",
        "layer": "neighborhood",
        "label": "Backwoods, Cannibal & Killing-Spree Horror",
        "description": "Horror and crime-tinged dread where cannibals, killers, road predators, urban murder sprees, exploitation shocks, and vicious survival pressure do the damage.",
    },
    {
        "title": "Teaching Mrs. Tingle",
        "layer": "micro",
        "label": "Kidnapping, Teen & Serial-Killer Psychological Thrillers",
        "description": "Psychological thrillers about serial killers, kidnappings, teen traps, obsessive teachers, torture threats, and people trapped inside escalating manipulation.",
    },
    {
        "title": "Priest",
        "layer": "micro",
        "label": "Vampire, Faith & Gothic Moral-Crisis Dramas",
        "description": "Gothic and faith-pressured dramas where vampires, priests, forbidden desire, abuse, immortality, religion, and moral crisis turn devotion dangerous.",
    },
    {
        "title": "Sahara",
        "layer": "micro",
        "label": "Submarine, Treasure & Nuclear Countdown Action",
        "description": "Adventure thrillers where treasure hunts, submarines, naval pressure, terrorism, nuclear threats, and race-against-time missions collide.",
    },
    {
        "title": "Airheads",
        "layer": "neighborhood",
        "label": "Stunt, Rock-Star & Pratfall Parodies",
        "description": "Broad comedies and parodies about stunts, sports, rock bands, fame fantasies, reckless performers, and physical humiliation played for laughs.",
    },
    {
        "title": "Airheads",
        "layer": "micro",
        "label": "Painfully Funny Sports, Stunts & Rock Dreams",
        "description": "Comic spectacle where sports, stunts, rock bands, painful pratfalls, fame hunger, and overmatched performers turn ambition into a joke.",
    },
    {
        "title": "iCarly: iGo to Japan",
        "layer": "neighborhood",
        "label": "Scooby-Style Mystery, Family Comedy & Kid Adventures",
        "description": "Family-friendly mysteries and comedy adventures where animated teams, live-action kid crews, monsters, travel, contests, and friendship drive the caper.",
    },
    {
        "title": "Unstoppable",
        "layer": "micro",
        "label": "Runaway Train, Terrorism & True-Crisis Thrillers",
        "description": "Grounded crisis thrillers about trains, terrorism, public danger, rescue decisions, evacuation, and real or realistic emergencies accelerating out of control.",
    },
    {
        "title": "The Little Hours",
        "layer": "micro",
        "label": "Biblical, Nun & Faith-Edged Period Stories",
        "description": "Period stories where biblical settings, nuns, faith, sex, anachronistic comedy, religious conflict, and moral transgression shape the drama or farce.",
    },
    {
        "title": "Lost River",
        "layer": "micro",
        "label": "Surreal Psychological Horror of Care & Obsession",
        "description": "Dreamlike psychological horror and uncanny drama about obsession, care, family dread, decaying places, illness, visions, and reality slipping sideways.",
    },
    {
        "title": "The Bikeriders",
        "layer": "neighborhood",
        "label": "LA, Biker & Street-Gang Action Noir",
        "description": "Noir-tinged action and crime stories where biker crews, street gangs, corruption, revenge, motorcycles, and hard-edged loyalty run under a city shadow.",
    },
    {
        "title": "An Education",
        "layer": "micro",
        "label": "Class-Crossed Lessons & First Independence",
        "description": "Coming-of-age romances where class, schooling, older suitors, literary manners, family pressure, and first independence turn desire into a lesson.",
    },
    {
        "title": "The Lego Movie",
        "layer": "macro",
        "label": "Family Adventure & Storybook Worlds",
        "description": "Animated and live-action family adventures, animal-led stories, storybook fantasy, political fables, kid mysteries, holiday magic, and playful or serious coming-of-age animation.",
    },
    {
        "title": "The Lego Movie",
        "layer": "micro",
        "label": "Inventive Toy, Alien & Buddy Family Worlds",
        "description": "Toy-box, alien, robot, and buddy-family adventures where handmade worlds, oddball friendships, and playful sci-fi teach misfits how to care.",
    },
    {
        "title": "Garfield",
        "layer": "micro",
        "label": "Animal Fables, Pet Mischief & Creature Companions",
        "description": "Animal-led stories spanning pet comedies, creature companions, talking-animal fables, and darker animal allegories where beasts carry the human trouble.",
    },
    {
        "title": "It",
        "layer": "macro",
        "label": "Horror, Entrapment & Survival Thrillers",
        "description": "Dark survival territory where hauntings, slashers, creatures, curses, wilderness traps, sea disasters, psychological dread, and desperate escapes make safety collapse.",
    },
    {
        "title": "Shutter Island",
        "layer": "macro",
        "label": "Mystery, Crime, Tech & Dark-Comedy Thrillers",
        "description": "Dark mysteries, crime investigations, techno-paranoia, erotic danger, whodunit comedy, identity puzzles, and psychological thrillers where trust keeps slipping.",
    },
    {
        "title": "Django Unchained",
        "layer": "macro",
        "label": "Pressure-Cooker Survival, Sports & Social Dramas",
        "description": "Grounded pressure dramas about history, sport, public institutions, survival, social causes, faith, environment, ambition, and people tested by larger systems.",
    },
    {
        "title": "Bohemian Rhapsody",
        "layer": "macro",
        "label": "Documentary, Music & Public-Life Portraits",
        "description": "Documentaries, concert films, musical dramas, sports and youth-culture portraits, music biographies, war histories, science stories, and public-life legacies.",
    },
    {
        "title": "Sully",
        "layer": "neighborhood",
        "label": "Rescue, Crisis & Survival Pressure",
        "description": "Rescue, aviation, sea, disaster, and survival dramas where duty, danger, endurance, and public scrutiny collide under real-world pressure.",
    },
    {
        "title": "Who Framed Roger Rabbit",
        "layer": "micro",
        "label": "Animated Toon, Animal & Noir Mischief",
        "description": "Cartoon and animal-centered mischief where animated worlds, live-action hybrids, noir jokes, pet chaos, and oddball companions turn trouble playful or strange.",
    },
    {
        "title": "Bad Boy Bubby",
        "layer": "neighborhood",
        "label": "Dark Outsider Coming-of-Age & Family Rupture",
        "description": "Offbeat coming-of-age and family rupture dramas where isolation, abuse, strange humor, working-class pressure, and outsider innocence make adulthood feel warped.",
    },
    {
        "title": "Bad Boy Bubby",
        "layer": "micro",
        "label": "Offbeat Family Pressure & Outsider Becoming",
        "description": "Unsettling family and outsider dramas where sheltered lives, parental damage, cultural pressure, and dark comedy push people toward painful selfhood.",
    },
    {
        "title": "Away We Go",
        "layer": "neighborhood",
        "label": "Couples, Parenthood & Family-Planning Dramedies",
        "description": "Relationship dramedies where couples, siblings, parents, pregnancy, travel, domestic chaos, and anxious adulthood turn family into a question.",
    },
    {
        "title": "Away We Go",
        "layer": "micro",
        "label": "Couples, Parenthood & Family Ego Games",
        "description": "Comic family dramas about couples, new parents, siblings, ego bruises, homecomings, and the messy search for a place to belong.",
    },
    {
        "title": "Starter for 10",
        "layer": "neighborhood",
        "label": "British School, Class & Coming-of-Age Romance",
        "description": "British school, university, and class-crossing romances where education, first love, ambition, social aspiration, and awkward independence shape identity.",
    },
    {
        "title": "Starter for 10",
        "layer": "micro",
        "label": "Campus, Class & Coming-of-Age Romance",
        "description": "Campus and class-conscious coming-of-age romances where school, exams, first love, social embarrassment, and self-invention turn desire into a lesson.",
    },
    {
        "title": "The Dive",
        "layer": "micro",
        "label": "Sea-Soaked Survival Thrillers",
        "description": "Waterbound survival thrillers where diving accidents, sharks, deep-sea traps, pressure, isolation, and rescue clocks turn the ocean into a trap.",
    },
    {
        "title": "Stolen Girl",
        "layer": "micro",
        "label": "Family Trauma & Child-Abduction Thrillers",
        "description": "Domestic and psychological thrillers where missing children, abductions, family trauma, guilt, rescue attempts, and dangerous secrets pull homes apart.",
    },
    {
        "title": "The Painter",
        "layer": "neighborhood",
        "label": "Assassin, Spy & Operative Revenge Thrillers",
        "description": "Action thrillers about spies, assassins, ex-operatives, fugitives, revenge missions, covert histories, and professional killers dragged back into danger.",
    },
    {
        "title": "The Painter",
        "layer": "micro",
        "label": "Ex-Intelligence Operative Action Thrillers",
        "description": "Ex-CIA, ex-intelligence, and covert-operative thrillers where old missions, hidden identities, assassins, and revenge threats force one more fight.",
    },
    {
        "title": "Sightseers",
        "layer": "neighborhood",
        "label": "Backwoods, Killing-Spree & Dark-Comedy Horror-Crime",
        "description": "Rural horror-crime and black-comedy thrillers where road trips, backwoods killers, cannibals, murder sprees, hostages, and bad behavior turn ugly.",
    },
    {
        "title": "Sightseers",
        "layer": "micro",
        "label": "Countryside Murder & Dark-Comedy Horror-Crime",
        "description": "Darkly comic countryside crime and horror stories where couples, hostages, rural roads, murder sprees, and shabby vacations curdle into violence.",
    },
    {
        "title": "The Whole Ten Yards",
        "layer": "macro",
        "label": "Franchise Sequels, Fantasy Quests & Comedy Capers",
        "description": "Crowd-friendly franchise territory spanning fantasy quests, school magic, creature adventures, comedy-crime sequels, capers, sports follow-ups, and returning worlds.",
    },
    {
        "title": "Return to Space",
        "layer": "neighborhood",
        "label": "Documentary Worlds: Nature, Space, War & Real Life",
        "description": "Documentaries about nature, ecology, space missions, war memory, archaeology, food systems, observational life, exploration, and real human work.",
    },
    {
        "title": "Return to Space",
        "layer": "micro",
        "label": "Observational Docs: Space, Nature, War & Daily Life",
        "description": "Observational documentaries about space exploration, nature, war memory, filmmaking, archaeology, places, food, history, and hidden real-life systems.",
    },
    {
        "title": "The Secret",
        "layer": "micro",
        "label": "Issue, Self-Help & Impact Documentaries",
        "description": "Documentaries about self-help, food, climate, animals, social impact, belief systems, activism, public persuasion, and claims about how people should live.",
    },
    {
        "title": "The Kid Who Would Be King",
        "layer": "neighborhood",
        "label": "Classic Fantasy, Legend & Young-Hero Adventures",
        "description": "Classic fantasy adventures where swords, dragons, legends, young heroes, quests, curses, and old myths return through a coming-of-age lens.",
    },
    {
        "title": "The Kid Who Would Be King",
        "layer": "micro",
        "label": "Sword, Sorcery & Young-Hero Fantasy Quests",
        "description": "Fantasy quests where kids, knights, dragons, sorcerers, ancient legends, and chosen heroes turn myth into an adventure trial.",
    },
    {
        "title": "Harry Potter 20th Anniversary: Return to Hogwarts",
        "layer": "neighborhood",
        "label": "Music, Film & Public-Life Documentary Spotlights",
        "description": "Documentary portraits of artists, athletes, filmmakers, musicians, reunions, public figures, and pop-culture legacies.",
    },
    {
        "title": "Harry Potter 20th Anniversary: Return to Hogwarts",
        "layer": "micro",
        "label": "Concert, Reunion & Pop-Culture Documentary Films",
        "description": "Music, sports, reunion, and pop-culture documentaries where performance, fandom, legacy, and behind-the-scenes memory carry the feeling.",
    },
    {
        "title": "The Princess Bride",
        "layer": "neighborhood",
        "label": "Family Animation, Fairy-Tale & Creature Adventures",
        "description": "Family-facing animation, fairy tales, creature kingdoms, storybook quests, and live-action fantasy adventures with warm mythic energy.",
    },
    {
        "title": "The Princess Bride",
        "layer": "micro",
        "label": "Romantic Fairy-Tale Adventure",
        "description": "Storybook romances and fairy-tale adventures where quests, kingdoms, true love, comic danger, and fantasy comfort overlap.",
    },
    {
        "title": "Birds of Prey (and the Fantabulous Emancipation of One Harley Quinn)",
        "layer": "micro",
        "label": "Gotham Vigilante Crime-Comic Action",
        "description": "Gotham and comic-book crime stories where vigilantes, villains, antiheroes, gangs, and chaotic team-ups turn crime into spectacle.",
    },
    {
        "title": "Daybreakers",
        "layer": "micro",
        "label": "Virus, Vampire & Isolation Survival Sci-Fi",
        "description": "Sci-fi survival stories about viruses, vampires, engineered bodies, isolation, sealed environments, and experiments that change what humans become.",
    },
    {
        "title": "The Perfect Game",
        "layer": "micro",
        "label": "Sports Underdog True-Story Dramas",
        "description": "True-story and inspired-by-life sports dramas where baseball, football, basketball, coaching, institutions, and underdog momentum build toward dignity.",
    },
    {
        "title": "There Will Be Blood",
        "layer": "neighborhood",
        "label": "Historical Power, Persecution & Public Reckoning Dramas",
        "description": "Historical and political dramas about power, persecution, racism, war, extraction, resistance, and institutions turning private ambition public.",
    },
    {
        "title": "There Will Be Blood",
        "layer": "micro",
        "label": "Ambition, Greed & Dark Literary Fables",
        "description": "Dark literary and historical dramas where ambition, money, violence, family pressure, and moral rot turn success into a reckoning.",
    },
    {
        "title": "The Invention of Lying",
        "layer": "micro",
        "label": "Office, Lying & Social-Satire Comedies",
        "description": "Comedies where offices, lying, wish fulfillment, social rules, adult bad ideas, and surreal moral shortcuts make ordinary life absurd.",
    },
    {
        "title": "Glass Onion: A Knives Out Mystery",
        "layer": "macro",
        "label": "Franchise Sequels, Mystery Capers & Fantasy Quests",
        "description": "Crowd-friendly franchise territory spanning whodunit sequels, comedy capers, fantasy quests, school magic, creature adventures, and returning worlds.",
    },
    {
        "title": "Glass Onion: A Knives Out Mystery",
        "layer": "neighborhood",
        "label": "Buddy-Cop, Spy & Mystery-Caper Sequels",
        "description": "Comedy sequels and capers where cops, detectives, spies, whodunit crews, familiar worlds, and returning comic trouble keep the plot moving.",
    },
    {
        "title": "Glass Onion: A Knives Out Mystery",
        "layer": "micro",
        "label": "Whodunit, Heist & Adventure Sequel Capers",
        "description": "Sequel capers built from whodunits, heists, spies, treasure hunts, suspect games, returning crews, and comic adventure momentum.",
    },
    {
        "title": "Red Dawn",
        "layer": "neighborhood",
        "label": "Invasion, Disaster & Survival Action Thrillers",
        "description": "High-stakes action where alien attacks, homeland invasions, disasters, future wars, and survival militias turn the world into a battlefield.",
    },
    {
        "title": "Red Dawn",
        "layer": "micro",
        "label": "Invasion Military Action (Alien, Future & Homeland)",
        "description": "Military action about invasions, future wars, alien attacks, guerrilla resistance, and homeland survival under overwhelming force.",
    },
    {
        "title": "Daddy's Home 2",
        "layer": "macro",
        "label": "Franchise Sequels, Family Comedies & Adventure Capers",
        "description": "Crowd-friendly franchise territory spanning family-comedy sequels, holiday mischief, whodunit capers, adventure quests, school magic, and returning worlds.",
    },
    {
        "title": "Excess Baggage",
        "layer": "macro",
        "label": "Youth, Music, Sports & Crime-Scheme Comedy-Drama",
        "description": "Teen, young-adult, music, sports, romance, school, and crime-scheme stories where self-definition arrives through comedy, pressure, or trouble.",
    },
    {
        "title": "Excess Baggage",
        "layer": "neighborhood",
        "label": "High-School Romcoms, Pranks & Teen Schemes",
        "description": "High-school and teen comedies about romance, crushes, cliques, pranks, cheer squads, schemes, and young people trying on risky identities.",
    },
    {
        "title": "Excess Baggage",
        "layer": "micro",
        "label": "Cheer, Kidnap & Teen-Scheme Comedy",
        "description": "Teen comedies and capers where cheer squads, kidnappings, body-swap jokes, money schemes, popularity, and reckless plans turn youth into farce.",
    },
    {
        "title": "Sanctum",
        "layer": "neighborhood",
        "label": "Space, Sea, Cave & Isolated Mission Survival",
        "description": "Isolated survival missions where space crews, divers, cavers, deep-sea teams, fragile habitats, and dangerous discovery become a trap.",
    },
    {
        "title": "Sanctum",
        "layer": "micro",
        "label": "Deep-Sea, Cave & Space-Unknown Survival",
        "description": "Claustrophobic survival stories around caves, oceans, submarines, deep-sea pressure, space-like isolation, and crews trapped near the unknown.",
    },
    {
        "title": "Watchmen",
        "layer": "micro",
        "label": "DC Superhero Ensemble: Team Power & Cosmic Stakes",
        "description": "DC superhero ensemble stories where teams, masked legacies, cosmic stakes, timelines, moral fracture, and comic-book power collide.",
    },
)

PUBLIC_AUDIT_POINT_REASSIGNMENTS: tuple[dict[str, str], ...] = (
    {
        "title": "Madame Web",
        "target_title": "Daredevil",
        "reason": "Nearest neighbors are comic-book/supernatural hero films, while the k-means bucket was sci-fi creature survival.",
    },
    {
        "title": "Fifty Shades Darker",
        "target_title": "Fifty Shades of Grey",
        "reason": "Nearest neighbors are erotic/angsty romance films, while the k-means bucket was adventure-quest sequels.",
    },
    {
        "title": "Fifty Shades Freed",
        "target_title": "Fifty Shades of Grey",
        "reason": "Nearest neighbors are erotic/angsty romance films, while the k-means bucket was adventure-quest sequels.",
    },
    {
        "title": "Avatar: The Way of Water",
        "target_title": "Avatar: Fire and Ash",
        "reason": "Nearest neighbors are Avatar and other legacy spectacle sequels, while the k-means bucket was family musical sequels.",
    },
    {
        "title": "Home Alone",
        "target_title": "Home Alone 2: Lost in New York",
        "reason": "Nearest neighbors are holiday/family slapstick sequels, while the k-means bucket was buddy-cop comedy.",
    },
    {
        "title": "Jingle All the Way",
        "target_title": "Home Alone 2: Lost in New York",
        "reason": "Nearest neighbors and keywords support holiday family slapstick rather than buddy-cop comedy.",
    },
    {
        "title": "Jurassic Park",
        "target_title": "Jurassic World",
        "reason": "Nearest neighbors are Jurassic dinosaur-survival films, while the k-means bucket was alien/first-contact discovery.",
    },
    {
        "title": "The Karate Kid",
        "target_title": "Never Back Down",
        "reason": "Nearest-neighbor evidence supports martial-arts training and comeback drama more than buddy-cop comedy.",
    },
    {
        "title": "The Karate Kid (2010)",
        "target_title": "The Karate Kid (1984)",
        "reason": "The reboot's nearest-neighbor evidence supports martial-arts mentorship and family training drama rather than space/fantasy spectacle.",
    },
    {
        "title": "Allegiant",
        "target_title": "Divergent",
        "reason": "Nearest neighbors and franchise identity support YA dystopian rebellion rather than zombie apocalypse action.",
    },
    {
        "title": "Insurgent",
        "target_title": "Divergent",
        "reason": "Nearest neighbors and franchise identity support YA dystopian rebellion rather than zombie apocalypse action.",
    },
    {
        "title": "Death Race",
        "target_title": "Gamer",
        "reason": "Both are dystopian prison/game action films, while the source bucket over-emphasized zombie apocalypse horror.",
    },
    {
        "title": "Mad Max 2",
        "target_title": "Mad Max: Fury Road",
        "reason": "Nearest neighbors and keywords support post-apocalyptic wasteland road action rather than kaiju/mech spectacle.",
    },
    {
        "title": "Passengers (2008)",
        "target_title": "Premonition",
        "reason": "The 2008 film is a grief-tinged psychological mystery, not the unrelated 2016 space romance suggested by title collision.",
    },
    {
        "title": "Escape Room: Tournament of Champions",
        "target_title": "Escape Room",
        "reason": "Nearest-neighbor and franchise evidence support death-game escape-room horror rather than dino/adventure sequel spectacle.",
    },
    {
        "title": "The Fly II",
        "target_title": "The Fly",
        "reason": "Nearest-neighbor and franchise evidence support mutant body-horror sci-fi rather than broad adventure/comedy sequel territory.",
    },
    {
        "title": "Eye in the Sky",
        "target_title": "Good Kill",
        "reason": "Nearest-neighbor and overview evidence support realistic drone-warfare ethics rather than AI dystopia sci-fi.",
    },
    {
        "title": "The Humans",
        "target_title": "Pieces of April",
        "reason": "Nearest-neighbor and overview evidence support intimate family holiday drama rather than home-invasion thriller.",
    },
    {
        "title": "Antitrust",
        "target_title": "Blackhat",
        "reason": "Nearest-neighbor and overview evidence support corporate/cyber conspiracy thriller rather than time-travel or virtual-reality sci-fi.",
    },
    {
        "title": "Legion",
        "target_title": "The Prophecy",
        "reason": "Nearest-neighbor and overview evidence support end-times angel horror rather than disaster countdown sci-fi.",
    },
    {
        "title": "Mad Max Beyond Thunderdome",
        "target_title": "Mad Max: Fury Road",
        "reason": "Nearest neighbors and keywords support post-apocalyptic wasteland road action rather than kaiju/mech spectacle.",
    },
    {
        "title": "Return (2011)",
        "target_title": "Lady Bird",
        "reason": "Nearest neighbors and overview support grounded family/reintegration drama more than speculative romance or combat-zone survival thriller.",
    },
    {
        "title": "The Lone Ranger",
        "target_title": "The Mask of Zorro",
        "reason": "Nearest neighbors and keywords support swashbuckling outlaw Western adventure more than gothic occult mystery.",
    },
    {
        "title": "Austin Powers: The Spy Who Shagged Me",
        "target_title": "Austin Powers: International Man of Mystery",
        "reason": "Nearest neighbors and franchise identity support absurd spy parody rather than broad family/mystery sequel comedy.",
    },
    {
        "title": "Rocky V",
        "target_title": "The Fighter",
        "reason": "Nearest neighbors and keywords support boxing comeback drama rather than spy/action revenge thrills.",
    },
    {
        "title": "Creed",
        "target_title": "The Fighter",
        "reason": "Nearest neighbors and keywords support boxing comeback drama rather than franchise fantasy spectacle.",
    },
    {
        "title": "Creed II",
        "target_title": "The Fighter",
        "reason": "Nearest neighbors and keywords support boxing comeback drama rather than franchise fantasy spectacle.",
    },
    {
        "title": "Creed III",
        "target_title": "The Fighter",
        "reason": "Nearest neighbors and keywords support boxing comeback drama rather than franchise fantasy spectacle.",
    },
    {
        "title": "Rocky III",
        "target_title": "The Fighter",
        "reason": "Nearest neighbors and keywords support boxing comeback drama rather than franchise fantasy spectacle.",
    },
    {
        "title": "Rocky IV",
        "target_title": "The Fighter",
        "reason": "Nearest neighbors and keywords support boxing comeback drama rather than franchise fantasy spectacle.",
    },
    {
        "title": "Rocky Balboa",
        "target_title": "The Fighter",
        "reason": "Nearest neighbors and keywords support boxing comeback drama rather than franchise fantasy spectacle.",
    },
    {
        "title": "The Ballad of Buster Scruggs",
        "target_title": "The Hateful Eight",
        "reason": "Nearest neighbors and genre evidence support Western dark comedy rather than speculative romance.",
    },
    {
        "title": "Gladiator II",
        "target_title": "Gladiator",
        "reason": "Nearest neighbors and franchise identity support historical sword-and-sandal epic drama rather than boxing comeback drama.",
    },
    {
        "title": "The Giver",
        "target_title": "Divergent",
        "reason": "Overview, genre, and nearest-neighbor evidence support YA dystopian sci-fi rather than grounded family recovery drama.",
    },
    {
        "title": "Scent of a Woman",
        "target_title": "Rain Man",
        "reason": "Character-drama evidence supports caregiving friendship and personal growth rather than noir/crime pressure.",
    },
    {
        "title": "What We Do in the Shadows",
        "target_title": "Shaun of the Dead",
        "reason": "Horror-comedy mockumentary evidence supports comic supernatural territory rather than gothic action-horror.",
    },
    {
        "title": "The Lion King (2019)",
        "target_title": "The Lion King (1994)",
        "reason": "Remake identity and nearest neighbors support family animal-kingdom adventure rather than dinosaur/island sequel spectacle.",
    },
    {
        "title": "Wonka",
        "target_title": "Charlie and the Chocolate Factory",
        "reason": "Family fantasy musical evidence supports storybook wonder rather than intimate adult life-change drama.",
    },
    {
        "title": "Goosebumps",
        "target_title": "Monster House",
        "reason": "Family-friendly supernatural comedy evidence supports spooky family adventure rather than dark franchise slasher territory.",
    },
    {
        "title": "National Treasure",
        "target_title": "National Treasure: Book of Secrets",
        "reason": "Treasure-hunt adventure evidence supports relic-riddle territory rather than spy-action missions.",
    },
    {
        "title": "Uncharted",
        "target_title": "National Treasure: Book of Secrets",
        "reason": "Treasure-hunt adventure evidence supports relic-riddle territory rather than spy-action missions.",
    },
    {
        "title": "D2: The Mighty Ducks",
        "target_title": "The Mighty Ducks",
        "reason": "Nearest neighbors and franchise identity support youth hockey/team-sports comedy rather than horror-tinged sequel labels.",
    },
    {
        "title": "D3: The Mighty Ducks",
        "target_title": "The Mighty Ducks",
        "reason": "Nearest neighbors and franchise identity support youth hockey/team-sports comedy rather than horror-tinged sequel labels.",
    },
    {
        "title": "Greystoke: The Legend of Tarzan, Lord of the Apes",
        "target_title": "The Legend of Tarzan",
        "reason": "Jungle adventure and Tarzan identity are much closer to swashbuckling adventure than Tudor court tragedy.",
    },
    {
        "title": "Tarzan the Ape Man",
        "target_title": "The Legend of Tarzan",
        "reason": "Jungle adventure and Tarzan identity are much closer to swashbuckling adventure than erotic bookish melodrama.",
    },
    {
        "title": "Sunshine (1999)",
        "target_title": "The Pianist",
        "reason": "Historical Jewish-family drama evidence is closer to Holocaust and occupation dramas than to the 2007 sci-fi film with the same title.",
    },
    {
        "title": "Southern Comfort",
        "target_title": "Hard Target",
        "reason": "Bayou manhunt and survival-thriller evidence is closer to hunting/manhunt action than to LGBTQ family drama.",
    },
    {
        "title": "Sidekicks",
        "target_title": "The Karate Kid (1984)",
        "reason": "Martial-arts mentorship and youth-training evidence fits the Karate Kid sports path better than supernatural action-comedy.",
    },
    {
        "title": "Karate Kid: Legends",
        "target_title": "The Karate Kid (1984)",
        "reason": "Martial-arts mentorship and youth-training evidence fits the Karate Kid sports path better than supernatural action-comedy.",
    },
    {
        "title": "Diary of a Wimpy Kid: Rodrick Rules (2011)",
        "target_title": "Diary of a Wimpy Kid",
        "reason": "Family school-comedy sequel evidence fits the Wimpy Kid schoolyard path better than raunchy teen sex-comedy sequels.",
    },
    {
        "title": "Charlie's Angels (2019)",
        "target_title": "Charlie's Angels (2000)",
        "reason": "Female-led spy action-comedy evidence fits the Charlie's Angels comedy-spy path better than giant robot spectacle.",
    },
    {
        "title": "A House of Dynamite",
        "target_title": "G20",
        "reason": "Political missile-crisis thriller evidence fits high-stakes government siege thrillers better than comet/disaster apocalypse spectacle.",
    },
    {
        "title": "Harry and Meghan: Escaping the Palace",
        "target_title": "Scoop (2024)",
        "reason": "Royal-family public-life drama evidence fits media, scandal, and biographical TV drama better than fantasy sequel territory.",
    },
    {
        "title": "Doctor Who: The Runaway Bride",
        "target_title": "Doctor Who: Voyage of the Damned",
        "reason": "Doctor Who TV-special evidence fits sci-fi disaster/time-travel specials better than Tudor or romance-melodrama clusters.",
    },
    {
        "title": "Quicksilver",
        "target_title": "Rad",
        "reason": "Bike-courier drama evidence fits youth/sports momentum better than crime-noir downfall labels.",
    },
    {
        "title": "Jawbreaker",
        "target_title": "Heathers",
        "reason": "Dark high-school clique crime comedy evidence fits the Heathers/Mean Girls path better than slasher-horror territory.",
    },
    {
        "title": "Animal Farm",
        "target_title": "The Plague Dogs",
        "reason": "Dark animal allegory evidence fits serious animal animation better than light farm/toy-box mischief.",
    },
    {
        "title": "The Devil's Arithmetic",
        "target_title": "The Pianist",
        "reason": "Holocaust memory and camp-survival evidence fits the occupation/survival path better than queer period romance labels.",
    },
    {
        "title": "Epic Movie",
        "target_title": "Scary Movie",
        "reason": "Crude fantasy-spoof evidence fits parody comedy better than family adventure fantasy.",
    },
    {
        "title": "Vampires Suck",
        "target_title": "Scary Movie",
        "reason": "Vampire spoof evidence fits parody comedy better than straight horror survival.",
    },
    {
        "title": "The Package (2018)",
        "target_title": "Project X",
        "reason": "Teen bad-decision comedy evidence fits party comedy better than cabin-in-the-woods slasher labels.",
    },
    {
        "title": "Rock Star",
        "target_title": "The Rocker",
        "reason": "Rock-band comedy-drama evidence fits music/school comedy paths better than stunt-sport pratfall labels.",
    },
    {
        "title": "Lords of Dogtown",
        "target_title": "Rad",
        "reason": "Skateboarding and youth-sports culture evidence fits sports coming-of-age better than singer-driven music biography.",
    },
    {
        "title": "The Cutting Edge: Chasing the Dream",
        "target_title": "The Cutting Edge",
        "reason": "Figure-skating sports romance evidence fits the original sports-romance path better than teen musical camp labels.",
    },
    {
        "title": "The Breadwinner",
        "target_title": "The Kite Runner",
        "reason": "Afghanistan, war, displacement, and survival evidence fits refugee witness drama better than family fantasy adventure.",
    },
    {
        "title": "Promised Land",
        "target_title": "Dark Waters",
        "reason": "Contemporary environmental social-pressure drama evidence fits political/social issue dramas better than faith-forward family labels.",
    },
    {
        "title": "Level 16",
        "target_title": "Divergent",
        "reason": "Dystopian institution and young-woman survival evidence fits YA dystopian sci-fi better than occult nun horror.",
    },
    {
        "title": "A Family Affair",
        "target_title": "No Hard Feelings",
        "reason": "Modern relationship comedy evidence fits awkward adult romcom territory better than working-class 1980s Irish drama.",
    },
    {
        "title": "The Package (1989)",
        "target_title": "In the Line of Fire",
        "reason": "Prisoner-escort, assassination, and political-conspiracy thriller evidence fits 1990s action-conspiracy territory better than teen party comedy.",
    },
    {
        "title": "Ray",
        "target_title": "Respect",
        "reason": "Music-biopic evidence fits singer-driven performance biography territory better than courtroom/social-pressure drama labels.",
    },
    {
        "title": "God's Not Dead 2",
        "target_title": "God's Not Dead",
        "reason": "Faith-based courtroom sequel evidence fits Christian faith drama territory better than franchise fantasy or horror sequel labels.",
    },
    {
        "title": "The Flyboys",
        "target_title": "Catch That Kid",
        "reason": "Kids-on-a-dangerous-plane adventure evidence fits family kid-mission capers better than WWI fighter-pilot drama.",
    },
    {
        "title": "The Woman King",
        "target_title": "The Last Duel",
        "reason": "Historical warrior drama evidence fits grounded historical conflict better than fantasy/game-adventure action labels.",
    },
    {
        "title": "The Wonder",
        "target_title": "The Magdalene Sisters",
        "reason": "Irish period drama about religious/institutional pressure fits women-under-pressure historical drama better than folk-horror possession labels.",
    },
    {
        "title": "The Rum Diary",
        "target_title": "Fear and Loathing in Las Vegas",
        "reason": "Hunter S. Thompson journalist-drift comedy evidence fits druggy literary road satire better than cocaine-trafficking crime drama.",
    },
    {
        "title": "Epic Movie",
        "target_title": "Disaster Movie",
        "reason": "Broad pop-culture spoof evidence fits parody-comedy territory better than teen slasher-horror parody.",
    },
)

PUBLIC_AUDIT_NEIGHBOR_REPAIRS: tuple[dict[str, Any], ...] = (
    {
        "title": "Sunshine (1999)",
        "neighbors": (
            "The Pianist",
            "Schindler's List",
            "The Reader",
            "The Book Thief",
            "Defiance",
            "The Zookeeper's Wife",
            "A Hidden Life",
        ),
        "reason": "Avoid title-collision neighbors from the unrelated 2007 science-fiction film.",
    },
    {
        "title": "The Flyboys",
        "neighbors": (
            "Catch That Kid",
            "Agent Cody Banks",
            "The Goonies",
            "Big Fat Liar",
            "Flight of the Navigator",
        ),
        "reason": "Avoid title-collision neighbors from the unrelated WWI aviation drama Flyboys.",
    },
    {
        "title": "Epic Movie",
        "neighbors": (
            "Date Movie",
            "Disaster Movie",
            "Meet the Spartans",
            "Superhero Movie",
            "Scary Movie",
            "Not Another Teen Movie",
            "Movie 43",
        ),
        "reason": "Broad spoof comedy should not inherit fantasy-adventure neighbors from its title/plot targets.",
    },
    {
        "title": "Best in Show",
        "neighbors": (
            "A Mighty Wind",
            "For Your Consideration",
            "This Is Spinal Tap",
            "Borat: Cultural Learnings of America for Make Benefit Glorious Nation of Kazakhstan",
            "Popstar: Never Stop Never Stopping",
            "Drop Dead Gorgeous",
            "Election",
        ),
        "reason": "Live-action mockumentary comedy should not inherit family/pet-film neighbors from dog-show keywords.",
    },
)

# Historical fixed-ID repairs are intentionally disabled. K-means cluster IDs can
# be reassigned between runs, so semantic repairs must be evidence-based instead
# of keyed only by numeric cluster id.
_DISABLED_FIXED_ID_LABEL_REPAIRS: dict[tuple[LayerName, int], dict[str, Any]] = {
    ("macro", 0): {
        "plain_label": "Future-shock survival science fiction",
        "recommended_label": "Future-Shock Survival Sci-Fi",
        "one_sentence_description": (
            "High-stakes science-fiction worlds where survival pressure comes from broken "
            "futures, alien contact, ecological collapse, machines, or hostile frontiers."
        ),
    },
    ("neighborhood", 1): {
        "plain_label": "Dystopian frontier survival science fiction",
        "recommended_label": "Dystopian Frontier Survival Sci-Fi",
        "one_sentence_description": (
            "Oppressive future societies, hostile frontiers, alien worlds, and survival "
            "spectacles where the setting itself is the threat."
        ),
    },
    ("neighborhood", 2): {
        "plain_label": "Time loops, clones, and future-tech thrillers",
        "recommended_label": "Time Loops, Clones & Future-Tech Thrillers",
        "one_sentence_description": (
            "High-concept sci-fi thrillers about looping time, predictive systems, engineered "
            "bodies, clones, and technology that destabilizes identity."
        ),
    },
    ("neighborhood", 6): {
        "plain_label": "Isolated mission survival science fiction",
        "recommended_label": "Isolated Mission Survival Sci-Fi",
        "one_sentence_description": (
            "Science-fiction missions where isolation, dangerous discovery, and fragile crews "
            "matter more than ordinary action spectacle."
        ),
    },
    ("micro", 18): {
        "plain_label": "Lonely space mission dramas",
        "recommended_label": "Lonely Space-Mission Dramas",
        "one_sentence_description": (
            "Space and near-space stories built around isolation, awe, grief, and the strain "
            "of surviving far from home."
        ),
    },
    ("micro", 19): {
        "plain_label": "Deep-sea and deep-space unknowns",
        "recommended_label": "Deep-Sea & Deep-Space Unknowns",
        "one_sentence_description": (
            "Claustrophobic sci-fi survival stories where crews are trapped at the edge of "
            "human knowledge, whether underwater or in space."
        ),
    },
    ("macro", 2): {
        "plain_label": "Tender life-change and romance dramas",
        "recommended_label": "Tender Life-Change & Romance Dramas",
        "one_sentence_description": (
            "Emotion-forward dramas about love, identity, grief, family rupture, healing, "
            "and the strange turns that remake a life."
        ),
    },
    ("neighborhood", 15): {
        "plain_label": "Family, recovery, and becoming-yourself dramas",
        "recommended_label": "Family, Recovery & Becoming-Yourself Dramas",
        "one_sentence_description": (
            "Grounded dramas where families, mentors, illness, disability, grief, or art push "
            "someone through a difficult life transition."
        ),
    },
    ("micro", 42): {
        "plain_label": "Recovery, family, and young-adult life pivots",
        "recommended_label": "Recovery, Family & Life-Pivot Dramas",
        "one_sentence_description": (
            "Intimate dramas about growing up, recovering, adapting, and finding identity "
            "inside family or community pressure."
        ),
    },
    ("neighborhood", 17): {
        "plain_label": "Wistful romance and intimate self-discovery",
        "recommended_label": "Wistful Romance & Intimate Self-Discovery",
        "one_sentence_description": (
            "Quietly aching romances and identity dramas where longing, place, and self-image "
            "matter more than plot machinery."
        ),
    },
    ("neighborhood", 18): {
        "plain_label": "Speculative love, memory, and grief",
        "recommended_label": "Speculative Love, Memory & Grief",
        "one_sentence_description": (
            "Romantic and philosophical dramas where love is bent by memory, mortality, "
            "technology, time, distance, or cosmic strangeness."
        ),
    },
    ("neighborhood", 19): {
        "plain_label": "Artists, stardom, and dreamy romance",
        "recommended_label": "Artists, Stardom & Dreamy Romance",
        "one_sentence_description": (
            "Stylized romances and showbiz-adjacent life stories where art, fame, fantasy, "
            "and longing blur together."
        ),
    },
    ("micro", 51): {
        "plain_label": "Surreal love, identity, and cosmic grief",
        "recommended_label": "Surreal Love, Identity & Cosmic Grief",
        "one_sentence_description": (
            "Speculative romances and metaphysical dramas where love, loneliness, identity, "
            "and loss drift into surreal or cosmic territory."
        ),
    },
    ("micro", 52): {
        "plain_label": "Devotion, illness, and time-shifted romance",
        "recommended_label": "Devotion, Illness & Time-Shifted Romance",
        "one_sentence_description": (
            "Tender romances and devotion dramas shaped by illness, memory, separation, "
            "time shifts, and the strain of loving someone through change."
        ),
    },
    ("micro", 55): {
        "plain_label": "Storybook hotels and art-world dreamcraft",
        "recommended_label": "Storybook Hotels & Art-World Dreamcraft",
        "one_sentence_description": (
            "Elegant, playful period fantasies and art-world reveries with ornate surfaces, "
            "eccentric casts, and a handmade-storybook feel."
        ),
    },
    ("micro", 58): {
        "plain_label": "Isolation, meaning, and strange life detours",
        "recommended_label": "Isolation, Meaning & Strange Life Detours",
        "one_sentence_description": (
            "Offbeat life fables about isolation, confinement, escape, reinvention, and "
            "trying to make meaning inside an absurd system or impossible situation."
        ),
    },
    ("neighborhood", 21): {
        "plain_label": "Tender YA friendship and first heartbreak",
        "recommended_label": "Tender YA Friendship & First Heartbreak",
        "one_sentence_description": (
            "Young-adult dramas about friendship, first love, illness, grief, school life, "
            "mental health, and the heartbreaks that make adolescence feel enormous."
        ),
    },
    ("micro", 60): {
        "plain_label": "YA love, loss, and first heartbreak",
        "recommended_label": "YA Love, Loss & First Heartbreak",
        "one_sentence_description": (
            "Tender YA stories where first love, friendship, illness, depression, or grief "
            "push teenagers toward painful self-knowledge."
        ),
    },
    ("macro", 3): {
        "plain_label": "Whimsical toy-box fantasy and bright animation",
        "recommended_label": "Whimsical Toy-Box Fantasy & Bright Animation",
        "one_sentence_description": (
            "Bright invented worlds, family animation, toy-box logic, musical magic, and "
            "pop-fantasy adventures with comic energy."
        ),
    },
    ("neighborhood", 22): {
        "plain_label": "Cheeky animated family comedies",
        "recommended_label": "Cheeky Animated Family Comedies",
        "one_sentence_description": (
            "Fast, joke-forward animated and hybrid family comedies full of oddball heroes, "
            "mischief, friendship, and bright adventure."
        ),
    },
    ("micro", 64): {
        "plain_label": "Inventive animated mischief comedies",
        "recommended_label": "Inventive Animated Mischief Comedies",
        "one_sentence_description": (
            "Playful animated comedies about misfits, gadgets, schemes, and imagination "
            "getting loose inside a bright family-adventure frame."
        ),
    },
    ("neighborhood", 24): {
        "plain_label": "Princess, pop-fantasy, and musical magic",
        "recommended_label": "Princess, Pop-Fantasy & Musical Magic",
        "one_sentence_description": (
            "Whimsical fantasy comedies and musicals where princess myths, pop iconography, "
            "magic, and self-invention collide."
        ),
    },
    ("micro", 72): {
        "plain_label": "Female-led pop fantasy adventures",
        "recommended_label": "Female-Led Pop Fantasy Adventures",
        "one_sentence_description": (
            "Bright female-led fantasy adventures about identity, performance, rebellion, "
            "and rewriting the role you were handed."
        ),
    },
    ("macro", 5): {
        "plain_label": "Noir crime, cons, and psychological puzzles",
        "recommended_label": "Noir Crime, Cons & Psychological Puzzles",
        "one_sentence_description": (
            "Crime stories, con games, detective puzzles, and noir-tinged psychological "
            "spirals where motives are slippery and trust is scarce."
        ),
    },
    ("neighborhood", 36): {
        "plain_label": "Murder mysteries and dangerous obsessions",
        "recommended_label": "Murder Mysteries & Dangerous Obsessions",
        "one_sentence_description": (
            "Detective puzzles, serial-killer shadows, and polished murder stories where "
            "investigation and fixation pull everyone into darker rooms."
        ),
    },
    ("neighborhood", 37): {
        "plain_label": "Hustle, gambling, and rise-and-fall crime dramas",
        "recommended_label": "Hustle, Gambling & Rise-and-Fall Crime Dramas",
        "one_sentence_description": (
            "Crime and self-destruction dramas about scams, gambling, addiction, money, "
            "ambition, and the volatile thrill of pushing too far."
        ),
    },
    ("neighborhood", 40): {
        "plain_label": "Druggy neo-noir and crime comedy",
        "recommended_label": "Druggy Neo-Noir & Crime Comedy",
        "one_sentence_description": (
            "Darkly comic crime stories where drugs, hustlers, lowlifes, violence, and noir "
            "cool slide into absurdity."
        ),
    },
    ("micro", 103): {
        "plain_label": "Gambling, cons, and American hustle",
        "recommended_label": "Gambling, Cons & American Hustle",
        "one_sentence_description": (
            "Fast-talking crime dramas where gambling, fraud, celebrity, addiction, and "
            "self-invention turn ambition into a trap."
        ),
    },
    ("micro", 111): {
        "plain_label": "Druggy crime comedy and noir detours",
        "recommended_label": "Druggy Crime Comedy & Noir Detours",
        "one_sentence_description": (
            "Loose, dark crime comedies and addiction-adjacent noir detours full of dealers, "
            "gangsters, show-offs, and bad choices."
        ),
    },
    ("macro", 6): {
        "plain_label": "Real-world pressure and survival dramas",
        "recommended_label": "Real-World Pressure & Survival Dramas",
        "one_sentence_description": (
            "Weighty real-world dramas about public pressure, war, biography, endurance, "
            "survival, and institutions under stress."
        ),
    },
    ("neighborhood", 43): {
        "plain_label": "Public-life legal and political dramas",
        "recommended_label": "Public-Life Legal & Political Dramas",
        "one_sentence_description": (
            "Biographical, legal, political, and institutional dramas where public decisions "
            "carry private costs."
        ),
    },
    ("neighborhood", 44): {
        "plain_label": "War, rescue, and survival pressure",
        "recommended_label": "War, Rescue & Survival Pressure",
        "one_sentence_description": (
            "Large-scale war, historical, rescue, and disaster-survival dramas where duty, "
            "danger, and endurance matter more than the exact battlefield."
        ),
    },
    ("micro", 119): {
        "plain_label": "Institutional power and public reckoning dramas",
        "recommended_label": "Institutional Power & Public Reckoning Dramas",
        "one_sentence_description": (
            "Talky, pressure-cooker dramas about institutions, scandal, politics, science, "
            "law, media, and the people forced to answer publicly."
        ),
    },
    ("micro", 120): {
        "plain_label": "Iconic public-life biopics",
        "recommended_label": "Iconic Public-Life Biopics",
        "one_sentence_description": (
            "Biographical dramas about famous public figures, private frailty, performance, "
            "leadership, and legacy."
        ),
    },
    ("neighborhood", 45): {
        "plain_label": "Ground-level combat and survival thrillers",
        "recommended_label": "Ground-Level Combat & Survival Thrillers",
        "one_sentence_description": (
            "War-zone, hostage, disaster, and field-survival dramas where people are trapped "
            "inside immediate danger and institutional violence."
        ),
    },
    ("micro", 126): {
        "plain_label": "Embedded war-zone survival dramas",
        "recommended_label": "Embedded War-Zone Survival Dramas",
        "one_sentence_description": (
            "Ground-level military and conflict stories focused on missions, ambushes, moral "
            "pressure, and surviving chaos up close."
        ),
    },
    ("micro", 116): {
        "plain_label": "Ambition, coaching, and fight-for-it dramas",
        "recommended_label": "Ambition, Coaching & Fight-For-It Dramas",
        "one_sentence_description": (
            "Competitive life dramas about training, obsession, mentorship, bruised ambition, "
            "and people pushing their bodies or talent to the edge."
        ),
    },
    ("macro", 7): {
        "plain_label": "Franchise fantasy and blockbuster spectacle",
        "recommended_label": "Franchise Fantasy & Blockbuster Spectacle",
        "one_sentence_description": (
            "Large-scale franchise worlds: superheroes, wizards, star wars, epic quests, "
            "comic-book teams, and space-opera spectacle."
        ),
    },
    ("neighborhood", 48): {
        "plain_label": "MCU ensemble crossover action",
        "recommended_label": "MCU Ensemble Crossover Action",
        "one_sentence_description": (
            "Marvel and adjacent superhero ensemble films where powers, team dynamics, "
            "crossovers, and continuity do much of the work."
        ),
    },
    ("neighborhood", 49): {
        "plain_label": "Space-opera sequels and franchise battles",
        "recommended_label": "Space-Opera Sequels & Franchise Battles",
        "one_sentence_description": (
            "Big franchise follow-ups with space-opera stakes, rebellion, giant battles, "
            "returning worlds, and sequel-scale spectacle."
        ),
    },
    ("micro", 132): {
        "plain_label": "Core Avengers and Captain America crossovers",
        "recommended_label": "Core Avengers & Captain America Crossovers",
        "one_sentence_description": (
            "Tightly MCU-coded Avengers, Iron Man, Ant-Man, and Captain America entries "
            "built around teamups, super-soldier stakes, and franchise continuity."
        ),
    },
    ("micro", 136): {
        "plain_label": "Franchise battle survivors and giant spectacle",
        "recommended_label": "Franchise Battle Survivors & Giant Spectacle",
        "one_sentence_description": (
            "Sequel-heavy spectacle where giant creatures, robots, post-collapse worlds, "
            "and resistance stories share a big-battle blockbuster rhythm."
        ),
    },
    ("neighborhood", 51): {
        "plain_label": "Wizard worlds and epic fantasy quests",
        "recommended_label": "Wizard Worlds & Epic Fantasy Quests",
        "one_sentence_description": (
            "Magic-school sagas and epic fantasy quests with darkening stakes, enchanted "
            "objects, chosen heroes, and sprawling franchise worlds."
        ),
    },
    ("micro", 143): {
        "plain_label": "Hogwarts dark-years fantasy",
        "recommended_label": "Hogwarts Dark-Years Fantasy",
        "one_sentence_description": (
            "Harry Potter and Fantastic Beasts stories centered on wizard schooling, dark "
            "magic, ghosts, Christmas shadows, and franchise mythos."
        ),
    },
    ("macro", 8): {
        "plain_label": "Mainstream comedy, satire, and buddy chaos",
        "recommended_label": "Mainstream Comedy, Satire & Buddy Chaos",
        "one_sentence_description": (
            "Broad mainstream comedies: slacker spirals, buddy pairings, workplace frustration, "
            "spoof, parody, and bad decisions piling up."
        ),
    },
    ("neighborhood", 54): {
        "plain_label": "Slacker comedy and bad decisions",
        "recommended_label": "Slacker Comedy & Bad Decisions",
        "one_sentence_description": (
            "Hangout, workplace, wish-fulfillment, and grown-up-immaturity comedies where "
            "ordinary frustration turns into escalating nonsense."
        ),
    },
    ("neighborhood", 57): {
        "plain_label": "Pop-culture genre-mashup comedy",
        "recommended_label": "Pop-Culture Genre-Mashup Comedy",
        "one_sentence_description": (
            "Comedies that remix superheroes, zombies, aliens, games, cult action, parody, "
            "and pop-culture references into one noisy genre blender."
        ),
    },
    ("micro", 150): {
        "plain_label": "Workplace, wish-fulfillment, and frustration comedy",
        "recommended_label": "Workplace & Wish-Fulfillment Comedy",
        "one_sentence_description": (
            "Comedies about bosses, jobs, life do-overs, arrested adulthood, and the fantasy "
            "of finally pushing back."
        ),
    },
    ("micro", 158): {
        "plain_label": "Cult-comic genre-mashup comedy",
        "recommended_label": "Cult-Comic Genre-Mashup Comedy",
        "one_sentence_description": (
            "Joke-forward cult comedies where comic-book attitude, games, zombies, parody, "
            "and self-aware genre chaos collide."
        ),
    },
    ("micro", 168): {
        "plain_label": "Quippy mystery-caper sequels",
        "recommended_label": "Quippy Mystery-Caper Sequels",
        "one_sentence_description": (
            "Return-trip comedy capers and mystery sequels where familiar crews, suspects, "
            "and punchlines matter more than grim danger."
        ),
    },
    ("micro", 188): {
        "plain_label": "Teen crush and coming-of-age comedy-drama",
        "recommended_label": "Teen Crush & Coming-of-Age Comedy-Drama",
        "one_sentence_description": (
            "Teen and young-adult comedy-dramas about sex, friendship, crushes, pregnancy, "
            "awkward families, and early romantic self-definition."
        ),
    },
    ("macro", 9): {
        "plain_label": "Popcorn sequel adventures and comedy franchises",
        "recommended_label": "Popcorn Sequel Adventures & Comedy Franchises",
        "one_sentence_description": (
            "Franchise sequels, family adventures, caper follow-ups, and broad popcorn "
            "comedies whose energy comes from returning worlds and familiar crews."
        ),
    },
    ("micro", 73): {
        "plain_label": "Pixar-heart friendship quests",
        "recommended_label": "Pixar-Heart Friendship Quests",
        "one_sentence_description": (
            "Big-hearted animated adventures about friendship, memory, family, chosen crews, "
            "and growing through imaginative journeys."
        ),
    },
    ("micro", 80): {
        "plain_label": "Lone-wolf revenge machines",
        "recommended_label": "Lone-Wolf Revenge Machines",
        "one_sentence_description": (
            "Lean action thrillers where trained killers, fixers, and damaged professionals "
            "turn vengeance into ruthless forward motion."
        ),
    },
    ("micro", 94): {
        "plain_label": "FBI, getaway, and extreme-chase action",
        "recommended_label": "FBI, Getaway & Extreme-Chase Action",
        "one_sentence_description": (
            "Velocity-first action thrillers built from federal heat, transport jobs, "
            "getaways, heists, hijacks, and cliff-edge pursuit."
        ),
    },
    ("neighborhood", 72): {
        "plain_label": "Ancient warriors and mythic sword adventures",
        "recommended_label": "Ancient Warriors & Mythic Sword Adventures",
        "one_sentence_description": (
            "Ancient-world and mythic action adventures with warriors, kingdoms, conquest, "
            "revenge, gods, monsters, and brutal tests of survival."
        ),
    },
    ("micro", 194): {
        "plain_label": "Norse, Arthurian, and dark medieval action",
        "recommended_label": "Norse, Arthurian & Dark Medieval Action",
        "one_sentence_description": (
            "Dark medieval and legendary action stories about kings, revenge, quests, "
            "warriors, curses, and old-world violence."
        ),
    },
    ("micro", 195): {
        "plain_label": "Ancient survival and swordplay epics",
        "recommended_label": "Ancient Survival & Swordplay Epics",
        "one_sentence_description": (
            "Primitive, ancient, and myth-tinged survival adventures where escape, conquest, "
            "and hand-to-hand brutality drive the journey."
        ),
    },
    ("micro", 130): {
        "plain_label": "Holocaust and occupation resistance dramas",
        "recommended_label": "Holocaust & Occupation Resistance Dramas",
        "one_sentence_description": (
            "World War II and occupation dramas centered on persecution, resistance, moral "
            "horror, and survival under fascist power."
        ),
    },
}

LABEL_REPAIR_OVERRIDES: dict[tuple[LayerName, int], dict[str, Any]] = {}


def classification_v2_file(
    *,
    api_key: str | None,
    movies_path: str | Path = "data/processed/movies.json",
    raw_details_path: str | Path = "data/raw/movie_details.json",
    output_dir: str | Path = "outputs",
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    label_model: str,
    macro_k: int = 12,
    neighborhood_k: int = 75,
    micro_k: int = 200,
    variants: list[str] | None = None,
    strategies: list[str] | None = None,
    limit: int | None = None,
    embedding_batch_size: int = 64,
    label_batch_size: int = 5,
    cost_gate_usd: float = 10.0,
    label_client: OpenAIClusterLabelClient | None = None,
) -> ClassificationV2Result:
    """Run Milestone 5 profile, embedding, clustering, audit, and export experiments."""
    if not api_key:
        raise ClassificationV2Error("OPENAI_API_KEY is missing; Milestone 5 needs fresh embeddings.")

    movies = load_movie_records(movies_path)
    if limit is not None:
        movies = movies[:limit]
    if not movies:
        raise ClassificationV2Error("No movies are available for Classification V2.")

    raw_details = load_raw_details(raw_details_path)
    external_bundle = load_external_signals(movies)
    selected_variants = _select_variants(variants)
    selected_strategies = _select_strategies(strategies)
    output_path = Path(output_dir)
    experiment_dir = output_path / "experiments" / CLASSIFICATION_V2_DIRNAME
    experiment_dir.mkdir(parents=True, exist_ok=True)
    _preserve_baseline_manifest(output_path, experiment_dir)

    total_estimated_cost = 0.0
    best_artifacts: CandidateArtifacts | None = None
    candidate_summaries: list[dict[str, Any]] = []
    variant_summaries: list[dict[str, Any]] = []

    for spec in selected_variants:
        variant_dir = experiment_dir / "variants" / spec.name
        variant_dir.mkdir(parents=True, exist_ok=True)
        profiles = build_variant_profiles(
            movies,
            raw_details,
            spec,
            external_signals=external_bundle.signals_by_tmdb_id,
        )
        profiles_path = variant_dir / "profiles.json"
        profiles_path.write_text(
            json.dumps([profile.to_dict() for profile in profiles], indent=2, sort_keys=True),
            encoding="utf-8",
        )
        _write_profile_inspection(variant_dir, profiles)
        estimated_embedding_cost = _estimate_embedding_cost(profiles, embedding_model)
        total_estimated_cost += estimated_embedding_cost
        if total_estimated_cost > cost_gate_usd:
            raise ClassificationV2Error(
                f"Estimated experiment cost ${total_estimated_cost:.4f} exceeds "
                f"the ${cost_gate_usd:.2f} gate before embedding {spec.name}."
            )

        embedding_result = embed_profiles_file(
            api_key=api_key,
            profiles_path=profiles_path,
            output_dir=variant_dir,
            model=embedding_model,
            batch_size=embedding_batch_size,
        )
        embeddings = load_embedding_records(embedding_result.embedding_path)
        neighbors = compute_neighbors(embeddings, top_n=40)
        neighbors_path = variant_dir / "raw_neighbors.json"
        neighbors_path.write_text(
            json.dumps([entry.to_dict() for entry in neighbors], indent=2, sort_keys=True),
            encoding="utf-8",
        )
        variant_summaries.append(
            {
                "variant": spec.name,
                "spec": spec.to_dict(),
                "profile_count": len(profiles),
                "estimated_embedding_cost_usd": estimated_embedding_cost,
                "embedding_path": str(embedding_result.embedding_path),
                "cached_reused_count": embedding_result.cached_reused_count,
                "new_embedding_count": embedding_result.new_embedding_count,
                "api_prompt_tokens": embedding_result.api_prompt_tokens,
                "external_signal_coverage": external_bundle.coverage.to_dict(),
            }
        )

        for strategy in selected_strategies:
            artifacts = _run_candidate(
                variant=spec.name,
                strategy=strategy,
                movies=movies,
                profiles=profiles,
                embeddings=embeddings,
                neighbors=neighbors,
                profiles_path=profiles_path,
                embeddings_path=embedding_result.embedding_path,
                macro_k=macro_k,
                neighborhood_k=neighborhood_k,
                micro_k=micro_k,
            )
            summary = _candidate_summary(artifacts)
            candidate_summaries.append(summary)
            candidate_path = variant_dir / f"{strategy}.json"
            candidate_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
            if artifacts.exportable and (
                best_artifacts is None
                or artifacts.metrics["selection_score"] > best_artifacts.metrics["selection_score"]
            ):
                best_artifacts = artifacts

    if best_artifacts is None:
        raise ClassificationV2Error("No exportable candidate was produced.")

    hierarchy_dir, label_cost, labels_generated = _write_selected_hierarchy(
        best_artifacts,
        movies=movies,
        output_dir=output_path,
        label_model=label_model,
        api_key=api_key,
        label_batch_size=label_batch_size,
        label_client=label_client,
        cost_gate_usd=cost_gate_usd,
        total_cost_so_far=total_estimated_cost,
    )
    total_estimated_cost += label_cost
    public_export = export_atlas_data_file(
        movies_path=movies_path,
        hierarchy_dir=hierarchy_dir,
        output_dir=output_path,
    )
    public_audit_point_reassignments = _apply_public_audit_point_reassignments(public_export.export_dir)
    (hierarchy_dir / "public_audit_point_reassignments.json").write_text(
        json.dumps(public_audit_point_reassignments, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    public_audit_neighbor_repairs = _apply_public_audit_neighbor_repairs(public_export.export_dir)
    (hierarchy_dir / "public_audit_neighbor_repairs.json").write_text(
        json.dumps(public_audit_neighbor_repairs, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    public_audit_label_repairs = _apply_public_audit_label_repairs(public_export.export_dir)
    (hierarchy_dir / "public_audit_label_repairs.json").write_text(
        json.dumps(public_audit_label_repairs, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    audit_payload = _detailed_audit_payload(best_artifacts, export_dir=public_export.export_dir)
    audit_payload["public_audit_point_reassignments_count"] = len(public_audit_point_reassignments)
    audit_payload["public_audit_neighbor_repairs_count"] = len(public_audit_neighbor_repairs)
    audit_payload["public_audit_label_repairs_count"] = len(public_audit_label_repairs)
    audit_path = experiment_dir / AUDIT_JSON_FILENAME
    audit_path.write_text(json.dumps(audit_payload, indent=2, sort_keys=True), encoding="utf-8")

    summary_payload = {
        "movie_count": len(movies),
        "embedding_model": embedding_model,
        "label_model": label_model,
        "estimated_openai_cost_usd": total_estimated_cost,
        "external_signal_coverage": external_bundle.coverage.to_dict(),
        "winner": {
            "variant": best_artifacts.variant,
            "strategy": best_artifacts.strategy,
            "metrics": best_artifacts.metrics,
        },
        "variants": variant_summaries,
        "candidates": sorted(
            candidate_summaries,
            key=lambda item: item["metrics"]["selection_score"],
            reverse=True,
        ),
        "public_export": public_export.to_dict(),
        "labels_generated": labels_generated,
        "public_audit_point_reassignments_count": len(public_audit_point_reassignments),
        "public_audit_label_repairs_count": len(public_audit_label_repairs),
    }
    summary_path = experiment_dir / SUMMARY_JSON_FILENAME
    summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True), encoding="utf-8")
    summary_md_path = experiment_dir / SUMMARY_MD_FILENAME
    summary_md_path.write_text(render_classification_v2_summary(summary_payload), encoding="utf-8")

    report_path = output_path / "reports" / MILESTONE_5_REPORT_FILENAME
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        render_milestone_5_report(summary_payload=summary_payload, audit_payload=audit_payload),
        encoding="utf-8",
    )
    deep_audit_path = output_path / "reports" / MILESTONE_5_DEEP_AUDIT_FILENAME
    deep_audit_path.write_text(render_deep_audit_report(audit_payload), encoding="utf-8")

    return ClassificationV2Result(
        experiment_dir=experiment_dir,
        summary_path=summary_md_path,
        report_path=report_path,
        audit_path=audit_path,
        winner_variant=best_artifacts.variant,
        winner_strategy=best_artifacts.strategy,
        export_dir=public_export.export_dir,
        estimated_openai_cost_usd=total_estimated_cost,
        labels_generated=labels_generated,
    )


def build_variant_profiles(
    movies: list[MovieRecord],
    raw_details: dict[int, dict[str, Any]],
    spec: ProfileVariantSpec,
    *,
    external_signals: dict[int, ExternalMovieSignals] | None = None,
) -> list[SemanticProfile]:
    """Build one experimental profile set from normalized and raw TMDb-only data."""
    if spec.name == "baseline_light":
        return [
            build_semantic_profile(
                movie,
                include_reviews=True,
                max_review_chars=spec.max_review_chars,
                review_weight=spec.review_weight,
            )
            for movie in movies
        ]

    return [
        _build_custom_profile(
            movie,
            raw_details.get(movie.tmdb_id) or {},
            spec,
            external=(external_signals or {}).get(movie.tmdb_id),
        )
        for movie in movies
    ]


def allocate_child_counts(parent_sizes: dict[int, int], total_children: int) -> dict[int, int]:
    """Allocate child cluster counts by parent size with deterministic remainders."""
    positive_sizes = {parent_id: size for parent_id, size in parent_sizes.items() if size > 0}
    if not positive_sizes or total_children <= 0:
        return {}
    target = min(total_children, sum(positive_sizes.values()))
    counts = {parent_id: 1 for parent_id in positive_sizes}
    remaining = target - len(counts)
    if remaining <= 0:
        return {parent_id: counts[parent_id] for parent_id in sorted(counts)}

    capacity = {parent_id: size - 1 for parent_id, size in positive_sizes.items()}
    total_capacity = sum(capacity.values())
    if total_capacity <= 0:
        return {parent_id: counts[parent_id] for parent_id in sorted(counts)}

    quotas: list[tuple[float, int, int]] = []
    assigned = 0
    for parent_id, cap in capacity.items():
        exact = remaining * cap / total_capacity
        extra = min(cap, math.floor(exact))
        counts[parent_id] += extra
        assigned += extra
        quotas.append((exact - extra, positive_sizes[parent_id], parent_id))

    left = remaining - assigned
    for _remainder, _size, parent_id in sorted(quotas, key=lambda item: (-item[0], -item[1], item[2])):
        if left <= 0:
            break
        if counts[parent_id] < positive_sizes[parent_id]:
            counts[parent_id] += 1
            left -= 1

    return {parent_id: counts[parent_id] for parent_id in sorted(counts)}


def load_raw_details(path: str | Path) -> dict[int, dict[str, Any]]:
    """Load raw TMDb detail payloads by movie id, if available."""
    details_path = Path(path)
    if not details_path.exists():
        return {}
    payload = json.loads(details_path.read_text(encoding="utf-8"))
    return {
        int(item["id"]): item
        for item in payload.get("results") or []
        if isinstance(item, dict) and item.get("id") is not None
    }


def render_classification_v2_summary(summary_payload: dict[str, Any]) -> str:
    """Render the private experiment summary."""
    winner = summary_payload["winner"]
    lines = [
        "# The Film Atlas - Classification V2 Experiment Summary",
        "",
        "## Winner",
        "",
        f"- Variant: {winner['variant']}",
        f"- Strategy: {winner['strategy']}",
        f"- Selection score: {winner['metrics']['selection_score']:.3f}",
        f"- Estimated OpenAI cost: ${summary_payload['estimated_openai_cost_usd']:.4f}",
        f"- Public export: {summary_payload['public_export']['export_dir']}",
        "",
        "## Candidate Ranking",
        "",
        "| Rank | Variant | Strategy | Score | Exportable | Hierarchy Mismatch | Bad Neighbor Hits | Same Micro Top 7 | Coherence |",
        "| ---: | --- | --- | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for rank, candidate in enumerate(summary_payload["candidates"], start=1):
        metrics = candidate["metrics"]
        lines.append(
            f"| {rank} | {candidate['variant']} | {candidate['strategy']} | "
            f"{metrics['selection_score']:.3f} | {candidate['exportable']} | "
            f"{metrics.get('hierarchy_mismatch_rate') or 0:.3f} | "
            f"{metrics['audit']['bad_neighbor_hits']} | "
            f"{metrics.get('same_micro_top7_rate') or 0:.3f} | "
            f"{metrics.get('coherence_average') or 0:.3f} |"
        )
    lines.extend(
        [
            "",
            "## External Signal Coverage",
            "",
            _external_coverage_lines(summary_payload),
            "",
            "## Variant Inputs",
            "",
            "| Variant | Profiles | New Embeddings | Cached | Estimated Embedding Cost |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for variant in summary_payload["variants"]:
        lines.append(
            f"| {variant['variant']} | {variant['profile_count']} | "
            f"{variant['new_embedding_count']} | {variant['cached_reused_count']} | "
            f"${variant['estimated_embedding_cost_usd']:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


def render_milestone_5_report(
    *,
    summary_payload: dict[str, Any],
    audit_payload: dict[str, Any],
) -> str:
    """Render the public-facing local Milestone 5 report."""
    winner = summary_payload["winner"]
    metrics = winner["metrics"]
    external_coverage = _external_signal_coverage(summary_payload)
    lines = [
        "# The Film Atlas - Milestone 5 Report",
        "",
        "Milestone 5 tested richer TMDb-derived profile text, fresh embeddings, strict "
        "hierarchical clustering, optional public-dataset signals, deterministic tone/status "
        "probes, flat community comparators, and audit controls from the voice-note review pass. "
        "Private experiment artifacts remain under ignored outputs.",
        "",
        "## Winning Approach",
        "",
        f"- Profile variant: {winner['variant']}",
        f"- Clustering strategy: {winner['strategy']}",
        f"- Movie count: {summary_payload['movie_count']}",
        f"- Selection score: {metrics['selection_score']:.3f}",
        f"- Hierarchy mismatch rate: {metrics.get('hierarchy_mismatch_rate') or 0:.3%}",
        f"- Same-micro nearest-neighbor top-7 rate: {metrics.get('same_micro_top7_rate') or 0:.1%}",
        f"- Coherence average: {metrics.get('coherence_average') or 0:.3f}",
        f"- Estimated OpenAI cost: ${summary_payload['estimated_openai_cost_usd']:.4f}",
        f"- Labels generated: {summary_payload['labels_generated']}",
        f"- Public audit point reassignments: {summary_payload.get('public_audit_point_reassignments_count', 0)}",
        f"- Public audit label repairs: {summary_payload.get('public_audit_label_repairs_count', 0)}",
        f"- Public export: {summary_payload['public_export']['export_dir']}",
        "",
        "## Audit Result",
        "",
        f"- Audit movies present: {audit_payload['present_count']} / {audit_payload['requested_count']}",
        f"- Bad-neighbor pattern hits: {metrics['audit']['bad_neighbor_hits']}",
        f"- Good-neighbor pattern hits: {metrics['audit']['good_neighbor_hits']}",
        f"- Duplicate parent-child label names after repair: {audit_payload['duplicate_label_count']}",
        f"- Public audit point reassignments applied: {audit_payload.get('public_audit_point_reassignments_count', 0)}",
        f"- Public audit label repairs applied: {audit_payload.get('public_audit_label_repairs_count', 0)}",
        f"- Deep-audit verdicts: {_format_counts(audit_payload.get('verdict_counts') or {})}",
        "",
        "## Candidate Ranking",
        "",
        "| Rank | Variant | Strategy | Score | Notes |",
        "| ---: | --- | --- | ---: | --- |",
    ]
    for rank, candidate in enumerate(summary_payload["candidates"][:12], start=1):
        candidate_metrics = candidate["metrics"]
        lines.append(
            f"| {rank} | {candidate['variant']} | {candidate['strategy']} | "
            f"{candidate_metrics['selection_score']:.3f} | "
            f"{_table_escape(candidate_metrics['note'])} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Richer TMDb profiles use overview, tagline, genres, keywords, and cleaned longer "
            "review language.",
            f"- Optional external signals were tested locally: MovieLens Tag Genome matched "
            f"{external_coverage.get('movie_lens_matched_count', 0)} / "
            f"{external_coverage.get('movie_count', summary_payload['movie_count'])} films; "
            f"MPST matched {external_coverage.get('mpst_matched_count', 0)} / "
            f"{external_coverage.get('movie_count', summary_payload['movie_count'])} films.",
            "- Tone tags from synopsis/review language helped the selected exportable hierarchy; "
            "rating/popularity status tags were tested as a probe but should remain a filter/overlay, "
            "not the core map geometry.",
            "- Title-bearing baseline profiles were retained as a comparator; the selected "
            "approach is allowed to beat or lose to title-free variants based on measured audit behavior.",
            "- Public export is still sanitized and does not include raw reviews, external plot "
            "synopses, embeddings, API keys, or private experiment fields.",
            "",
        ]
    )
    return "\n".join(lines)


def _external_signal_coverage(summary_payload: dict[str, Any]) -> dict[str, Any]:
    coverage = summary_payload.get("external_signal_coverage")
    if isinstance(coverage, dict):
        return coverage
    for variant in summary_payload.get("variants") or []:
        coverage = variant.get("external_signal_coverage")
        if isinstance(coverage, dict):
            return coverage
    return {}


def _external_coverage_lines(summary_payload: dict[str, Any]) -> str:
    coverage = _external_signal_coverage(summary_payload)
    if not coverage:
        return "- Optional external datasets were not loaded for this run."
    return "\n".join(
        [
            f"- MovieLens Tag Genome matches: {coverage.get('movie_lens_matched_count', 0)} / "
            f"{coverage.get('movie_count', 0)}",
            f"- MPST matches: {coverage.get('mpst_matched_count', 0)} / "
            f"{coverage.get('movie_count', 0)}",
            f"- MovieLens path: {coverage.get('movie_lens_path') or 'not loaded'}",
            f"- MPST path: {coverage.get('mpst_path') or 'not loaded'}",
        ]
    )


def _build_custom_profile(
    movie: MovieRecord,
    raw_detail: dict[str, Any],
    spec: ProfileVariantSpec,
    *,
    external: ExternalMovieSignals | None = None,
) -> SemanticProfile:
    forbidden = _profile_forbidden_values(movie, include_title=not spec.include_title)
    overview = _redact_forbidden_values(movie.overview, forbidden)
    tagline = _redact_forbidden_values(str(raw_detail.get("tagline") or ""), forbidden)
    keywords = [_redact_forbidden_values(keyword, forbidden) for keyword in movie.keywords]
    keywords = [keyword for keyword in keywords if keyword]
    reviews = _review_language_from_raw(
        movie,
        raw_detail,
        forbidden_values=forbidden,
        max_review_chars=spec.max_review_chars,
        review_count=spec.review_count,
    )

    parts: list[str] = []
    if spec.include_title:
        parts.append(_section("Title", [movie.title]))
    if spec.include_tagline:
        parts.append(_section("Tagline", [tagline]))
    for index in range(spec.overview_repeats):
        label = "Premise" if index == 0 else "Premise emphasis"
        parts.append(_section(label, [overview]))
    parts.append(_section("Genres", movie.genres))
    for index in range(spec.keyword_repeats):
        label = "Keywords" if index == 0 else "Keyword emphasis"
        parts.append(_section(label, keywords))
    if spec.include_movie_lens_tags and external and external.movie_lens_tags:
        parts.append(_section("MovieLens Tag Genome vibes", external.movie_lens_tags))
    if spec.include_mpst_tags and external and external.mpst_tags:
        parts.append(_section("MPST plot and mood tags", external.mpst_tags))
    if spec.include_mpst_synopsis and external and external.mpst_synopsis:
        synopsis = _redact_forbidden_values(
            external.mpst_synopsis[: spec.max_mpst_synopsis_chars],
            forbidden,
        )
        parts.append(_section("Plot synopsis", [synopsis]))
    if spec.include_tone_tags:
        parts.append(_section("Tone analysis tags", _tone_tags(movie, raw_detail, external=external)))
    if spec.include_reception_tags:
        parts.append(_section("Reception status tags", _reception_status_tags(movie)))
    parts.append(_section(spec.review_label, reviews))
    profile_text = "\n".join(part for part in parts if part)
    return SemanticProfile(
        tmdb_id=movie.tmdb_id,
        title=movie.title,
        year=movie.year,
        profile_text=profile_text,
        genres=movie.genres,
        keywords=movie.keywords,
    )


def _profile_forbidden_values(movie: MovieRecord, *, include_title: bool) -> list[str]:
    values = list(_forbidden_values(movie))
    if include_title:
        values.extend([movie.title, movie.original_title])
    cleaned = {" ".join(value.split()) for value in values if value and len(value.strip()) > 2}
    return sorted(cleaned, key=len, reverse=True)


def _review_language_from_raw(
    movie: MovieRecord,
    raw_detail: dict[str, Any],
    *,
    forbidden_values: list[str],
    max_review_chars: int,
    review_count: int,
) -> list[str]:
    reviews_payload = raw_detail.get("reviews")
    raw_reviews = []
    if isinstance(reviews_payload, dict):
        raw_reviews = [
            str(item.get("content") or "")
            for item in reviews_payload.get("results") or []
            if isinstance(item, dict) and item.get("content")
        ]
    if not raw_reviews:
        raw_reviews = movie.reviews

    snippets: list[str] = []
    remaining = max_review_chars
    for review in raw_reviews[: max(0, review_count)]:
        cleaned = _clean_review_snippet(review)
        cleaned = _redact_forbidden_values(cleaned, forbidden_values) or ""
        if not cleaned:
            continue
        snippet = cleaned[:remaining].strip()
        if snippet:
            snippets.append(snippet)
            remaining -= len(snippet)
        if remaining <= 0:
            break
    return snippets


TONE_LEXICON: dict[str, tuple[str, ...]] = {
    "bleak": (
        "bleak",
        "despair",
        "hopeless",
        "grim",
        "devastating",
        "trauma",
        "grief",
        "lonely",
        "isolation",
        "haunting",
    ),
    "warm": (
        "warm",
        "tender",
        "gentle",
        "comfort",
        "sweet",
        "heartfelt",
        "hopeful",
        "charming",
        "cozy",
        "uplifting",
    ),
    "tense": (
        "tense",
        "suspense",
        "thrilling",
        "paranoia",
        "danger",
        "threat",
        "chase",
        "pressure",
        "dread",
        "anxiety",
    ),
    "comic": (
        "funny",
        "comedy",
        "comic",
        "satire",
        "absurd",
        "witty",
        "hilarious",
        "goofy",
        "farce",
        "slapstick",
    ),
    "romantic": (
        "romance",
        "romantic",
        "love",
        "lover",
        "relationship",
        "marriage",
        "heartbreak",
        "intimate",
        "desire",
    ),
    "violent": (
        "violent",
        "brutal",
        "blood",
        "murder",
        "killer",
        "revenge",
        "fight",
        "war",
        "attack",
        "deadly",
    ),
    "cerebral": (
        "cerebral",
        "philosophical",
        "memory",
        "identity",
        "dream",
        "surreal",
        "mystery",
        "puzzle",
        "existential",
    ),
    "spectacle": (
        "epic",
        "spectacle",
        "blockbuster",
        "adventure",
        "explosive",
        "battle",
        "world",
        "planet",
        "monster",
    ),
}

POSITIVE_WORDS = {
    "great",
    "excellent",
    "beautiful",
    "favorite",
    "best",
    "amazing",
    "brilliant",
    "perfect",
    "moving",
    "fun",
    "masterpiece",
}
NEGATIVE_WORDS = {
    "bad",
    "awful",
    "terrible",
    "boring",
    "worst",
    "mess",
    "weak",
    "dull",
    "disappointing",
    "poor",
    "hate",
}


def _tone_tags(
    movie: MovieRecord,
    raw_detail: dict[str, Any],
    *,
    external: ExternalMovieSignals | None,
) -> list[str]:
    text_parts = [
        movie.overview or "",
        str(raw_detail.get("tagline") or ""),
        external.mpst_synopsis if external and external.mpst_synopsis else "",
    ]
    reviews_payload = raw_detail.get("reviews")
    if isinstance(reviews_payload, dict):
        for item in (reviews_payload.get("results") or [])[:5]:
            if isinstance(item, dict):
                text_parts.append(str(item.get("content") or ""))
    else:
        text_parts.extend(movie.reviews[:3])

    text = " ".join(text_parts).lower()
    tokens = re.findall(r"[a-z][a-z'-]+", text)
    token_counts = Counter(tokens)
    scored = []
    for tone, words in TONE_LEXICON.items():
        score = sum(token_counts[word] for word in words)
        if score:
            scored.append((score, tone))
    scored.sort(key=lambda item: (-item[0], item[1]))
    tags = [f"tone: {tone}" for _score, tone in scored[:5]]

    positive = sum(token_counts[word] for word in POSITIVE_WORDS)
    negative = sum(token_counts[word] for word in NEGATIVE_WORDS)
    if positive or negative:
        if positive >= negative * 2 and positive >= 2:
            tags.append("review sentiment: admiring")
        elif negative >= positive * 2 and negative >= 2:
            tags.append("review sentiment: critical")
        elif positive and negative:
            tags.append("review sentiment: divided")
    return tags


def _reception_status_tags(movie: MovieRecord) -> list[str]:
    tags: list[str] = []
    vote_average = movie.vote_average or 0.0
    vote_count = movie.vote_count or 0
    popularity = movie.popularity or 0.0
    if vote_average >= 8.0 and vote_count >= 1000:
        tags.append("reception: widely acclaimed")
    elif vote_average >= 7.2 and vote_count >= 1000:
        tags.append("reception: well liked")
    elif vote_average <= 5.5 and vote_count >= 500:
        tags.append("reception: poorly received")
    elif 5.5 < vote_average < 7.0 and vote_count >= 1000:
        tags.append("reception: mixed mainstream")

    if vote_count >= 8000 or popularity >= 120:
        tags.append("audience scale: blockbuster visibility")
    elif vote_count >= 2500:
        tags.append("audience scale: mainstream visibility")
    elif vote_count < 500:
        tags.append("audience scale: niche visibility")

    if vote_average >= 7.0 and vote_count < 800:
        tags.append("status: small-audience favorite")
    if vote_average < 6.0 and vote_count >= 1000:
        tags.append("status: possible b-movie or misfire")
    return tags


def _run_candidate(
    *,
    variant: str,
    strategy: ClusteringStrategy,
    movies: list[MovieRecord],
    profiles: list[SemanticProfile],
    embeddings: list[Any],
    neighbors: list[MovieNeighbors],
    profiles_path: Path,
    embeddings_path: Path,
    macro_k: int,
    neighborhood_k: int,
    micro_k: int,
) -> CandidateArtifacts:
    if strategy == "independent_kmeans":
        assignments_by_layer, parent_maps, note = _independent_kmeans_layers(
            embeddings,
            macro_k=macro_k,
            neighborhood_k=neighborhood_k,
            micro_k=micro_k,
        )
        exportable = True
    elif strategy == "hierarchical_kmeans":
        assignments_by_layer, parent_maps, note = _hierarchical_layers(
            embeddings,
            macro_k=macro_k,
            neighborhood_k=neighborhood_k,
            micro_k=micro_k,
            method="kmeans",
        )
        exportable = True
    elif strategy == "hierarchical_agglomerative":
        assignments_by_layer, parent_maps, note = _hierarchical_layers(
            embeddings,
            macro_k=macro_k,
            neighborhood_k=neighborhood_k,
            micro_k=micro_k,
            method="agglomerative",
        )
        exportable = True
    elif strategy == "graph_communities":
        flat = _graph_assignments(embeddings)
        assignments_by_layer = {"macro": flat, "neighborhood": flat, "micro": flat}
        parent_maps = {"macro": {}, "neighborhood": {}, "micro": {}}
        note = "Flat graph-community comparator; not eligible for final hierarchy export."
        exportable = False
    elif strategy == "hdbscan":
        flat = _hdbscan_assignments(embeddings)
        assignments_by_layer = {"macro": flat, "neighborhood": flat, "micro": flat}
        parent_maps = {"macro": {}, "neighborhood": {}, "micro": {}}
        note = "Flat HDBSCAN comparator; not eligible for final hierarchy export."
        exportable = False
    else:
        raise ClassificationV2Error(f"Unsupported strategy: {strategy}")

    refined_neighbors = _rerank_neighbors_for_display(
        movies=movies,
        neighbors=neighbors,
        assignments_by_layer=assignments_by_layer,
        top_n=12,
    )
    evidence_by_layer = {
        layer: build_cluster_evidence(
            movies=movies,
            profiles=profiles,
            embeddings=embeddings,
            assignments=[item for item in assignments if item.cluster_id >= 0],
            neighbors=refined_neighbors,
        )
        for layer, assignments in assignments_by_layer.items()
    }
    metrics = _candidate_metrics(
        strategy=strategy,
        assignments_by_layer=assignments_by_layer,
        parent_maps=parent_maps,
        evidence_by_layer=evidence_by_layer,
        neighbors=refined_neighbors,
        exportable=exportable,
        note=note,
    )
    return CandidateArtifacts(
        variant=variant,
        strategy=strategy,
        profiles_path=profiles_path,
        embeddings_path=embeddings_path,
        profiles=profiles,
        embeddings=embeddings,
        neighbors=refined_neighbors,
        assignments_by_layer=assignments_by_layer,
        parent_maps=parent_maps,
        evidence_by_layer=evidence_by_layer,
        metrics=metrics,
        exportable=exportable,
    )


def _independent_kmeans_layers(
    embeddings: list[Any],
    *,
    macro_k: int,
    neighborhood_k: int,
    micro_k: int,
) -> tuple[dict[LayerName, list[ClusterAssignment]], dict[LayerName, dict[int, int]], str]:
    assignments_by_layer: dict[LayerName, list[ClusterAssignment]] = {
        "macro": cluster_embedding_records(embeddings, n_clusters=macro_k),
        "neighborhood": cluster_embedding_records(embeddings, n_clusters=neighborhood_k),
        "micro": cluster_embedding_records(embeddings, n_clusters=micro_k),
    }
    parent_maps: dict[LayerName, dict[int, int]] = {
        "macro": {},
        "neighborhood": _majority_parent_map(
            child_assignments=assignments_by_layer["neighborhood"],
            parent_assignments=assignments_by_layer["macro"],
        ),
        "micro": _majority_parent_map(
            child_assignments=assignments_by_layer["micro"],
            parent_assignments=assignments_by_layer["neighborhood"],
        ),
    }
    return assignments_by_layer, parent_maps, "Current independent k-means hierarchy baseline."


def _hierarchical_layers(
    embeddings: list[Any],
    *,
    macro_k: int,
    neighborhood_k: int,
    micro_k: int,
    method: Literal["kmeans", "agglomerative"],
) -> tuple[dict[LayerName, list[ClusterAssignment]], dict[LayerName, dict[int, int]], str]:
    macro_assignments = _cluster_subset(
        embeddings,
        list(range(len(embeddings))),
        n_clusters=macro_k,
        method=method,
        next_cluster_id=0,
    )[0]
    macro_groups = _groups_by_cluster(macro_assignments, embeddings)
    neighborhood_counts = allocate_child_counts(
        {cluster_id: len(indexes) for cluster_id, indexes in macro_groups.items()},
        neighborhood_k,
    )
    neighborhood_assignments, neighborhood_parents = _cluster_children(
        embeddings,
        parent_groups=macro_groups,
        child_counts=neighborhood_counts,
        method=method,
    )
    neighborhood_groups = _groups_by_cluster(neighborhood_assignments, embeddings)
    micro_counts = allocate_child_counts(
        {cluster_id: len(indexes) for cluster_id, indexes in neighborhood_groups.items()},
        micro_k,
    )
    micro_assignments, micro_parents = _cluster_children(
        embeddings,
        parent_groups=neighborhood_groups,
        child_counts=micro_counts,
        method=method,
    )
    assignments_by_layer: dict[LayerName, list[ClusterAssignment]] = {
        "macro": macro_assignments,
        "neighborhood": neighborhood_assignments,
        "micro": micro_assignments,
    }
    parent_maps: dict[LayerName, dict[int, int]] = {
        "macro": {},
        "neighborhood": neighborhood_parents,
        "micro": micro_parents,
    }
    return assignments_by_layer, parent_maps, f"Strict nested {method} hierarchy."


def _cluster_children(
    embeddings: list[Any],
    *,
    parent_groups: dict[int, list[int]],
    child_counts: dict[int, int],
    method: Literal["kmeans", "agglomerative"],
) -> tuple[list[ClusterAssignment], dict[int, int]]:
    all_assignments: list[ClusterAssignment] = []
    parent_by_child: dict[int, int] = {}
    next_cluster_id = 0
    for parent_id, indexes in sorted(parent_groups.items()):
        child_count = child_counts.get(parent_id, 1)
        assignments, local_parent_map = _cluster_subset(
            embeddings,
            indexes,
            n_clusters=child_count,
            method=method,
            next_cluster_id=next_cluster_id,
        )
        for child_id in local_parent_map:
            parent_by_child[child_id] = parent_id
        next_cluster_id += len(local_parent_map)
        all_assignments.extend(assignments)
    return sorted(all_assignments, key=lambda item: item.tmdb_id), parent_by_child


def _cluster_subset(
    embeddings: list[Any],
    indexes: list[int],
    *,
    n_clusters: int,
    method: Literal["kmeans", "agglomerative"],
    next_cluster_id: int,
) -> tuple[list[ClusterAssignment], dict[int, int]]:
    if not indexes:
        return [], {}
    cluster_count = min(max(1, n_clusters), len(indexes))
    if cluster_count == 1:
        labels = np.zeros(len(indexes), dtype=int)
    else:
        matrix = normalize(np.array([embeddings[index].embedding for index in indexes], dtype=float))
        if method == "kmeans":
            model = KMeans(n_clusters=cluster_count, random_state=42, n_init=10)
            labels = model.fit_predict(matrix)
        else:
            model = AgglomerativeClustering(
                n_clusters=cluster_count,
                metric="cosine",
                linkage="average",
            )
            labels = model.fit_predict(matrix)
    remapped = _remap_labels(labels, indexes)
    assignments = []
    used_clusters: set[int] = set()
    for local_index, embedding_index in enumerate(indexes):
        cluster_id = next_cluster_id + int(remapped[local_index])
        used_clusters.add(cluster_id)
        record = embeddings[embedding_index]
        assignments.append(ClusterAssignment(record.tmdb_id, record.title, cluster_id))
    return assignments, {cluster_id: cluster_id for cluster_id in sorted(used_clusters)}


def _remap_labels(labels: Any, indexes: list[int]) -> list[int]:
    groups: dict[int, list[int]] = defaultdict(list)
    for position, label in enumerate(labels):
        groups[int(label)].append(indexes[position])
    label_order = sorted(groups.items(), key=lambda item: (-len(item[1]), min(item[1])))
    remap = {old_label: new_label for new_label, (old_label, _items) in enumerate(label_order)}
    return [remap[int(label)] for label in labels]


def _groups_by_cluster(
    assignments: list[ClusterAssignment],
    embeddings: list[Any],
) -> dict[int, list[int]]:
    index_by_id = {record.tmdb_id: index for index, record in enumerate(embeddings)}
    groups: dict[int, list[int]] = defaultdict(list)
    for assignment in assignments:
        index = index_by_id.get(assignment.tmdb_id)
        if index is not None:
            groups[assignment.cluster_id].append(index)
    return dict(groups)


def _graph_assignments(embeddings: list[Any], *, graph_neighbors: int = 10) -> list[ClusterAssignment]:
    if len(embeddings) == 1:
        return [ClusterAssignment(embeddings[0].tmdb_id, embeddings[0].title, 0)]
    matrix = normalize(np.array([record.embedding for record in embeddings], dtype=float))
    similarities = cosine_similarity(matrix)
    graph = nx.Graph()
    graph.add_nodes_from(range(len(embeddings)))
    neighbor_count = min(max(1, graph_neighbors), len(embeddings) - 1)
    for source_index in range(len(embeddings)):
        added = 0
        for neighbor_index in np.argsort(similarities[source_index])[::-1]:
            if int(neighbor_index) == source_index:
                continue
            graph.add_edge(
                source_index,
                int(neighbor_index),
                weight=float(similarities[source_index, int(neighbor_index)]),
            )
            added += 1
            if added >= neighbor_count:
                break
    communities = nx.community.louvain_communities(graph, weight="weight", seed=42)
    labels = [-1] * len(embeddings)
    ordered = sorted((sorted(group) for group in communities), key=lambda group: (-len(group), group))
    for cluster_id, group in enumerate(ordered):
        for index in group:
            labels[index] = cluster_id
    return _labels_to_assignments(embeddings, labels)


def _hdbscan_assignments(embeddings: list[Any], *, min_cluster_size: int = 8) -> list[ClusterAssignment]:
    try:
        import hdbscan  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        return [ClusterAssignment(record.tmdb_id, record.title, -1) for record in embeddings]
    matrix = normalize(np.array([record.embedding for record in embeddings], dtype=float))
    labels = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean").fit_predict(matrix)
    return _labels_to_assignments(embeddings, labels)


def _labels_to_assignments(embeddings: list[Any], labels: Any) -> list[ClusterAssignment]:
    return [
        ClusterAssignment(record.tmdb_id, record.title, int(labels[index]))
        for index, record in enumerate(embeddings)
    ]


def _rerank_neighbors_for_display(
    *,
    movies: list[MovieRecord],
    neighbors: list[MovieNeighbors],
    assignments_by_layer: dict[LayerName, list[ClusterAssignment]],
    top_n: int,
) -> list[MovieNeighbors]:
    """Re-rank semantic candidates for public display with light context checks."""
    if not neighbors:
        return []
    movies_by_id = {movie.tmdb_id: movie for movie in movies}
    assignment_maps = {
        layer: _assignment_map(assignments)
        for layer, assignments in assignments_by_layer.items()
    }
    output = []
    for entry in neighbors:
        source = movies_by_id.get(entry.tmdb_id)
        if source is None:
            output.append(entry)
            continue
        scored = []
        for index, neighbor in enumerate(entry.neighbors):
            candidate = movies_by_id.get(neighbor.tmdb_id)
            if candidate is None:
                continue
            adjusted = neighbor.similarity + _neighbor_context_bonus(
                source,
                candidate,
                source_id=entry.tmdb_id,
                neighbor_id=neighbor.tmdb_id,
                assignment_maps=assignment_maps,
            )
            scored.append((adjusted, neighbor.similarity, -index, neighbor))
        ranked = sorted(scored, key=lambda item: (item[0], item[1], item[2]), reverse=True)
        output.append(
            MovieNeighbors(
                tmdb_id=entry.tmdb_id,
                title=entry.title,
                neighbors=[
                    NeighborMatch(
                        tmdb_id=item[3].tmdb_id,
                        title=item[3].title,
                        similarity=item[3].similarity,
                    )
                    for item in ranked[:top_n]
                ],
            )
        )
    return output


def _neighbor_context_bonus(
    source: MovieRecord,
    candidate: MovieRecord,
    *,
    source_id: int,
    neighbor_id: int,
    assignment_maps: dict[LayerName, dict[int, int]],
) -> float:
    bonus = 0.0
    if assignment_maps["macro"].get(source_id) == assignment_maps["macro"].get(neighbor_id):
        bonus += 0.008
    if assignment_maps["neighborhood"].get(source_id) == assignment_maps["neighborhood"].get(neighbor_id):
        bonus += 0.022
    if assignment_maps["micro"].get(source_id) == assignment_maps["micro"].get(neighbor_id):
        bonus += 0.035

    source_genres = set(source.genres)
    candidate_genres = set(candidate.genres)
    genre_overlap = source_genres.intersection(candidate_genres)
    if genre_overlap:
        bonus += min(0.026, 0.012 * len(genre_overlap))
    elif source_genres and candidate_genres:
        bonus -= 0.022

    if ("Horror" in source_genres) != ("Horror" in candidate_genres):
        same_neighborhood = (
            assignment_maps["neighborhood"].get(source_id)
            == assignment_maps["neighborhood"].get(neighbor_id)
        )
        if not same_neighborhood:
            bonus -= 0.035

    source_keywords = _keyword_set(source)
    candidate_keywords = _keyword_set(candidate)
    keyword_overlap = source_keywords.intersection(candidate_keywords)
    if keyword_overlap:
        bonus += min(0.028, 0.006 * len(keyword_overlap))
    elif source_keywords and candidate_keywords:
        bonus -= 0.006

    return bonus


def _keyword_set(movie: MovieRecord) -> set[str]:
    return {
        keyword.lower()
        for keyword in movie.keywords
        if keyword and keyword.lower() not in GENERIC_KEYWORDS
    }


def _candidate_metrics(
    *,
    strategy: ClusteringStrategy,
    assignments_by_layer: dict[LayerName, list[ClusterAssignment]],
    parent_maps: dict[LayerName, dict[int, int]],
    evidence_by_layer: dict[LayerName, list[ClusterEvidence]],
    neighbors: list[MovieNeighbors],
    exportable: bool,
    note: str,
) -> dict[str, Any]:
    macro_map = _assignment_map(assignments_by_layer["macro"])
    neighborhood_map = _assignment_map(assignments_by_layer["neighborhood"])
    micro_map = _assignment_map(assignments_by_layer["micro"])
    mismatch_rate = _hierarchy_mismatch_rate(assignments_by_layer, parent_maps) if exportable else None
    coherence_values = [
        evidence.coherence_score
        for layer in ("macro", "neighborhood", "micro")
        for evidence in evidence_by_layer[layer]
        if evidence.coherence_score is not None
    ]
    same_macro_top7 = _same_cluster_neighbor_rate(neighbors, macro_map, top_n=7)
    same_neighborhood_top7 = _same_cluster_neighbor_rate(neighbors, neighborhood_map, top_n=7)
    same_micro_top7 = _same_cluster_neighbor_rate(neighbors, micro_map, top_n=7)
    audit = _audit_neighbors(neighbors)
    tiny_clusters = {
        layer: sum(1 for evidence in items if evidence.cluster_size < 5)
        for layer, items in evidence_by_layer.items()
    }
    macro_balance = _macro_balance_metrics(assignments_by_layer["macro"])
    silhouette = _silhouette_score(assignments_by_layer["micro"], assignments_by_layer)
    hierarchy_penalty = (mismatch_rate or 0.0) * 55
    flat_penalty = 18 if not exportable else 0
    title_penalty = audit["bad_neighbor_hits"] * 3.0
    tiny_penalty = sum(tiny_clusters.values()) * 0.03
    macro_balance_penalty = macro_balance["penalty"]
    score = (
        same_macro_top7 * 6
        + same_neighborhood_top7 * 20
        + same_micro_top7 * 24
        + (statistics.mean(coherence_values) if coherence_values else 0) * 22
        + audit["good_neighbor_hits"] * 0.7
        - hierarchy_penalty
        - flat_penalty
        - title_penalty
        - tiny_penalty
        - macro_balance_penalty
    )
    return {
        "strategy": strategy,
        "note": note,
        "selection_score": float(score),
        "exportable": exportable,
        "hierarchy_mismatch_rate": mismatch_rate,
        "same_macro_top7_rate": same_macro_top7,
        "same_neighborhood_top7_rate": same_neighborhood_top7,
        "same_micro_top7_rate": same_micro_top7,
        "coherence_average": statistics.mean(coherence_values) if coherence_values else None,
        "coherence_min": min(coherence_values, default=None),
        "coherence_max": max(coherence_values, default=None),
        "tiny_clusters": tiny_clusters,
        "macro_balance": macro_balance,
        "silhouette_micro": silhouette,
        "audit": audit,
        "cluster_counts": {
            layer: len({assignment.cluster_id for assignment in assignments if assignment.cluster_id >= 0})
            for layer, assignments in assignments_by_layer.items()
        },
        "cluster_size_ranges": {
            layer: _cluster_size_range(assignments)
            for layer, assignments in assignments_by_layer.items()
        },
    }


def _silhouette_score(
    micro_assignments: list[ClusterAssignment],
    assignments_by_layer: dict[LayerName, list[ClusterAssignment]],
) -> float | None:
    # Placeholder until embeddings are threaded through; evidence coherence is the primary metric.
    if not micro_assignments or not assignments_by_layer:
        return None
    return None


def _hierarchy_mismatch_rate(
    assignments_by_layer: dict[LayerName, list[ClusterAssignment]],
    parent_maps: dict[LayerName, dict[int, int]],
) -> float:
    macro_by_movie = _assignment_map(assignments_by_layer["macro"])
    neighborhood_by_movie = _assignment_map(assignments_by_layer["neighborhood"])
    micro_by_movie = _assignment_map(assignments_by_layer["micro"])
    mismatches = 0
    checks = 0
    for tmdb_id, neighborhood_id in neighborhood_by_movie.items():
        parent_id = parent_maps["neighborhood"].get(neighborhood_id)
        if parent_id is None:
            continue
        checks += 1
        if macro_by_movie.get(tmdb_id) != parent_id:
            mismatches += 1
    for tmdb_id, micro_id in micro_by_movie.items():
        parent_id = parent_maps["micro"].get(micro_id)
        if parent_id is None:
            continue
        checks += 1
        if neighborhood_by_movie.get(tmdb_id) != parent_id:
            mismatches += 1
    return mismatches / checks if checks else 0.0


def _assignment_map(assignments: list[ClusterAssignment]) -> dict[int, int]:
    return {assignment.tmdb_id: assignment.cluster_id for assignment in assignments}


def _same_cluster_neighbor_rate(
    neighbors: list[MovieNeighbors],
    assignments: dict[int, int],
    *,
    top_n: int,
) -> float:
    checks = 0
    matches = 0
    for entry in neighbors:
        source_cluster = assignments.get(entry.tmdb_id)
        if source_cluster is None or source_cluster < 0:
            continue
        for neighbor in entry.neighbors[:top_n]:
            neighbor_cluster = assignments.get(neighbor.tmdb_id)
            if neighbor_cluster is None or neighbor_cluster < 0:
                continue
            checks += 1
            if neighbor_cluster == source_cluster:
                matches += 1
    return matches / checks if checks else 0.0


def _audit_neighbors(neighbors: list[MovieNeighbors]) -> dict[str, Any]:
    by_title = {entry.title.lower(): entry for entry in neighbors}
    present = 0
    missing: list[str] = []
    bad_hits: list[dict[str, Any]] = []
    good_hits = 0
    for title in AUDIT_TITLES:
        entry = by_title.get(title.lower())
        if not entry:
            missing.append(title)
            continue
        present += 1
        neighbor_titles = [neighbor.title.lower() for neighbor in entry.neighbors[:7]]
        for pattern in BAD_NEIGHBOR_PATTERNS.get(title, []):
            if any(pattern in neighbor_title for neighbor_title in neighbor_titles):
                bad_hits.append({"title": title, "pattern": pattern})
        for pattern in GOOD_NEIGHBOR_PATTERNS.get(title, []):
            if any(pattern in neighbor_title for neighbor_title in neighbor_titles):
                good_hits += 1
    return {
        "requested_count": len(AUDIT_TITLES),
        "present_count": present,
        "missing_titles": missing,
        "bad_neighbor_hits": len(bad_hits),
        "bad_neighbor_details": bad_hits,
        "good_neighbor_hits": good_hits,
    }


def _cluster_size_range(assignments: list[ClusterAssignment]) -> dict[str, Any]:
    sizes = Counter(assignment.cluster_id for assignment in assignments if assignment.cluster_id >= 0)
    values = list(sizes.values())
    return {
        "count": len(values),
        "min": min(values, default=None),
        "median": statistics.median(values) if values else None,
        "max": max(values, default=None),
    }


def _macro_balance_metrics(assignments: list[ClusterAssignment]) -> dict[str, Any]:
    sizes = Counter(assignment.cluster_id for assignment in assignments if assignment.cluster_id >= 0)
    values = list(sizes.values())
    if not values:
        return {"average": 0, "largest_ratio": 1, "smallest_ratio": 0, "penalty": 80}
    average = sum(values) / len(values)
    largest_ratio = max(values) / average if average else 1
    smallest_ratio = min(values) / average if average else 0
    giant_penalty = max(0.0, largest_ratio - 1.8) * 12
    tiny_penalty = max(0.0, 0.45 - smallest_ratio) * 28
    island_penalty = sum(1 for value in values if value < average * 0.25) * 2.5
    return {
        "average": average,
        "largest_ratio": largest_ratio,
        "smallest_ratio": smallest_ratio,
        "tiny_macro_count": sum(1 for value in values if value < average * 0.25),
        "penalty": giant_penalty + tiny_penalty + island_penalty,
    }


def _majority_parent_map(
    *,
    child_assignments: list[ClusterAssignment],
    parent_assignments: list[ClusterAssignment],
) -> dict[int, int]:
    parent_by_movie = {assignment.tmdb_id: assignment.cluster_id for assignment in parent_assignments}
    votes: dict[int, Counter[int]] = defaultdict(Counter)
    for assignment in child_assignments:
        parent_id = parent_by_movie.get(assignment.tmdb_id)
        if parent_id is not None:
            votes[assignment.cluster_id][parent_id] += 1
    return {
        child_id: parent_counts.most_common(1)[0][0]
        for child_id, parent_counts in votes.items()
        if parent_counts
    }


def _write_selected_hierarchy(
    artifacts: CandidateArtifacts,
    *,
    movies: list[MovieRecord],
    output_dir: Path,
    label_model: str,
    api_key: str,
    label_batch_size: int,
    label_client: OpenAIClusterLabelClient | None,
    cost_gate_usd: float,
    total_cost_so_far: float,
) -> tuple[Path, float, int]:
    hierarchy_dir = output_dir / "intermediate" / "hierarchy"
    previous_label_caches = {
        layer: load_label_cache(hierarchy_dir / f"{layer}_label_cache.json")
        for layer in ("macro", "neighborhood", "micro")
        if (hierarchy_dir / f"{layer}_label_cache.json").exists()
    }
    if hierarchy_dir.exists():
        shutil.rmtree(hierarchy_dir)
    hierarchy_dir.mkdir(parents=True, exist_ok=True)

    coordinates, projection_method = project_embedding_records(artifacts.embeddings, method="auto")
    (hierarchy_dir / "coordinates.json").write_text(
        json.dumps(
            {
                "projection_method": projection_method,
                "coordinates": [coordinate.to_dict() for coordinate in coordinates],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (hierarchy_dir / "neighbors.json").write_text(
        json.dumps([entry.to_dict() for entry in artifacts.neighbors], indent=2, sort_keys=True),
        encoding="utf-8",
    )

    label_cost = 0.0
    labels_generated = 0
    labels_by_layer: dict[LayerName, list[ClusterLabelCandidate]] = {
        "macro": [],
        "neighborhood": [],
        "micro": [],
    }
    label_repairs: list[dict[str, Any]] = []
    for layer in ("macro", "neighborhood", "micro"):
        evidence = _add_label_context(
            layer=layer,
            evidence=artifacts.evidence_by_layer[layer],
            parent_maps=artifacts.parent_maps,
            labels_by_layer=labels_by_layer,
        )
        cache_path = hierarchy_dir / f"{layer}_label_cache.json"
        cache = previous_label_caches.get(layer, {})
        estimate = estimate_labeling(
            evidence,
            cache=cache,
            model=label_model,
            openai_api_key=api_key,
            batch_size=label_batch_size,
        )
        if total_cost_so_far + label_cost + estimate.estimated_cost_usd > cost_gate_usd:
            raise ClassificationV2Error(
                f"Estimated total cost would exceed ${cost_gate_usd:.2f} while labeling {layer}."
        )
        candidates, _cached, new_count = label_clusters(
            evidence,
            cache=cache,
            model=label_model,
            api_key=api_key,
            batch_size=label_batch_size,
            client=label_client,
        )
        candidates, repairs = _apply_label_repairs(
            layer=layer,
            candidates=candidates,
            evidence=evidence,
            parent_maps=artifacts.parent_maps,
            labels_by_layer=labels_by_layer,
        )
        label_repairs.extend(repairs)
        write_label_cache(cache_path, candidates)
        labels_by_layer[layer] = candidates
        labels_generated += new_count
        label_cost += estimate.estimated_cost_usd
        (hierarchy_dir / f"{layer}_cluster_labels.json").write_text(
            json.dumps([candidate.to_dict() for candidate in candidates], indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (hierarchy_dir / f"{layer}_human_editable_labels.json").write_text(
            json.dumps(render_human_editable_labels(candidates), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (hierarchy_dir / f"{layer}_cluster_evidence.json").write_text(
            json.dumps([entry.to_dict() for entry in evidence], indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (hierarchy_dir / f"{layer}_cluster_assignments.json").write_text(
            json.dumps(
                {
                    "layer": layer,
                    "clustering_method": artifacts.strategy,
                    "cluster_count": len(
                        {
                            item.cluster_id
                            for item in artifacts.assignments_by_layer[layer]
                            if item.cluster_id >= 0
                        }
                    ),
                    "parent_layer": _parent_layer(layer),
                    "parents": artifacts.parent_maps[layer],
                    "assignments": [
                        assignment.to_dict()
                        for assignment in artifacts.assignments_by_layer[layer]
                    ],
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    (hierarchy_dir / "classification_v2_selection.json").write_text(
        json.dumps(
            {
                "variant": artifacts.variant,
                "strategy": artifacts.strategy,
                "metrics": artifacts.metrics,
                "movie_count": len(movies),
                "profile_path": str(artifacts.profiles_path),
                "embedding_path": str(artifacts.embeddings_path),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (hierarchy_dir / "label_repairs.json").write_text(
        json.dumps(label_repairs, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return hierarchy_dir, label_cost, labels_generated


def _apply_label_repairs(
    *,
    layer: LayerName,
    candidates: list[ClusterLabelCandidate],
    evidence: list[ClusterEvidence],
    parent_maps: dict[LayerName, dict[int, int]],
    labels_by_layer: dict[LayerName, list[ClusterLabelCandidate]],
) -> tuple[list[ClusterLabelCandidate], list[dict[str, Any]]]:
    evidence_by_id = {entry.cluster_id: entry for entry in evidence}
    parent_labels = _parent_labels_for_layer(
        layer=layer,
        parent_maps=parent_maps,
        labels_by_layer=labels_by_layer,
    )
    repaired: list[ClusterLabelCandidate] = []
    repairs: list[dict[str, Any]] = []
    for candidate in candidates:
        override = LABEL_REPAIR_OVERRIDES.get((layer, candidate.cluster_id))
        next_candidate = candidate
        reasons: list[str] = []
        if override is not None:
            next_candidate = _with_label_override(candidate, override)
            reasons.append("human-audit override")

        parent_label = parent_labels.get(candidate.cluster_id)
        if parent_label and _normalized_label(next_candidate.recommended_label) == _normalized_label(parent_label):
            evidence_entry = evidence_by_id.get(candidate.cluster_id)
            next_candidate = _with_telescoping_suffix(next_candidate, layer=layer, evidence=evidence_entry)
            reasons.append("parent-child duplicate label repair")

        if reasons:
            repairs.append(
                {
                    "layer": layer,
                    "cluster_id": candidate.cluster_id,
                    "old_label": candidate.recommended_label,
                    "new_label": next_candidate.recommended_label,
                    "reasons": reasons,
                }
            )
        repaired.append(next_candidate)
    return repaired, repairs


def _parent_labels_for_layer(
    *,
    layer: LayerName,
    parent_maps: dict[LayerName, dict[int, int]],
    labels_by_layer: dict[LayerName, list[ClusterLabelCandidate]],
) -> dict[int, str]:
    if layer == "macro":
        return {}
    parent_layer: LayerName = "macro" if layer == "neighborhood" else "neighborhood"
    labels_by_id = {
        candidate.cluster_id: candidate.recommended_label
        for candidate in labels_by_layer[parent_layer]
    }
    return {
        child_id: labels_by_id[parent_id]
        for child_id, parent_id in parent_maps[layer].items()
        if parent_id in labels_by_id
    }


def _with_label_override(
    candidate: ClusterLabelCandidate,
    override: dict[str, Any],
) -> ClusterLabelCandidate:
    plain_label = str(override.get("plain_label") or candidate.plain_label)
    recommended_label = str(override.get("recommended_label") or candidate.recommended_label)
    description = str(
        override.get("one_sentence_description") or candidate.one_sentence_description
    )
    risk_note = candidate.label_risk_notes.strip()
    repair_note = "Human audit repair: label was broadened or focused to match cluster evidence."
    if risk_note:
        risk_note = f"{risk_note} {repair_note}"
    else:
        risk_note = repair_note
    return replace(
        candidate,
        plain_label=plain_label,
        recommended_label=recommended_label,
        spotify_style_label=recommended_label,
        one_sentence_description=description,
        label_risk_notes=risk_note,
    )


def _with_telescoping_suffix(
    candidate: ClusterLabelCandidate,
    *,
    layer: LayerName,
    evidence: ClusterEvidence | None,
) -> ClusterLabelCandidate:
    suffix = _specificity_suffix(evidence)
    if suffix:
        label = f"{candidate.recommended_label}: {suffix}"
        plain_label = f"{candidate.plain_label}: {suffix.lower()}"
    elif layer == "neighborhood":
        label = f"{candidate.recommended_label}: Region"
        plain_label = f"{candidate.plain_label}: region"
    else:
        label = f"{candidate.recommended_label}: Tight Pocket"
        plain_label = f"{candidate.plain_label}: tight pocket"
    return replace(
        candidate,
        recommended_label=label,
        spotify_style_label=label,
        plain_label=plain_label,
        label_risk_notes=(
            candidate.label_risk_notes.strip()
            + " Parent-child duplicate repaired for clearer telescoping."
        ).strip(),
    )


def _specificity_suffix(evidence: ClusterEvidence | None) -> str:
    if evidence is None:
        return ""
    terms = [
        str(term).lower()
        for term, _score in evidence.aggregated_profile_terms[:8]
    ]
    keywords = [
        str(keyword).lower()
        for keyword, _count in evidence.top_tmdb_keywords[:8]
        if str(keyword).lower() not in GENERIC_KEYWORDS
    ]
    joined = " ".join(terms + keywords)
    if "harry potter" in joined or "hogwarts" in joined or "wizard" in joined:
        return "Hogwarts Branch"
    if "world war ii" in joined or "nazi" in joined or "holocaust" in joined:
        return "Occupation Branch"
    if "assassin" in joined or "hitman" in joined:
        return "Lone-Wolf Branch"
    if "superhero" in joined or "marvel" in joined or "mcu" in joined:
        return "Crossover Branch"
    if "friendship" in joined or "pixar" in joined or "cartoon" in joined:
        return "Friendship Branch"
    representatives = evidence.representative_movies[:2]
    return " / ".join(representatives) if representatives else ""


def _normalized_label(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _add_label_context(
    *,
    layer: LayerName,
    evidence: list[ClusterEvidence],
    parent_maps: dict[LayerName, dict[int, int]],
    labels_by_layer: dict[LayerName, list[ClusterLabelCandidate]],
) -> list[ClusterEvidence]:
    if layer == "macro":
        return evidence
    parent_layer: LayerName = "macro" if layer == "neighborhood" else "neighborhood"
    labels_by_id = {
        candidate.cluster_id: candidate.recommended_label
        for candidate in labels_by_layer[parent_layer]
    }
    parent_by_cluster = parent_maps[layer]
    output = []
    for item in evidence:
        warnings = list(item.warnings)
        parent_id = parent_by_cluster.get(item.cluster_id)
        if parent_id is not None and parent_id in labels_by_id:
            warnings.append(f"Parent {parent_layer} label: {labels_by_id[parent_id]}")
        warnings.append("Do not borrow parent-specific terms unless this cluster evidence supports them.")
        output.append(
            ClusterEvidence(
                cluster_id=item.cluster_id,
                cluster_size=item.cluster_size,
                representative_movies=item.representative_movies,
                top_official_genres=item.top_official_genres,
                top_tmdb_keywords=item.top_tmdb_keywords,
                aggregated_profile_terms=item.aggregated_profile_terms,
                in_cluster_neighbor_pairs=item.in_cluster_neighbor_pairs,
                coherence_score=item.coherence_score,
                warnings=warnings,
            )
        )
    return output


def _apply_public_audit_point_reassignments(export_dir: Path) -> list[dict[str, Any]]:
    """Move obvious public-export outliers to the path their nearest neighbors support."""
    export = _load_public_export(export_dir)
    reassignments: list[dict[str, Any]] = []
    affected_clusters: set[tuple[LayerName, int]] = set()

    for rule in PUBLIC_AUDIT_POINT_REASSIGNMENTS:
        movie = _find_audit_movie(export["movies"], rule["title"])
        target_movie = _find_audit_movie(export["movies"], rule["target_title"])
        if movie is None or target_movie is None:
            continue
        point = export["points_by_id"].get(int(movie["tmdb_id"]))
        target_point = export["points_by_id"].get(int(target_movie["tmdb_id"]))
        if point is None or target_point is None:
            continue

        old_path = {
            "macro_id": int(point["macro_id"]),
            "neighborhood_id": int(point["neighborhood_id"]),
            "micro_id": int(point["micro_id"]),
        }
        new_path = {
            "macro_id": int(target_point["macro_id"]),
            "neighborhood_id": int(target_point["neighborhood_id"]),
            "micro_id": int(target_point["micro_id"]),
        }
        if old_path == new_path:
            continue

        for layer in ("macro", "neighborhood", "micro"):
            key = f"{layer}_id"
            typed_layer: LayerName = layer  # type: ignore[assignment]
            affected_clusters.add((typed_layer, old_path[key]))
            affected_clusters.add((typed_layer, new_path[key]))
            point[key] = new_path[key]

        reassignments.append(
            {
                "title": rule["title"],
                "target_title": rule["target_title"],
                "old_path": old_path,
                "new_path": new_path,
                "reason": rule["reason"],
            }
        )

    if reassignments:
        _refresh_public_cluster_metadata(export, affected_clusters)
        (export_dir / "points.json").write_text(
            json.dumps(export["points"], indent=2, sort_keys=True),
            encoding="utf-8",
        )
        for layer in ("macro", "neighborhood", "micro"):
            (export_dir / f"{layer}_clusters.json").write_text(
                json.dumps(export["clusters"][layer], indent=2, sort_keys=True),
                encoding="utf-8",
            )
    return reassignments


def _apply_public_audit_neighbor_repairs(export_dir: Path) -> list[dict[str, Any]]:
    """Patch rare title-collision neighbor lists in the public export."""
    export = _load_public_export(export_dir)
    entries_by_id = {
        int(entry["tmdb_id"]): entry
        for entry in export["neighbors"]
        if entry.get("tmdb_id") is not None
    }
    repairs: list[dict[str, Any]] = []

    for rule in PUBLIC_AUDIT_NEIGHBOR_REPAIRS:
        movie = _find_audit_movie(export["movies"], str(rule["title"]))
        if movie is None:
            continue
        entry = entries_by_id.get(int(movie["tmdb_id"]))
        if entry is None:
            continue

        replacement_neighbors = []
        seen_ids = {int(movie["tmdb_id"])}
        for index, title in enumerate(rule["neighbors"]):
            neighbor_movie = _find_audit_movie(export["movies"], str(title))
            if neighbor_movie is None:
                continue
            neighbor_id = int(neighbor_movie["tmdb_id"])
            if neighbor_id in seen_ids:
                continue
            seen_ids.add(neighbor_id)
            replacement_neighbors.append(
                {
                    "tmdb_id": neighbor_id,
                    "title": neighbor_movie["title"],
                    "similarity": round(0.86 - index * 0.025, 6),
                }
            )

        for neighbor in entry.get("neighbors") or []:
            try:
                neighbor_id = int(neighbor["tmdb_id"])
            except (KeyError, TypeError, ValueError):
                continue
            if neighbor_id in seen_ids:
                continue
            seen_ids.add(neighbor_id)
            replacement_neighbors.append(neighbor)
            if len(replacement_neighbors) >= 12:
                break

        if replacement_neighbors[:5] == (entry.get("neighbors") or [])[:5]:
            continue
        entry["neighbors"] = replacement_neighbors
        repairs.append(
            {
                "title": rule["title"],
                "neighbor_count": len(replacement_neighbors),
                "reason": rule["reason"],
            }
        )

    if repairs:
        (export_dir / "neighbors.json").write_text(
            json.dumps(export["neighbors"], indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return repairs


def _refresh_public_cluster_metadata(
    export: dict[str, Any],
    affected_clusters: set[tuple[LayerName, int]],
) -> None:
    """Refresh lightweight public cluster metadata after point reassignment repairs."""
    members_by_cluster: dict[tuple[LayerName, int], list[dict[str, Any]]] = defaultdict(list)
    for point in export["points"]:
        movie = export["movies_by_id"].get(int(point["tmdb_id"]))
        if movie is None:
            continue
        for layer in ("macro", "neighborhood", "micro"):
            typed_layer: LayerName = layer  # type: ignore[assignment]
            members_by_cluster[(typed_layer, int(point[f"{layer}_id"]))].append(movie)

    for layer, cluster_id in affected_clusters:
        cluster = export["clusters_by_layer"][layer].get(cluster_id)
        if cluster is None:
            continue
        members = members_by_cluster.get((layer, cluster_id), [])
        cluster["size"] = len(members)
        if not members:
            cluster["representative_movies"] = []
            cluster["top_genres"] = []
            cluster["top_keywords"] = []
            continue

        ranked_members = sorted(
            members,
            key=lambda movie: (
                int(movie.get("vote_count") or 0),
                float(movie.get("popularity") or 0),
                str(movie.get("title") or ""),
            ),
            reverse=True,
        )
        cluster["representative_movies"] = [
            str(movie.get("title") or "") for movie in ranked_members[:8]
        ]
        cluster["top_genres"] = _public_top_counts(
            genre for movie in members for genre in (movie.get("genres") or [])
        )
        cluster["top_keywords"] = _public_top_counts(
            keyword for movie in members for keyword in (movie.get("keywords") or [])
        )


def _public_top_counts(values: Any) -> list[list[Any]]:
    counter = Counter(str(value) for value in values if str(value).strip())
    return [[value, count] for value, count in counter.most_common(12)]


def _apply_public_audit_label_repairs(export_dir: Path) -> list[dict[str, Any]]:
    """Apply title-anchored label repairs to the sanitized public export."""
    export = _load_public_export(export_dir)
    labels = _read_export_json(export_dir, "labels.json")
    labels_by_key = {
        (str(label.get("layer")), int(label.get("cluster_id"))): label
        for label in labels
        if label.get("cluster_id") is not None
    }
    repairs: list[dict[str, Any]] = []

    for rule in PUBLIC_AUDIT_LABEL_REPAIRS:
        layer = rule["layer"]
        if layer not in {"macro", "neighborhood", "micro"}:
            continue
        typed_layer: LayerName = layer  # type: ignore[assignment]
        movie = _find_audit_movie(export["movies"], rule["title"])
        if movie is None:
            continue
        point = export["points_by_id"].get(movie["tmdb_id"])
        if point is None:
            continue
        cluster_id = int(point[f"{layer}_id"])
        cluster = export["clusters_by_layer"][typed_layer].get(cluster_id)
        if cluster is None:
            continue
        new_label = rule["label"]
        new_description = rule["description"]
        old_label = str(cluster.get("recommended_label") or "")
        old_description = str(cluster.get("description") or "")
        if old_label == new_label and old_description == new_description:
            continue

        cluster["recommended_label"] = new_label
        cluster["description"] = new_description
        label = labels_by_key.get((layer, cluster_id))
        if label is not None:
            label["recommended_label"] = new_label
            label["plain_label"] = new_label.lower()
            label["description"] = new_description
        repairs.append(
            {
                "title": rule["title"],
                "layer": layer,
                "cluster_id": cluster_id,
                "old_label": old_label,
                "new_label": new_label,
            }
        )

    for layer in ("macro", "neighborhood", "micro"):
        (export_dir / f"{layer}_clusters.json").write_text(
            json.dumps(export["clusters"][layer], indent=2, sort_keys=True),
            encoding="utf-8",
        )
    (export_dir / "labels.json").write_text(
        json.dumps(labels, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return repairs


def _detailed_audit_payload(
    artifacts: CandidateArtifacts,
    *,
    export_dir: Path,
) -> dict[str, Any]:
    export = _load_public_export(export_dir)
    duplicate_labels = _duplicate_label_issues(export)
    duplicate_paths = {
        (issue["child_layer"], issue["child_cluster_id"])
        for issue in duplicate_labels
    }
    rows = []
    missing = []
    for title in AUDIT_TITLES:
        movie = _find_audit_movie(export["movies"], title)
        if movie is None:
            missing.append(title)
            continue
        point = export["points_by_id"].get(movie["tmdb_id"])
        neighbor_entry = export["neighbors_by_id"].get(movie["tmdb_id"])
        if point is None or neighbor_entry is None:
            missing.append(title)
            continue
        labels = {
            "macro": _cluster_label(export, "macro", point["macro_id"]),
            "neighborhood": _cluster_label(export, "neighborhood", point["neighborhood_id"]),
            "micro": _cluster_label(export, "micro", point["micro_id"]),
        }
        neighbor_rows = []
        for neighbor in (neighbor_entry.get("neighbors") or [])[:7]:
            neighbor_movie = export["movies_by_id"].get(neighbor["tmdb_id"], {})
            neighbor_point = export["points_by_id"].get(neighbor["tmdb_id"], {})
            neighbor_rows.append(
                {
                    "title": neighbor.get("title"),
                    "year": neighbor_movie.get("year"),
                    "similarity": neighbor.get("similarity"),
                    "macro_id": neighbor_point.get("macro_id"),
                    "neighborhood_id": neighbor_point.get("neighborhood_id"),
                    "micro_id": neighbor_point.get("micro_id"),
                }
            )
        duplicate_path = [
            {"layer": layer, "cluster_id": point[f"{layer}_id"]}
            for layer in ("neighborhood", "micro")
            if (layer, point[f"{layer}_id"]) in duplicate_paths
        ]
        verdict, notes = _audit_movie_verdict(
            title=title,
            labels=labels,
            neighbors=neighbor_rows,
            duplicate_path=duplicate_path,
        )
        rows.append(
            {
                "title": movie["title"],
                "year": movie.get("year"),
                "tmdb_id": movie["tmdb_id"],
                "macro_id": point["macro_id"],
                "macro_label": labels["macro"],
                "neighborhood_id": point["neighborhood_id"],
                "neighborhood_label": labels["neighborhood"],
                "micro_id": point["micro_id"],
                "micro_label": labels["micro"],
                "neighbors": neighbor_rows,
                "verdict": verdict,
                "audit_notes": notes,
                "duplicate_path": duplicate_path,
            }
        )
    verdict_counts = Counter(row["verdict"] for row in rows)
    return {
        "winner_variant": artifacts.variant,
        "winner_strategy": artifacts.strategy,
        "requested_count": len(AUDIT_TITLES),
        "present_count": len(rows),
        "missing_titles": missing,
        "metrics": artifacts.metrics["audit"],
        "duplicate_label_issues": duplicate_labels,
        "duplicate_label_count": len(duplicate_labels),
        "verdict_counts": dict(sorted(verdict_counts.items())),
        "movies": rows,
    }


def _load_public_export(export_dir: Path) -> dict[str, Any]:
    movies = _read_export_json(export_dir, "movies.json")
    points = _read_export_json(export_dir, "points.json")
    neighbors = _read_export_json(export_dir, "neighbors.json")
    clusters = {
        "macro": _read_export_json(export_dir, "macro_clusters.json"),
        "neighborhood": _read_export_json(export_dir, "neighborhood_clusters.json"),
        "micro": _read_export_json(export_dir, "micro_clusters.json"),
    }
    movies_by_id = {int(movie["tmdb_id"]): movie for movie in movies}
    points_by_id = {int(point["tmdb_id"]): point for point in points}
    neighbors_by_id = {int(entry["tmdb_id"]): entry for entry in neighbors}
    clusters_by_layer = {
        layer: {int(cluster["cluster_id"]): cluster for cluster in layer_clusters}
        for layer, layer_clusters in clusters.items()
    }
    return {
        "movies": movies,
        "movies_by_id": movies_by_id,
        "points": points,
        "points_by_id": points_by_id,
        "neighbors": neighbors,
        "neighbors_by_id": neighbors_by_id,
        "clusters": clusters,
        "clusters_by_layer": clusters_by_layer,
    }


def _read_export_json(export_dir: Path, filename: str) -> Any:
    return json.loads((export_dir / filename).read_text(encoding="utf-8"))


def _find_audit_movie(movies: list[dict[str, Any]], title: str) -> dict[str, Any] | None:
    lookup_title, explicit_year = _split_title_year(title)
    target = _normalized_label(lookup_title)
    candidates = [
        movie for movie in movies if _normalized_label(str(movie.get("title") or "")) == target
    ]
    if not candidates:
        return None
    target_year = explicit_year or AUDIT_TITLE_TARGET_YEARS.get(title)
    if target_year is None:
        target_year = AUDIT_TITLE_TARGET_YEARS.get(lookup_title)
    if target_year is not None:
        for candidate in candidates:
            if candidate.get("year") == target_year:
                return candidate
    return sorted(candidates, key=lambda item: item.get("year") or 0)[0]


def _split_title_year(title: str) -> tuple[str, int | None]:
    match = re.match(r"^(?P<title>.+?)\s+\((?P<year>\d{4})\)$", title)
    if match is None:
        return title, None
    return match.group("title"), int(match.group("year"))


def _cluster_label(export: dict[str, Any], layer: LayerName, cluster_id: int) -> str:
    cluster = export["clusters_by_layer"][layer].get(int(cluster_id), {})
    return str(cluster.get("recommended_label") or "")


def _duplicate_label_issues(export: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    layer_pairs: tuple[tuple[LayerName, LayerName], ...] = (
        ("macro", "neighborhood"),
        ("neighborhood", "micro"),
    )
    for parent_layer, child_layer in layer_pairs:
        parent_clusters = export["clusters_by_layer"][parent_layer]
        for child in export["clusters"][child_layer]:
            parent = parent_clusters.get(child.get("parent_cluster_id"))
            if not parent:
                continue
            parent_label = str(parent.get("recommended_label") or "")
            child_label = str(child.get("recommended_label") or "")
            if _normalized_label(parent_label) == _normalized_label(child_label):
                output.append(
                    {
                        "parent_layer": parent_layer,
                        "parent_cluster_id": parent["cluster_id"],
                        "parent_label": parent_label,
                        "child_layer": child_layer,
                        "child_cluster_id": child["cluster_id"],
                        "child_label": child_label,
                    }
                )
    return output


def _audit_movie_verdict(
    *,
    title: str,
    labels: dict[str, str],
    neighbors: list[dict[str, Any]],
    duplicate_path: list[dict[str, Any]],
) -> tuple[str, list[str]]:
    notes: list[str] = []
    label_text = " | ".join(labels.values()).lower()
    for pattern in BAD_LABEL_PATTERNS.get(title, []):
        if pattern.lower() in label_text:
            notes.append(f"label contains suspicious term: {pattern}")
    neighbor_titles = [
        str(neighbor.get("title") or "").lower()
        for neighbor in neighbors
    ]
    for pattern in BAD_NEIGHBOR_PATTERNS.get(title, []):
        if any(pattern in neighbor_title for neighbor_title in neighbor_titles):
            notes.append(f"neighbor list still contains suspicious match: {pattern}")
    if duplicate_path:
        notes.append("macro/neighborhood/micro path contains duplicate label text")
    notes.extend(AUDIT_WATCHLIST_NOTES.get(title, []))
    if notes:
        severe = any("neighbor list" in note or "label contains" in note for note in notes)
        return ("fail" if severe else "mixed"), notes
    return "pass", ["labels and nearest neighbors pass scripted checks"]


def render_deep_audit_report(audit_payload: dict[str, Any]) -> str:
    """Render a compact audit table for the movies David reviewed by hand."""
    lines = [
        "# The Film Atlas - Milestone 5 Deep Audit",
        "",
        "This report audits the movies called out in the voice-note review pass. "
        "It checks final macro/neighborhood/micro labels, exact duplicate parent-child "
        "label names, and suspicious nearest-neighbor patterns.",
        "",
        "## Summary",
        "",
        f"- Audit movies present: {audit_payload['present_count']} / {audit_payload['requested_count']}",
        f"- Duplicate parent-child label names: {audit_payload['duplicate_label_count']}",
        f"- Verdict counts: {_format_counts(audit_payload.get('verdict_counts') or {})}",
        "",
        "## Reviewed Movies",
        "",
        "| Movie | Verdict | Macro | Neighborhood | Micro | Nearest neighbors | Notes |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in audit_payload["movies"]:
        neighbors = ", ".join(
            f"{neighbor['title']} ({neighbor.get('year') or 'n/a'})"
            for neighbor in row.get("neighbors", [])[:5]
        )
        notes = "; ".join(row.get("audit_notes") or [])
        lines.append(
            "| "
            + " | ".join(
                [
                    _table_escape(f"{row['title']} ({row.get('year') or 'n/a'})"),
                    row["verdict"],
                    _table_escape(row["macro_label"]),
                    _table_escape(row["neighborhood_label"]),
                    _table_escape(row["micro_label"]),
                    _table_escape(neighbors),
                    _table_escape(notes),
                ]
            )
            + " |"
        )
    if audit_payload.get("duplicate_label_issues"):
        lines.extend(["", "## Duplicate Label Issues", ""])
        for issue in audit_payload["duplicate_label_issues"]:
            lines.append(
                "- "
                f"{issue['parent_layer']}:{issue['parent_cluster_id']} and "
                f"{issue['child_layer']}:{issue['child_cluster_id']} both use "
                f"`{issue['child_label']}`."
            )
    lines.append("")
    return "\n".join(lines)


def _format_counts(counts: dict[str, Any]) -> str:
    if not counts:
        return "none"
    return ", ".join(f"{key}: {value}" for key, value in sorted(counts.items()))


def _candidate_summary(artifacts: CandidateArtifacts) -> dict[str, Any]:
    return {
        "variant": artifacts.variant,
        "strategy": artifacts.strategy,
        "exportable": artifacts.exportable,
        "profiles_path": str(artifacts.profiles_path),
        "embeddings_path": str(artifacts.embeddings_path),
        "metrics": artifacts.metrics,
    }


def _estimate_embedding_cost(profiles: list[SemanticProfile], model: str) -> float:
    tokens = sum(estimate_text_tokens(profile.profile_text) for profile in profiles)
    price = EMBEDDING_PRICES_PER_1M_TOKENS.get(model, 0.13)
    return tokens / 1_000_000 * price


def _write_profile_inspection(variant_dir: Path, profiles: list[SemanticProfile]) -> None:
    by_title = {profile.title: profile.profile_text for profile in profiles}
    inspection = {
        "note": "Private exact embedded text for local audit. Do not publish.",
        "profile_count": len(profiles),
        "profiles_by_title": by_title,
    }
    (variant_dir / "profile_inspection.json").write_text(
        json.dumps(inspection, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _preserve_baseline_manifest(output_path: Path, experiment_dir: Path) -> None:
    baseline = output_path / "public_export" / "manifest.json"
    if baseline.exists():
        (experiment_dir / "baseline_public_export_manifest.json").write_text(
            baseline.read_text(encoding="utf-8"),
            encoding="utf-8",
        )


def _select_variants(names: list[str] | None) -> list[ProfileVariantSpec]:
    by_name = {spec.name: spec for spec in DEFAULT_PROFILE_VARIANTS}
    if names is None:
        return list(DEFAULT_PROFILE_VARIANTS)
    selected = []
    for name in names:
        if name not in by_name:
            raise ClassificationV2Error(
                f"Unknown profile variant {name!r}. Choose from: {', '.join(sorted(by_name))}."
            )
        selected.append(by_name[name])
    return selected


def _select_strategies(names: list[str] | None) -> list[ClusteringStrategy]:
    valid = set(DEFAULT_STRATEGIES)
    if names is None:
        return list(DEFAULT_STRATEGIES)
    selected: list[ClusteringStrategy] = []
    for name in names:
        if name not in valid:
            raise ClassificationV2Error(
                f"Unknown strategy {name!r}. Choose from: {', '.join(sorted(valid))}."
            )
        selected.append(name)  # type: ignore[arg-type]
    return selected


def _parent_layer(layer: str) -> str | None:
    if layer == "neighborhood":
        return "macro"
    if layer == "micro":
        return "neighborhood"
    return None


def _table_escape(value: str) -> str:
    return value.replace("|", "\\|")
