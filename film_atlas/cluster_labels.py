"""Draft microgenre labeling for cluster-level Film Atlas evidence."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx

from film_atlas.cluster import cluster_embedding_records
from film_atlas.embedding import estimate_text_tokens, load_embedding_records
from film_atlas.inspect_clusters import ClusterEvidence, build_cluster_evidence, load_cluster_evidence
from film_atlas.normalize import load_movie_records
from film_atlas.profiles import load_profiles

LABEL_PROMPT_VERSION = "cluster-labels-v2"
DEFAULT_LABEL_MODEL = "gpt-4.1-mini"
LABEL_CACHE_FILENAME = "cluster_label_cache.json"
LABEL_CANDIDATES_JSON_FILENAME = "cluster_label_candidates.json"
HUMAN_EDITABLE_LABELS_FILENAME = "human_editable_cluster_labels.json"
LABEL_CANDIDATES_REPORT_FILENAME = "cluster_label_candidates.md"
LABEL_REVIEW_REPORT_FILENAME = "cluster_label_review.md"
LABEL_MODEL_PRICES_PER_1M_TOKENS = {
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}
FALLBACK_PRICE_PER_1M_TOKENS = {"input": 2.00, "output": 8.00}


class ClusterLabelError(RuntimeError):
    """Raised when cluster labeling cannot proceed."""


@dataclass(frozen=True, slots=True)
class EvidenceSummary:
    genres: list[tuple[str, int]]
    tmdb_keywords: list[tuple[str, int]]
    aggregated_terms: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ClusterLabelCandidate:
    cluster_id: int
    cluster_size: int
    evidence_hash: str
    model: str
    prompt_version: str
    plain_label: str
    poetic_label: str
    spotify_style_label: str
    recommended_label: str
    one_sentence_description: str
    why_this_label_fits: str
    representative_movies: list[str]
    edge_case_movies: list[str]
    possible_misfits: list[str]
    confidence_score: float
    label_risk_notes: str
    evidence_summary: EvidenceSummary
    cached: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ClusterLabelCandidate:
        summary = value.get("evidence_summary") or {}
        return cls(
            cluster_id=int(value["cluster_id"]),
            cluster_size=int(value.get("cluster_size") or 0),
            evidence_hash=str(value["evidence_hash"]),
            model=str(value.get("model") or ""),
            prompt_version=str(value.get("prompt_version") or LABEL_PROMPT_VERSION),
            plain_label=str(value.get("plain_label") or ""),
            poetic_label=str(value.get("poetic_label") or ""),
            spotify_style_label=str(value.get("spotify_style_label") or ""),
            recommended_label=str(value.get("recommended_label") or ""),
            one_sentence_description=str(value.get("one_sentence_description") or ""),
            why_this_label_fits=str(value.get("why_this_label_fits") or ""),
            representative_movies=[str(item) for item in value.get("representative_movies") or []],
            edge_case_movies=[str(item) for item in value.get("edge_case_movies") or []],
            possible_misfits=[str(item) for item in value.get("possible_misfits") or []],
            confidence_score=float(value.get("confidence_score") or 0),
            label_risk_notes=str(value.get("label_risk_notes") or ""),
            evidence_summary=EvidenceSummary(
                genres=[(str(row[0]), int(row[1])) for row in summary.get("genres") or []],
                tmdb_keywords=[
                    (str(row[0]), int(row[1])) for row in summary.get("tmdb_keywords") or []
                ],
                aggregated_terms=[str(item) for item in summary.get("aggregated_terms") or []],
            ),
            cached=bool(value.get("cached") or False),
        )


@dataclass(frozen=True, slots=True)
class LabelingEstimate:
    cluster_count: int
    cached_count: int
    clusters_to_label: int
    model: str
    estimated_input_tokens: int
    estimated_output_tokens: int
    estimated_cost_usd: float
    openai_api_key_status: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class LabelingRunResult:
    candidates: list[ClusterLabelCandidate]
    estimate: LabelingEstimate
    json_path: Path
    report_path: Path
    editable_json_path: Path
    cached_reused_count: int
    new_label_count: int


class OpenAIClusterLabelClient:
    """Small HTTP client for draft cluster labels."""

    def __init__(
        self,
        api_key: str | None,
        *,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 90.0,
        max_retries: int = 3,
        backoff_seconds: float = 1.0,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.api_key = api_key
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self.sleep = sleep
        self.client = httpx.Client(base_url=base_url, timeout=timeout, transport=transport)

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> OpenAIClusterLabelClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def label_batch(self, evidences: list[ClusterEvidence], *, model: str) -> list[dict[str, Any]]:
        """Request labels for a batch of cluster evidence records."""
        if not self.api_key:
            raise ClusterLabelError(
                "OPENAI_API_KEY is required for live cluster labeling. "
                "Add it to .env, then run the command again."
            )
        messages = build_label_messages(evidences)
        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.7,
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        response_json = self._post_with_retries("/chat/completions", payload=payload, headers=headers)
        content = response_json["choices"][0]["message"]["content"]
        return parse_label_response(content)

    def _post_with_retries(
        self,
        endpoint: str,
        *,
        payload: dict[str, Any],
        headers: dict[str, str],
    ) -> dict[str, Any]:
        retry_statuses = {429, 500, 502, 503, 504}
        last_response: httpx.Response | None = None
        for attempt in range(self.max_retries + 1):
            response = self.client.post(endpoint, json=payload, headers=headers)
            last_response = response
            if response.status_code not in retry_statuses:
                response.raise_for_status()
                return response.json()
            if attempt < self.max_retries:
                retry_after = response.headers.get("Retry-After")
                delay = float(retry_after) if retry_after else self.backoff_seconds * 2**attempt
                self.sleep(delay)
        if last_response is None:
            raise ClusterLabelError("OpenAI label request failed before receiving a response.")
        last_response.raise_for_status()
        return last_response.json()


def estimate_labeling_file(
    *,
    evidence_path: str | Path | None = None,
    embeddings_path: str | Path = "outputs/intermediate/embeddings.jsonl",
    movies_path: str | Path = "data/processed/movies.json",
    profiles_path: str | Path = "data/processed/profiles.json",
    cache_path: str | Path = "outputs/intermediate/cluster_label_cache.json",
    model: str = DEFAULT_LABEL_MODEL,
    k: int = 35,
    openai_api_key: str | None = None,
) -> LabelingEstimate:
    """Estimate live labeling cost without calling OpenAI."""
    evidence = load_or_build_label_evidence(
        evidence_path=evidence_path,
        embeddings_path=embeddings_path,
        movies_path=movies_path,
        profiles_path=profiles_path,
        k=k,
    )
    return estimate_labeling(
        evidence,
        cache=load_label_cache(cache_path),
        model=model,
        openai_api_key=openai_api_key,
    )


def label_clusters_file(
    *,
    api_key: str | None,
    evidence_path: str | Path | None = None,
    embeddings_path: str | Path = "outputs/intermediate/embeddings.jsonl",
    movies_path: str | Path = "data/processed/movies.json",
    profiles_path: str | Path = "data/processed/profiles.json",
    output_dir: str | Path = "outputs",
    model: str = DEFAULT_LABEL_MODEL,
    method: str = "kmeans",
    k: int = 35,
    batch_size: int = 5,
    client: OpenAIClusterLabelClient | None = None,
) -> LabelingRunResult:
    """Generate draft cluster labels and write Milestone 3 review artifacts."""
    if method != "kmeans":
        raise ClusterLabelError("Milestone 3 labeling currently supports --method kmeans only.")
    evidence = load_or_build_label_evidence(
        evidence_path=evidence_path,
        embeddings_path=embeddings_path,
        movies_path=movies_path,
        profiles_path=profiles_path,
        k=k,
    )
    output_path = Path(output_dir)
    cache_path = output_path / "intermediate" / LABEL_CACHE_FILENAME
    cache = load_label_cache(cache_path)
    estimate = estimate_labeling(evidence, cache=cache, model=model, openai_api_key=api_key)
    if estimate.estimated_cost_usd > 1:
        raise ClusterLabelError(
            f"Estimated labeling cost is ${estimate.estimated_cost_usd:.4f}, which exceeds $1. "
            "Run estimate-labeling and get explicit approval before proceeding."
        )

    candidates, cached_reused, new_count = label_clusters(
        evidence,
        cache=cache,
        model=model,
        api_key=api_key,
        batch_size=batch_size,
        client=client,
    )
    write_label_cache(cache_path, candidates)
    json_path = output_path / "intermediate" / LABEL_CANDIDATES_JSON_FILENAME
    editable_json_path = output_path / "intermediate" / HUMAN_EDITABLE_LABELS_FILENAME
    report_path = output_path / "reports" / LABEL_CANDIDATES_REPORT_FILENAME
    review_path = output_path / "reports" / LABEL_REVIEW_REPORT_FILENAME
    json_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps([candidate.to_dict() for candidate in candidates], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    editable_json_path.write_text(
        json.dumps(render_human_editable_labels(candidates), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    report_text = render_label_candidates_report(candidates, estimate=estimate)
    report_path.write_text(report_text, encoding="utf-8")
    review_path.write_text(render_label_review_markdown(candidates), encoding="utf-8")
    return LabelingRunResult(
        candidates=candidates,
        estimate=estimate,
        json_path=json_path,
        report_path=report_path,
        editable_json_path=editable_json_path,
        cached_reused_count=cached_reused,
        new_label_count=new_count,
    )


def render_label_review_file(
    *,
    candidates_path: str | Path = "outputs/intermediate/cluster_label_candidates.json",
    output_dir: str | Path = "outputs",
) -> Path:
    """Render the human review report from existing cluster label candidates."""
    candidates = load_label_candidates(candidates_path)
    path = Path(output_dir) / "reports" / LABEL_REVIEW_REPORT_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_label_review_markdown(candidates), encoding="utf-8")
    editable_path = Path(output_dir) / "intermediate" / HUMAN_EDITABLE_LABELS_FILENAME
    editable_path.parent.mkdir(parents=True, exist_ok=True)
    editable_path.write_text(
        json.dumps(render_human_editable_labels(candidates), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path


def label_clusters(
    evidence: list[ClusterEvidence],
    *,
    cache: dict[str, ClusterLabelCandidate],
    model: str,
    api_key: str | None,
    batch_size: int,
    client: OpenAIClusterLabelClient | None = None,
) -> tuple[list[ClusterLabelCandidate], int, int]:
    """Generate labels for evidence, reusing cache entries where possible."""
    candidates_by_cluster: dict[int, ClusterLabelCandidate] = {}
    uncached: list[ClusterEvidence] = []
    cached_reused = 0
    for entry in evidence:
        key = label_cache_key(model, evidence_hash(entry))
        cached = cache.get(key)
        if cached:
            candidates_by_cluster[entry.cluster_id] = _with_cached(cached)
            cached_reused += 1
        else:
            uncached.append(entry)

    owns_client = client is None
    active_client = client or OpenAIClusterLabelClient(api_key)
    try:
        for batch in _chunks(uncached, max(1, batch_size)):
            responses = active_client.label_batch(batch, model=model)
            if len(responses) != len(batch):
                raise ClusterLabelError("Label response count did not match cluster batch size.")
            for entry, response in zip(batch, responses, strict=True):
                candidate = candidate_from_response(entry, response, model=model)
                candidates_by_cluster[entry.cluster_id] = candidate
    finally:
        if owns_client:
            active_client.close()

    ordered = [candidates_by_cluster[entry.cluster_id] for entry in sorted(evidence, key=_cluster_sort)]
    return ordered, cached_reused, len(uncached)


def estimate_labeling(
    evidence: list[ClusterEvidence],
    *,
    cache: dict[str, ClusterLabelCandidate],
    model: str,
    openai_api_key: str | None,
    batch_size: int = 5,
) -> LabelingEstimate:
    """Estimate labeling cost using prompt text and a conservative output allowance."""
    uncached = [
        entry for entry in evidence if label_cache_key(model, evidence_hash(entry)) not in cache
    ]
    input_tokens = sum(
        estimate_text_tokens(json.dumps(build_label_messages(batch), ensure_ascii=False))
        for batch in _chunks(uncached, max(1, batch_size))
    )
    output_tokens = len(uncached) * 450
    prices = LABEL_MODEL_PRICES_PER_1M_TOKENS.get(model, FALLBACK_PRICE_PER_1M_TOKENS)
    cost = (
        input_tokens / 1_000_000 * prices["input"]
        + output_tokens / 1_000_000 * prices["output"]
    )
    return LabelingEstimate(
        cluster_count=len(evidence),
        cached_count=len(evidence) - len(uncached),
        clusters_to_label=len(uncached),
        model=model,
        estimated_input_tokens=input_tokens,
        estimated_output_tokens=output_tokens,
        estimated_cost_usd=cost,
        openai_api_key_status="set" if openai_api_key else "missing",
    )


def build_label_messages(evidence: list[ClusterEvidence]) -> list[dict[str, str]]:
    """Build the label prompt for a batch of cluster evidence records."""
    system = (
        "You name clusters of movies for The Film Atlas, a non-commercial portfolio project. "
        "Create vivid but useful movie microgenre labels. The labels should feel alive, like "
        "a good playlist name, but they must stay faithful to the evidence. Avoid cringe "
        "phrasing, avoid generic official-genre restatements, and avoid franchise-only labels "
        "unless the evidence is truly franchise-specific. Return strict JSON only."
    )
    user_payload = {
        "task": "Draft human-reviewable Spotify-style movie microgenre labels.",
        "required_fields_per_cluster": [
            "cluster_id",
            "plain_label",
            "poetic_label",
            "spotify_style_label",
            "recommended_label",
            "one_sentence_description",
            "why_this_label_fits",
            "representative_movies",
            "edge_case_movies",
            "possible_misfits",
            "confidence_score",
            "label_risk_notes",
        ],
        "clusters": [prompt_evidence(entry) for entry in evidence],
        "style_rules": [
            "recommended_label should be concise, vivid, and usable in a UI.",
            "plain_label can be direct; poetic_label can be more evocative.",
            "spotify_style_label should feel like a playlist or microgenre name.",
            "Prefer poetic-but-true over specific-but-false.",
            "The recommended_label must fit the cluster as a whole, not only the first "
            "or most famous representative movie.",
            "If top genres/keywords show multiple subfamilies, name the shared axis "
            "broadly instead of choosing one narrow subfamily.",
            "Do not use specifics like moon, arctic, spacefaring, battlefield, alien, heist, "
            "rom-com, witchcraft, road trip, or phone unless cluster evidence strongly supports them.",
            "Likewise, avoid animation, spooky, dinosaur, wizard, pirate, war, boxing, or "
            "music unless those terms are broadly supported across genres, keywords, and representatives.",
            "For mixed franchise clusters, prefer inclusive labels like legacy sequels, "
            "creature adventures, comeback battles, or fantasy quests over a single franchise setting.",
            "Use parent-context warnings only as context; the child cluster evidence must still "
            "justify the label.",
            "If evidence is broad or mixed, choose a broader evocative label and explain the risk.",
            "confidence_score must be between 0 and 1.",
            "edge_case_movies and possible_misfits should only use titles from the evidence.",
        ],
        "response_shape": {"clusters": ["one object per input cluster"]},
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]


def prompt_evidence(entry: ClusterEvidence) -> dict[str, Any]:
    """Compact one cluster's evidence for label prompting."""
    return {
        "cluster_id": entry.cluster_id,
        "cluster_size": entry.cluster_size,
        "representative_movies": entry.representative_movies[:10],
        "top_official_genres": entry.top_official_genres[:8],
        "top_tmdb_keywords": entry.top_tmdb_keywords[:12],
        "aggregated_profile_terms": [term for term, _score in entry.aggregated_profile_terms[:15]],
        "coherence_score": entry.coherence_score,
        "warnings": entry.warnings,
    }


def parse_label_response(content: str) -> list[dict[str, Any]]:
    """Parse a model JSON response into cluster label objects."""
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ClusterLabelError("OpenAI label response was not valid JSON.") from exc
    clusters = payload.get("clusters")
    if not isinstance(clusters, list):
        raise ClusterLabelError("OpenAI label response must contain a clusters array.")
    return [dict(item) for item in clusters]


def candidate_from_response(
    evidence: ClusterEvidence,
    response: dict[str, Any],
    *,
    model: str,
) -> ClusterLabelCandidate:
    """Normalize a raw label response into a candidate record."""
    response_cluster_id = int(response.get("cluster_id", evidence.cluster_id))
    if response_cluster_id != evidence.cluster_id:
        raise ClusterLabelError(
            f"Label response cluster_id {response_cluster_id} did not match {evidence.cluster_id}."
        )
    return ClusterLabelCandidate(
        cluster_id=evidence.cluster_id,
        cluster_size=evidence.cluster_size,
        evidence_hash=evidence_hash(evidence),
        model=model,
        prompt_version=LABEL_PROMPT_VERSION,
        plain_label=str(response.get("plain_label") or ""),
        poetic_label=str(response.get("poetic_label") or ""),
        spotify_style_label=str(response.get("spotify_style_label") or ""),
        recommended_label=str(response.get("recommended_label") or ""),
        one_sentence_description=str(response.get("one_sentence_description") or ""),
        why_this_label_fits=str(response.get("why_this_label_fits") or ""),
        representative_movies=_string_list(
            response.get("representative_movies") or evidence.representative_movies[:8]
        ),
        edge_case_movies=_string_list(response.get("edge_case_movies") or []),
        possible_misfits=_string_list(response.get("possible_misfits") or []),
        confidence_score=_bounded_confidence(response.get("confidence_score")),
        label_risk_notes=str(response.get("label_risk_notes") or ""),
        evidence_summary=evidence_summary(evidence),
    )


def load_or_build_label_evidence(
    *,
    evidence_path: str | Path | None,
    embeddings_path: str | Path,
    movies_path: str | Path,
    profiles_path: str | Path,
    k: int,
) -> list[ClusterEvidence]:
    """Load cluster evidence or regenerate k-means evidence from existing embeddings."""
    if evidence_path is not None:
        path = Path(evidence_path)
        if not path.exists():
            raise ClusterLabelError(f"Cluster evidence not found at {path}.")
        evidence = load_cluster_evidence(path)
        if not evidence:
            raise ClusterLabelError(f"Cluster evidence at {path} is empty.")
        return sorted(evidence, key=_cluster_sort)

    embeddings_file = Path(embeddings_path)
    if not embeddings_file.exists():
        raise ClusterLabelError(
            f"No embeddings found at {embeddings_file}. Milestone 3 reuses existing embeddings "
            "and does not re-embed profiles."
        )
    embeddings = load_embedding_records(embeddings_file)
    if not embeddings:
        raise ClusterLabelError(f"No embedding records found at {embeddings_file}.")
    assignments = cluster_embedding_records(embeddings, n_clusters=k)
    evidence = build_cluster_evidence(
        movies=load_movie_records(movies_path),
        profiles=load_profiles(profiles_path),
        embeddings=embeddings,
        assignments=assignments,
        neighbors=[],
    )
    return sorted(evidence, key=_cluster_sort)


def evidence_hash(evidence: ClusterEvidence) -> str:
    """Hash the prompt-relevant evidence for one cluster."""
    payload = {
        "prompt_version": LABEL_PROMPT_VERSION,
        "cluster_id": evidence.cluster_id,
        "cluster_size": evidence.cluster_size,
        "representative_movies": evidence.representative_movies,
        "top_official_genres": evidence.top_official_genres,
        "top_tmdb_keywords": evidence.top_tmdb_keywords,
        "aggregated_profile_terms": evidence.aggregated_profile_terms,
        "coherence_score": evidence.coherence_score,
        "warnings": evidence.warnings,
    }
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def label_cache_key(model: str, digest: str) -> str:
    """Return the stable cache key for a model/evidence pair."""
    return f"{model}:{LABEL_PROMPT_VERSION}:{digest}"


def load_label_cache(path: str | Path) -> dict[str, ClusterLabelCandidate]:
    """Load cached label candidates."""
    cache_path = Path(path)
    if not cache_path.exists():
        return {}
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    return {
        str(key): ClusterLabelCandidate.from_dict(value)
        for key, value in (payload.get("entries") or {}).items()
    }


def write_label_cache(path: str | Path, candidates: list[ClusterLabelCandidate]) -> Path:
    """Write label cache entries keyed by model and evidence hash."""
    cache_path = Path(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "prompt_version": LABEL_PROMPT_VERSION,
        "entries": {
            label_cache_key(candidate.model, candidate.evidence_hash): candidate.to_dict()
            for candidate in candidates
        },
    }
    cache_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return cache_path


def load_label_candidates(path: str | Path) -> list[ClusterLabelCandidate]:
    """Load generated label candidates."""
    candidate_path = Path(path)
    if not candidate_path.exists():
        raise ClusterLabelError(f"Cluster label candidates not found at {candidate_path}.")
    payload = json.loads(candidate_path.read_text(encoding="utf-8"))
    return [ClusterLabelCandidate.from_dict(item) for item in payload]


def render_human_editable_labels(candidates: list[ClusterLabelCandidate]) -> list[dict[str, Any]]:
    """Render an easy-to-edit draft label JSON structure."""
    return [
        {
            "cluster_id": candidate.cluster_id,
            "cluster_size": candidate.cluster_size,
            "recommended_label": candidate.recommended_label,
            "plain_label": candidate.plain_label,
            "poetic_label": candidate.poetic_label,
            "spotify_style_label": candidate.spotify_style_label,
            "one_sentence_description": candidate.one_sentence_description,
            "why_this_label_fits": candidate.why_this_label_fits,
            "representative_movies": candidate.representative_movies,
            "edge_case_movies": candidate.edge_case_movies,
            "possible_misfits": candidate.possible_misfits,
            "confidence_score": candidate.confidence_score,
            "label_risk_notes": candidate.label_risk_notes,
            "human_review": {
                "approved": False,
                "final_label": candidate.recommended_label,
                "final_description": candidate.one_sentence_description,
                "notes": "",
            },
            "evidence_summary": candidate.evidence_summary.to_dict(),
        }
        for candidate in sorted(candidates, key=lambda item: item.cluster_id)
    ]


def render_label_candidates_report(
    candidates: list[ClusterLabelCandidate],
    *,
    estimate: LabelingEstimate,
) -> str:
    """Render the candidate label report."""
    lines = [
        "# The Film Atlas - Milestone 3 Draft Cluster Label Candidates",
        "",
        "These are draft, human-reviewable microgenre labels for existing k-means k=35 "
        "clusters. They are not final public website labels or export JSON.",
        "",
        "## Summary",
        "",
        f"- Clusters labeled: {len(candidates)}",
        f"- Label model: {estimate.model}",
        f"- Cached labels reused: {estimate.cached_count}",
        f"- New labels estimated/requested: {estimate.clusters_to_label}",
        f"- Estimated labeling cost: ${estimate.estimated_cost_usd:.4f}",
        "- Clustering method: kmeans",
        "- Cluster count target: 35",
        "",
        "## Candidate Labels",
        "",
        _candidate_table(candidates),
        "",
        "## Clusters Needing Care",
        "",
        _risk_notes(candidates),
        "",
    ]
    return "\n".join(lines)


def render_label_review_markdown(candidates: list[ClusterLabelCandidate]) -> str:
    """Render a detailed human review worksheet."""
    sections = [
        "# The Film Atlas - Milestone 3 Cluster Label Review",
        "",
        "Use this worksheet to accept, revise, or reject draft microgenre labels before any "
        "public export or frontend integration.",
        "",
    ]
    for candidate in sorted(candidates, key=lambda item: item.cluster_id):
        sections.append(
            "\n".join(
                [
                    f"## Cluster {candidate.cluster_id}: {candidate.recommended_label}",
                    "",
                    f"- Size: {candidate.cluster_size}",
                    f"- Confidence: {candidate.confidence_score:.2f}",
                    f"- Plain label: {candidate.plain_label}",
                    f"- Poetic label: {candidate.poetic_label}",
                    f"- Spotify-style label: {candidate.spotify_style_label}",
                    f"- Description: {candidate.one_sentence_description}",
                    f"- Why it fits: {candidate.why_this_label_fits}",
                    f"- Risk notes: {candidate.label_risk_notes or 'none'}",
                    f"- Representative movies: {', '.join(candidate.representative_movies)}",
                    f"- Edge cases: {', '.join(candidate.edge_case_movies) or 'none listed'}",
                    f"- Possible misfits: {', '.join(candidate.possible_misfits) or 'none listed'}",
                    f"- Genres: {_count_terms(candidate.evidence_summary.genres)}",
                    f"- TMDb keywords: {_count_terms(candidate.evidence_summary.tmdb_keywords)}",
                    f"- Aggregated terms: {', '.join(candidate.evidence_summary.aggregated_terms)}",
                    "",
                    "Review fields:",
                    "",
                    "- Approved:",
                    "- Final label:",
                    "- Final description:",
                    "- Notes:",
                    "",
                ]
            )
        )
    return "\n".join(sections)


def evidence_summary(evidence: ClusterEvidence) -> EvidenceSummary:
    """Build compact evidence summary for output artifacts."""
    return EvidenceSummary(
        genres=evidence.top_official_genres[:8],
        tmdb_keywords=evidence.top_tmdb_keywords[:12],
        aggregated_terms=[term for term, _score in evidence.aggregated_profile_terms[:15]],
    )


def _candidate_table(candidates: list[ClusterLabelCandidate]) -> str:
    lines = [
        "| Cluster | Size | Recommended Label | Confidence | Representative Movies | Risk Notes |",
        "| ---: | ---: | --- | ---: | --- | --- |",
    ]
    for candidate in sorted(candidates, key=lambda item: item.cluster_id):
        lines.append(
            f"| {candidate.cluster_id} | {candidate.cluster_size} | "
            f"{_escape_table(candidate.recommended_label)} | {candidate.confidence_score:.2f} | "
            f"{_escape_table(', '.join(candidate.representative_movies[:5]))} | "
            f"{_escape_table(candidate.label_risk_notes or 'none')} |"
        )
    return "\n".join(lines)


def _risk_notes(candidates: list[ClusterLabelCandidate]) -> str:
    risky = [
        candidate
        for candidate in candidates
        if candidate.confidence_score < 0.7 or candidate.label_risk_notes.strip()
    ]
    if not risky:
        return "_No low-confidence or risk-flagged labels._"
    return "\n".join(
        f"- Cluster {candidate.cluster_id} ({candidate.recommended_label}): "
        f"confidence {candidate.confidence_score:.2f}; "
        f"{candidate.label_risk_notes or 'review recommended.'}"
        for candidate in sorted(risky, key=lambda item: (item.confidence_score, item.cluster_id))
    )


def _count_terms(items: list[tuple[str, int]]) -> str:
    return ", ".join(f"{name} ({count})" for name, count in items) or "none"


def _bounded_confidence(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.5
    return min(1.0, max(0.0, number))


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if value is None:
        return []
    return [str(value)]


def _with_cached(candidate: ClusterLabelCandidate) -> ClusterLabelCandidate:
    return ClusterLabelCandidate(
        **{**candidate.to_dict(), "cached": True, "evidence_summary": candidate.evidence_summary}
    )


def _chunks(items: list[Any], size: int) -> list[list[Any]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def _cluster_sort(evidence: ClusterEvidence) -> tuple[int, int]:
    return (evidence.cluster_id, evidence.cluster_size)


def _escape_table(value: str) -> str:
    return value.replace("|", "\\|")
