"""Command line interface for The Film Atlas Milestone 1 pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, cast

import typer

from film_atlas.config import MissingCredentialsError, load_settings
from film_atlas.classification_v2 import ClassificationV2Error, classification_v2_file
from film_atlas.cluster import cluster_embeddings_file
from film_atlas.cluster_sweep import ClusterSweepError, parse_ks, sweep_clusters_file
from film_atlas.clustering_methods import (
    ClusteringMethodComparisonError,
    compare_clustering_methods_file,
    parse_methods,
)
from film_atlas.cluster_labels import (
    ClusterLabelError,
    DEFAULT_LABEL_MODEL,
    estimate_labeling_file,
    label_clusters_file,
    render_label_review_file,
)
from film_atlas.embedding import (
    DEFAULT_EMBEDDING_MODEL,
    embed_profiles_file,
    estimate_profiles_file,
)
from film_atlas.fetch import fetch_balanced as fetch_balanced_step
from film_atlas.fetch import fetch_details as fetch_details_step
from film_atlas.fetch import fetch_discover as fetch_discover_step
from film_atlas.inspect_clusters import inspect_clusters_file
from film_atlas.large_audit import LargeAuditError, large_audit_file
from film_atlas.milestone2_report import generate_milestone_2_report_file
from film_atlas.milestone4 import (
    Milestone4Error,
    build_hierarchy_file,
    export_atlas_data_file,
    milestone_4_file,
    scale_dataset_file,
)
from film_atlas.neighbors import compute_neighbors_file
from film_atlas.normalize import normalize_details_file
from film_atlas.profiles import ReviewWeight, build_profiles_file
from film_atlas.reduce import reduce_embeddings_file
from film_atlas.report import generate_report_file
from film_atlas.review_ablation import (
    ReviewAblationError,
    parse_review_variants,
    review_ablation_file,
)
from film_atlas.sample_map import make_sample_map_file
from film_atlas.territory_layout import TerritoryLayoutError, build_territory_layouts_file
from film_atlas.tmdb_client import TMDbClient

app = typer.Typer(help="The Film Atlas offline data pipeline.")


def main() -> None:
    """Entrypoint for console scripts."""
    app()


@app.command()
def fetch_discover(
    limit: Annotated[int, typer.Option(help="Maximum number of discovered movies.")] = 500,
    min_votes: Annotated[int, typer.Option(help="Minimum TMDb vote count.")] = 500,
    output_dir: Annotated[Path | None, typer.Option(help="Base data output directory.")] = None,
    sort_by: Annotated[str, typer.Option(help="TMDb discover sort order.")] = "popularity.desc",
    min_runtime: Annotated[int, typer.Option(help="Minimum runtime in minutes.")] = 60,
    release_date_gte: Annotated[
        str | None,
        typer.Option(help="Earliest primary release date, YYYY-MM-DD."),
    ] = None,
    release_date_lte: Annotated[
        str | None,
        typer.Option(help="Latest primary release date, YYYY-MM-DD."),
    ] = None,
    exclude_future: Annotated[
        bool,
        typer.Option(
            "--exclude-future/--include-future",
            help="Exclude future/unreleased films by capping release date at today.",
        ),
    ] = True,
    refresh: Annotated[bool, typer.Option(help="Ignore cached responses.")] = False,
) -> None:
    """Fetch a controlled sample from TMDb /discover/movie."""
    settings = load_settings(data_dir=output_dir)
    try:
        with TMDbClient(
            settings.tmdb_bearer_token,
            cache_dir=settings.data_dir / "cache",
        ) as client:
            path = fetch_discover_step(
                client,
                limit=limit,
                min_votes=min_votes,
                output_dir=settings.data_dir,
                sort_by=sort_by,
                min_runtime=min_runtime,
                release_date_gte=release_date_gte,
                release_date_lte=release_date_lte,
                exclude_future=exclude_future,
                refresh=refresh,
            )
    except MissingCredentialsError as exc:
        _missing_token_exit(exc)
    typer.echo(f"Wrote discovered movies to {path}")


@app.command()
def fetch_balanced(
    per_decade: Annotated[int, typer.Option(help="Movies to fetch per decade bucket.")] = 100,
    start_year: Annotated[int, typer.Option(help="First release year to include.")] = 1980,
    end_year: Annotated[int, typer.Option(help="Last release year to include.")] = 2026,
    min_votes: Annotated[int, typer.Option(help="Minimum TMDb vote count.")] = 500,
    output_dir: Annotated[Path | None, typer.Option(help="Base data output directory.")] = None,
    sort_by: Annotated[str, typer.Option(help="TMDb discover sort order.")] = "vote_count.desc",
    min_runtime: Annotated[int, typer.Option(help="Minimum runtime in minutes.")] = 60,
    exclude_future: Annotated[
        bool,
        typer.Option(
            "--exclude-future/--include-future",
            help="Exclude future/unreleased films by capping release date at today.",
        ),
    ] = True,
    refresh: Annotated[bool, typer.Option(help="Ignore cached responses.")] = False,
) -> None:
    """Fetch a decade-balanced TMDb sample into data/raw/discover_movies.json."""
    settings = load_settings(data_dir=output_dir)
    try:
        with TMDbClient(
            settings.tmdb_bearer_token,
            cache_dir=settings.data_dir / "cache",
        ) as client:
            path = fetch_balanced_step(
                client,
                per_decade=per_decade,
                start_year=start_year,
                end_year=end_year,
                min_votes=min_votes,
                output_dir=settings.data_dir,
                sort_by=sort_by,
                min_runtime=min_runtime,
                exclude_future=exclude_future,
                refresh=refresh,
            )
    except MissingCredentialsError as exc:
        _missing_token_exit(exc)
    typer.echo(f"Wrote balanced discovered movies to {path}")


@app.command()
def fetch_details(
    discover_path: Annotated[
        Path | None,
        typer.Option(help="Path to data/raw/discover_movies.json."),
    ] = None,
    output_dir: Annotated[Path | None, typer.Option(help="Base data output directory.")] = None,
    limit: Annotated[int | None, typer.Option(help="Optional detail-fetch limit.")] = None,
    refresh: Annotated[bool, typer.Option(help="Ignore cached responses.")] = False,
) -> None:
    """Fetch TMDb details for discovered movie IDs."""
    settings = load_settings(data_dir=output_dir)
    resolved_discover_path = discover_path or settings.data_dir / "raw" / "discover_movies.json"
    try:
        with TMDbClient(
            settings.tmdb_bearer_token,
            cache_dir=settings.data_dir / "cache",
        ) as client:
            path = fetch_details_step(
                client,
                discover_path=resolved_discover_path,
                output_dir=settings.data_dir,
                limit=limit,
                refresh=refresh,
            )
    except MissingCredentialsError as exc:
        _missing_token_exit(exc)
    typer.echo(f"Wrote movie details to {path}")


@app.command()
def normalize(
    details_path: Annotated[
        Path | None,
        typer.Option(help="Path to data/raw/movie_details.json."),
    ] = None,
    output_dir: Annotated[Path | None, typer.Option(help="Base data output directory.")] = None,
) -> None:
    """Normalize raw TMDb details into processed JSON and Parquet files."""
    settings = load_settings(data_dir=output_dir)
    resolved_details_path = details_path or settings.data_dir / "raw" / "movie_details.json"
    path = normalize_details_file(details_path=resolved_details_path, output_dir=settings.data_dir)
    typer.echo(f"Wrote normalized movies to {path}")


@app.command()
def build_profiles(
    movies_path: Annotated[
        Path | None,
        typer.Option(help="Path to data/processed/movies.json."),
    ] = None,
    output_dir: Annotated[Path | None, typer.Option(help="Base data output directory.")] = None,
    include_reviews: Annotated[
        bool,
        typer.Option(
            "--include-reviews/--no-include-reviews",
            help="Include conservative review-language snippets in profile text.",
        ),
    ] = True,
    max_review_chars: Annotated[
        int,
        typer.Option(help="Maximum total review-language characters per movie profile."),
    ] = 180,
    review_weight: Annotated[
        str,
        typer.Option(help="Review contribution weight: light, medium, or heavy."),
    ] = "light",
) -> None:
    """Build semantic profile text for each normalized movie."""
    settings = load_settings(data_dir=output_dir)
    resolved_movies_path = movies_path or settings.data_dir / "processed" / "movies.json"
    path = build_profiles_file(
        movies_path=resolved_movies_path,
        output_dir=settings.data_dir,
        include_reviews=include_reviews,
        max_review_chars=max_review_chars,
        review_weight=_parse_review_weight(review_weight),
    )
    typer.echo(f"Wrote semantic profiles to {path}")


@app.command()
def make_sample_map(
    profiles_path: Annotated[
        Path | None,
        typer.Option(help="Path to data/processed/profiles.json."),
    ] = None,
    output_dir: Annotated[Path | None, typer.Option(help="Base outputs directory.")] = None,
) -> None:
    """Generate a rough TF-IDF/SVD 2D sample map."""
    settings = load_settings(output_dir=output_dir)
    resolved_profiles_path = profiles_path or settings.data_dir / "processed" / "profiles.json"
    result = make_sample_map_file(
        profiles_path=resolved_profiles_path,
        output_dir=settings.output_dir,
    )
    typer.echo(f"Wrote sample map CSV to {result.csv_path}")
    typer.echo(f"Wrote sample map JSON to {result.json_path}")
    typer.echo(f"Wrote sample map HTML to {result.html_path}")
    typer.echo(f"Wrote nearest-neighbor pairs to {result.neighbors_path}")


@app.command()
def report(
    data_dir: Annotated[Path | None, typer.Option(help="Base data directory.")] = None,
    output_dir: Annotated[Path | None, typer.Option(help="Base outputs directory.")] = None,
) -> None:
    """Generate the Milestone 1 data-quality report."""
    settings = load_settings(data_dir=data_dir, output_dir=output_dir)
    path = generate_report_file(data_dir=settings.data_dir, output_dir=settings.output_dir)
    typer.echo(f"Wrote Milestone 1 report to {path}")


@app.command()
def estimate_embeddings(
    profiles_path: Annotated[
        Path | None,
        typer.Option(help="Path to data/processed/profiles.json."),
    ] = None,
    limit: Annotated[int | None, typer.Option(help="Optional profile limit.")] = None,
    model: Annotated[str | None, typer.Option(help="OpenAI embedding model override.")] = None,
) -> None:
    """Estimate OpenAI embedding tokens and cost before live API calls."""
    settings = load_settings()
    resolved_profiles_path = profiles_path or settings.data_dir / "processed" / "profiles.json"
    resolved_model = model or settings.openai_embedding_model or DEFAULT_EMBEDDING_MODEL
    estimate = estimate_profiles_file(
        profiles_path=resolved_profiles_path,
        model=resolved_model,
        limit=limit,
    )
    typer.echo(f"Profiles available: {estimate.profile_count}")
    typer.echo(f"Profiles selected: {estimate.selected_count}")
    typer.echo(f"Embedding model: {estimate.model}")
    typer.echo(f"Estimated tokens: {estimate.estimated_tokens}")
    typer.echo(f"Estimated cost: ${estimate.estimated_cost_usd:.4f}")


@app.command()
def embed_profiles(
    profiles_path: Annotated[
        Path | None,
        typer.Option(help="Path to data/processed/profiles.json."),
    ] = None,
    output_dir: Annotated[Path | None, typer.Option(help="Base outputs directory.")] = None,
    limit: Annotated[int | None, typer.Option(help="Optional profile limit.")] = None,
    model: Annotated[str | None, typer.Option(help="OpenAI embedding model override.")] = None,
    batch_size: Annotated[int, typer.Option(help="OpenAI embedding request batch size.")] = 64,
) -> None:
    """Generate or reuse cached OpenAI embeddings for semantic profiles."""
    settings = load_settings(output_dir=output_dir)
    resolved_profiles_path = profiles_path or settings.data_dir / "processed" / "profiles.json"
    resolved_model = model or settings.openai_embedding_model or DEFAULT_EMBEDDING_MODEL
    estimate = estimate_profiles_file(
        profiles_path=resolved_profiles_path,
        model=resolved_model,
        limit=limit,
    )
    if estimate.estimated_cost_usd > 1:
        typer.echo(
            f"Estimated cost is ${estimate.estimated_cost_usd:.4f}, which exceeds $1. "
            "Run with a smaller --limit or confirm manually before proceeding.",
            err=True,
        )
        raise typer.Exit(code=2)
    try:
        result = embed_profiles_file(
            api_key=settings.require_openai_api_key(),
            profiles_path=resolved_profiles_path,
            output_dir=settings.output_dir,
            model=resolved_model,
            limit=limit,
            batch_size=batch_size,
        )
    except MissingCredentialsError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    typer.echo(f"Wrote embeddings to {result.embedding_path}")
    typer.echo(f"Wrote embedding manifest to {result.manifest_path}")
    typer.echo(f"Cached reused: {result.cached_reused_count}")
    typer.echo(f"New embeddings: {result.new_embedding_count}")


@app.command()
def reduce_embeddings(
    embeddings_path: Annotated[
        Path,
        typer.Option(help="Path to outputs/intermediate/embeddings.jsonl."),
    ] = Path("outputs/intermediate/embeddings.jsonl"),
    output_dir: Annotated[Path | None, typer.Option(help="Base outputs directory.")] = None,
) -> None:
    """Project embeddings into 2D coordinates."""
    settings = load_settings(output_dir=output_dir)
    path = reduce_embeddings_file(embeddings_path=embeddings_path, output_dir=settings.output_dir)
    typer.echo(f"Wrote coordinates to {path}")


@app.command()
def cluster_movies(
    embeddings_path: Annotated[
        Path,
        typer.Option(help="Path to outputs/intermediate/embeddings.jsonl."),
    ] = Path("outputs/intermediate/embeddings.jsonl"),
    output_dir: Annotated[Path | None, typer.Option(help="Base outputs directory.")] = None,
    n_clusters: Annotated[int | None, typer.Option(help="Optional k-means cluster count.")] = None,
) -> None:
    """Cluster embedded movies."""
    settings = load_settings(output_dir=output_dir)
    path = cluster_embeddings_file(
        embeddings_path=embeddings_path,
        output_dir=settings.output_dir,
        n_clusters=n_clusters,
    )
    typer.echo(f"Wrote cluster assignments to {path}")


@app.command()
def compute_neighbors(
    embeddings_path: Annotated[
        Path,
        typer.Option(help="Path to outputs/intermediate/embeddings.jsonl."),
    ] = Path("outputs/intermediate/embeddings.jsonl"),
    output_dir: Annotated[Path | None, typer.Option(help="Base outputs directory.")] = None,
    top_n: Annotated[int, typer.Option(help="Neighbors per movie.")] = 10,
) -> None:
    """Compute cosine-similarity nearest neighbors."""
    settings = load_settings(output_dir=output_dir)
    path = compute_neighbors_file(
        embeddings_path=embeddings_path,
        output_dir=settings.output_dir,
        top_n=top_n,
    )
    typer.echo(f"Wrote neighbors to {path}")


@app.command()
def inspect_clusters(
    movies_path: Annotated[
        Path | None,
        typer.Option(help="Path to data/processed/movies.json."),
    ] = None,
    profiles_path: Annotated[
        Path | None,
        typer.Option(help="Path to data/processed/profiles.json."),
    ] = None,
    embeddings_path: Annotated[
        Path,
        typer.Option(help="Path to outputs/intermediate/embeddings.jsonl."),
    ] = Path("outputs/intermediate/embeddings.jsonl"),
    assignments_path: Annotated[
        Path,
        typer.Option(help="Path to outputs/intermediate/cluster_assignments.json."),
    ] = Path("outputs/intermediate/cluster_assignments.json"),
    neighbors_path: Annotated[
        Path,
        typer.Option(help="Path to outputs/intermediate/neighbors.json."),
    ] = Path("outputs/intermediate/neighbors.json"),
    output_dir: Annotated[Path | None, typer.Option(help="Base outputs directory.")] = None,
) -> None:
    """Generate cluster-level evidence for inspection."""
    settings = load_settings(output_dir=output_dir)
    resolved_movies_path = movies_path or settings.data_dir / "processed" / "movies.json"
    resolved_profiles_path = profiles_path or settings.data_dir / "processed" / "profiles.json"
    path = inspect_clusters_file(
        movies_path=resolved_movies_path,
        profiles_path=resolved_profiles_path,
        embeddings_path=embeddings_path,
        assignments_path=assignments_path,
        neighbors_path=neighbors_path,
        output_dir=settings.output_dir,
    )
    typer.echo(f"Wrote cluster evidence to {path}")


@app.command()
def sweep_clusters(
    ks: Annotated[
        str,
        typer.Option(help="Comma-separated k-means cluster counts, for example: 15,25,35,50."),
    ] = "15,25,35,50",
    movies_path: Annotated[
        Path,
        typer.Option(help="Path to data/processed/movies.json."),
    ] = Path("data/processed/movies.json"),
    profiles_path: Annotated[
        Path,
        typer.Option(help="Path to data/processed/profiles.json."),
    ] = Path("data/processed/profiles.json"),
    embeddings_path: Annotated[
        Path,
        typer.Option(help="Path to outputs/intermediate/embeddings.jsonl."),
    ] = Path("outputs/intermediate/embeddings.jsonl"),
    neighbors_path: Annotated[
        Path,
        typer.Option(help="Path to outputs/intermediate/neighbors.json."),
    ] = Path("outputs/intermediate/neighbors.json"),
    output_dir: Annotated[Path, typer.Option(help="Base outputs directory.")] = Path("outputs"),
) -> None:
    """Compare local k-means cluster counts over existing embeddings."""
    try:
        json_path, report_path = sweep_clusters_file(
            ks=parse_ks(ks),
            movies_path=movies_path,
            profiles_path=profiles_path,
            embeddings_path=embeddings_path,
            neighbors_path=neighbors_path,
            output_dir=output_dir,
        )
    except ClusterSweepError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    typer.echo(f"Wrote cluster sweep JSON to {json_path}")
    typer.echo(f"Wrote cluster sweep report to {report_path}")


@app.command()
def compare_clustering_methods(
    methods: Annotated[
        str,
        typer.Option(
            help="Comma-separated clustering methods: kmeans,agglomerative,graph,hdbscan.",
        ),
    ] = "kmeans,agglomerative,graph,hdbscan",
    kmeans_k: Annotated[int, typer.Option(help="Cluster count for k-means baseline.")] = 35,
    agglomerative_k: Annotated[
        int,
        typer.Option(help="Cluster count for agglomerative clustering."),
    ] = 35,
    graph_neighbors: Annotated[
        int,
        typer.Option(help="Nearest neighbors per movie for graph/community clustering."),
    ] = 10,
    min_cluster_size: Annotated[
        int,
        typer.Option(help="Minimum cluster size for optional HDBSCAN."),
    ] = 5,
    limit: Annotated[
        int | None,
        typer.Option(help="Optional embedding-record limit for local experiments."),
    ] = None,
    movies_path: Annotated[
        Path,
        typer.Option(help="Path to data/processed/movies.json."),
    ] = Path("data/processed/movies.json"),
    profiles_path: Annotated[
        Path,
        typer.Option(help="Path to data/processed/profiles.json."),
    ] = Path("data/processed/profiles.json"),
    embeddings_path: Annotated[
        Path,
        typer.Option(help="Path to outputs/intermediate/embeddings.jsonl."),
    ] = Path("outputs/intermediate/embeddings.jsonl"),
    output_dir: Annotated[Path, typer.Option(help="Base outputs directory.")] = Path("outputs"),
) -> None:
    """Compare local clustering methods over existing full embedding vectors."""
    try:
        json_path, report_path = compare_clustering_methods_file(
            methods=parse_methods(methods),
            movies_path=movies_path,
            profiles_path=profiles_path,
            embeddings_path=embeddings_path,
            output_dir=output_dir,
            kmeans_k=kmeans_k,
            agglomerative_k=agglomerative_k,
            graph_neighbors=graph_neighbors,
            min_cluster_size=min_cluster_size,
            limit=limit,
        )
    except ClusteringMethodComparisonError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    typer.echo(f"Wrote clustering method comparison JSON to {json_path}")
    typer.echo(f"Wrote clustering method comparison report to {report_path}")


@app.command()
def estimate_labeling(
    evidence_path: Annotated[
        Path | None,
        typer.Option(help="Optional path to precomputed cluster evidence JSON."),
    ] = None,
    embeddings_path: Annotated[
        Path,
        typer.Option(help="Path to outputs/intermediate/embeddings.jsonl."),
    ] = Path("outputs/intermediate/embeddings.jsonl"),
    movies_path: Annotated[
        Path,
        typer.Option(help="Path to data/processed/movies.json."),
    ] = Path("data/processed/movies.json"),
    profiles_path: Annotated[
        Path,
        typer.Option(help="Path to data/processed/profiles.json."),
    ] = Path("data/processed/profiles.json"),
    output_dir: Annotated[Path, typer.Option(help="Base outputs directory.")] = Path("outputs"),
    model: Annotated[str | None, typer.Option(help="OpenAI label model override.")] = None,
    k: Annotated[int, typer.Option(help="k-means cluster count to label.")] = 35,
) -> None:
    """Estimate Milestone 3 draft cluster-labeling cost without calling OpenAI."""
    settings = load_settings(output_dir=output_dir)
    resolved_model = model or settings.openai_label_model or DEFAULT_LABEL_MODEL
    try:
        estimate = estimate_labeling_file(
            evidence_path=evidence_path,
            embeddings_path=embeddings_path,
            movies_path=movies_path,
            profiles_path=profiles_path,
            cache_path=settings.output_dir / "intermediate" / "cluster_label_cache.json",
            model=resolved_model,
            k=k,
            openai_api_key=settings.openai_api_key,
        )
    except ClusterLabelError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    typer.echo(f"Clusters available: {estimate.cluster_count}")
    typer.echo(f"Cached labels reusable: {estimate.cached_count}")
    typer.echo(f"Clusters to label: {estimate.clusters_to_label}")
    typer.echo(f"Label model: {estimate.model}")
    typer.echo(f"Estimated input tokens: {estimate.estimated_input_tokens}")
    typer.echo(f"Estimated output tokens: {estimate.estimated_output_tokens}")
    typer.echo(f"Estimated cost: ${estimate.estimated_cost_usd:.4f}")
    typer.echo(f"OPENAI_API_KEY: {estimate.openai_api_key_status}")


@app.command()
def label_clusters(
    method: Annotated[str, typer.Option(help="Clustering method to label.")] = "kmeans",
    k: Annotated[int, typer.Option(help="k-means cluster count to label.")] = 35,
    evidence_path: Annotated[
        Path | None,
        typer.Option(help="Optional path to precomputed cluster evidence JSON."),
    ] = None,
    embeddings_path: Annotated[
        Path,
        typer.Option(help="Path to outputs/intermediate/embeddings.jsonl."),
    ] = Path("outputs/intermediate/embeddings.jsonl"),
    movies_path: Annotated[
        Path,
        typer.Option(help="Path to data/processed/movies.json."),
    ] = Path("data/processed/movies.json"),
    profiles_path: Annotated[
        Path,
        typer.Option(help="Path to data/processed/profiles.json."),
    ] = Path("data/processed/profiles.json"),
    output_dir: Annotated[Path | None, typer.Option(help="Base outputs directory.")] = None,
    model: Annotated[str | None, typer.Option(help="OpenAI label model override.")] = None,
    batch_size: Annotated[int, typer.Option(help="Clusters per label prompt.")] = 5,
) -> None:
    """Generate draft human-reviewable microgenre labels for k-means clusters."""
    settings = load_settings(output_dir=output_dir)
    resolved_model = model or settings.openai_label_model or DEFAULT_LABEL_MODEL
    try:
        result = label_clusters_file(
            api_key=settings.require_openai_api_key(),
            evidence_path=evidence_path,
            embeddings_path=embeddings_path,
            movies_path=movies_path,
            profiles_path=profiles_path,
            output_dir=settings.output_dir,
            model=resolved_model,
            method=method,
            k=k,
            batch_size=batch_size,
        )
    except (MissingCredentialsError, ClusterLabelError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    typer.echo(f"Wrote cluster label candidates JSON to {result.json_path}")
    typer.echo(f"Wrote cluster label candidates report to {result.report_path}")
    typer.echo(f"Wrote human-editable labels JSON to {result.editable_json_path}")
    typer.echo(f"Cached labels reused: {result.cached_reused_count}")
    typer.echo(f"New labels generated: {result.new_label_count}")


@app.command()
def render_label_review(
    candidates_path: Annotated[
        Path,
        typer.Option(help="Path to outputs/intermediate/cluster_label_candidates.json."),
    ] = Path("outputs/intermediate/cluster_label_candidates.json"),
    output_dir: Annotated[Path | None, typer.Option(help="Base outputs directory.")] = None,
) -> None:
    """Render the Milestone 3 human label-review worksheet."""
    settings = load_settings(output_dir=output_dir)
    try:
        path = render_label_review_file(
            candidates_path=candidates_path,
            output_dir=settings.output_dir,
        )
    except ClusterLabelError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    typer.echo(f"Wrote cluster label review to {path}")


@app.command()
def review_ablation(
    variants: Annotated[
        str,
        typer.Option(help="Comma-separated variants: no_reviews,light_reviews,medium_reviews."),
    ] = "no_reviews,light_reviews,medium_reviews",
    k: Annotated[int, typer.Option(help="k-means cluster count.")] = 35,
    limit: Annotated[int | None, typer.Option(help="Optional movie/profile limit.")] = 500,
    movies_path: Annotated[
        Path,
        typer.Option(help="Path to data/processed/movies.json."),
    ] = Path("data/processed/movies.json"),
    output_dir: Annotated[Path | None, typer.Option(help="Base outputs directory.")] = None,
    embedding_model: Annotated[
        str | None,
        typer.Option(help="OpenAI embedding model override."),
    ] = None,
    label_model: Annotated[str | None, typer.Option(help="OpenAI label model override.")] = None,
    embedding_batch_size: Annotated[
        int,
        typer.Option(help="OpenAI embedding request batch size."),
    ] = 64,
    label_batch_size: Annotated[int, typer.Option(help="Clusters per label prompt.")] = 5,
) -> None:
    """Run the Milestone 3.25 review-weight ablation."""
    settings = load_settings(output_dir=output_dir)
    resolved_embedding_model = (
        embedding_model or settings.openai_embedding_model or DEFAULT_EMBEDDING_MODEL
    )
    resolved_label_model = label_model or settings.openai_label_model or DEFAULT_LABEL_MODEL
    try:
        summary_path, report_path, summary = review_ablation_file(
            variants=parse_review_variants(variants),
            movies_path=movies_path,
            output_dir=settings.output_dir,
            global_embeddings_path=settings.output_dir / "intermediate" / "embeddings.jsonl",
            global_label_cache_path=settings.output_dir / "intermediate" / "cluster_label_cache.json",
            embedding_model=resolved_embedding_model,
            label_model=resolved_label_model,
            openai_api_key=settings.require_openai_api_key(),
            k=k,
            limit=limit,
            embedding_batch_size=embedding_batch_size,
            label_batch_size=label_batch_size,
        )
    except (MissingCredentialsError, ReviewAblationError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    typer.echo(f"Wrote review ablation summary to {summary_path}")
    typer.echo(f"Wrote review ablation report to {report_path}")
    typer.echo(f"Recommended variant: {summary.recommended_variant}")
    typer.echo(f"Estimated live cost: ${summary.total_estimated_cost_usd:.4f}")


@app.command()
def scale_dataset(
    target: Annotated[int, typer.Option(help="Target eligible movie count.")] = 2000,
    since_year: Annotated[int, typer.Option(help="Earliest release year to include.")] = 1980,
    min_votes: Annotated[int, typer.Option(help="Minimum TMDb vote count.")] = 100,
    data_dir: Annotated[Path | None, typer.Option(help="Base data directory.")] = None,
    output_dir: Annotated[Path | None, typer.Option(help="Base outputs directory.")] = None,
    candidate_limit: Annotated[
        int | None,
        typer.Option(help="Optional discover candidate count before filtering."),
    ] = None,
    refresh: Annotated[bool, typer.Option(help="Ignore cached TMDb responses.")] = False,
) -> None:
    """Fetch and profile a scaled English-language TMDb dataset."""
    settings = load_settings(data_dir=data_dir, output_dir=output_dir)
    try:
        with TMDbClient(
            settings.tmdb_bearer_token,
            cache_dir=settings.data_dir / "cache",
        ) as client:
            result = scale_dataset_file(
                client,
                target=target,
                since_year=since_year,
                min_votes=min_votes,
                data_dir=settings.data_dir,
                output_dir=settings.output_dir,
                candidate_limit=candidate_limit,
                refresh=refresh,
            )
    except (MissingCredentialsError, Milestone4Error) as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    typer.echo(f"Wrote scaled movies to {result.movies_path}")
    typer.echo(f"Wrote light-review profiles to {result.profiles_path}")
    typer.echo(f"Selected movies: {result.selected_count}")
    typer.echo(f"TMDb details fetched: {result.detail_count}")


@app.command()
def build_hierarchy(
    macro_k: Annotated[int, typer.Option(help="Macro region k-means cluster count.")] = 12,
    neighborhood_k: Annotated[
        int,
        typer.Option(help="Neighborhood k-means cluster count."),
    ] = 75,
    micro_k: Annotated[int, typer.Option(help="Microcluster k-means cluster count.")] = 200,
    movies_path: Annotated[
        Path | None,
        typer.Option(help="Path to data/processed/movies.json."),
    ] = None,
    profiles_path: Annotated[
        Path | None,
        typer.Option(help="Path to data/processed/profiles.json."),
    ] = None,
    embeddings_path: Annotated[
        Path,
        typer.Option(help="Path to outputs/intermediate/embeddings.jsonl."),
    ] = Path("outputs/intermediate/embeddings.jsonl"),
    output_dir: Annotated[Path | None, typer.Option(help="Base outputs directory.")] = None,
    label_model: Annotated[str | None, typer.Option(help="OpenAI label model override.")] = None,
    label_batch_size: Annotated[int, typer.Option(help="Clusters per label prompt.")] = 5,
) -> None:
    """Build Milestone 4 hierarchy layers, labels, neighbors, and projection."""
    settings = load_settings(output_dir=output_dir)
    resolved_movies_path = movies_path or settings.data_dir / "processed" / "movies.json"
    resolved_profiles_path = profiles_path or settings.data_dir / "processed" / "profiles.json"
    resolved_label_model = label_model or settings.openai_label_model or DEFAULT_LABEL_MODEL
    try:
        result = build_hierarchy_file(
            movies_path=resolved_movies_path,
            profiles_path=resolved_profiles_path,
            embeddings_path=embeddings_path,
            output_dir=settings.output_dir,
            macro_k=macro_k,
            neighborhood_k=neighborhood_k,
            micro_k=micro_k,
            label_model=resolved_label_model,
            openai_api_key=settings.require_openai_api_key(),
            label_batch_size=label_batch_size,
        )
    except (MissingCredentialsError, Milestone4Error) as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    typer.echo(f"Wrote hierarchy summary to {result.summary_path}")
    typer.echo(f"Projection method: {result.projection_method}")
    typer.echo(f"Labels generated: {result.labels_generated_count}")
    typer.echo(f"Estimated label cost: ${result.labels_total_estimated_cost_usd:.4f}")


@app.command()
def export_atlas_data(
    movies_path: Annotated[
        Path | None,
        typer.Option(help="Path to data/processed/movies.json."),
    ] = None,
    hierarchy_dir: Annotated[
        Path | None,
        typer.Option(help="Path to outputs/intermediate/hierarchy."),
    ] = None,
    output_dir: Annotated[Path | None, typer.Option(help="Base outputs directory.")] = None,
) -> None:
    """Export sanitized static JSON for the future Astro frontend."""
    settings = load_settings(output_dir=output_dir)
    resolved_movies_path = movies_path or settings.data_dir / "processed" / "movies.json"
    resolved_hierarchy_dir = hierarchy_dir or settings.output_dir / "intermediate" / "hierarchy"
    try:
        result = export_atlas_data_file(
            movies_path=resolved_movies_path,
            hierarchy_dir=resolved_hierarchy_dir,
            output_dir=settings.output_dir,
        )
    except (FileNotFoundError, Milestone4Error) as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    typer.echo(f"Wrote public export manifest to {result.manifest_path}")
    typer.echo(f"Wrote public export files to {result.export_dir}")


@app.command()
def build_territory_layouts(
    export_dir: Annotated[
        Path,
        typer.Option(help="Path to the public export directory."),
    ] = Path("outputs/public_export"),
) -> None:
    """Build experimental nested territory layouts for frontend review."""
    try:
        result = build_territory_layouts_file(export_dir=export_dir)
    except TerritoryLayoutError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    typer.echo(f"Wrote {result.variant_count} territory layout variants to {result.layout_path}")
    typer.echo(
        f"Layouts cover {result.movie_count:,} movies and {result.region_count:,} regions."
    )


@app.command()
def milestone_4(
    target: Annotated[int, typer.Option(help="Target eligible movie count.")] = 2000,
    since_year: Annotated[int, typer.Option(help="Earliest release year to include.")] = 1980,
    min_votes: Annotated[int, typer.Option(help="Minimum TMDb vote count.")] = 100,
    macro_k: Annotated[int, typer.Option(help="Macro region cluster count.")] = 12,
    neighborhood_k: Annotated[int, typer.Option(help="Neighborhood cluster count.")] = 75,
    micro_k: Annotated[int, typer.Option(help="Microcluster count.")] = 200,
    data_dir: Annotated[Path | None, typer.Option(help="Base data directory.")] = None,
    output_dir: Annotated[Path | None, typer.Option(help="Base outputs directory.")] = None,
    embedding_model: Annotated[
        str | None,
        typer.Option(help="OpenAI embedding model override."),
    ] = None,
    label_model: Annotated[str | None, typer.Option(help="OpenAI label model override.")] = None,
    embedding_batch_size: Annotated[
        int,
        typer.Option(help="OpenAI embedding request batch size."),
    ] = 64,
    label_batch_size: Annotated[int, typer.Option(help="Clusters per label prompt.")] = 5,
    refresh: Annotated[bool, typer.Option(help="Ignore cached TMDb responses.")] = False,
) -> None:
    """Run the full Milestone 4 scaled atlas pipeline."""
    settings = load_settings(data_dir=data_dir, output_dir=output_dir)
    resolved_embedding_model = (
        embedding_model or settings.openai_embedding_model or DEFAULT_EMBEDDING_MODEL
    )
    resolved_label_model = label_model or settings.openai_label_model or DEFAULT_LABEL_MODEL
    try:
        with TMDbClient(
            settings.tmdb_bearer_token,
            cache_dir=settings.data_dir / "cache",
        ) as client:
            result = milestone_4_file(
                client,
                target=target,
                since_year=since_year,
                min_votes=min_votes,
                data_dir=settings.data_dir,
                output_dir=settings.output_dir,
                embedding_model=resolved_embedding_model,
                label_model=resolved_label_model,
                openai_api_key=settings.require_openai_api_key(),
                macro_k=macro_k,
                neighborhood_k=neighborhood_k,
                micro_k=micro_k,
                embedding_batch_size=embedding_batch_size,
                label_batch_size=label_batch_size,
                refresh=refresh,
            )
    except (MissingCredentialsError, Milestone4Error) as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    typer.echo(f"Wrote Milestone 4 report to {result.report_path}")
    typer.echo(f"Wrote public export to {result.export.export_dir}")
    typer.echo(f"Movies exported: {result.scale.selected_count}")


@app.command()
def classification_v2(
    movies_path: Annotated[
        Path | None,
        typer.Option(help="Path to data/processed/movies.json."),
    ] = None,
    raw_details_path: Annotated[
        Path | None,
        typer.Option(help="Path to data/raw/movie_details.json for richer TMDb-only text."),
    ] = None,
    output_dir: Annotated[Path | None, typer.Option(help="Base outputs directory.")] = None,
    variants: Annotated[
        str | None,
        typer.Option(help="Comma-separated profile variants. Defaults to all Milestone 5 variants."),
    ] = None,
    strategies: Annotated[
        str | None,
        typer.Option(help="Comma-separated clustering strategies. Defaults to all Milestone 5 strategies."),
    ] = None,
    limit: Annotated[int | None, typer.Option(help="Optional movie/profile limit.")] = None,
    macro_k: Annotated[int, typer.Option(help="Macro cluster count.")] = 12,
    neighborhood_k: Annotated[int, typer.Option(help="Neighborhood cluster count.")] = 75,
    micro_k: Annotated[int, typer.Option(help="Microcluster count.")] = 200,
    embedding_batch_size: Annotated[int, typer.Option(help="Embedding API batch size.")] = 64,
    label_batch_size: Annotated[int, typer.Option(help="Cluster label API batch size.")] = 5,
    cost_gate_usd: Annotated[float, typer.Option(help="Maximum estimated OpenAI spend.")] = 10.0,
) -> None:
    """Run Milestone 5 classification/profile experiments and export the selected atlas."""
    settings = load_settings(output_dir=output_dir)
    resolved_movies_path = movies_path or settings.data_dir / "processed" / "movies.json"
    resolved_raw_details_path = raw_details_path or settings.data_dir / "raw" / "movie_details.json"
    try:
        result = classification_v2_file(
            api_key=settings.openai_api_key,
            movies_path=resolved_movies_path,
            raw_details_path=resolved_raw_details_path,
            output_dir=settings.output_dir,
            embedding_model=settings.openai_embedding_model,
            label_model=settings.openai_label_model,
            macro_k=macro_k,
            neighborhood_k=neighborhood_k,
            micro_k=micro_k,
            variants=_parse_optional_csv(variants),
            strategies=_parse_optional_csv(strategies),
            limit=limit,
            embedding_batch_size=embedding_batch_size,
            label_batch_size=label_batch_size,
            cost_gate_usd=cost_gate_usd,
        )
    except (ClassificationV2Error, MissingCredentialsError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(f"Wrote Classification V2 summary to {result.summary_path}")
    typer.echo(f"Wrote Milestone 5 report to {result.report_path}")
    typer.echo(f"Wrote audit report to {result.audit_path}")
    typer.echo(f"Wrote selected public export to {result.export_dir}")
    typer.echo(f"Winner: {result.winner_variant} + {result.winner_strategy}")
    typer.echo(f"Estimated OpenAI cost: ${result.estimated_openai_cost_usd:.4f}")
    typer.echo(f"Labels generated: {result.labels_generated}")


@app.command()
def large_audit(
    export_dir: Annotated[Path | None, typer.Option(help="Path to public export directory.")] = None,
    output_dir: Annotated[Path | None, typer.Option(help="Base outputs directory.")] = None,
    model: Annotated[str, typer.Option(help="OpenAI chat model for the audit.")] = "gpt-4.1-mini",
    review_count: Annotated[int, typer.Option(help="Number of movies to review.")] = 1000,
    batch_size: Annotated[int, typer.Option(help="Movies per audit request.")] = 25,
    workers: Annotated[int, typer.Option(help="Concurrent audit batch requests.")] = 1,
    seed: Annotated[int, typer.Option(help="Deterministic sample seed.")] = 42,
) -> None:
    """Run an LLM-assisted large QA pass over the public atlas export."""
    settings = load_settings(output_dir=output_dir)
    resolved_export_dir = export_dir or settings.output_dir / "public_export"
    try:
        result = large_audit_file(
            api_key=settings.openai_api_key,
            export_dir=resolved_export_dir,
            output_dir=settings.output_dir,
            model=model,
            review_count=review_count,
            batch_size=batch_size,
            workers=workers,
            seed=seed,
        )
    except (LargeAuditError, MissingCredentialsError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(f"Wrote large audit JSON to {result.json_path}")
    typer.echo(f"Wrote large audit report to {result.report_path}")
    typer.echo(f"Reviewed movies: {result.reviewed_count}")
    typer.echo(f"Verdicts: {result.verdict_counts}")
    typer.echo(f"Estimated audit cost: ${result.estimated_cost_usd:.4f}")


@app.command()
def milestone_2(
    limit: Annotated[int | None, typer.Option(help="Optional profile limit.")] = 100,
    profiles_path: Annotated[
        Path | None,
        typer.Option(help="Path to data/processed/profiles.json."),
    ] = None,
    movies_path: Annotated[
        Path | None,
        typer.Option(help="Path to data/processed/movies.json."),
    ] = None,
    output_dir: Annotated[Path | None, typer.Option(help="Base outputs directory.")] = None,
    model: Annotated[str | None, typer.Option(help="OpenAI embedding model override.")] = None,
    batch_size: Annotated[int, typer.Option(help="OpenAI embedding request batch size.")] = 64,
) -> None:
    """Run Milestone 2 embedding, projection, clustering, neighbors, evidence, and report."""
    settings = load_settings(output_dir=output_dir)
    resolved_profiles_path = profiles_path or settings.data_dir / "processed" / "profiles.json"
    resolved_movies_path = movies_path or settings.data_dir / "processed" / "movies.json"
    resolved_model = model or settings.openai_embedding_model or DEFAULT_EMBEDDING_MODEL
    estimate = estimate_profiles_file(
        profiles_path=resolved_profiles_path,
        model=resolved_model,
        limit=limit,
    )
    typer.echo(f"Estimated tokens: {estimate.estimated_tokens}")
    typer.echo(f"Estimated cost: ${estimate.estimated_cost_usd:.4f}")
    if estimate.estimated_cost_usd > 1:
        typer.echo(
            "Estimated live embedding cost exceeds $1. "
            "Pause here and rerun with a smaller --limit or explicit approval.",
            err=True,
        )
        raise typer.Exit(code=2)

    try:
        embed_result = embed_profiles_file(
            api_key=settings.require_openai_api_key(),
            profiles_path=resolved_profiles_path,
            output_dir=settings.output_dir,
            model=resolved_model,
            limit=limit,
            batch_size=batch_size,
        )
    except MissingCredentialsError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    embeddings_path = embed_result.embedding_path
    reduce_embeddings_file(embeddings_path=embeddings_path, output_dir=settings.output_dir)
    assignments_path = cluster_embeddings_file(
        embeddings_path=embeddings_path,
        output_dir=settings.output_dir,
    )
    neighbors_path = compute_neighbors_file(
        embeddings_path=embeddings_path,
        output_dir=settings.output_dir,
        top_n=10,
    )
    evidence_path = inspect_clusters_file(
        movies_path=resolved_movies_path,
        profiles_path=resolved_profiles_path,
        embeddings_path=embeddings_path,
        assignments_path=assignments_path,
        neighbors_path=neighbors_path,
        output_dir=settings.output_dir,
    )
    report_path = generate_milestone_2_report_file(
        profiles_path=resolved_profiles_path,
        embeddings_path=embeddings_path,
        manifest_path=embed_result.manifest_path,
        assignments_path=assignments_path,
        neighbors_path=neighbors_path,
        evidence_path=evidence_path,
        output_dir=settings.output_dir,
    )
    typer.echo(f"Milestone 2 complete. Report: {report_path}")


@app.command()
def quickstart(
    limit: Annotated[int, typer.Option(help="Maximum number of movies to fetch.")] = 100,
    min_votes: Annotated[int, typer.Option(help="Minimum TMDb vote count.")] = 500,
    data_dir: Annotated[Path | None, typer.Option(help="Base data directory.")] = None,
    output_dir: Annotated[Path | None, typer.Option(help="Base outputs directory.")] = None,
    release_date_gte: Annotated[
        str | None,
        typer.Option(help="Earliest primary release date, YYYY-MM-DD."),
    ] = None,
    release_date_lte: Annotated[
        str | None,
        typer.Option(help="Latest primary release date, YYYY-MM-DD."),
    ] = None,
    exclude_future: Annotated[
        bool,
        typer.Option(
            "--exclude-future/--include-future",
            help="Exclude future/unreleased films by capping release date at today.",
        ),
    ] = True,
    include_reviews: Annotated[
        bool,
        typer.Option(
            "--include-reviews/--no-include-reviews",
            help="Include conservative review-language snippets in profile text.",
        ),
    ] = True,
    max_review_chars: Annotated[
        int,
        typer.Option(help="Maximum total review-language characters per movie profile."),
    ] = 180,
    review_weight: Annotated[
        str,
        typer.Option(help="Review contribution weight: light, medium, or heavy."),
    ] = "light",
) -> None:
    """Run the full Milestone 1 pipeline after TMDB_BEARER_TOKEN is configured."""
    settings = load_settings(data_dir=data_dir, output_dir=output_dir)
    try:
        with TMDbClient(
            settings.tmdb_bearer_token,
            cache_dir=settings.data_dir / "cache",
        ) as client:
            discover_path = fetch_discover_step(
                client,
                limit=limit,
                min_votes=min_votes,
                output_dir=settings.data_dir,
                release_date_gte=release_date_gte,
                release_date_lte=release_date_lte,
                exclude_future=exclude_future,
            )
            details_path = fetch_details_step(
                client,
                discover_path=discover_path,
                output_dir=settings.data_dir,
            )
    except MissingCredentialsError as exc:
        _missing_token_exit(exc)

    movies_path = normalize_details_file(details_path=details_path, output_dir=settings.data_dir)
    profiles_path = build_profiles_file(
        movies_path=movies_path,
        output_dir=settings.data_dir,
        include_reviews=include_reviews,
        max_review_chars=max_review_chars,
        review_weight=_parse_review_weight(review_weight),
    )
    make_sample_map_file(profiles_path=profiles_path, output_dir=settings.output_dir)
    report_path = generate_report_file(data_dir=settings.data_dir, output_dir=settings.output_dir)
    typer.echo(f"Milestone 1 quickstart complete. Report: {report_path}")


def _missing_token_exit(exc: MissingCredentialsError) -> None:
    typer.secho(str(exc), fg=typer.colors.RED, err=True)
    typer.echo("After adding TMDB_BEARER_TOKEN to .env, run:", err=True)
    typer.echo("  uv run film-atlas quickstart --limit 100", err=True)
    raise typer.Exit(code=1)


def _parse_optional_csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


def _parse_review_weight(value: str) -> ReviewWeight:
    if value in {"light", "medium", "heavy"}:
        return cast(ReviewWeight, value)
    raise typer.BadParameter("review-weight must be one of: light, medium, heavy")
