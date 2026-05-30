"""Command line interface for The Film Atlas Milestone 1 pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, cast

import typer

from film_atlas.config import MissingCredentialsError, load_settings
from film_atlas.cluster import cluster_embeddings_file
from film_atlas.cluster_sweep import ClusterSweepError, parse_ks, sweep_clusters_file
from film_atlas.embedding import (
    DEFAULT_EMBEDDING_MODEL,
    embed_profiles_file,
    estimate_profiles_file,
)
from film_atlas.fetch import fetch_balanced as fetch_balanced_step
from film_atlas.fetch import fetch_details as fetch_details_step
from film_atlas.fetch import fetch_discover as fetch_discover_step
from film_atlas.inspect_clusters import inspect_clusters_file
from film_atlas.milestone2_report import generate_milestone_2_report_file
from film_atlas.neighbors import compute_neighbors_file
from film_atlas.normalize import normalize_details_file
from film_atlas.profiles import ReviewWeight, build_profiles_file
from film_atlas.reduce import reduce_embeddings_file
from film_atlas.report import generate_report_file
from film_atlas.sample_map import make_sample_map_file
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


def _parse_review_weight(value: str) -> ReviewWeight:
    if value in {"light", "medium", "heavy"}:
        return cast(ReviewWeight, value)
    raise typer.BadParameter("review-weight must be one of: light, medium, heavy")
