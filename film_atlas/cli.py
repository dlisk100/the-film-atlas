"""Command line interface for The Film Atlas Milestone 1 pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, cast

import typer

from film_atlas.config import MissingCredentialsError, load_settings
from film_atlas.fetch import fetch_balanced as fetch_balanced_step
from film_atlas.fetch import fetch_details as fetch_details_step
from film_atlas.fetch import fetch_discover as fetch_discover_step
from film_atlas.normalize import normalize_details_file
from film_atlas.profiles import ReviewWeight, build_profiles_file
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
