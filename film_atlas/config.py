"""Configuration and environment loading for the Film Atlas pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


class MissingCredentialsError(RuntimeError):
    """Raised when a live TMDb command is run without a bearer token."""


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime settings loaded from environment variables."""

    tmdb_bearer_token: str | None
    data_dir: Path
    output_dir: Path

    def require_tmdb_token(self) -> str:
        """Return the TMDb bearer token or raise a clear runtime error."""
        if self.tmdb_bearer_token:
            return self.tmdb_bearer_token
        raise MissingCredentialsError(
            "TMDB_BEARER_TOKEN is required for live TMDb fetches. "
            "Copy .env.example to .env, add the token, then run the fetch command again."
        )


def load_settings(
    *,
    env_file: str | Path = ".env",
    data_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> Settings:
    """Load project settings from .env and optional CLI overrides."""
    load_dotenv(env_file)

    env_data_dir = os.getenv("FILM_ATLAS_DATA_DIR", "data")
    env_output_dir = os.getenv("FILM_ATLAS_OUTPUT_DIR", "outputs")

    token = os.getenv("TMDB_BEARER_TOKEN") or None
    return Settings(
        tmdb_bearer_token=token,
        data_dir=Path(data_dir or env_data_dir),
        output_dir=Path(output_dir or env_output_dir),
    )
