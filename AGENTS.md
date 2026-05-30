# The Film Atlas Agent Notes

This repository is for the standalone Film Atlas data pipeline. Do not modify
David's personal website repo from here.

Milestone 1 boundaries:

- Use only TMDb's official API for live data.
- Do not use OpenAI APIs.
- Do not scrape Letterboxd, IMDb, or any website.
- Keep raw TMDb responses and processed local data out of git.
- Build semantic profile text from title, overview, genres, keywords, and short
  review-language snippets only.
- Do not include year, decade, country, language, cast, director, or production
  company fields in semantic profile text.

Verification commands:

```bash
uv run pytest
uv run ruff check .
```
