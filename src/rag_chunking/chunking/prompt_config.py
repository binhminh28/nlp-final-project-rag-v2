"""Minimal project environment loading without adding a runtime dependency."""

from __future__ import annotations

import os
from pathlib import Path


def load_project_dotenv(start: Path | None = None) -> Path | None:
    """Load the nearest project `.env`, preserving existing process variables."""

    current = (start or Path.cwd()).resolve()
    if current.is_file():
        current = current.parent
    for directory in (current, *current.parents):
        candidate = directory / ".env"
        if candidate.is_file():
            _load_env_file(candidate)
            return candidate
        if (directory / "pyproject.toml").is_file() and directory != current:
            break
    return None


def _load_env_file(path: Path) -> None:
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise ValueError(f"Invalid .env assignment at {path}:{line_number}")
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or not key.replace("_", "a").isalnum() or key[0].isdigit():
            raise ValueError(f"Invalid .env key at {path}:{line_number}")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)
