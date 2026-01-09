from __future__ import annotations

from pathlib import Path


def find_repo_root(start: Path | None = None) -> Path:
    """
    Find the repository root by walking upward until we see a marker file/folder.
    Markers include: pyproject.toml, .git, or a top-level 'data' directory.

    """
    here = (start or Path(__file__)).resolve()

    # If called from a file path, use its parent directory as the starting point.
    cur = here if here.is_dir() else here.parent

    markers = {"pyproject.toml", ".git", "data"}

    while True:
        if any((cur / m).exists() for m in markers):
            return cur

        if cur.parent == cur:
            raise RuntimeError(
                "Could not find repo root. Expected one of: pyproject.toml, .git, or data/ "
                "somewhere above this file."
            )
        cur = cur.parent


def repo_path(*parts: str) -> Path:
    """Convenience: repo_path('data','raw') -> <repo>/data/raw"""
    return find_repo_root() / Path(*parts)
