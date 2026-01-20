"""Core utility functions for path management and project setup."""

from __future__ import annotations

import sys
from pathlib import Path


def get_repo_root() -> Path:
    """Get the repository root directory.

    This file lives in `backend/core/`, so the repo root is three parents up:
    repo/
      backend/
        core/
          utils.py

    Returns:
        Path to the repository root directory.
    """
    current_file = Path(__file__).resolve()
    return current_file.parent.parent.parent


def setup_project_path() -> None:
    """Add the repository root to sys.path if not already present.

    This allows running scripts with `python scripts/...` without packaging.
    """
    repo_root = get_repo_root()
    repo_root_str = str(repo_root)
    
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

