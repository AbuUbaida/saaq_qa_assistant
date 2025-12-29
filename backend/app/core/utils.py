"""Core utility functions for path management and project setup."""

from __future__ import annotations

import sys
from pathlib import Path


def get_repo_root() -> Path:
    """Get the repository root directory.
    
    Assumes this file is in backend/app/core/, so goes up 3 levels.
    
    Returns:
        Path to the repository root directory.
    """
    # This file is in backend/app/core/utils.py
    # So repo root is 3 levels up
    current_file = Path(__file__).resolve()
    repo_root = current_file.parent.parent.parent.parent
    return repo_root


def setup_project_path() -> None:
    """Add the repository root to sys.path if not already present.
    
    This allows imports from backend.app modules to work correctly.
    """
    repo_root = get_repo_root()
    repo_root_str = str(repo_root)
    
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

