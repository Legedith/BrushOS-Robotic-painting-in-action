from __future__ import annotations

import sys
from pathlib import Path


def add_repo_root() -> None:
    """Ensure the repo root is on sys.path for local imports."""
    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
