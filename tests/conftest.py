from __future__ import annotations

from pathlib import Path
import shutil


def pytest_sessionstart(session: object) -> None:
    """Clear test_results before running tests."""
    results_dir = Path("test_results")
    if results_dir.exists():
        shutil.rmtree(results_dir)
