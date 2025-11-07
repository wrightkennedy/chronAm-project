"""
Utilities to prepare a ready-to-run ChronAm environment on first launch.

Creates a default project folder within the user's Documents directory,
copies the bundled sample parquet dataset, and returns the paths for the
caller to plug into preferences.
"""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .config import init_project

SAMPLE_PARQUET_FILENAME = "AmericanStories_1800.parquet"


def _runtime_root() -> Path:
    """Return the folder containing packaged data, handling PyInstaller."""
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path.cwd()))
    return Path(__file__).resolve().parents[1]


def bundled_sample_parquet() -> Optional[Path]:
    """Locate the bundled demo parquet file, if present."""
    candidate = _runtime_root() / "data" / "parquet" / SAMPLE_PARQUET_FILENAME
    return candidate if candidate.exists() else None


def _documents_base() -> Path:
    """Best-effort detection of a writable user Documents directory."""
    home = Path.home()
    candidates = [
        os.environ.get("CHRONAM_DOCUMENTS_HOME"),
        home / "Documents",
        home / "OneDrive" / "Documents",
        home,
    ]
    for path in candidates:
        if not path:
            continue
        p = Path(path).expanduser()
        if p.exists() and os.access(p, os.W_OK):
            return p
    return home


@dataclass
class DefaultEnvironment:
    project_dir: Path
    dataset_dir: Path
    sample_path: Optional[Path]
    created: bool = False


def ensure_default_environment(project_name: str = "ChronAm") -> Optional[DefaultEnvironment]:
    """
    Ensure a default ChronAm project folder and sample dataset exist.

    Returns DefaultEnvironment describing the resulting paths, or None if the
    bundled sample parquet is unavailable.
    """
    sample_source = bundled_sample_parquet()
    if not sample_source:
        return None

    project_dir = _documents_base() / project_name
    init_project(str(project_dir))

    dataset_dir = project_dir / "data" / "parquet"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    dest = dataset_dir / SAMPLE_PARQUET_FILENAME
    created = False
    try:
        if not dest.exists() or dest.stat().st_size != sample_source.stat().st_size:
            shutil.copy2(sample_source, dest)
            created = True
    except OSError:
        # On failure we still return the environment so the UI can fall back.
        pass

    return DefaultEnvironment(
        project_dir=project_dir,
        dataset_dir=dataset_dir,
        sample_path=dest if dest.exists() else None,
        created=created,
    )
