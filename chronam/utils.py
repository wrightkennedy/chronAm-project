"""Shared utility helpers for path-safe keyword handling and metadata files."""

import json
import os
import re
from datetime import datetime, date
from typing import Optional, Dict, Any


_INVALID_CHARS = re.compile(r"[<>:\\|?*\"\n\r\t]+")


def term_directory_name(term: Optional[str]) -> str:
    """Return a filesystem-safe directory name for the provided search term."""
    raw = (term or "").strip()
    if not raw:
        raw = "term"
    sanitized = re.sub(r"[\\/]+", "_", raw)
    sanitized = _INVALID_CHARS.sub("_", sanitized)
    sanitized = sanitized.strip(" .")
    if not sanitized:
        sanitized = re.sub(r"[^A-Za-z0-9._-]", "", re.sub(r"\s+", "_", term or "term"))
    return sanitized or "term"


def _json_default(value: Any):
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, tuple):
        return list(value)
    return str(value)


def write_metadata_file(
    project_dir: Optional[str],
    output_path: Optional[str],
    metadata: Optional[Dict[str, Any]],
    *,
    enabled: bool = True,
) -> Optional[str]:
    """Persist a JSON metadata companion file for the given output.

    Returns the metadata path when written, else None.
    """
    if not enabled or not project_dir or not output_path or not metadata:
        return None

    metadata_dir = os.path.join(project_dir, 'data', 'metadata')
    try:
        os.makedirs(metadata_dir, exist_ok=True)
    except OSError:
        return None

    base_name = os.path.basename(output_path)
    name_root, _ = os.path.splitext(base_name)
    meta_path = os.path.join(metadata_dir, f"{name_root}_metadata.json")

    payload = dict(metadata)
    payload.setdefault('output_file', os.path.abspath(output_path))
    payload.setdefault('metadata_version', 1)
    payload['metadata_created_at'] = datetime.utcnow().isoformat() + 'Z'

    try:
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, default=_json_default)
    except OSError:
        return None

    return meta_path
