"""Shared utility helpers for path-safe keyword handling."""

import re
from typing import Optional


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
