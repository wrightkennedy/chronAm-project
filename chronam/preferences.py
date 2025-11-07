"""
Application-wide preferences for ChronAm.

Preferences are stored outside of the git workspace so that user-specific
choices such as dataset paths or quit behaviors are not accidentally committed.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

PREFERENCES_VERSION = 1
PREFERENCES_FILENAME = "preferences.json"


def _default_config_dir() -> Path:
    """Return an OS-appropriate directory for storing preferences."""
    custom = os.environ.get("CHRONAM_CONFIG_HOME")
    if custom:
        return Path(custom).expanduser()

    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "ChronAm"
    if os.name == "nt":
        base = os.environ.get("APPDATA")
        if base:
            return Path(base) / "ChronAm"
        return Path.home() / "AppData" / "Roaming" / "ChronAm"
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else Path.home() / ".config"
    return base / "chronam"


def _normalize_path(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    try:
        return str(Path(value).expanduser())
    except Exception:
        return None


@dataclass
class AppPreferences:
    dataset_folder_override: Optional[str] = None
    last_dataset_folder: Optional[str] = None
    metadata_enabled: bool = True
    open_last_project: bool = False
    warn_on_quit: bool = True
    save_on_quit: bool = False
    offer_clear_on_quit: bool = False
    warn_data_folder: bool = True
    warn_data_folder_limit_gb: float = 5.0
    last_project_path: Optional[str] = None

    def resolved_dataset_folder(self) -> Optional[str]:
        return self.dataset_folder_override or self.last_dataset_folder

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["version"] = PREFERENCES_VERSION
        return data

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "AppPreferences":
        if not isinstance(data, dict):
            return cls()

        def coerce_bool(key: str, default: bool) -> bool:
            value = data.get(key, default)
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.lower() in {"1", "true", "yes", "on"}
            if isinstance(value, (int, float)):
                return bool(value)
            return default

        def coerce_float(key: str, default: float) -> float:
            value = data.get(key, default)
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        return cls(
            dataset_folder_override=_normalize_path(data.get("dataset_folder_override")),
            last_dataset_folder=_normalize_path(data.get("last_dataset_folder")),
            metadata_enabled=coerce_bool("metadata_enabled", True),
            open_last_project=coerce_bool("open_last_project", False),
            warn_on_quit=coerce_bool("warn_on_quit", True),
            save_on_quit=coerce_bool("save_on_quit", False),
            offer_clear_on_quit=coerce_bool("offer_clear_on_quit", False),
            warn_data_folder=coerce_bool("warn_data_folder", True),
            warn_data_folder_limit_gb=coerce_float("warn_data_folder_limit_gb", 5.0),
            last_project_path=_normalize_path(data.get("last_project_path")),
        )


class PreferenceStore:
    """Helper to manage reading/writing ChronAm preferences."""

    def __init__(self, path: Optional[Path] = None):
        self._path = Path(path) if path else _default_config_dir() / PREFERENCES_FILENAME
        self.preferences = AppPreferences()
        self.load()

    @property
    def path(self) -> Path:
        return self._path

    def load(self) -> None:
        try:
            with open(self._path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except FileNotFoundError:
            self.preferences = AppPreferences()
            return
        except (OSError, json.JSONDecodeError):
            self.preferences = AppPreferences()
            return
        self.preferences = AppPreferences.from_dict(data)

    def save(self) -> None:
        directory = self._path.parent
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError:
            return
        try:
            with open(self._path, "w", encoding="utf-8") as fh:
                json.dump(self.preferences.to_dict(), fh, ensure_ascii=False, indent=2)
        except OSError:
            return

    def reset(self) -> None:
        self.preferences = AppPreferences()
        self.save()

    def update_from_dict(self, payload: Dict[str, Any]) -> None:
        current = self.preferences.to_dict()
        current.update(payload)
        self.preferences = AppPreferences.from_dict(current)

    def export_for_project(self) -> Dict[str, Any]:
        return self.preferences.to_dict()

    def apply_project_preferences(self, payload: Optional[Dict[str, Any]]) -> None:
        if not isinstance(payload, dict):
            return
        merged = self.preferences.to_dict()
        merged.update(payload)
        self.preferences = AppPreferences.from_dict(merged)
