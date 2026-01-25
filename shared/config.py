"""YAML configuration loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        path: Path to the YAML file.

    Returns:
        Dictionary with configuration values.
    """
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def dump_yaml(path: Path, data: Dict[str, Any]) -> None:
    """Write data to a YAML file.

    Args:
        path: Path to write the YAML file.
        data: Dictionary to serialize.
    """
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle)
