"""
echoguard.utils.config
~~~~~~~~~~~~~~~~~~~~~~
Configuration loading utility.

Loads YAML config files and exposes them as plain dicts so that every
module can call ``load_config()`` without importing PyYAML directly.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

# Default config path — resolved relative to the project root
_DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


def load_config(config_path: Optional[str | Path] = None) -> Dict[str, Any]:
    """Load a YAML configuration file.

    Parameters
    ----------
    config_path:
        Path to the YAML file.  If ``None``, the project default config at
        ``echoguard/configs/default.yaml`` is loaded.

    Returns
    -------
    dict
        Parsed configuration as a nested dictionary.

    Raises
    ------
    FileNotFoundError
        If the specified config file does not exist.
    """
    path = Path(config_path) if config_path is not None else _DEFAULT_CONFIG

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    return config or {}


def get_section(config: Dict[str, Any], section: str) -> Dict[str, Any]:
    """Safely retrieve a named section from a config dict.

    Parameters
    ----------
    config:
        Full config dictionary returned by :func:`load_config`.
    section:
        Top-level key to retrieve.

    Returns
    -------
    dict
        The section dict, or an empty dict if the key is missing.
    """
    return config.get(section, {})
