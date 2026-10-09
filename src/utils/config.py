"""Configuration management with YAML loading and experiment overrides."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override dict into base dict."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(
    config_path: str = "config/default.yaml",
    experiment_path: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load configuration with optional experiment overrides.

    Args:
        config_path: Path to the default config YAML.
        experiment_path: Optional path to experiment-specific overrides.
        overrides: Optional dict of runtime overrides.

    Returns:
        Merged configuration dictionary.
    """
    project_root = _find_project_root()

    # Load base config
    full_config_path = project_root / config_path
    with open(full_config_path) as f:
        config = yaml.safe_load(f)

    # Apply experiment overrides
    if experiment_path:
        full_experiment_path = project_root / experiment_path
        with open(full_experiment_path) as f:
            experiment_config = yaml.safe_load(f)
        if experiment_config:
            config = _deep_merge(config, experiment_config)

    # Apply runtime overrides
    if overrides:
        config = _deep_merge(config, overrides)

    # Resolve paths relative to project root
    config = _resolve_paths(config, project_root)
    config["_project_root"] = str(project_root)

    return config


def _find_project_root() -> Path:
    """Find the project root by looking for pyproject.toml."""
    current = Path.cwd()
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return current


def _resolve_paths(config: dict, root: Path) -> dict:
    """Resolve directory paths relative to project root."""
    path_keys = {"raw_dir", "labeled_dir", "features_dir", "splits_dir"}
    if "data" in config:
        for key in path_keys:
            if key in config["data"]:
                config["data"][key] = str(root / config["data"][key])
    return config
