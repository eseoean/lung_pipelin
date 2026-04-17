from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def read_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} must be a mapping.")
    return data


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _repo_root_from_config(config_path: Path) -> Path:
    if config_path.parent.name == "configs":
        return config_path.parent.parent
    return config_path.parent


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path).resolve()
    raw = read_yaml(path)
    parents = raw.pop("inherits", [])
    merged: dict[str, Any] = {}
    for parent in parents:
        parent_path = (path.parent / parent).resolve()
        merged = deep_merge(merged, load_config(parent_path))
    merged = deep_merge(merged, raw)
    merged["_meta"] = {
        "config_path": str(path),
        "repo_root": str(_repo_root_from_config(path)),
    }
    return merged


def repo_root(cfg: dict[str, Any]) -> Path:
    return Path(cfg["_meta"]["repo_root"]).resolve()


def resolve_repo_path(cfg: dict[str, Any], raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return repo_root(cfg) / path


def stage_output_dir(cfg: dict[str, Any], stage_name: str) -> Path:
    return resolve_repo_path(cfg, cfg["stages"][stage_name]["output_dir"])

