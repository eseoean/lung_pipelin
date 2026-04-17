from __future__ import annotations

from pathlib import Path

from lung_pipeline.config import deep_merge, load_config, repo_root


def test_deep_merge_keeps_nested_keys() -> None:
    merged = deep_merge({"a": {"x": 1, "y": 2}}, {"a": {"y": 3}, "b": 4})
    assert merged == {"a": {"x": 1, "y": 3}, "b": 4}


def test_load_config_resolves_repo_root() -> None:
    cfg = load_config(Path("configs/lung.yaml"))
    assert repo_root(cfg).name == "lung_pipelin"
    assert cfg["study"]["label_source"] == "GDSC"


def test_load_ipf_config_overrides_lung_defaults() -> None:
    cfg = load_config(Path("configs/ipf.yaml"))
    assert cfg["study"]["objective"] == "disease_signature_reversal"
    assert "geo_ipf_bulk" in cfg["dataset_buckets"]["disease_context"]
