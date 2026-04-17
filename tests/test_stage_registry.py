from __future__ import annotations

from pathlib import Path

from lung_pipeline.cli import RUNNERS
from lung_pipeline.config import load_config
from lung_pipeline.registry import PIPELINE_STEPS


def test_all_pipeline_steps_have_runners() -> None:
    assert set(PIPELINE_STEPS) == set(RUNNERS)


def test_dry_run_emits_stage_manifest() -> None:
    cfg = load_config(Path("configs/lung.yaml"))
    result = RUNNERS["standardize_tables"](cfg, dry_run=True)
    assert result["stage"] == "standardize_tables"
    assert result["status"] == "dry_run"


def test_ipf_dry_run_uses_ipf_contract_outputs() -> None:
    cfg = load_config(Path("configs/ipf.yaml"))
    result = RUNNERS["build_model_inputs"](cfg, dry_run=True)
    assert result["stage"] == "build_model_inputs"
    assert any("pseudo_labels.parquet" in output for output in result["outputs"])
