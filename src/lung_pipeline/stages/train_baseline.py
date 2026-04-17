from __future__ import annotations

from typing import Any

from ._common import build_stage_manifest, resolve_stage_inputs, resolve_stage_notes


def run(cfg: dict[str, Any], dry_run: bool = False) -> dict[str, Any]:
    default_notes = [
        "Start with simple baseline models before stacking more context.",
        "Use both GroupCV and random-split evaluation tracks.",
        "Group key defaults to canonical_drug_id from configs/lung.yaml.",
    ]
    return build_stage_manifest(
        cfg,
        "train_baseline",
        resolve_stage_inputs(cfg, "train_baseline", ["model_input_table"]),
        resolve_stage_notes(cfg, "train_baseline", default_notes),
        dry_run,
    )
