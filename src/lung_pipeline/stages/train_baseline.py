from __future__ import annotations

from typing import Any

from ._common import build_stage_manifest


def run(cfg: dict[str, Any], dry_run: bool = False) -> dict[str, Any]:
    notes = [
        "Start with simple baseline models before stacking more context.",
        "Use both GroupCV and random-split evaluation tracks.",
        "Group key defaults to canonical_drug_id from configs/lung.yaml.",
    ]
    return build_stage_manifest(cfg, "train_baseline", ["model_input_table"], notes, dry_run)

