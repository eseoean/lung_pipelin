from __future__ import annotations

from typing import Any

from ..registry import dataset_buckets
from ._common import build_stage_manifest, resolve_stage_inputs, resolve_stage_notes


def run(cfg: dict[str, Any], dry_run: bool = False) -> dict[str, Any]:
    buckets = dataset_buckets(cfg)
    default_notes = [
        "Fuse model scores with safety and translational evidence.",
        "ClinicalTrials and ADMET live in the final interpretation layer.",
        "SIDER is reserved for external validation or plausibility checks.",
    ]
    return build_stage_manifest(
        cfg,
        "rerank_outputs",
        resolve_stage_inputs(
            cfg,
            "rerank_outputs",
            buckets.get("validation_filter", []) + ["patient_scores"],
        ),
        resolve_stage_notes(cfg, "rerank_outputs", default_notes),
        dry_run,
    )
