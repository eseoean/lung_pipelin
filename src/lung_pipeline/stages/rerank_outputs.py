from __future__ import annotations

from typing import Any

from ..registry import DATASET_BUCKETS
from ._common import build_stage_manifest


def run(cfg: dict[str, Any], dry_run: bool = False) -> dict[str, Any]:
    notes = [
        "Fuse model scores with safety and translational evidence.",
        "ClinicalTrials and ADMET live in the final interpretation layer.",
        "SIDER is reserved for external validation or plausibility checks.",
    ]
    return build_stage_manifest(
        cfg,
        "rerank_outputs",
        DATASET_BUCKETS["validation_filter"] + ["patient_scores"],
        notes,
        dry_run,
    )

