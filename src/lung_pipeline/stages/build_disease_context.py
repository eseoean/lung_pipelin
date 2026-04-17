from __future__ import annotations

from typing import Any

from ..registry import DATASET_BUCKETS
from ._common import build_stage_manifest


def run(cfg: dict[str, Any], dry_run: bool = False) -> dict[str, Any]:
    inputs = DATASET_BUCKETS["disease_context"] + ["msigdb"]
    notes = [
        "Build LUAD/LUSC disease signatures first.",
        "Use GTEx as the normal-lung baseline.",
        "Treat CPTAC as validation and pathway support before direct training features.",
    ]
    return build_stage_manifest(cfg, "build_disease_context", inputs, notes, dry_run)

