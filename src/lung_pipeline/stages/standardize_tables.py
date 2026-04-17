from __future__ import annotations

from typing import Any

from ..registry import DATASET_BUCKETS, PIPELINE_PRIORITIES
from ._common import build_stage_manifest


def run(cfg: dict[str, Any], dry_run: bool = False) -> dict[str, Any]:
    inputs = DATASET_BUCKETS["supervision"] + DATASET_BUCKETS["cell_context"] + DATASET_BUCKETS["drug_knowledge"]
    notes = [
        "Priority: build drug master table.",
        "Priority: build cell line master table.",
        "Expected joins: GDSC/PRISM to DrugBank/ChEMBL and GDSC to DepMap/COSMIC.",
        f"Top repo priorities: {', '.join(PIPELINE_PRIORITIES[:2])}.",
    ]
    return build_stage_manifest(cfg, "standardize_tables", inputs, notes, dry_run)

