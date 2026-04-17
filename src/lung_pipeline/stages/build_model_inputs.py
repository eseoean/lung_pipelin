from __future__ import annotations

from typing import Any

from ..registry import DATASET_BUCKETS
from ._common import build_stage_manifest


def run(cfg: dict[str, Any], dry_run: bool = False) -> dict[str, Any]:
    inputs = (
        DATASET_BUCKETS["supervision"]
        + DATASET_BUCKETS["cell_context"]
        + DATASET_BUCKETS["drug_knowledge"]
        + DATASET_BUCKETS["perturbation_pathway"]
        + ["disease_context_outputs"]
    )
    notes = [
        "Final learning unit is cell line x drug.",
        "Separate outputs into sample X, drug X, pair X, and labels y.",
        "LINCS is marked as very important in the planning doc.",
    ]
    return build_stage_manifest(cfg, "build_model_inputs", inputs, notes, dry_run)

