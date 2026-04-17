from __future__ import annotations

from typing import Any

from ..registry import dataset_buckets
from ._common import build_stage_manifest, resolve_stage_inputs, resolve_stage_notes


def run(cfg: dict[str, Any], dry_run: bool = False) -> dict[str, Any]:
    buckets = dataset_buckets(cfg)
    default_inputs = (
        buckets.get("supervision", [])
        + buckets.get("cell_context", [])
        + buckets.get("drug_knowledge", [])
        + buckets.get("perturbation_pathway", [])
        + ["disease_context_outputs"]
    )
    default_notes = [
        "Final learning unit is cell line x drug.",
        "Separate outputs into sample X, drug X, pair X, and labels y.",
        "LINCS is marked as very important in the planning doc.",
    ]
    return build_stage_manifest(
        cfg,
        "build_model_inputs",
        resolve_stage_inputs(cfg, "build_model_inputs", default_inputs),
        resolve_stage_notes(cfg, "build_model_inputs", default_notes),
        dry_run,
    )
