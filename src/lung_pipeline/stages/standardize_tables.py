from __future__ import annotations

from typing import Any

from ..registry import dataset_buckets, pipeline_priorities
from ._common import build_stage_manifest, resolve_stage_inputs, resolve_stage_notes


def run(cfg: dict[str, Any], dry_run: bool = False) -> dict[str, Any]:
    buckets = dataset_buckets(cfg)
    default_inputs = (
        buckets.get("supervision", [])
        + buckets.get("cell_context", [])
        + buckets.get("drug_knowledge", [])
    )
    default_notes = [
        "Priority: build drug master table.",
        "Priority: build cell line master table.",
        "Expected joins: GDSC/PRISM to DrugBank/ChEMBL and GDSC to DepMap/COSMIC.",
        f"Top repo priorities: {', '.join(pipeline_priorities(cfg)[:2])}.",
    ]
    return build_stage_manifest(
        cfg,
        "standardize_tables",
        resolve_stage_inputs(cfg, "standardize_tables", default_inputs),
        resolve_stage_notes(cfg, "standardize_tables", default_notes),
        dry_run,
    )
