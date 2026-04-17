from __future__ import annotations

from typing import Any

from ..registry import dataset_buckets
from ._common import build_stage_manifest, resolve_stage_inputs, resolve_stage_notes


def run(cfg: dict[str, Any], dry_run: bool = False) -> dict[str, Any]:
    buckets = dataset_buckets(cfg)
    default_inputs = buckets.get("disease_context", []) + ["msigdb"]
    default_notes = [
        "Build LUAD/LUSC disease signatures first.",
        "Use GTEx as the normal-lung baseline.",
        "Treat CPTAC as validation and pathway support before direct training features.",
    ]
    return build_stage_manifest(
        cfg,
        "build_disease_context",
        resolve_stage_inputs(cfg, "build_disease_context", default_inputs),
        resolve_stage_notes(cfg, "build_disease_context", default_notes),
        dry_run,
    )
