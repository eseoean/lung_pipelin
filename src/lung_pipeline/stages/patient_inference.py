from __future__ import annotations

from typing import Any

from ._common import build_stage_manifest, resolve_stage_inputs, resolve_stage_notes


def run(cfg: dict[str, Any], dry_run: bool = False) -> dict[str, Any]:
    default_notes = [
        "Patient inference consumes disease-context outputs plus trained models.",
        "TCGA patient-side features are the default inference input family.",
    ]
    return build_stage_manifest(
        cfg,
        "patient_inference",
        resolve_stage_inputs(cfg, "patient_inference", ["patient_features", "trained_models"]),
        resolve_stage_notes(cfg, "patient_inference", default_notes),
        dry_run,
    )
