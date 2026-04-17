from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import repo_root, resolve_repo_path
from ..ipf_cell_state import (
    build_gse136831_cell_state_reference,
    default_gse136831_paths,
)
from ..io import write_json
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
    manifest = build_stage_manifest(
        cfg,
        "build_disease_context",
        resolve_stage_inputs(cfg, "build_disease_context", default_inputs),
        resolve_stage_notes(cfg, "build_disease_context", default_notes),
        dry_run,
    )

    is_ipf = str(cfg.get("study", {}).get("disease", "")).upper() == "IPF"
    if dry_run or not is_ipf:
        return manifest

    root = repo_root(cfg)
    paths = default_gse136831_paths(root)
    metadata_path = paths["metadata"]
    gene_ids_path = paths["gene_ids"]
    stage_outputs = [
        resolve_repo_path(cfg, item)
        for item in cfg["stage_contracts"]["build_disease_context"]["outputs"]
    ]
    target_parquet = next(
        (
            path
            for path in stage_outputs
            if Path(path).name == "ipf_cell_state_reference.parquet"
        ),
        resolve_repo_path(cfg, "data/processed/disease_context/ipf_cell_state_reference.parquet"),
    )

    if metadata_path.exists() and gene_ids_path.exists():
        summary = build_gse136831_cell_state_reference(
            metadata_path=metadata_path,
            gene_ids_path=gene_ids_path,
            output_parquet=target_parquet,
            output_csv=root / "docs/ipf/gse136831_cell_state_reference.csv",
            summary_json=root / "docs/ipf/gse136831_cell_state_reference_summary.json",
        )
        manifest["status"] = "partial_built"
        manifest["artifacts"] = {
            "source_accession": "GSE136831",
            "source_mode": "metadata_driven_reference",
            "cell_state_reference_parquet": str(target_parquet),
            "cell_state_reference_csv": str(root / "docs/ipf/gse136831_cell_state_reference.csv"),
            "cell_state_reference_summary": str(
                root / "docs/ipf/gse136831_cell_state_reference_summary.json"
            ),
            "total_cells": summary["total_cells"],
            "cell_state_rows": summary["cell_state_rows"],
        }
    else:
        missing_inputs = []
        if not metadata_path.exists():
            missing_inputs.append(str(metadata_path))
        if not gene_ids_path.exists():
            missing_inputs.append(str(gene_ids_path))
        manifest["status"] = "partial_built_missing_inputs"
        manifest["artifacts"] = {
            "source_accession": "GSE136831",
            "missing_inputs": missing_inputs,
        }

    write_json(resolve_repo_path(cfg, "outputs/manifests/build_disease_context.json"), manifest)
    return manifest
