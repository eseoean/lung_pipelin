from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import repo_root, resolve_repo_path
from ..ipf_cell_state import (
    build_gse136831_cell_state_reference,
    default_gse136831_paths,
)
from ..ipf_geo_metadata import (
    build_gse122960_sample_reference,
    default_gse122960_paths,
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
    gse122960_paths = default_gse122960_paths(root)
    series_matrix_path = gse122960_paths["series_matrix"]
    filelist_path = gse122960_paths["filelist"]
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

    artifacts: dict[str, Any] = {}

    if metadata_path.exists() and gene_ids_path.exists():
        summary = build_gse136831_cell_state_reference(
            metadata_path=metadata_path,
            gene_ids_path=gene_ids_path,
            output_parquet=target_parquet,
            output_csv=root / "docs/ipf/gse136831_cell_state_reference.csv",
            summary_json=root / "docs/ipf/gse136831_cell_state_reference_summary.json",
        )
        artifacts["gse136831_cell_state_reference"] = {
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
        artifacts["gse136831_cell_state_reference"] = {
            "source_accession": "GSE136831",
            "missing_inputs": missing_inputs,
        }

    if series_matrix_path.exists() and filelist_path.exists():
        gse122960_summary = build_gse122960_sample_reference(
            series_matrix_path=series_matrix_path,
            filelist_path=filelist_path,
            output_parquet=root / "data/processed/disease_context/ipf_gse122960_sample_reference.parquet",
            output_csv=root / "docs/ipf/gse122960_sample_reference.csv",
            summary_json=root / "docs/ipf/gse122960_sample_reference_summary.json",
        )
        artifacts["gse122960_sample_reference"] = {
            "source_accession": "GSE122960",
            "source_mode": "sample_metadata_reference",
            "sample_reference_parquet": str(
                root / "data/processed/disease_context/ipf_gse122960_sample_reference.parquet"
            ),
            "sample_reference_csv": str(root / "docs/ipf/gse122960_sample_reference.csv"),
            "sample_reference_summary": str(
                root / "docs/ipf/gse122960_sample_reference_summary.json"
            ),
            "sample_count": gse122960_summary["sample_count"],
            "filtered_h5_available_count": gse122960_summary["filtered_h5_available_count"],
            "raw_h5_available_count": gse122960_summary["raw_h5_available_count"],
        }
    else:
        missing_inputs = []
        if not series_matrix_path.exists():
            missing_inputs.append(str(series_matrix_path))
        if not filelist_path.exists():
            missing_inputs.append(str(filelist_path))
        artifacts["gse122960_sample_reference"] = {
            "source_accession": "GSE122960",
            "missing_inputs": missing_inputs,
        }

    has_real_artifacts = any(
        "source_mode" in item for item in artifacts.values()
    )
    manifest["status"] = "partial_built" if has_real_artifacts else "partial_built_missing_inputs"
    manifest["artifacts"] = artifacts

    write_json(resolve_repo_path(cfg, "outputs/manifests/build_disease_context.json"), manifest)
    return manifest
