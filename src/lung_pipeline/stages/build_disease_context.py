from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import repo_root, resolve_repo_path
from ..ipf_cell_state import (
    build_gse136831_cell_state_reference,
    default_gse136831_paths,
)
from ..ipf_cell_state_expression import build_gse136831_expression_reference
from ..ipf_geo_metadata import (
    build_gse122960_sample_reference,
    default_gse122960_paths,
)
from ..ipf_geo_expression import build_gse122960_expression_reference
from ..ipf_pbmc_validation import (
    build_gse233844_pbmc_expression_reference,
    build_gse233844_pbmc_sample_reference,
    default_gse233844_paths,
)
from ..ipf_bulk_geo import (
    build_gse32537_bulk_reference,
    build_gse47460_bulk_sample_reference,
    default_gse32537_paths,
    default_gse47460_paths,
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
    barcodes_path = paths["barcodes"]
    raw_counts_matrix_path = paths["raw_counts_matrix"]
    gse122960_paths = default_gse122960_paths(root)
    series_matrix_path = gse122960_paths["series_matrix"]
    filelist_path = gse122960_paths["filelist"]
    raw_tar_path = gse122960_paths["raw_tar"]
    gse233844_paths = default_gse233844_paths(root)
    gse233844_series_matrix_path = gse233844_paths["series_matrix"]
    gse233844_filelist_path = gse233844_paths["filelist"]
    gse233844_raw_tar_path = gse233844_paths["raw_tar"]
    gse32537_paths = default_gse32537_paths(root)
    gse32537_series_matrix_path = gse32537_paths["series_matrix"]
    gse32537_family_soft_path = gse32537_paths["family_soft"]
    gse47460_paths = default_gse47460_paths(root)
    gse47460_family_soft_path = gse47460_paths["family_soft"]
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

    if metadata_path.exists() and gene_ids_path.exists() and barcodes_path.exists() and raw_counts_matrix_path.exists():
        gse136831_expression_summary = build_gse136831_expression_reference(
            metadata_path=metadata_path,
            gene_ids_path=gene_ids_path,
            barcodes_path=barcodes_path,
            matrix_path=raw_counts_matrix_path,
            output_parquet=root / "data/processed/disease_context/ipf_gse136831_expression_reference.parquet",
            output_csv=root / "docs/ipf/gse136831_expression_group_reference.csv",
            top_genes_csv=root / "docs/ipf/gse136831_expression_ipf_vs_control_top_genes.csv",
            summary_json=root / "docs/ipf/gse136831_expression_reference_summary.json",
        )
        artifacts["gse136831_expression_reference"] = {
            "source_accession": "GSE136831",
            "source_mode": "sparse_matrix_expression_reference",
            "expression_reference_parquet": str(
                root / "data/processed/disease_context/ipf_gse136831_expression_reference.parquet"
            ),
            "expression_group_reference_csv": str(
                root / "docs/ipf/gse136831_expression_group_reference.csv"
            ),
            "expression_top_genes_csv": str(
                root / "docs/ipf/gse136831_expression_ipf_vs_control_top_genes.csv"
            ),
            "expression_reference_summary": str(
                root / "docs/ipf/gse136831_expression_reference_summary.json"
            ),
            "group_rows": gse136831_expression_summary["group_rows"],
            "total_cells": gse136831_expression_summary["total_cells"],
            "manuscripts_compared_ipf_vs_control": gse136831_expression_summary[
                "manuscripts_compared_ipf_vs_control"
            ],
        }
    else:
        missing_inputs = []
        for candidate in [metadata_path, gene_ids_path, barcodes_path, raw_counts_matrix_path]:
            if not candidate.exists():
                missing_inputs.append(str(candidate))
        artifacts["gse136831_expression_reference"] = {
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

    if raw_tar_path.exists() and series_matrix_path.exists() and filelist_path.exists():
        expression_summary = build_gse122960_expression_reference(
            sample_reference_path=root / "docs/ipf/gse122960_sample_reference.csv",
            raw_tar_path=raw_tar_path,
            output_parquet=root
            / "data/processed/disease_context/ipf_gse122960_expression_reference.parquet",
            output_csv=root / "docs/ipf/gse122960_expression_sample_summary.csv",
            top_genes_csv=root / "docs/ipf/gse122960_expression_ipf_vs_control_top_genes.csv",
            summary_json=root / "docs/ipf/gse122960_expression_reference_summary.json",
        )
        artifacts["gse122960_expression_reference"] = {
            "source_accession": "GSE122960",
            "source_mode": "filtered_h5_expression_reference",
            "expression_reference_parquet": str(
                root / "data/processed/disease_context/ipf_gse122960_expression_reference.parquet"
            ),
            "expression_sample_summary_csv": str(
                root / "docs/ipf/gse122960_expression_sample_summary.csv"
            ),
            "expression_top_genes_csv": str(
                root / "docs/ipf/gse122960_expression_ipf_vs_control_top_genes.csv"
            ),
            "expression_reference_summary": str(
                root / "docs/ipf/gse122960_expression_reference_summary.json"
            ),
            "sample_count": expression_summary["sample_count"],
            "total_cells": expression_summary["total_cells"],
            "top_gene_count": expression_summary.get("top_gene_count", 0),
        }
    else:
        missing_inputs = []
        if not raw_tar_path.exists():
            missing_inputs.append(str(raw_tar_path))
        if not (root / "docs/ipf/gse122960_sample_reference.csv").exists():
            missing_inputs.append(str(root / "docs/ipf/gse122960_sample_reference.csv"))
        artifacts["gse122960_expression_reference"] = {
            "source_accession": "GSE122960",
            "missing_inputs": missing_inputs,
        }

    if gse233844_series_matrix_path.exists() and gse233844_filelist_path.exists():
        gse233844_sample_summary = build_gse233844_pbmc_sample_reference(
            series_matrix_path=gse233844_series_matrix_path,
            filelist_path=gse233844_filelist_path,
            output_parquet=root / "data/processed/disease_context/ipf_gse233844_pbmc_sample_reference.parquet",
            output_csv=root / "docs/ipf/gse233844_pbmc_sample_reference.csv",
            summary_json=root / "docs/ipf/gse233844_pbmc_sample_reference_summary.json",
        )
        artifacts["gse233844_pbmc_sample_reference"] = {
            "source_accession": "GSE233844",
            "source_mode": "pbmc_metadata_reference",
            "sample_reference_parquet": str(
                root / "data/processed/disease_context/ipf_gse233844_pbmc_sample_reference.parquet"
            ),
            "sample_reference_csv": str(root / "docs/ipf/gse233844_pbmc_sample_reference.csv"),
            "sample_reference_summary": str(
                root / "docs/ipf/gse233844_pbmc_sample_reference_summary.json"
            ),
            "sample_count": gse233844_sample_summary["sample_count"],
            "matrix_available_count": gse233844_sample_summary["matrix_available_count"],
        }
    else:
        missing_inputs = []
        if not gse233844_series_matrix_path.exists():
            missing_inputs.append(str(gse233844_series_matrix_path))
        if not gse233844_filelist_path.exists():
            missing_inputs.append(str(gse233844_filelist_path))
        artifacts["gse233844_pbmc_sample_reference"] = {
            "source_accession": "GSE233844",
            "missing_inputs": missing_inputs,
        }

    if gse233844_raw_tar_path.exists() and (root / "docs/ipf/gse233844_pbmc_sample_reference.csv").exists():
        gse233844_expression_summary = build_gse233844_pbmc_expression_reference(
            sample_reference_path=root / "docs/ipf/gse233844_pbmc_sample_reference.csv",
            raw_tar_path=gse233844_raw_tar_path,
            output_parquet=root / "data/processed/disease_context/ipf_gse233844_pbmc_expression_reference.parquet",
            output_csv=root / "docs/ipf/gse233844_pbmc_expression_sample_summary.csv",
            top_genes_csv=root / "docs/ipf/gse233844_pbmc_expression_top_genes.csv",
            summary_json=root / "docs/ipf/gse233844_pbmc_expression_reference_summary.json",
        )
        artifacts["gse233844_pbmc_expression_reference"] = {
            "source_accession": "GSE233844",
            "source_mode": "pbmc_expression_validation_reference",
            "expression_reference_parquet": str(
                root / "data/processed/disease_context/ipf_gse233844_pbmc_expression_reference.parquet"
            ),
            "expression_sample_summary_csv": str(
                root / "docs/ipf/gse233844_pbmc_expression_sample_summary.csv"
            ),
            "expression_top_genes_csv": str(
                root / "docs/ipf/gse233844_pbmc_expression_top_genes.csv"
            ),
            "expression_reference_summary": str(
                root / "docs/ipf/gse233844_pbmc_expression_reference_summary.json"
            ),
            "sample_count": gse233844_expression_summary["sample_count"],
            "total_cells": gse233844_expression_summary["total_cells"],
            "comparison_count": gse233844_expression_summary["comparison_count"],
        }
    else:
        missing_inputs = []
        if not gse233844_raw_tar_path.exists():
            missing_inputs.append(str(gse233844_raw_tar_path))
        if not (root / "docs/ipf/gse233844_pbmc_sample_reference.csv").exists():
            missing_inputs.append(str(root / "docs/ipf/gse233844_pbmc_sample_reference.csv"))
        artifacts["gse233844_pbmc_expression_reference"] = {
            "source_accession": "GSE233844",
            "missing_inputs": missing_inputs,
        }

    if gse32537_series_matrix_path.exists() and gse32537_family_soft_path.exists():
        gse32537_summary = build_gse32537_bulk_reference(
            series_matrix_path=gse32537_series_matrix_path,
            family_soft_path=gse32537_family_soft_path,
            sample_reference_parquet=root / "data/processed/disease_context/ipf_gse32537_bulk_sample_reference.parquet",
            sample_reference_csv=root / "docs/ipf/gse32537_bulk_sample_reference.csv",
            expression_summary_parquet=root / "data/processed/disease_context/ipf_gse32537_bulk_expression_reference.parquet",
            expression_summary_csv=root / "docs/ipf/gse32537_bulk_expression_reference.csv",
            top_genes_csv=root / "docs/ipf/gse32537_bulk_ipf_vs_control_top_genes.csv",
            summary_json=root / "docs/ipf/gse32537_bulk_reference_summary.json",
        )
        artifacts["gse32537_bulk_reference"] = {
            "source_accession": "GSE32537",
            "source_mode": "bulk_series_matrix_reference",
            "sample_reference_parquet": str(
                root / "data/processed/disease_context/ipf_gse32537_bulk_sample_reference.parquet"
            ),
            "sample_reference_csv": str(root / "docs/ipf/gse32537_bulk_sample_reference.csv"),
            "expression_reference_parquet": str(
                root / "data/processed/disease_context/ipf_gse32537_bulk_expression_reference.parquet"
            ),
            "expression_reference_csv": str(root / "docs/ipf/gse32537_bulk_expression_reference.csv"),
            "top_genes_csv": str(root / "docs/ipf/gse32537_bulk_ipf_vs_control_top_genes.csv"),
            "reference_summary": str(root / "docs/ipf/gse32537_bulk_reference_summary.json"),
            "sample_count": gse32537_summary["sample_count"],
            "ipf_sample_count": gse32537_summary["ipf_sample_count"],
            "control_sample_count": gse32537_summary["control_sample_count"],
        }
    else:
        missing_inputs = []
        if not gse32537_series_matrix_path.exists():
            missing_inputs.append(str(gse32537_series_matrix_path))
        if not gse32537_family_soft_path.exists():
            missing_inputs.append(str(gse32537_family_soft_path))
        artifacts["gse32537_bulk_reference"] = {
            "source_accession": "GSE32537",
            "missing_inputs": missing_inputs,
        }

    if gse47460_family_soft_path.exists():
        gse47460_summary = build_gse47460_bulk_sample_reference(
            family_soft_path=gse47460_family_soft_path,
            output_parquet=root / "data/processed/disease_context/ipf_gse47460_bulk_sample_reference.parquet",
            output_csv=root / "docs/ipf/gse47460_bulk_sample_reference.csv",
            summary_json=root / "docs/ipf/gse47460_bulk_sample_reference_summary.json",
        )
        artifacts["gse47460_bulk_sample_reference"] = {
            "source_accession": "GSE47460",
            "source_mode": "bulk_family_soft_metadata_reference",
            "sample_reference_parquet": str(
                root / "data/processed/disease_context/ipf_gse47460_bulk_sample_reference.parquet"
            ),
            "sample_reference_csv": str(root / "docs/ipf/gse47460_bulk_sample_reference.csv"),
            "sample_reference_summary": str(
                root / "docs/ipf/gse47460_bulk_sample_reference_summary.json"
            ),
            "sample_count": gse47460_summary["sample_count"],
            "disease_bucket_distribution": gse47460_summary["disease_bucket_distribution"],
        }
    else:
        artifacts["gse47460_bulk_sample_reference"] = {
            "source_accession": "GSE47460",
            "missing_inputs": [str(gse47460_family_soft_path)],
        }

    has_real_artifacts = any(
        "source_mode" in item for item in artifacts.values()
    )
    manifest["status"] = "partial_built" if has_real_artifacts else "partial_built_missing_inputs"
    manifest["artifacts"] = artifacts

    write_json(resolve_repo_path(cfg, "outputs/manifests/build_disease_context.json"), manifest)
    return manifest
