from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .ipf_cell_state import (
    DEFAULT_ACCESSION,
    default_gse136831_paths,
    load_gse136831_gene_map,
    load_gse136831_metadata,
)


GROUP_COLS = [
    "Disease_Identity",
    "CellType_Category",
    "Manuscript_Identity",
    "Subclass_Cell_Identity",
]


def load_gse136831_barcodes(barcodes_path: Path) -> pd.DataFrame:
    with gzip.open(barcodes_path, "rt", errors="ignore") as fh:
        barcodes = [line.strip() for line in fh if line.strip()]
    return pd.DataFrame(
        {
            "CellBarcode_Identity": barcodes,
            "matrix_col_idx": np.arange(len(barcodes), dtype=np.int64),
        }
    )


def _prepare_group_reference(metadata: pd.DataFrame, barcodes: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    merged = barcodes.merge(metadata, on="CellBarcode_Identity", how="left", validate="1:1")
    if merged[GROUP_COLS].isna().any().any():
        missing = int(merged[GROUP_COLS].isna().any(axis=1).sum())
        raise ValueError(f"{missing} barcodes could not be matched to metadata groups")

    grouped = (
        merged.groupby(GROUP_COLS, dropna=False)
        .agg(
            cell_count=("CellBarcode_Identity", "size"),
            mean_nUMI=("nUMI", "mean"),
            mean_nGene=("nGene", "mean"),
            subject_count=("Subject_Identity", "nunique"),
            library_count=("Library_Identity", "nunique"),
        )
        .reset_index()
        .sort_values(["Disease_Identity", "Manuscript_Identity", "cell_count"], ascending=[True, True, False])
        .reset_index(drop=True)
    )
    grouped["group_id"] = np.arange(len(grouped), dtype=np.int64)

    merged = merged.merge(grouped[GROUP_COLS + ["group_id"]], on=GROUP_COLS, how="left", validate="many_to_one")
    group_ids = merged.sort_values("matrix_col_idx")["group_id"].to_numpy(dtype=np.int64)
    return grouped, group_ids


def _read_matrix_dimensions(matrix_path: Path) -> tuple[int, int, int]:
    with gzip.open(matrix_path, "rt", errors="ignore") as fh:
        header = fh.readline().strip()
        if not header.startswith("%%MatrixMarket"):
            raise ValueError(f"Unexpected MatrixMarket header in {matrix_path}: {header}")
        dims = fh.readline().strip().split()
        if len(dims) != 3:
            raise ValueError(f"Malformed dimensions line in {matrix_path}: {' '.join(dims)}")
        return int(dims[0]), int(dims[1]), int(dims[2])


def _aggregate_group_gene_counts(
    matrix_path: Path,
    n_genes: int,
    n_groups: int,
    cell_to_group: np.ndarray,
    chunksize: int = 2_000_000,
) -> np.ndarray:
    aggregated = np.zeros((n_groups, n_genes), dtype=np.float64)

    with gzip.open(matrix_path, "rt", errors="ignore") as fh:
        fh.readline()
        fh.readline()
        reader = pd.read_csv(
            fh,
            sep=" ",
            header=None,
            names=["gene_idx", "cell_idx", "count"],
            chunksize=chunksize,
            dtype={"gene_idx": np.int64, "cell_idx": np.int64, "count": np.float64},
        )
        for chunk in reader:
            gene_idx = chunk["gene_idx"].to_numpy(dtype=np.int64) - 1
            cell_idx = chunk["cell_idx"].to_numpy(dtype=np.int64) - 1
            counts = chunk["count"].to_numpy(dtype=np.float64)
            group_idx = cell_to_group[cell_idx]
            flat_idx = group_idx * n_genes + gene_idx
            chunk_counts = np.bincount(flat_idx, weights=counts, minlength=n_groups * n_genes)
            aggregated += chunk_counts.reshape(n_groups, n_genes)

    return aggregated


def build_gse136831_expression_reference(
    metadata_path: Path,
    gene_ids_path: Path,
    barcodes_path: Path,
    matrix_path: Path,
    output_parquet: Path,
    output_csv: Path | None = None,
    top_genes_csv: Path | None = None,
    summary_json: Path | None = None,
    top_gene_limit_per_manuscript: int = 15,
) -> dict[str, Any]:
    metadata = load_gse136831_metadata(metadata_path)
    genes = load_gse136831_gene_map(gene_ids_path)
    barcodes = load_gse136831_barcodes(barcodes_path)

    n_genes, n_cells, _ = _read_matrix_dimensions(matrix_path)
    if n_genes != len(genes):
        raise ValueError(f"Matrix genes ({n_genes}) do not match gene map rows ({len(genes)})")
    if n_cells != len(barcodes):
        raise ValueError(f"Matrix cells ({n_cells}) do not match barcode rows ({len(barcodes)})")

    grouped, cell_to_group = _prepare_group_reference(metadata, barcodes)
    counts_by_group = _aggregate_group_gene_counts(
        matrix_path=matrix_path,
        n_genes=n_genes,
        n_groups=len(grouped),
        cell_to_group=cell_to_group,
    )

    total_umis = counts_by_group.sum(axis=1)
    detected_gene_count = (counts_by_group > 0).sum(axis=1)

    reference = grouped.copy()
    reference["accession"] = DEFAULT_ACCESSION
    reference["source_type"] = "scRNA-seq"
    reference["total_umis"] = total_umis
    reference["detected_gene_count"] = detected_gene_count
    reference["mean_umis_from_matrix"] = np.where(reference["cell_count"] > 0, total_umis / reference["cell_count"], 0.0)
    reference = reference[
        [
            "accession",
            "source_type",
            "group_id",
            "Disease_Identity",
            "CellType_Category",
            "Manuscript_Identity",
            "Subclass_Cell_Identity",
            "cell_count",
            "mean_nUMI",
            "mean_nGene",
            "subject_count",
            "library_count",
            "total_umis",
            "mean_umis_from_matrix",
            "detected_gene_count",
        ]
    ].rename(
        columns={
            "Disease_Identity": "disease_identity",
            "CellType_Category": "celltype_category",
            "Manuscript_Identity": "manuscript_identity",
            "Subclass_Cell_Identity": "subclass_cell_identity",
            "mean_nUMI": "mean_numi",
            "mean_nGene": "mean_ngene",
        }
    )

    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    reference.to_parquet(output_parquet, index=False)
    if output_csv:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        reference.to_csv(output_csv, index=False)

    manuscript_rows: list[dict[str, Any]] = []
    gene_id_values = genes.iloc[:, 0].astype(str).to_numpy()
    gene_name_values = genes.iloc[:, 1].astype(str).to_numpy()

    for manuscript_identity in sorted(reference["manuscript_identity"].unique()):
        manuscript_groups = reference[reference["manuscript_identity"] == manuscript_identity]
        diseases = set(manuscript_groups["disease_identity"].tolist())
        if "IPF" not in diseases or "Control" not in diseases:
            continue

        ipf_group_ids = manuscript_groups.loc[manuscript_groups["disease_identity"] == "IPF", "group_id"].to_numpy(dtype=np.int64)
        control_group_ids = manuscript_groups.loc[manuscript_groups["disease_identity"] == "Control", "group_id"].to_numpy(dtype=np.int64)

        ipf_counts = counts_by_group[ipf_group_ids].sum(axis=0)
        control_counts = counts_by_group[control_group_ids].sum(axis=0)
        ipf_total = float(ipf_counts.sum())
        control_total = float(control_counts.sum())
        ipf_cpm = (ipf_counts * 1_000_000.0 / ipf_total) if ipf_total > 0 else ipf_counts
        control_cpm = (control_counts * 1_000_000.0 / control_total) if control_total > 0 else control_counts
        log2_fc = np.log2(ipf_cpm + 1.0) - np.log2(control_cpm + 1.0)
        order = np.argsort(np.abs(log2_fc))[::-1][: min(top_gene_limit_per_manuscript, len(log2_fc))]

        for rank, gene_idx in enumerate(order, start=1):
            manuscript_rows.append(
                {
                    "manuscript_identity": manuscript_identity,
                    "rank_within_manuscript": rank,
                    "gene_id": gene_id_values[gene_idx],
                    "gene_name": gene_name_values[gene_idx],
                    "ipf_cpm": float(ipf_cpm[gene_idx]),
                    "control_cpm": float(control_cpm[gene_idx]),
                    "log2_fc_ipf_vs_control": float(log2_fc[gene_idx]),
                    "ipf_group_count": int(len(ipf_group_ids)),
                    "control_group_count": int(len(control_group_ids)),
                }
            )

    top_genes = pd.DataFrame(manuscript_rows)
    if top_genes_csv:
        top_genes_csv.parent.mkdir(parents=True, exist_ok=True)
        top_genes.to_csv(top_genes_csv, index=False)

    summary: dict[str, Any] = {
        "accession": DEFAULT_ACCESSION,
        "total_cells": int(n_cells),
        "group_rows": int(len(reference)),
        "matrix_gene_rows": int(n_genes),
        "manuscripts_compared_ipf_vs_control": int(top_genes["manuscript_identity"].nunique()) if not top_genes.empty else 0,
        "top_gene_rows_exported": int(len(top_genes)),
        "disease_distribution": metadata["Disease_Identity"].value_counts().to_dict(),
        "output_parquet": str(output_parquet),
    }
    if output_csv:
        summary["output_csv"] = str(output_csv)
    if top_genes_csv:
        summary["top_genes_csv"] = str(top_genes_csv)

    if summary_json:
        summary_json.parent.mkdir(parents=True, exist_ok=True)
        summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")

    return summary


def default_gse136831_expression_paths(repo_root: Path) -> dict[str, Path]:
    paths = default_gse136831_paths(repo_root)
    return {
        "metadata": paths["metadata"],
        "gene_ids": paths["gene_ids"],
        "barcodes": paths["barcodes"],
        "raw_counts_matrix": paths["raw_counts_matrix"],
    }
