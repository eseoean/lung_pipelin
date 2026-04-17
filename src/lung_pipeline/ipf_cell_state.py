from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_ACCESSION = "GSE136831"


def default_gse136831_paths(repo_root: Path) -> dict[str, Path]:
    base = repo_root / "data" / "raw" / "geo" / "ipf" / DEFAULT_ACCESSION
    supp = base / "supplementary"
    return {
        "metadata": supp / f"{DEFAULT_ACCESSION}_AllCells.Samples.CellType.MetadataTable.txt.gz",
        "gene_ids": supp / f"{DEFAULT_ACCESSION}_AllCells.GeneIDs.txt.gz",
        "barcodes": supp / f"{DEFAULT_ACCESSION}_AllCells.cellBarcodes.txt.gz",
    }


def _clean_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [str(col).strip().strip('"') for col in frame.columns]
    return frame


def load_gse136831_metadata(metadata_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(metadata_path, sep="\t", compression="gzip")
    frame = _clean_columns(frame)
    text_cols = [
        "CellBarcode_Identity",
        "CellType_Category",
        "Manuscript_Identity",
        "Subclass_Cell_Identity",
        "Disease_Identity",
        "Subject_Identity",
        "Library_Identity",
    ]
    for col in text_cols:
        if col in frame.columns:
            frame[col] = frame[col].astype(str).str.strip().str.strip('"')
    return frame


def load_gse136831_gene_map(gene_ids_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(gene_ids_path, sep="\t", compression="gzip")
    frame = _clean_columns(frame)
    return frame


def build_gse136831_cell_state_reference(
    metadata_path: Path,
    gene_ids_path: Path,
    output_parquet: Path,
    output_csv: Path | None = None,
    summary_json: Path | None = None,
) -> dict[str, Any]:
    metadata = load_gse136831_metadata(metadata_path)
    genes = load_gse136831_gene_map(gene_ids_path)

    grouped = (
        metadata.groupby(
            [
                "Disease_Identity",
                "CellType_Category",
                "Manuscript_Identity",
                "Subclass_Cell_Identity",
            ],
            dropna=False,
        )
        .agg(
            cell_count=("CellBarcode_Identity", "size"),
            mean_nUMI=("nUMI", "mean"),
            mean_nGene=("nGene", "mean"),
            subject_count=("Subject_Identity", "nunique"),
            library_count=("Library_Identity", "nunique"),
        )
        .reset_index()
        .sort_values(["Disease_Identity", "cell_count"], ascending=[True, False])
    )

    grouped["accession"] = DEFAULT_ACCESSION
    grouped["source_type"] = "scRNA-seq"
    grouped = grouped[
        [
            "accession",
            "source_type",
            "Disease_Identity",
            "CellType_Category",
            "Manuscript_Identity",
            "Subclass_Cell_Identity",
            "cell_count",
            "mean_nUMI",
            "mean_nGene",
            "subject_count",
            "library_count",
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
    grouped.to_parquet(output_parquet, index=False)

    if output_csv:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        grouped.to_csv(output_csv, index=False)

    summary: dict[str, Any] = {
        "accession": DEFAULT_ACCESSION,
        "total_cells": int(len(metadata)),
        "cell_state_rows": int(len(grouped)),
        "gene_reference_rows": int(len(genes)),
        "disease_distribution": metadata["Disease_Identity"].value_counts().to_dict(),
        "celltype_distribution": metadata["CellType_Category"].value_counts().head(12).to_dict(),
        "top_subclass_distribution": metadata["Subclass_Cell_Identity"].value_counts().head(20).to_dict(),
        "output_parquet": str(output_parquet),
    }

    if output_csv:
        summary["output_csv"] = str(output_csv)

    if summary_json:
        summary_json.parent.mkdir(parents=True, exist_ok=True)
        summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")

    return summary
