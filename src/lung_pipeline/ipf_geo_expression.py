from __future__ import annotations

import json
import tarfile
import tempfile
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

from .ipf_geo_metadata import DEFAULT_GSE122960, default_gse122960_paths


def _decode_array(values: Any) -> list[str]:
    decoded: list[str] = []
    for item in values:
        if isinstance(item, bytes):
            decoded.append(item.decode("utf-8"))
        else:
            decoded.append(str(item))
    return decoded


def _load_reference_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _read_10x_filtered_h5_summary(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as h5:
        genome_key = next(iter(h5.keys()))
        group = h5[genome_key]

        data = np.asarray(group["data"])
        indices = np.asarray(group["indices"])
        indptr = np.asarray(group["indptr"])
        shape = tuple(int(x) for x in np.asarray(group["shape"]))
        gene_ids = _decode_array(group["genes"])
        gene_names = _decode_array(group["gene_names"])

        n_genes, n_cells = shape
        counts_per_gene = np.bincount(indices, weights=data, minlength=n_genes).astype(np.float64)
        genes_per_cell = np.diff(indptr).astype(np.int64)
        counts_per_cell = np.array(
            [float(data[indptr[i] : indptr[i + 1]].sum()) for i in range(n_cells)],
            dtype=np.float64,
        )

    return {
        "gene_ids": gene_ids,
        "gene_names": gene_names,
        "counts_per_gene": counts_per_gene,
        "n_genes": n_genes,
        "n_cells": n_cells,
        "nnz": int(len(data)),
        "total_umis": float(counts_per_cell.sum()),
        "mean_umis_per_cell": float(counts_per_cell.mean()) if n_cells else 0.0,
        "median_umis_per_cell": float(np.median(counts_per_cell)) if n_cells else 0.0,
        "mean_genes_per_cell": float(genes_per_cell.mean()) if n_cells else 0.0,
        "median_genes_per_cell": float(np.median(genes_per_cell)) if n_cells else 0.0,
        "detected_genes": int((counts_per_gene > 0).sum()),
    }


def build_gse122960_expression_reference(
    sample_reference_path: Path,
    raw_tar_path: Path,
    output_parquet: Path,
    output_csv: Path | None = None,
    top_genes_csv: Path | None = None,
    summary_json: Path | None = None,
    top_gene_limit: int = 200,
) -> dict[str, Any]:
    sample_reference = _load_reference_table(sample_reference_path).copy()
    if "filtered_h5_name" not in sample_reference.columns:
        raise KeyError(f"'filtered_h5_name' column is required in {sample_reference_path}")

    sample_reference["filtered_h5_name"] = sample_reference["filtered_h5_name"].astype(str)
    sample_lookup = {
        row["filtered_h5_name"]: row.to_dict()
        for _, row in sample_reference.iterrows()
        if row["filtered_h5_name"]
    }

    sample_rows: list[dict[str, Any]] = []
    sample_order: list[str] = []
    count_vectors: list[np.ndarray] = []
    gene_ids: list[str] | None = None
    gene_names: list[str] | None = None

    with tarfile.open(raw_tar_path, "r") as tf:
        filtered_members = [
            member
            for member in tf.getmembers()
            if member.isfile() and member.name.endswith("_filtered_gene_bc_matrices_h5.h5")
        ]

        for member in filtered_members:
            member_name = Path(member.name).name
            if member_name not in sample_lookup:
                continue

            with tempfile.TemporaryDirectory() as tmpdir:
                tf.extract(member, path=tmpdir, filter="data")
                extracted = Path(tmpdir) / member.name
                summary = _read_10x_filtered_h5_summary(extracted)

            if gene_ids is None:
                gene_ids = summary["gene_ids"]
                gene_names = summary["gene_names"]
            else:
                if gene_ids != summary["gene_ids"] or gene_names != summary["gene_names"]:
                    raise ValueError("Gene ordering mismatch across GSE122960 filtered H5 files")

            sample_meta = sample_lookup[member_name]
            sample_rows.append(
                {
                    "accession": sample_meta.get("accession", DEFAULT_GSE122960),
                    "gsm_accession": sample_meta.get("gsm_accession", member_name.split("_", 1)[0]),
                    "sample_title": sample_meta.get("sample_title", member_name),
                    "disease_condition": sample_meta.get("disease_condition", ""),
                    "disease_bucket": sample_meta.get("disease_bucket", "Other"),
                    "filtered_h5_name": member_name,
                    "cell_count": summary["n_cells"],
                    "detected_gene_count": summary["detected_genes"],
                    "nnz": summary["nnz"],
                    "total_umis": summary["total_umis"],
                    "mean_umis_per_cell": summary["mean_umis_per_cell"],
                    "median_umis_per_cell": summary["median_umis_per_cell"],
                    "mean_genes_per_cell": summary["mean_genes_per_cell"],
                    "median_genes_per_cell": summary["median_genes_per_cell"],
                }
            )
            sample_order.append(sample_meta.get("gsm_accession", member_name.split("_", 1)[0]))
            count_vectors.append(summary["counts_per_gene"])

    if not sample_rows or gene_ids is None or gene_names is None:
        raise ValueError(f"No filtered H5 samples from {raw_tar_path} matched {sample_reference_path}")

    sample_summary = pd.DataFrame(sample_rows).sort_values(["disease_bucket", "gsm_accession"])
    counts_matrix = np.column_stack(count_vectors)

    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    sample_summary.to_parquet(output_parquet, index=False)

    if output_csv:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        sample_summary.to_csv(output_csv, index=False)

    top_gene_rows: list[dict[str, Any]] = []
    bucket_names = list(dict.fromkeys(sample_summary["disease_bucket"].tolist()))
    sample_index_lookup = {sample_id: idx for idx, sample_id in enumerate(sample_order)}
    bucket_indices = {
        bucket: [
            sample_index_lookup[gsm]
            for gsm in sample_summary.loc[sample_summary["disease_bucket"] == bucket, "gsm_accession"]
            if gsm in sample_index_lookup
        ]
        for bucket in bucket_names
    }

    bucket_totals: dict[str, float] = {}
    bucket_cpm: dict[str, np.ndarray] = {}
    for bucket, idxs in bucket_indices.items():
        bucket_counts = counts_matrix[:, idxs].sum(axis=1) if idxs else np.zeros(len(gene_ids))
        total = float(bucket_counts.sum())
        bucket_totals[bucket] = total
        bucket_cpm[bucket] = (bucket_counts * 1_000_000.0 / total) if total > 0 else bucket_counts

    if "IPF" in bucket_cpm and "Control" in bucket_cpm:
        ipf_counts = counts_matrix[:, bucket_indices["IPF"]]
        control_counts = counts_matrix[:, bucket_indices["Control"]]
        log2_fc = np.log2(bucket_cpm["IPF"] + 1.0) - np.log2(bucket_cpm["Control"] + 1.0)
        order = np.argsort(np.abs(log2_fc))[::-1][: min(top_gene_limit, len(log2_fc))]
        for rank, gene_idx in enumerate(order, start=1):
            top_gene_rows.append(
                {
                    "rank": rank,
                    "gene_id": gene_ids[gene_idx],
                    "gene_name": gene_names[gene_idx],
                    "ipf_cpm": float(bucket_cpm["IPF"][gene_idx]),
                    "control_cpm": float(bucket_cpm["Control"][gene_idx]),
                    "log2_fc_ipf_vs_control": float(log2_fc[gene_idx]),
                    "ipf_detected_samples": int((ipf_counts[gene_idx] > 0).sum()),
                    "control_detected_samples": int((control_counts[gene_idx] > 0).sum()),
                }
            )

    top_genes_frame = pd.DataFrame(top_gene_rows)
    if top_genes_csv:
        top_genes_csv.parent.mkdir(parents=True, exist_ok=True)
        top_genes_frame.to_csv(top_genes_csv, index=False)

    summary = {
        "accession": DEFAULT_GSE122960,
        "sample_count": int(len(sample_summary)),
        "matched_filtered_h5_count": int(len(sample_rows)),
        "total_cells": int(sample_summary["cell_count"].sum()),
        "median_cells_per_sample": float(sample_summary["cell_count"].median()),
        "total_detected_genes_union": int(len(gene_ids)),
        "disease_bucket_distribution": sample_summary["disease_bucket"].value_counts().to_dict(),
        "bucket_total_umis": {bucket: round(total, 2) for bucket, total in bucket_totals.items()},
        "output_parquet": str(output_parquet),
    }
    if output_csv:
        summary["output_csv"] = str(output_csv)
    if top_genes_csv:
        summary["top_genes_csv"] = str(top_genes_csv)
        summary["top_gene_count"] = int(len(top_genes_frame))

    if summary_json:
        summary_json.parent.mkdir(parents=True, exist_ok=True)
        summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")

    return summary


def default_gse122960_expression_paths(repo_root: Path) -> dict[str, Path]:
    paths = default_gse122960_paths(repo_root)
    return {
        "sample_reference": repo_root / "docs/ipf/gse122960_sample_reference.csv",
        "raw_tar": paths["raw_tar"],
    }
