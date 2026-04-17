from __future__ import annotations

import csv
import gzip
import json
import tarfile
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_ACCESSION = "GSE233844"


def default_gse233844_paths(repo_root: Path) -> dict[str, Path]:
    base = repo_root / "data" / "raw" / "geo" / "ipf" / DEFAULT_ACCESSION
    supp = base / "supplementary"
    return {
        "series_matrix": base / f"{DEFAULT_ACCESSION}_series_matrix.txt.gz",
        "filelist": supp / "filelist.txt",
        "raw_tar": supp / f"{DEFAULT_ACCESSION}_RAW.tar",
    }


def default_gse233844_expression_paths(repo_root: Path) -> dict[str, Path]:
    root = default_gse233844_paths(repo_root)
    return {
        "sample_reference": repo_root / "docs/ipf/gse233844_pbmc_sample_reference.csv",
        "raw_tar": root["raw_tar"],
    }


def _strip_quotes(value: str) -> str:
    return value.strip().strip('"')


def _normalize_sample_field(raw_field: str, seen: Counter[str]) -> str:
    field = raw_field.removeprefix("!Sample_").lower()
    field = field.replace("/", "_").replace(" ", "_")
    count = seen[field]
    seen[field] += 1
    return field if count == 0 else f"{field}_{count + 1}"


def parse_geo_series_matrix_sample_table(series_matrix_path: Path) -> pd.DataFrame:
    sample_fields: list[tuple[str, list[str]]] = []
    seen: Counter[str] = Counter()

    with gzip.open(series_matrix_path, "rt", errors="ignore") as fh:
        for line in fh:
            if not line.startswith("!Sample_"):
                continue
            parts = line.rstrip("\n").split("\t")
            field = _normalize_sample_field(parts[0], seen)
            values = [_strip_quotes(item) for item in parts[1:]]
            sample_fields.append((field, values))

    if not sample_fields:
        raise ValueError(f"No !Sample_ rows found in {series_matrix_path}")

    sample_count = max(len(values) for _, values in sample_fields)
    table: dict[str, list[str]] = {}
    for field, values in sample_fields:
        padded = values + [""] * (sample_count - len(values))
        table[field] = padded
    return pd.DataFrame(table)


def load_geo_filelist(filelist_path: Path) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    with filelist_path.open() as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            rows.append({str(k): str(v) for k, v in row.items()})
    return pd.DataFrame(rows)


def _extract_value(raw_value: str) -> str:
    if ":" in raw_value:
        return raw_value.split(":", 1)[1].strip()
    return raw_value.strip()


def _map_group_bucket(raw_value: str) -> str:
    value = raw_value.lower()
    if value == "progressive ipf":
        return "IPF-Progressive"
    if value == "stable ipf":
        return "IPF-Stable"
    if value == "control":
        return "Control"
    return "Other"


def _resolve_first_existing(frame: pd.DataFrame, candidates: list[str]) -> str:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    raise KeyError(f"None of {candidates} found in columns")


def build_gse233844_pbmc_sample_reference(
    series_matrix_path: Path,
    filelist_path: Path,
    output_parquet: Path,
    output_csv: Path | None = None,
    summary_json: Path | None = None,
) -> dict[str, Any]:
    samples = parse_geo_series_matrix_sample_table(series_matrix_path)
    filelist = load_geo_filelist(filelist_path)

    samples = samples.copy()
    samples["accession"] = DEFAULT_ACCESSION
    samples["group_assignment"] = samples["characteristics_ch1_2"].map(_extract_value)
    samples["group_bucket"] = samples["group_assignment"].map(_map_group_bucket)
    samples["age_years"] = pd.to_numeric(samples["characteristics_ch1_3"].map(_extract_value), errors="coerce")
    samples["sex"] = samples["characteristics_ch1_4"].map(_extract_value)
    samples["race_ethnicity"] = samples["characteristics_ch1_5"].map(_extract_value)
    samples["smoking_status"] = samples["characteristics_ch1_6"].map(_extract_value)
    samples["pack_years"] = pd.to_numeric(samples["characteristics_ch1_7"].map(_extract_value).replace({"NA": np.nan}), errors="coerce")
    samples["dlco_percent_predicted"] = pd.to_numeric(samples["characteristics_ch1_8"].map(_extract_value).replace({"NA": np.nan}), errors="coerce")
    samples["fvc_percent_predicted"] = pd.to_numeric(samples["characteristics_ch1_9"].map(_extract_value).replace({"NA": np.nan}), errors="coerce")
    samples["barcodes_name"] = samples["supplementary_file_1"].str.rsplit("/", n=1).str[-1]
    samples["features_name"] = samples["supplementary_file_2"].str.rsplit("/", n=1).str[-1]
    samples["matrix_name"] = samples["supplementary_file_3"].str.rsplit("/", n=1).str[-1]

    available_names = set(filelist["Name"].tolist()) if "Name" in filelist.columns else set()
    samples["barcodes_available"] = samples["barcodes_name"].isin(available_names)
    samples["features_available"] = samples["features_name"].isin(available_names)
    samples["matrix_available"] = samples["matrix_name"].isin(available_names)

    selected_cols = {
        "accession": _resolve_first_existing(samples, ["accession"]),
        "sample_geo_accession": _resolve_first_existing(samples, ["sample_geo_accession", "geo_accession"]),
        "title": _resolve_first_existing(samples, ["title"]),
        "source_name_ch1": _resolve_first_existing(samples, ["source_name_ch1"]),
        "group_assignment": _resolve_first_existing(samples, ["group_assignment"]),
        "group_bucket": _resolve_first_existing(samples, ["group_bucket"]),
        "age_years": _resolve_first_existing(samples, ["age_years"]),
        "sex": _resolve_first_existing(samples, ["sex"]),
        "race_ethnicity": _resolve_first_existing(samples, ["race_ethnicity"]),
        "smoking_status": _resolve_first_existing(samples, ["smoking_status"]),
        "pack_years": _resolve_first_existing(samples, ["pack_years"]),
        "dlco_percent_predicted": _resolve_first_existing(samples, ["dlco_percent_predicted"]),
        "fvc_percent_predicted": _resolve_first_existing(samples, ["fvc_percent_predicted"]),
        "barcodes_name": _resolve_first_existing(samples, ["barcodes_name"]),
        "features_name": _resolve_first_existing(samples, ["features_name"]),
        "matrix_name": _resolve_first_existing(samples, ["matrix_name"]),
        "barcodes_available": _resolve_first_existing(samples, ["barcodes_available"]),
        "features_available": _resolve_first_existing(samples, ["features_available"]),
        "matrix_available": _resolve_first_existing(samples, ["matrix_available"]),
    }

    reference = samples[list(selected_cols.values())].rename(
        columns={
            selected_cols["sample_geo_accession"]: "gsm_accession",
            selected_cols["title"]: "sample_title",
            selected_cols["source_name_ch1"]: "source_name",
        }
    )

    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    reference.to_parquet(output_parquet, index=False)
    if output_csv:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        reference.to_csv(output_csv, index=False)

    summary = {
        "accession": DEFAULT_ACCESSION,
        "sample_count": int(len(reference)),
        "group_distribution": reference["group_assignment"].value_counts().to_dict(),
        "group_bucket_distribution": reference["group_bucket"].value_counts().to_dict(),
        "matrix_available_count": int(reference["matrix_available"].sum()),
        "barcodes_available_count": int(reference["barcodes_available"].sum()),
        "features_available_count": int(reference["features_available"].sum()),
        "output_parquet": str(output_parquet),
    }
    if output_csv:
        summary["output_csv"] = str(output_csv)
    if summary_json:
        summary_json.parent.mkdir(parents=True, exist_ok=True)
        summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary


def _load_gzip_lines_from_tar(tf: tarfile.TarFile, member_name: str) -> list[str]:
    member = tf.getmember(member_name)
    with tempfile.TemporaryDirectory() as tmpdir:
        tf.extract(member, path=tmpdir, filter="data")
        extracted = Path(tmpdir) / member_name
        with gzip.open(extracted, "rt", errors="ignore") as fh:
            return [line.strip() for line in fh if line.strip()]


def _read_sample_matrix_from_tar(
    tf: tarfile.TarFile,
    matrix_name: str,
    feature_name: str,
    barcode_name: str,
) -> dict[str, Any]:
    features = _load_gzip_lines_from_tar(tf, feature_name)
    barcodes = _load_gzip_lines_from_tar(tf, barcode_name)
    feature_rows = [row.split("\t") for row in features]
    gene_ids = np.array([row[0] for row in feature_rows], dtype=object)
    gene_names = np.array([row[1] if len(row) > 1 else row[0] for row in feature_rows], dtype=object)

    member = tf.getmember(matrix_name)
    with tempfile.TemporaryDirectory() as tmpdir:
        tf.extract(member, path=tmpdir, filter="data")
        extracted = Path(tmpdir) / matrix_name
        with gzip.open(extracted, "rt", errors="ignore") as fh:
            header = fh.readline().strip()
            if not header.startswith("%%MatrixMarket"):
                raise ValueError(f"Unexpected MatrixMarket header in {matrix_name}: {header}")
            dims_line = fh.readline().strip()
            while dims_line.startswith("%"):
                dims_line = fh.readline().strip()
            dims = dims_line.split()
            if len(dims) != 3:
                raise ValueError(f"Malformed dimensions in {matrix_name}: {dims_line}")
            n_genes, n_cells, _ = map(int, dims)
            counts_per_gene = np.zeros(n_genes, dtype=np.float64)
            total_umis = 0.0
            for line in fh:
                gene_idx_str, _cell_idx_str, count_str = line.strip().split()
                gene_idx = int(gene_idx_str) - 1
                count = float(count_str)
                counts_per_gene[gene_idx] += count
                total_umis += count

    return {
        "gene_ids": gene_ids,
        "gene_names": gene_names,
        "counts_per_gene": counts_per_gene,
        "n_genes": n_genes,
        "n_cells": len(barcodes),
        "total_umis": total_umis,
        "detected_gene_count": int((counts_per_gene > 0).sum()),
    }


def build_gse233844_pbmc_expression_reference(
    sample_reference_path: Path,
    raw_tar_path: Path,
    output_parquet: Path,
    output_csv: Path | None = None,
    top_genes_csv: Path | None = None,
    summary_json: Path | None = None,
    top_gene_limit_per_comparison: int = 150,
) -> dict[str, Any]:
    if sample_reference_path.suffix == ".parquet":
        sample_reference = pd.read_parquet(sample_reference_path)
    else:
        sample_reference = pd.read_csv(sample_reference_path)

    sample_reference = sample_reference.copy()
    sample_reference["matrix_name"] = sample_reference["matrix_name"].astype(str)
    sample_lookup = {
        row["matrix_name"]: row.to_dict()
        for _, row in sample_reference.iterrows()
        if row["matrix_name"]
    }

    sample_rows: list[dict[str, Any]] = []
    sample_ids: list[str] = []
    count_vectors: list[np.ndarray] = []
    gene_ids: np.ndarray | None = None
    gene_names: np.ndarray | None = None

    with tarfile.open(raw_tar_path, "r") as tf:
        matrix_members = [
            member.name
            for member in tf.getmembers()
            if member.isfile() and member.name.endswith("_matrix.mtx.gz")
        ]
        for matrix_name in matrix_members:
            if matrix_name not in sample_lookup:
                continue
            sample_meta = sample_lookup[matrix_name]
            matrix_summary = _read_sample_matrix_from_tar(
                tf=tf,
                matrix_name=matrix_name,
                feature_name=str(sample_meta["features_name"]),
                barcode_name=str(sample_meta["barcodes_name"]),
            )
            if gene_ids is None:
                gene_ids = matrix_summary["gene_ids"]
                gene_names = matrix_summary["gene_names"]
            else:
                if not np.array_equal(gene_ids, matrix_summary["gene_ids"]):
                    raise ValueError("Gene id ordering mismatch across GSE233844 samples")

            sample_rows.append(
                {
                    "accession": sample_meta.get("accession", DEFAULT_ACCESSION),
                    "gsm_accession": sample_meta.get("gsm_accession"),
                    "sample_title": sample_meta.get("sample_title"),
                    "group_assignment": sample_meta.get("group_assignment"),
                    "group_bucket": sample_meta.get("group_bucket"),
                    "source_name": sample_meta.get("source_name"),
                    "matrix_name": matrix_name,
                    "cell_count": matrix_summary["n_cells"],
                    "detected_gene_count": matrix_summary["detected_gene_count"],
                    "total_umis": matrix_summary["total_umis"],
                    "mean_umis_per_cell": (
                        matrix_summary["total_umis"] / matrix_summary["n_cells"]
                        if matrix_summary["n_cells"] > 0
                        else 0.0
                    ),
                }
            )
            sample_ids.append(sample_meta.get("gsm_accession"))
            count_vectors.append(matrix_summary["counts_per_gene"])

    if not sample_rows or gene_ids is None or gene_names is None:
        raise ValueError(f"No sample matrices from {raw_tar_path} matched {sample_reference_path}")

    sample_summary = pd.DataFrame(sample_rows).sort_values(["group_bucket", "gsm_accession"])
    counts_matrix = np.column_stack(count_vectors)

    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    sample_summary.to_parquet(output_parquet, index=False)
    if output_csv:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        sample_summary.to_csv(output_csv, index=False)

    bucket_order = ["IPF-Progressive", "IPF-Stable", "Control"]
    sample_index_lookup = {sample_id: idx for idx, sample_id in enumerate(sample_ids)}
    bucket_indices = {
        bucket: [
            sample_index_lookup[gsm]
            for gsm in sample_summary.loc[sample_summary["group_bucket"] == bucket, "gsm_accession"]
            if gsm in sample_index_lookup
        ]
        for bucket in bucket_order
    }
    bucket_cpm: dict[str, np.ndarray] = {}
    bucket_total_umis: dict[str, float] = {}
    for bucket, idxs in bucket_indices.items():
        bucket_counts = counts_matrix[:, idxs].sum(axis=1) if idxs else np.zeros(len(gene_ids))
        total = float(bucket_counts.sum())
        bucket_total_umis[bucket] = total
        bucket_cpm[bucket] = (bucket_counts * 1_000_000.0 / total) if total > 0 else bucket_counts

    comparisons = [
        ("IPF-Progressive", "Control"),
        ("IPF-Stable", "Control"),
        ("IPF-Progressive", "IPF-Stable"),
    ]
    top_gene_rows: list[dict[str, Any]] = []
    for lhs, rhs in comparisons:
        if bucket_total_umis.get(lhs, 0.0) <= 0 or bucket_total_umis.get(rhs, 0.0) <= 0:
            continue
        log2_fc = np.log2(bucket_cpm[lhs] + 1.0) - np.log2(bucket_cpm[rhs] + 1.0)
        order = np.argsort(np.abs(log2_fc))[::-1][: min(top_gene_limit_per_comparison, len(log2_fc))]
        for rank, gene_idx in enumerate(order, start=1):
            top_gene_rows.append(
                {
                    "comparison": f"{lhs} vs {rhs}",
                    "rank_within_comparison": rank,
                    "gene_id": gene_ids[gene_idx],
                    "gene_name": gene_names[gene_idx],
                    "lhs_cpm": float(bucket_cpm[lhs][gene_idx]),
                    "rhs_cpm": float(bucket_cpm[rhs][gene_idx]),
                    "log2_fc": float(log2_fc[gene_idx]),
                }
            )

    top_genes = pd.DataFrame(top_gene_rows)
    if top_genes_csv:
        top_genes_csv.parent.mkdir(parents=True, exist_ok=True)
        top_genes.to_csv(top_genes_csv, index=False)

    summary = {
        "accession": DEFAULT_ACCESSION,
        "sample_count": int(len(sample_summary)),
        "matched_matrix_count": int(len(sample_rows)),
        "total_cells": int(sample_summary["cell_count"].sum()),
        "median_cells_per_sample": float(sample_summary["cell_count"].median()),
        "detected_genes_union": int((counts_matrix.sum(axis=1) > 0).sum()),
        "group_bucket_distribution": sample_summary["group_bucket"].value_counts().to_dict(),
        "bucket_total_umis": {bucket: round(total, 2) for bucket, total in bucket_total_umis.items() if total > 0},
        "comparison_count": int(top_genes["comparison"].nunique()) if not top_genes.empty else 0,
        "top_gene_rows_exported": int(len(top_genes)),
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
