from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .source_io import ensure_local_copy

COUNT_SUFFIX = ".rna_seq.augmented_star_gene_counts.tsv"


def load_count_entries(
    counts_source: str | Path,
    cache_dir: Path,
    *,
    manifest_source: str | Path | None = None,
    max_files: int | None = None,
) -> list[dict[str, Any]]:
    filenames_with_meta = _resolve_manifest_or_listing(
        counts_source,
        cache_dir,
        manifest_source=manifest_source,
    )
    if max_files is not None:
        filenames_with_meta = filenames_with_meta[:max_files]

    source_value = str(counts_source).strip()
    entries: list[dict[str, Any]] = []
    for item in filenames_with_meta:
        filename = item["filename"]
        if source_value.startswith("s3://"):
            source_path = f"{source_value.rstrip('/')}/{filename}"
        else:
            source_path = str((Path(source_value).expanduser().resolve() / filename))
        sample_id = filename.removesuffix(COUNT_SUFFIX)
        entries.append(
            {
                "sample_id": sample_id,
                "filename": filename,
                "source_path": source_path,
                "gdc_file_id": item.get("gdc_file_id"),
                "md5": item.get("md5"),
                "size": item.get("size"),
            }
        )
    return entries


def read_expression_profile(
    source_path: str | Path,
    cache_dir: Path,
    *,
    value_column: str = "tpm_unstranded",
    gene_types: list[str] | None = None,
) -> tuple[pd.Series, dict[str, float]]:
    path = ensure_local_copy(source_path, cache_dir)
    df = pd.read_csv(
        path,
        sep="\t",
        comment="#",
        usecols=lambda col: col in {"gene_id", "gene_name", "gene_type", value_column},
    )
    if value_column not in df.columns:
        raise ValueError(f"Expression value column not found: {value_column}")

    gene_id = df["gene_id"].astype(str)
    mask = df["gene_name"].notna() & ~gene_id.str.startswith("N_")
    if gene_types:
        mask &= df["gene_type"].astype(str).isin(gene_types)

    filtered = df.loc[mask, ["gene_name", value_column]].copy()
    filtered["gene_name"] = filtered["gene_name"].astype(str).str.upper().str.strip()
    filtered = filtered[filtered["gene_name"] != ""]
    filtered[value_column] = pd.to_numeric(filtered[value_column], errors="coerce").fillna(0.0)
    filtered = filtered.groupby("gene_name", as_index=False)[value_column].sum()

    raw_values = filtered[value_column].to_numpy(dtype=float)
    log_values = np.log2(np.clip(raw_values, a_min=0.0, a_max=None) + 1.0)
    series = pd.Series(log_values, index=filtered["gene_name"].to_numpy(), dtype=float)
    stats = {
        "expr__n_genes": float(series.shape[0]),
        "expr__detected_genes": float((raw_values > 0).sum()),
        "expr__mean_log2_tpm": float(series.mean()) if not series.empty else 0.0,
        "expr__std_log2_tpm": float(series.std(ddof=0)) if not series.empty else 0.0,
        "expr__median_log2_tpm": float(series.median()) if not series.empty else 0.0,
        "expr__max_log2_tpm": float(series.max()) if not series.empty else 0.0,
    }
    return series.sort_index(), stats


def _resolve_manifest_or_listing(
    counts_source: str | Path,
    cache_dir: Path,
    *,
    manifest_source: str | Path | None = None,
) -> list[dict[str, Any]]:
    if manifest_source is not None:
        manifest_path = ensure_local_copy(manifest_source, cache_dir)
        manifest = pd.read_csv(manifest_path, sep="\t")
        if "filename" not in manifest.columns:
            raise ValueError(f"TCGA RNA manifest is missing filename column: {manifest_path}")
        rows: list[dict[str, Any]] = []
        for row in manifest.to_dict(orient="records"):
            filename = str(row["filename"]).strip()
            if filename.endswith(COUNT_SUFFIX):
                rows.append(
                    {
                        "filename": filename,
                        "gdc_file_id": row.get("id"),
                        "md5": row.get("md5"),
                        "size": row.get("size"),
                    }
                )
        return rows

    counts_value = str(counts_source).strip()
    if counts_value.startswith("s3://"):
        result = subprocess.run(
            ["aws", "s3", "ls", counts_value.rstrip("/") + "/"],
            capture_output=True,
            text=True,
            check=True,
        )
        rows = []
        for line in result.stdout.splitlines():
            parts = line.split()
            if parts and parts[-1].endswith(COUNT_SUFFIX):
                rows.append({"filename": parts[-1]})
        return rows

    counts_dir = Path(counts_value).expanduser().resolve()
    if not counts_dir.exists():
        raise FileNotFoundError(f"TCGA counts directory not found: {counts_dir}")
    return [{"filename": path.name} for path in sorted(counts_dir.glob(f"*{COUNT_SUFFIX}"))]
