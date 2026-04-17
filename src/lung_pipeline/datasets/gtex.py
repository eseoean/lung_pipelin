from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .source_io import ensure_local_copy


def load_gtex_lung_reference(
    source_path: str | Path,
    cache_dir: Path,
) -> tuple[pd.DataFrame, dict[str, float]]:
    path = ensure_local_copy(source_path, cache_dir)
    df = pd.read_csv(path, sep="\t", compression="infer", skiprows=2)

    if "Description" not in df.columns:
        raise ValueError(f"GTEx GCT is missing Description column: {path}")

    sample_columns = [
        column for column in df.columns if column not in {"id", "Name", "Description"}
    ]
    if not sample_columns:
        raise ValueError(f"GTEx GCT has no sample columns: {path}")

    work = df[["Description", *sample_columns]].copy()
    work["Description"] = work["Description"].fillna("").astype(str).str.upper().str.strip()
    work = work[work["Description"] != ""]
    work[sample_columns] = work[sample_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    grouped = work.groupby("Description", as_index=False)[sample_columns].sum()
    values = grouped[sample_columns].to_numpy(dtype=float)
    log_values = np.log2(np.clip(values, a_min=0.0, a_max=None) + 1.0)

    reference = pd.DataFrame(
        {
            "gene_symbol": grouped["Description"].to_numpy(),
            "normal_lung_n_samples": int(len(sample_columns)),
            "normal_lung_mean_log2_tpm": log_values.mean(axis=1),
            "normal_lung_std_log2_tpm": log_values.std(axis=1),
            "normal_lung_median_log2_tpm": np.median(log_values, axis=1),
            "normal_lung_detected_fraction": (values > 0).mean(axis=1),
        }
    ).sort_values("normal_lung_mean_log2_tpm", ascending=False).reset_index(drop=True)

    stats = {
        "n_genes": float(reference.shape[0]),
        "n_samples": float(len(sample_columns)),
        "mean_detected_fraction": float(reference["normal_lung_detected_fraction"].mean()),
        "mean_log2_tpm": float(reference["normal_lung_mean_log2_tpm"].mean()),
    }
    return reference, stats
