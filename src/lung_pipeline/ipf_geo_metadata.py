from __future__ import annotations

import csv
import gzip
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_GSE122960 = "GSE122960"


def default_gse122960_paths(repo_root: Path) -> dict[str, Path]:
    base = repo_root / "data" / "raw" / "geo" / "ipf" / DEFAULT_GSE122960
    return {
        "series_matrix": base / f"{DEFAULT_GSE122960}_series_matrix.txt.gz",
        "family_soft": base / f"{DEFAULT_GSE122960}_family.soft.gz",
        "filelist": base / "supplementary" / "filelist.txt",
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


def _extract_disease_condition(raw_value: str) -> str:
    if ":" in raw_value:
        return raw_value.split(":", 1)[1].strip()
    return raw_value.strip()


def _normalize_disease_condition_label(raw_value: str) -> str:
    normalized = raw_value.strip()
    lookup = {
        "donor": "Donor",
        "idiopathic pulmonary fibrosis": "Idiopathic pulmonary fibrosis",
        "hypersensitivity pneumonitis": "Hypersensitivity pneumonitis",
        "systemic slcerosis-associated interstitial lung disease": (
            "Systemic sclerosis-associated interstitial lung disease"
        ),
        "myositis-associated interstitial lng disease": (
            "Myositis-associated interstitial lung disease"
        ),
        "cryobiopsy": "Cryobiopsy",
    }
    return lookup.get(normalized.lower(), normalized)


def _map_disease_bucket(raw_value: str) -> str:
    value = raw_value.lower()
    if value == "donor":
        return "Control"
    if "idiopathic pulmonary fibrosis" in value:
        return "IPF"
    if "cryobiopsy" in value:
        return "Early-IPF/Cryobiopsy"
    if "hypersensitivity pneumonitis" in value:
        return "Other-ILD"
    if "interstitial lung disease" in value:
        return "Other-ILD"
    return "Other"


def _resolve_first_existing(frame: pd.DataFrame, candidates: list[str]) -> str:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    raise KeyError(f"None of {candidates} found in columns")


def build_gse122960_sample_reference(
    series_matrix_path: Path,
    filelist_path: Path,
    output_parquet: Path,
    output_csv: Path | None = None,
    summary_json: Path | None = None,
) -> dict[str, Any]:
    samples = parse_geo_series_matrix_sample_table(series_matrix_path)
    filelist = load_geo_filelist(filelist_path)

    samples = samples.copy()
    samples["accession"] = DEFAULT_GSE122960
    samples["disease_condition"] = (
        samples["characteristics_ch1"]
        .map(_extract_disease_condition)
        .map(_normalize_disease_condition_label)
    )
    samples["disease_bucket"] = samples["disease_condition"].map(_map_disease_bucket)
    samples["filtered_h5_name"] = samples["supplementary_file_1"].str.rsplit("/", n=1).str[-1]
    samples["raw_h5_name"] = samples["supplementary_file_2"].str.rsplit("/", n=1).str[-1]

    available_names = set(filelist["Name"].tolist()) if "Name" in filelist.columns else set()
    samples["filtered_h5_available"] = samples["filtered_h5_name"].isin(available_names)
    samples["raw_h5_available"] = samples["raw_h5_name"].isin(available_names)

    selected_cols = {
        "accession": _resolve_first_existing(samples, ["accession"]),
        "sample_geo_accession": _resolve_first_existing(samples, ["sample_geo_accession", "geo_accession"]),
        "title": _resolve_first_existing(samples, ["title"]),
        "disease_condition": _resolve_first_existing(samples, ["disease_condition"]),
        "disease_bucket": _resolve_first_existing(samples, ["disease_bucket"]),
        "source_name_ch1": _resolve_first_existing(samples, ["source_name_ch1"]),
        "organism_ch1": _resolve_first_existing(samples, ["organism_ch1"]),
        "library_strategy": _resolve_first_existing(samples, ["library_strategy"]),
        "supplementary_file_1": _resolve_first_existing(samples, ["supplementary_file_1"]),
        "supplementary_file_2": _resolve_first_existing(samples, ["supplementary_file_2"]),
        "filtered_h5_name": _resolve_first_existing(samples, ["filtered_h5_name"]),
        "raw_h5_name": _resolve_first_existing(samples, ["raw_h5_name"]),
        "filtered_h5_available": _resolve_first_existing(samples, ["filtered_h5_available"]),
        "raw_h5_available": _resolve_first_existing(samples, ["raw_h5_available"]),
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
        "accession": DEFAULT_GSE122960,
        "sample_count": int(len(reference)),
        "disease_distribution": reference["disease_condition"].value_counts().to_dict(),
        "disease_bucket_distribution": reference["disease_bucket"].value_counts().to_dict(),
        "filtered_h5_available_count": int(reference["filtered_h5_available"].sum()),
        "raw_h5_available_count": int(reference["raw_h5_available"].sum()),
        "output_parquet": str(output_parquet),
    }
    if output_csv:
        summary["output_csv"] = str(output_csv)

    if summary_json:
        summary_json.parent.mkdir(parents=True, exist_ok=True)
        summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")

    return summary
