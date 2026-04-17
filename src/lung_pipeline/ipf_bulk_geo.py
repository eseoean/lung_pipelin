from __future__ import annotations

import gzip
import json
import tarfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _strip_quotes(value: str) -> str:
    return value.strip().strip('"')


def _normalize_sample_field(raw_field: str, seen: Counter[str]) -> str:
    field = raw_field.removeprefix("!Sample_").lower()
    field = field.replace("/", "_").replace(" ", "_").replace("%", "pct")
    count = seen[field]
    seen[field] += 1
    return field if count == 0 else f"{field}_{count + 1}"


def default_gse32537_paths(repo_root: Path) -> dict[str, Path]:
    base = repo_root / "data" / "raw" / "geo" / "ipf" / "GSE32537"
    return {
        "series_matrix": base / "GSE32537_series_matrix.txt.gz",
        "family_soft": base / "GSE32537_family.soft.gz",
    }


def default_gse47460_paths(repo_root: Path) -> dict[str, Path]:
    base = repo_root / "data" / "raw" / "geo" / "ipf" / "GSE47460"
    return {
        "family_soft": base / "GSE47460_family.soft.gz",
        "raw_tar": base / "supplementary" / "GSE47460_RAW.tar",
    }


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


def parse_family_soft_sample_blocks(family_soft_path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    with gzip.open(family_soft_path, "rt", errors="ignore") as fh:
        for raw_line in fh:
            line = raw_line.rstrip("\n")
            if line.startswith("^SAMPLE ="):
                if current is not None:
                    rows.append(current)
                current = {"gsm_accession": line.split("=", 1)[1].strip()}
                continue
            if current is None:
                continue
            if line.startswith("^") and not line.startswith("^SAMPLE ="):
                rows.append(current)
                current = None
                continue
            if line.startswith("!Sample_"):
                key_raw, value = line.split("=", 1)
                key_raw = key_raw.strip()
                value = value.strip()
                if key_raw == "!Sample_characteristics_ch1":
                    current.setdefault("characteristics_ch1", []).append(value)
                else:
                    key = key_raw.removeprefix("!Sample_").strip().lower()
                    key = key.replace("/", "_").replace(" ", "_").replace("%", "pct")
                    current[key] = value

    if current is not None:
        rows.append(current)

    if not rows:
        raise ValueError(f"No ^SAMPLE blocks found in {family_soft_path}")

    normalized_rows: list[dict[str, Any]] = []
    for row in rows:
        normalized = dict(row)
        for idx, item in enumerate(row.get("characteristics_ch1", []), start=1):
            normalized[f"characteristics_ch1_{idx}"] = item
        normalized_rows.append(normalized)
    return pd.DataFrame(normalized_rows)


def _extract_kv(raw_value: str) -> tuple[str, str]:
    if ":" in raw_value:
        key, value = raw_value.split(":", 1)
        return key.strip().lower(), value.strip()
    return raw_value.strip().lower(), raw_value.strip()


def _clean_numeric(value: str) -> float | None:
    value = str(value).strip()
    if value in {"", "--", "NA", "na"}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _extract_first_gene_symbol(gene_assignment: str) -> str:
    for chunk in gene_assignment.split("///"):
        parts = [part.strip() for part in chunk.split("//")]
        if len(parts) >= 2:
            symbol = parts[1].strip()
            if symbol and symbol != "---":
                return symbol
    return ""


def _normalize_gse47460_feature_label(probe_name: str, systematic_name: str) -> str:
    systematic_name = str(systematic_name).strip()
    probe_name = str(probe_name).strip()
    if systematic_name and systematic_name.lower() not in {"nan", "---"}:
        return systematic_name
    return probe_name


def parse_gse47460_feature_table(raw_member_bytes: bytes) -> pd.DataFrame:
    lines = gzip.decompress(raw_member_bytes).decode("utf-8", "ignore").splitlines()

    header: list[str] | None = None
    records: list[dict[str, Any]] = []
    capture = False
    for line in lines:
        if line.startswith("FEATURES\t"):
            header = line.split("\t")
            capture = True
            continue
        if not capture or not line.startswith("DATA\t") or header is None:
            continue
        parts = line.split("\t")
        if len(parts) != len(header):
            continue
        row = dict(zip(header, parts))
        if row.get("ControlType") != "0":
            continue
        feature_label = _normalize_gse47460_feature_label(
            row.get("ProbeName", ""),
            row.get("SystematicName", ""),
        )
        signal = _clean_numeric(row.get("gProcessedSignal", ""))
        if not feature_label or signal is None:
            continue
        records.append(
            {
                "feature_label": feature_label,
                "probe_name": row.get("ProbeName", ""),
                "systematic_name": row.get("SystematicName", ""),
                "g_processed_signal": float(signal),
            }
        )

    if not records:
        raise ValueError("No gene-level feature rows parsed from GSE47460 raw member")

    feature_frame = pd.DataFrame(records)
    return (
        feature_frame.groupby("feature_label", as_index=False)
        .agg(
            probe_name=("probe_name", "first"),
            systematic_name=("systematic_name", "first"),
            g_processed_signal=("g_processed_signal", "mean"),
        )
    )


def parse_gse32537_platform_gene_map(family_soft_path: Path) -> pd.DataFrame:
    header: list[str] | None = None
    rows: list[dict[str, str]] = []
    capture = False

    with gzip.open(family_soft_path, "rt", errors="ignore") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("!platform_table_begin"):
                capture = True
                continue
            if not capture:
                continue
            if line.startswith("!platform_table_end"):
                break
            if header is None:
                header = line.split("\t")
                continue
            parts = line.split("\t")
            if len(parts) != len(header):
                continue
            rows.append(dict(zip(header, parts)))

    if not rows:
        raise ValueError(f"No platform table found in {family_soft_path}")

    frame = pd.DataFrame(rows)
    frame["gene_symbol"] = frame["gene_assignment"].map(_extract_first_gene_symbol)
    frame["feature_label"] = frame["gene_symbol"].where(frame["gene_symbol"] != "", frame["ID"])
    return frame[["ID", "feature_label", "gene_symbol", "gene_assignment"]]


def build_gse32537_bulk_reference(
    series_matrix_path: Path,
    family_soft_path: Path,
    sample_reference_parquet: Path,
    sample_reference_csv: Path | None = None,
    expression_summary_parquet: Path | None = None,
    expression_summary_csv: Path | None = None,
    top_genes_csv: Path | None = None,
    summary_json: Path | None = None,
    top_gene_limit: int = 200,
) -> dict[str, Any]:
    sample_table = parse_geo_series_matrix_sample_table(series_matrix_path).copy()

    sample_table["accession"] = "GSE32537"
    sample_table["age_years"] = pd.to_numeric(sample_table["characteristics_ch1"].str.split(":").str[-1].str.strip(), errors="coerce")
    sample_table["sex"] = sample_table["characteristics_ch1_2"].str.split(":").str[-1].str.strip()
    sample_table["final_diagnosis"] = sample_table["characteristics_ch1_3"].str.split(":").str[-1].str.strip()
    sample_table["repository"] = sample_table["characteristics_ch1_4"].str.split(":").str[-1].str.strip()
    sample_table["tissue_source"] = sample_table["characteristics_ch1_5"].str.split(":").str[-1].str.strip()
    sample_table["rin"] = pd.to_numeric(sample_table["characteristics_ch1_7"].str.split(":").str[-1].str.strip(), errors="coerce")
    sample_table["smoking_status"] = sample_table["characteristics_ch1_8"].str.split(":").str[-1].str.strip()
    sample_table["pack_years"] = pd.to_numeric(sample_table["characteristics_ch1_10"].str.split(":").str[-1].str.strip(), errors="coerce")
    sample_table["fvc_pct_predicted"] = pd.to_numeric(sample_table["characteristics_ch1_12"].str.split(":").str[-1].str.strip(), errors="coerce")
    sample_table["dlco_pct_predicted"] = pd.to_numeric(sample_table["characteristics_ch1_13"].str.split(":").str[-1].str.strip(), errors="coerce")
    sample_table["disease_bucket"] = sample_table["final_diagnosis"].map(
        lambda x: "IPF" if x == "IPF/UIP" else ("Control" if str(x).lower() == "control" else "Other-IIP")
    )

    sample_reference = sample_table[
        [
            "accession",
            "geo_accession",
            "title",
            "source_name_ch1",
            "age_years",
            "sex",
            "final_diagnosis",
            "repository",
            "tissue_source",
            "rin",
            "smoking_status",
            "pack_years",
            "fvc_pct_predicted",
            "dlco_pct_predicted",
            "disease_bucket",
        ]
    ].rename(columns={"geo_accession": "gsm_accession", "title": "sample_title", "source_name_ch1": "source_name"})

    sample_reference_parquet.parent.mkdir(parents=True, exist_ok=True)
    sample_reference.to_parquet(sample_reference_parquet, index=False)
    if sample_reference_csv:
        sample_reference_csv.parent.mkdir(parents=True, exist_ok=True)
        sample_reference.to_csv(sample_reference_csv, index=False)

    platform_map = parse_gse32537_platform_gene_map(family_soft_path)

    # Read only the matrix section for predictable parsing.
    matrix_lines: list[str] = []
    with gzip.open(series_matrix_path, "rt", errors="ignore") as fh:
        capture = False
        for line in fh:
            if line.startswith("!series_matrix_table_begin"):
                capture = True
                continue
            if line.startswith("!series_matrix_table_end"):
                break
            if capture:
                matrix_lines.append(line)
    from io import StringIO

    expression = pd.read_csv(StringIO("".join(matrix_lines)), sep="\t")
    expression = expression.rename(columns={"ID_REF": "feature_id"})
    expression["feature_id"] = expression["feature_id"].astype(str)
    expression = expression.merge(platform_map, how="left", left_on="feature_id", right_on="ID")
    expression["feature_label"] = expression["feature_label"].fillna(expression["feature_id"])

    sample_ids = [col for col in expression.columns if col.startswith("GSM")]
    grouped = expression[["feature_label", *sample_ids]].groupby("feature_label", as_index=False).mean()
    grouped = grouped.set_index("feature_label")

    sample_lookup = sample_reference.set_index("gsm_accession")
    ipf_ids = [gsm for gsm in sample_ids if gsm in sample_lookup.index and sample_lookup.loc[gsm, "disease_bucket"] == "IPF"]
    control_ids = [gsm for gsm in sample_ids if gsm in sample_lookup.index and sample_lookup.loc[gsm, "disease_bucket"] == "Control"]

    expression_summary = pd.DataFrame(
        {
            "feature_label": grouped.index,
            "ipf_mean_expression": grouped[ipf_ids].mean(axis=1) if ipf_ids else np.nan,
            "control_mean_expression": grouped[control_ids].mean(axis=1) if control_ids else np.nan,
        }
    )
    expression_summary["delta_ipf_vs_control"] = (
        expression_summary["ipf_mean_expression"] - expression_summary["control_mean_expression"]
    )
    top_genes = expression_summary.reindex(
        expression_summary["delta_ipf_vs_control"].abs().sort_values(ascending=False).index
    ).head(top_gene_limit)

    if expression_summary_parquet:
        expression_summary_parquet.parent.mkdir(parents=True, exist_ok=True)
        expression_summary.to_parquet(expression_summary_parquet, index=False)
    if expression_summary_csv:
        expression_summary_csv.parent.mkdir(parents=True, exist_ok=True)
        expression_summary.to_csv(expression_summary_csv, index=False)
    if top_genes_csv:
        top_genes_csv.parent.mkdir(parents=True, exist_ok=True)
        top_genes.to_csv(top_genes_csv, index=False)

    summary = {
        "accession": "GSE32537",
        "sample_count": int(len(sample_reference)),
        "disease_bucket_distribution": sample_reference["disease_bucket"].value_counts().to_dict(),
        "matrix_feature_rows": int(len(expression)),
        "gene_level_rows": int(len(expression_summary)),
        "ipf_sample_count": int(len(ipf_ids)),
        "control_sample_count": int(len(control_ids)),
        "top_gene_count": int(len(top_genes)),
        "sample_reference_parquet": str(sample_reference_parquet),
    }
    if sample_reference_csv:
        summary["sample_reference_csv"] = str(sample_reference_csv)
    if expression_summary_parquet:
        summary["expression_summary_parquet"] = str(expression_summary_parquet)
    if expression_summary_csv:
        summary["expression_summary_csv"] = str(expression_summary_csv)
    if top_genes_csv:
        summary["top_genes_csv"] = str(top_genes_csv)
    if summary_json:
        summary_json.parent.mkdir(parents=True, exist_ok=True)
        summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary


def build_gse47460_bulk_sample_reference(
    family_soft_path: Path,
    output_parquet: Path,
    output_csv: Path | None = None,
    summary_json: Path | None = None,
) -> dict[str, Any]:
    sample_blocks = parse_family_soft_sample_blocks(family_soft_path)

    meta_rows: list[dict[str, Any]] = []
    for _, row in sample_blocks.iterrows():
        characteristics = {}
        for col, value in row.items():
            if not str(col).startswith("characteristics_ch1_"):
                continue
            key, parsed = _extract_kv(str(value))
            characteristics[key] = parsed

        disease_state = str(characteristics.get("disease state", "")).strip().strip('"')
        ild_subtype = str(characteristics.get("ild subtype", "")).strip().strip('"')
        disease_state_lower = disease_state.lower()
        ild_subtype_lower = ild_subtype.lower()
        if disease_state_lower == "control":
            bucket = "Control"
        elif disease_state_lower in {"ild", "interstitial lung disease"} and (
            "ipf" in ild_subtype_lower or "uip/ipf" in ild_subtype_lower
        ):
            bucket = "IPF"
        elif disease_state_lower in {"ild", "interstitial lung disease"}:
            bucket = "Other-ILD"
        elif disease_state_lower in {"copd", "chronic obstructive lung disease"}:
            bucket = "COPD"
        else:
            bucket = "Other"

        meta_rows.append(
            {
                "accession": "GSE47460",
                "gsm_accession": row["gsm_accession"],
                "sample_title": row.get("title", ""),
                "source_name": row.get("source_name_ch1", ""),
                "disease_state": disease_state,
                "ild_subtype": ild_subtype,
                "sex": characteristics.get("sex", ""),
                "age_years": _clean_numeric(characteristics.get("age", "")),
                "smoker_status": characteristics.get("smoker?", ""),
                "fev1_pct_predicted_pre_bd": _clean_numeric(characteristics.get("%predicted fev1 (pre-bd)", "")),
                "fvc_pct_predicted_pre_bd": _clean_numeric(characteristics.get("%predicted fvc (pre-bd)", "")),
                "dlco_pct_predicted": _clean_numeric(characteristics.get("%predicted dlco", "")),
                "disease_bucket": bucket,
            }
        )

    reference = pd.DataFrame(meta_rows)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    reference.to_parquet(output_parquet, index=False)
    if output_csv:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        reference.to_csv(output_csv, index=False)

    summary = {
        "accession": "GSE47460",
        "sample_count": int(len(reference)),
        "disease_state_distribution": reference["disease_state"].value_counts().to_dict(),
        "disease_bucket_distribution": reference["disease_bucket"].value_counts().to_dict(),
        "output_parquet": str(output_parquet),
    }
    if output_csv:
        summary["output_csv"] = str(output_csv)
    if summary_json:
        summary_json.parent.mkdir(parents=True, exist_ok=True)
        summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary


def build_gse47460_bulk_expression_reference(
    sample_reference_path: Path,
    raw_tar_path: Path,
    output_parquet: Path,
    output_csv: Path | None = None,
    top_genes_csv: Path | None = None,
    summary_json: Path | None = None,
    top_gene_limit: int = 200,
) -> dict[str, Any]:
    sample_reference = pd.read_csv(sample_reference_path)
    sample_lookup = sample_reference.set_index("gsm_accession")

    sample_rows: list[dict[str, Any]] = []
    bucket_feature_sums: dict[str, Counter[str]] = {
        bucket: Counter() for bucket in sample_reference["disease_bucket"].dropna().unique()
    }
    bucket_feature_counts: dict[str, Counter[str]] = {
        bucket: Counter() for bucket in sample_reference["disease_bucket"].dropna().unique()
    }

    with tarfile.open(raw_tar_path, "r") as tf:
        for member in tf.getmembers():
            if not member.isfile() or not member.name.endswith(".txt.gz"):
                continue
            gsm_accession = Path(member.name).name.split("_", 1)[0]
            if gsm_accession not in sample_lookup.index:
                continue
            extracted = tf.extractfile(member)
            if extracted is None:
                continue
            feature_frame = parse_gse47460_feature_table(extracted.read())
            bucket = str(sample_lookup.loc[gsm_accession, "disease_bucket"])

            feature_means = dict(
                zip(
                    feature_frame["feature_label"].astype(str),
                    feature_frame["g_processed_signal"].astype(float),
                )
            )
            bucket_feature_sums.setdefault(bucket, Counter()).update(feature_means)
            bucket_feature_counts.setdefault(bucket, Counter()).update(
                {feature: 1 for feature in feature_means}
            )

            sample_rows.append(
                {
                    "gsm_accession": gsm_accession,
                    "sample_title": sample_lookup.loc[gsm_accession, "sample_title"],
                    "disease_bucket": bucket,
                    "feature_count": int(len(feature_frame)),
                    "mean_processed_signal": float(feature_frame["g_processed_signal"].mean()),
                    "median_processed_signal": float(feature_frame["g_processed_signal"].median()),
                }
            )

    if not sample_rows:
        raise ValueError(f"No GSE47460 sample rows were parsed from {raw_tar_path}")

    all_features = sorted(
        {
            feature
            for bucket_counter in bucket_feature_sums.values()
            for feature in bucket_counter.keys()
        }
    )
    expression_summary = pd.DataFrame({"feature_label": all_features})
    for bucket in sorted(bucket_feature_sums):
        sums = bucket_feature_sums[bucket]
        counts = bucket_feature_counts[bucket]
        expression_summary[f"{bucket.lower().replace('-', '_')}_mean_expression"] = [
            (sums[feature] / counts[feature]) if counts[feature] else np.nan
            for feature in all_features
        ]

    if {
        "ipf_mean_expression",
        "control_mean_expression",
    }.issubset(expression_summary.columns):
        expression_summary["delta_ipf_vs_control"] = (
            expression_summary["ipf_mean_expression"]
            - expression_summary["control_mean_expression"]
        )
    else:
        expression_summary["delta_ipf_vs_control"] = np.nan

    top_genes = expression_summary.reindex(
        expression_summary["delta_ipf_vs_control"].abs().sort_values(ascending=False).index
    ).head(top_gene_limit)

    sample_summary = pd.DataFrame(sample_rows).sort_values(
        ["disease_bucket", "gsm_accession"]
    )
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    sample_summary.to_parquet(output_parquet, index=False)
    if output_csv:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        sample_summary.to_csv(output_csv, index=False)
    if top_genes_csv:
        top_genes_csv.parent.mkdir(parents=True, exist_ok=True)
        top_genes.to_csv(top_genes_csv, index=False)

    summary = {
        "accession": "GSE47460",
        "sample_count": int(len(sample_summary)),
        "disease_bucket_distribution": sample_summary["disease_bucket"].value_counts().to_dict(),
        "feature_union_count": int(len(all_features)),
        "ipf_sample_count": int((sample_summary["disease_bucket"] == "IPF").sum()),
        "control_sample_count": int((sample_summary["disease_bucket"] == "Control").sum()),
        "top_gene_count": int(len(top_genes)),
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
