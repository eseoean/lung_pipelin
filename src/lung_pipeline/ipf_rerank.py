from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .io import write_json


def _minmax(series: pd.Series) -> pd.Series:
    numeric = series.astype(float)
    min_value = float(numeric.min())
    max_value = float(numeric.max())
    if max_value <= min_value:
        return pd.Series(0.0, index=series.index, dtype=float)
    return (numeric - min_value) / (max_value - min_value)


def _as_markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows available._"
    columns = frame.columns.tolist()
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in frame.astype(str).itertuples(index=False, name=None)
    ]
    return "\n".join([header, divider, *rows])


def _download_gap_rows(knowledge_root: Path) -> list[dict[str, str]]:
    expected = [
        ("admet", knowledge_root / "admet", "Safety and pharmacokinetic filtering"),
        ("siderside", knowledge_root / "siderside", "Side-effect plausibility checks"),
        ("clinicaltrials", knowledge_root / "clinicaltrials", "Trial-stage translation support"),
        ("string", knowledge_root / "string", "Network proximity / PPI support"),
    ]
    rows: list[dict[str, str]] = []
    for bucket, path, purpose in expected:
        rows.append(
            {
                "source": bucket,
                "status": "available" if path.exists() else "missing",
                "path": str(path),
                "purpose": purpose,
            }
        )

    opentargets_path = knowledge_root / "opentargets" / "association_by_overall_indirect"
    rows.append(
        {
            "source": "opentargets",
            "status": "available_schema_gap" if opentargets_path.exists() else "missing",
            "path": str(opentargets_path),
            "purpose": "Disease-target association scores are present, but a stable ENSG-to-symbol mapping is still missing in the local IPF branch.",
        }
    )
    return rows


def _support_note(row: pd.Series) -> str:
    notes: list[str] = []
    if int(row.get("is_approved", 0)) == 1:
        notes.append("approved")
    elif int(row.get("is_investigational", 0)) == 1:
        notes.append("investigational")
    if int(row.get("has_lincs_match", 0)) == 1:
        notes.append("LINCS")
    if float(row.get("fibrosis_priority_overlap_count", 0)) > 0:
        notes.append(f"fibrosis_targets={int(row['fibrosis_priority_overlap_count'])}")
    if float(row.get("broad_chemistry_penalty", 0)) > 0:
        notes.append("broad-chemistry-penalty")
    return ", ".join(notes) if notes else "target-overlap only"


def run_ipf_rerank(
    *,
    patient_scores_parquet: Path,
    final_ranked_parquet: Path,
    manifest_json: Path,
    summary_json: Path,
    review_csv: Path,
    translation_report_md: Path,
    download_gap_report_md: Path,
    knowledge_root: Path,
) -> dict[str, Any]:
    candidates = pd.read_parquet(patient_scores_parquet).copy()
    if candidates.empty:
        raise ValueError("Patient inference table is empty.")

    candidates["consensus_model_norm"] = _minmax(candidates["consensus_model_score"])
    candidates["pseudo_label_norm"] = _minmax(candidates["pseudo_label_score"])
    candidates["translation_readiness_score"] = (
        0.45 * candidates["approval_bonus"].astype(float)
        + 0.20 * candidates["lincs_support_norm"].astype(float)
        + 0.20 * candidates["target_overlap_count_norm"].astype(float)
        + 0.15 * candidates["fibrosis_priority_overlap_norm"].astype(float)
    )
    candidates["final_rerank_score"] = (
        0.40 * candidates["consensus_model_norm"].astype(float)
        + 0.20 * candidates["pseudo_label_norm"].astype(float)
        + 0.15 * candidates["specific_overlap_norm"].astype(float)
        + 0.10 * candidates["fibrosis_priority_overlap_norm"].astype(float)
        + 0.10 * candidates["lincs_support_norm"].astype(float)
        + 0.05 * candidates["approval_bonus"].astype(float)
        - 0.10 * candidates["broad_target_penalty"].astype(float)
        - 0.15 * candidates["broad_chemistry_penalty"].astype(float)
    )
    candidates = candidates.sort_values(
        [
            "final_rerank_score",
            "consensus_model_score",
            "pseudo_label_score",
            "fibrosis_priority_overlap_count",
            "is_approved",
        ],
        ascending=False,
    ).reset_index(drop=True)
    candidates["final_rerank_rank"] = range(1, len(candidates) + 1)
    candidates["support_note"] = candidates.apply(_support_note, axis=1)

    final_ranked_parquet.parent.mkdir(parents=True, exist_ok=True)
    review_csv.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_parquet(final_ranked_parquet, index=False)
    candidates.to_csv(review_csv, index=False)

    top_columns = [
        "final_rerank_rank",
        "drug_name",
        "final_rerank_score",
        "consensus_model_score",
        "pseudo_label_score",
        "target_overlap_count",
        "fibrosis_priority_overlap_count",
        "has_lincs_match",
        "is_approved",
        "support_note",
    ]
    for column in top_columns:
        if column not in candidates.columns:
            candidates[column] = 0
    top_table = candidates.loc[:, top_columns].head(20).copy()
    top_table["final_rerank_score"] = top_table["final_rerank_score"].map(lambda x: f"{x:.4f}")
    top_table["consensus_model_score"] = top_table["consensus_model_score"].map(lambda x: f"{x:.4f}")
    top_table["pseudo_label_score"] = top_table["pseudo_label_score"].map(lambda x: f"{x:.4f}")
    translation_report_md.parent.mkdir(parents=True, exist_ok=True)
    translation_report_md.write_text(
        "\n".join(
            [
                "# IPF Translation Support Report",
                "",
                "This report fuses currently available local evidence only:",
                "",
                "- patient-inference consensus model score",
                "- pseudo-label reversal score",
                "- fibrosis-priority target overlap",
                "- LINCS perturbation support",
                "- approval / investigational status",
                "- broad-target and broad-chemistry penalties",
                "",
                "## Top 20 reranked candidates",
                "",
                _as_markdown_table(top_table),
                "",
            ]
        )
        + "\n"
    )

    gap_rows = _download_gap_rows(knowledge_root)
    gap_frame = pd.DataFrame(gap_rows)
    download_gap_report_md.parent.mkdir(parents=True, exist_ok=True)
    download_gap_report_md.write_text(
        "\n".join(
            [
                "# IPF Download Gap Report",
                "",
                "Current reranking used only locally available evidence. Missing or partially blocked sources are listed below.",
                "",
                _as_markdown_table(gap_frame),
                "",
            ]
        )
        + "\n"
    )

    summary = {
        "study": "IPF",
        "row_count": int(len(candidates)),
        "top10_drugs": candidates["drug_name"].head(10).tolist(),
        "approved_in_top20": int(candidates.head(20)["is_approved"].sum()),
        "lincs_in_top20": int(candidates.head(20)["has_lincs_match"].sum()),
        "top_final_rerank_score": float(candidates.iloc[0]["final_rerank_score"]),
        "final_ranked_parquet": str(final_ranked_parquet),
        "review_csv": str(review_csv),
        "translation_report_md": str(translation_report_md),
        "download_gap_report_md": str(download_gap_report_md),
    }
    write_json(summary_json, summary)

    manifest = {
        "stage": "rerank_outputs",
        "study": "IPF",
        "objective": "disease_signature_reversal",
        "row_count": int(len(candidates)),
        "top10_drugs": summary["top10_drugs"],
        "artifacts": {
            "patient_scores_parquet": str(patient_scores_parquet),
            "final_ranked_parquet": str(final_ranked_parquet),
            "review_csv": str(review_csv),
            "summary_json": str(summary_json),
            "translation_report_md": str(translation_report_md),
            "download_gap_report_md": str(download_gap_report_md),
        },
        "local_gap_status": gap_rows,
    }
    write_json(manifest_json, manifest)
    return manifest
