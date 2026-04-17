from __future__ import annotations

import pandas as pd
from pathlib import Path

from lung_pipeline.ipf_rerank import run_ipf_rerank


def test_run_ipf_rerank(tmp_path: Path) -> None:
    candidates = pd.DataFrame(
        [
            {
                "drug_name": "Drug_A",
                "consensus_model_score": 0.91,
                "pseudo_label_score": 0.88,
                "specific_overlap_norm": 0.42,
                "fibrosis_priority_overlap_norm": 1.0,
                "fibrosis_priority_overlap_count": 3,
                "lincs_support_norm": 0.7,
                "approval_bonus": 1.0,
                "has_lincs_match": 1,
                "is_approved": 1,
                "is_investigational": 0,
                "target_overlap_count_norm": 0.7,
                "broad_target_penalty": 0.0,
                "broad_chemistry_penalty": 0.0,
            },
            {
                "drug_name": "Drug_B",
                "consensus_model_score": 0.85,
                "pseudo_label_score": 0.90,
                "specific_overlap_norm": 0.35,
                "fibrosis_priority_overlap_norm": 0.67,
                "fibrosis_priority_overlap_count": 2,
                "lincs_support_norm": 0.0,
                "approval_bonus": 0.0,
                "has_lincs_match": 0,
                "is_approved": 0,
                "is_investigational": 1,
                "target_overlap_count_norm": 0.5,
                "broad_target_penalty": 0.0,
                "broad_chemistry_penalty": 0.0,
            },
            {
                "drug_name": "Drug_C",
                "consensus_model_score": 0.70,
                "pseudo_label_score": 0.86,
                "specific_overlap_norm": 0.25,
                "fibrosis_priority_overlap_norm": 0.33,
                "fibrosis_priority_overlap_count": 1,
                "lincs_support_norm": 0.0,
                "approval_bonus": 1.0,
                "has_lincs_match": 0,
                "is_approved": 1,
                "is_investigational": 0,
                "target_overlap_count_norm": 0.3,
                "broad_target_penalty": 0.1,
                "broad_chemistry_penalty": 1.0,
            },
        ]
    )
    patient_scores = tmp_path / "patient_scores.parquet"
    candidates.to_parquet(patient_scores, index=False)

    knowledge_root = tmp_path / "knowledge"
    (knowledge_root / "opentargets" / "association_by_overall_indirect").mkdir(parents=True, exist_ok=True)

    manifest = run_ipf_rerank(
        patient_scores_parquet=patient_scores,
        final_ranked_parquet=tmp_path / "final_ranked.parquet",
        manifest_json=tmp_path / "rerank_manifest.json",
        summary_json=tmp_path / "summary.json",
        review_csv=tmp_path / "review.csv",
        translation_report_md=tmp_path / "translation_support_report.md",
        download_gap_report_md=tmp_path / "download_gap_report.md",
        knowledge_root=knowledge_root,
    )

    ranked = pd.read_parquet(tmp_path / "final_ranked.parquet")
    assert ranked.iloc[0]["drug_name"] == "Drug_A"
    assert "final_rerank_score" in ranked.columns
    assert (tmp_path / "translation_support_report.md").exists()
    gap_text = (tmp_path / "download_gap_report.md").read_text()
    assert "admet" in gap_text
    assert "clinicaltrials" in gap_text
    assert manifest["stage"] == "rerank_outputs"
