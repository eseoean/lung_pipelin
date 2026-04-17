from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from lung_pipeline.ipf_patient_inference import run_ipf_patient_inference


def test_run_ipf_patient_inference(tmp_path: Path) -> None:
    train_table = pd.DataFrame(
        [
            {
                "drugbank_id": f"DB{i:04d}",
                "drug_name": f"Drug_{i}",
                "target_overlap_count": overlap,
                "target_overlap_weight": overlap * 1.2,
                "target_overlap_norm": overlap / 6.0,
                "lincs_support_norm": lincs / 10.0,
                "target_overlap_count_norm": overlap / 6.0,
                "target_overlap_density": density,
                "target_overlap_density_norm": density,
                "specific_overlap_norm": density * 0.5,
                "approval_bonus": float(approved),
                "smiles_bonus": 1.0,
                "fibrosis_priority_overlap_count": fibrosis,
                "fibrosis_priority_overlap_norm": fibrosis / 3.0,
                "broad_target_penalty": penalty,
                "mineral_keyword_penalty": 0.0,
                "inorganic_like_penalty": 0.0,
                "broad_chemistry_penalty": penalty,
                "pseudo_label_score": score,
                "pseudo_label_rank": i + 1,
                "has_lincs_match": int(lincs > 0),
                "is_approved": approved,
                "is_investigational": int(not approved),
            }
            for i, (overlap, lincs, density, approved, fibrosis, penalty, score) in enumerate(
                [
                    (6, 9, 0.9, 1, 3, 0.0, 0.95),
                    (5, 8, 0.8, 1, 2, 0.0, 0.88),
                    (4, 7, 0.7, 0, 2, 0.0, 0.80),
                    (3, 5, 0.6, 1, 1, 0.0, 0.69),
                    (2, 3, 0.4, 0, 1, 0.1, 0.43),
                    (1, 0, 0.2, 0, 0, 0.2, 0.21),
                ]
            )
        ]
    )
    train_table_path = tmp_path / "train_table.parquet"
    train_table.to_parquet(train_table_path, index=False)

    metrics_payload = {
        "best_model": "Ridge",
        "models": [
            {"model_name": "Ridge", "spearman": 0.98},
            {"model_name": "DummyMean", "spearman": 0.10},
        ],
    }
    accession_metrics = tmp_path / "accession.json"
    random_metrics = tmp_path / "random.json"
    accession_metrics.write_text(json.dumps(metrics_payload))
    random_metrics.write_text(json.dumps(metrics_payload))

    manifest = run_ipf_patient_inference(
        train_table_parquet=train_table_path,
        accession_metrics_json=accession_metrics,
        random_metrics_json=random_metrics,
        output_parquet=tmp_path / "patient_ranked_candidates.parquet",
        manifest_json=tmp_path / "patient_manifest.json",
        summary_json=tmp_path / "summary.json",
        review_csv=tmp_path / "review.csv",
        random_seed=42,
    )

    ranked = pd.read_parquet(tmp_path / "patient_ranked_candidates.parquet")
    assert len(ranked) == 6
    assert ranked.iloc[0]["drug_name"] == "Drug_0"
    assert "consensus_model_score" in ranked.columns
    assert manifest["selected_models"]["accessioncv"] == "Ridge"
    assert (tmp_path / "review.csv").exists()
