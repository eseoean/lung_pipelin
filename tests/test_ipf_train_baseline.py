from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from lung_pipeline.ipf_train_baseline import run_ipf_train_baselines


def test_run_ipf_train_baselines(tmp_path: Path) -> None:
    train_table = pd.DataFrame(
        [
            {
                "drugbank_id": f"DB{i:04d}",
                "drug_name": f"Drug_{i}",
                "target_overlap_genes": gene,
                "target_overlap_count": overlap,
                "target_overlap_weight": overlap * 0.8,
                "target_overlap_norm": overlap / 6.0,
                "lincs_signature_count": lincs,
                "lincs_support_norm": lincs / 10.0,
                "target_overlap_count_norm": overlap / 6.0,
                "target_overlap_density": density,
                "target_overlap_density_norm": density,
                "specific_overlap_norm": density * (overlap / 6.0),
                "is_approved": approved,
                "is_investigational": int(not approved),
                "has_lincs_match": int(lincs > 0),
                "smiles_bonus": 1.0,
                "approval_bonus": float(approved),
                "fibrosis_priority_overlap_count": fibrosis,
                "fibrosis_priority_overlap_norm": fibrosis / 3.0,
                "broad_target_penalty": penalty,
                "mineral_keyword_penalty": 0.0,
                "inorganic_like_penalty": 0.0,
                "broad_chemistry_penalty": 0.0,
                "pseudo_label_score": score,
                "pseudo_label_rank": i + 1,
            }
            for i, (gene, overlap, lincs, density, approved, fibrosis, penalty, score) in enumerate(
                [
                    ("GENE_A|GENE_B", 5, 9, 0.9, 1, 3, 0.0, 0.92),
                    ("GENE_A|GENE_C", 4, 8, 0.8, 1, 2, 0.0, 0.84),
                    ("GENE_B|GENE_D", 4, 7, 0.7, 0, 2, 0.0, 0.77),
                    ("GENE_C|GENE_D", 3, 6, 0.6, 1, 1, 0.0, 0.69),
                    ("GENE_E", 3, 5, 0.5, 0, 1, 0.0, 0.61),
                    ("GENE_F", 2, 3, 0.4, 1, 1, 0.0, 0.53),
                    ("GENE_G", 2, 2, 0.3, 0, 0, 0.0, 0.44),
                    ("GENE_H", 1, 1, 0.2, 0, 0, 0.1, 0.33),
                    ("GENE_I", 1, 0, 0.1, 0, 0, 0.2, 0.25),
                    ("GENE_J", 1, 0, 0.1, 1, 0, 0.1, 0.23),
                    ("", 0, 0, 0.0, 0, 0, 0.3, 0.10),
                    ("", 0, 0, 0.0, 0, 0, 0.4, 0.05),
                ]
            )
        ]
    )
    disease_features = pd.DataFrame(
        [
            {"gene_name": "GENE_A", "evidence_sources": "GSE136831|GSE122960"},
            {"gene_name": "GENE_B", "evidence_sources": "GSE136831"},
            {"gene_name": "GENE_C", "evidence_sources": "GSE122960"},
            {"gene_name": "GENE_D", "evidence_sources": "GSE32537"},
            {"gene_name": "GENE_E", "evidence_sources": "GSE233844"},
            {"gene_name": "GENE_F", "evidence_sources": "GSE32537"},
            {"gene_name": "GENE_G", "evidence_sources": "GSE47460"},
            {"gene_name": "GENE_H", "evidence_sources": "GSE47460"},
            {"gene_name": "GENE_I", "evidence_sources": "GSE233844"},
            {"gene_name": "GENE_J", "evidence_sources": "GSE122960"},
        ]
    )

    train_table_path = tmp_path / "train_table.parquet"
    disease_features_path = tmp_path / "disease_features.parquet"
    train_table.to_parquet(train_table_path, index=False)
    disease_features.to_parquet(disease_features_path, index=False)

    summary = run_ipf_train_baselines(
        train_table_parquet=train_table_path,
        disease_features_parquet=disease_features_path,
        random_metrics_json=tmp_path / "randomcv_metrics.json",
        accession_metrics_json=tmp_path / "accessioncv_metrics.json",
        ranking_manifest_json=tmp_path / "ranking_manifest.json",
        random_oof_csv=tmp_path / "random_oof.csv",
        accession_oof_csv=tmp_path / "accession_oof.csv",
        random_seed=42,
        n_splits=3,
    )

    assert summary["row_count"] == 12
    assert summary["feature_count"] > 5
    assert summary["randomcv_best_model"] in {"Ridge", "RandomForest", "HistGradientBoosting", "DummyMean"}
    assert summary["accessioncv_best_model"] in {"Ridge", "RandomForest", "HistGradientBoosting", "DummyMean"}

    random_metrics = json.loads((tmp_path / "randomcv_metrics.json").read_text())
    accession_metrics = json.loads((tmp_path / "accessioncv_metrics.json").read_text())
    assert len(random_metrics["models"]) == 4
    assert len(accession_metrics["models"]) == 4
    assert (tmp_path / "random_oof.csv").exists()
    assert (tmp_path / "accession_oof.csv").exists()
