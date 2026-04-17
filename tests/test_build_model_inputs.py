from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from lung_pipeline.config import load_config
from lung_pipeline.stages.build_model_inputs import run


def test_build_model_inputs_builds_expected_outputs(tmp_path: Path) -> None:
    masters_dir = tmp_path / "data" / "interim" / "masters"
    disease_dir = tmp_path / "data" / "processed" / "disease_context"
    masters_dir.mkdir(parents=True, exist_ok=True)
    disease_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [
            {
                "sample_id": "A549",
                "cell_line_name": "A549",
                "canonical_drug_id": "1",
                "DRUG_ID": 1,
                "drug_name": "Gefitinib",
                "TCGA_DESC": "LUAD",
                "gdsc_version": "GDSC2",
                "putative_target": "EGFR",
                "pathway_name": "RTK",
                "WEBRELEASE": "Y",
                "label_regression": 0.1,
                "label_binary": 1,
                "label_main": 0.1,
                "label_aux": 1,
                "label_main_type": "regression",
                "label_aux_type": "binary",
                "binary_threshold": 0.5,
                "model_id": "ACH-1",
                "is_depmap_mapped": 1,
                "gdsc_cosmic_id": "1001",
            },
            {
                "sample_id": "H2170",
                "cell_line_name": "H2170",
                "canonical_drug_id": "2",
                "DRUG_ID": 2,
                "drug_name": "SOX2-inhibitor",
                "TCGA_DESC": "LUSC",
                "gdsc_version": "GDSC2",
                "putative_target": "SOX2",
                "pathway_name": "TF",
                "WEBRELEASE": "Y",
                "label_regression": 1.5,
                "label_binary": 0,
                "label_main": 1.5,
                "label_aux": 0,
                "label_main_type": "regression",
                "label_aux_type": "binary",
                "binary_threshold": 0.5,
                "model_id": "ACH-2",
                "is_depmap_mapped": 1,
                "gdsc_cosmic_id": "1002",
            },
        ]
    ).to_parquet(masters_dir / "response_labels.parquet", index=False)

    pd.DataFrame(
        [
            {
                "sample_id": "A549",
                "cell_line_name": "A549",
                "gdsc_cosmic_id": "1001",
                "tcga_desc_values": ["LUAD"],
                "gdsc_versions": ["GDSC2"],
                "model_id": "ACH-1",
                "depmap_cosmic_id": "1001",
                "depmap_cell_line_name": "A549",
                "depmap_stripped_name": "A549",
                "depmap_oncotree_code": "LUAD",
                "depmap_primary_disease": "Lung Cancer",
                "matched_alias": "a549",
                "matched_alias_source": "CellLineName",
                "mapping_rule": "exact",
                "is_depmap_mapped": 1,
            },
            {
                "sample_id": "H2170",
                "cell_line_name": "H2170",
                "gdsc_cosmic_id": "1002",
                "tcga_desc_values": ["LUSC"],
                "gdsc_versions": ["GDSC2"],
                "model_id": "ACH-2",
                "depmap_cosmic_id": "1002",
                "depmap_cell_line_name": "H2170",
                "depmap_stripped_name": "H2170",
                "depmap_oncotree_code": "LUSC",
                "depmap_primary_disease": "Lung Cancer",
                "matched_alias": "h2170",
                "matched_alias_source": "CellLineName",
                "mapping_rule": "exact",
                "is_depmap_mapped": 1,
            },
        ]
    ).to_parquet(masters_dir / "cell_line_master.parquet", index=False)

    pd.DataFrame(
        [
            {
                "canonical_drug_id": "1",
                "DRUG_ID": 1,
                "drug_name": "Gefitinib",
                "drug_name_norm": "gefitinib",
                "match_source": "exact",
                "target_pathway": "RTK",
                "synonyms": "Iressa",
                "canonical_smiles": "CCN",
                "has_smiles": 1,
            },
            {
                "canonical_drug_id": "2",
                "DRUG_ID": 2,
                "drug_name": "SOX2-inhibitor",
                "drug_name_norm": "sox2_inhibitor",
                "match_source": "exact",
                "target_pathway": "TF",
                "synonyms": "",
                "canonical_smiles": "",
                "has_smiles": 0,
            },
        ]
    ).to_parquet(masters_dir / "drug_master.parquet", index=False)

    pd.DataFrame(
        [
            {"canonical_drug_id": "1", "target_gene_symbol": "EGFR"},
            {"canonical_drug_id": "2", "target_gene_symbol": "SOX2"},
        ]
    ).to_parquet(masters_dir / "drug_target_mapping.parquet", index=False)

    pd.DataFrame(
        [
            {"sample_id": "A549", "sample__crispr__EGFR": 0.8, "sample__crispr__KRAS": 0.3},
            {"sample_id": "H2170", "sample__crispr__EGFR": 0.2, "sample__crispr__KRAS": 0.9},
        ]
    ).to_parquet(masters_dir / "sample_crispr_wide.parquet", index=False)

    pd.DataFrame(
        [
            {
                "gene_symbol": "EGFR",
                "cohort": "LUAD",
                "tcga_project_id": "TCGA-LUAD",
                "n_samples": 2,
                "mean_log2_tpm": 8.0,
                "mean_within_sample_z": 1.5,
                "other_cohort_mean_log2_tpm": 2.0,
                "pooled_lung_mean_log2_tpm": 5.0,
                "delta_vs_other_cohort": 6.0,
                "delta_vs_pooled_lung": 3.0,
                "abs_delta_rank": 1,
            },
            {
                "gene_symbol": "MUC1",
                "cohort": "LUAD",
                "tcga_project_id": "TCGA-LUAD",
                "n_samples": 2,
                "mean_log2_tpm": 7.0,
                "mean_within_sample_z": 1.2,
                "other_cohort_mean_log2_tpm": 2.5,
                "pooled_lung_mean_log2_tpm": 4.75,
                "delta_vs_other_cohort": 4.5,
                "delta_vs_pooled_lung": 2.25,
                "abs_delta_rank": 2,
            },
        ]
    ).to_parquet(disease_dir / "luad_signature.parquet", index=False)

    pd.DataFrame(
        [
            {
                "gene_symbol": "SOX2",
                "cohort": "LUSC",
                "tcga_project_id": "TCGA-LUSC",
                "n_samples": 2,
                "mean_log2_tpm": 9.0,
                "mean_within_sample_z": 1.8,
                "other_cohort_mean_log2_tpm": 1.0,
                "pooled_lung_mean_log2_tpm": 5.0,
                "delta_vs_other_cohort": 8.0,
                "delta_vs_pooled_lung": 4.0,
                "abs_delta_rank": 1,
            },
            {
                "gene_symbol": "TP63",
                "cohort": "LUSC",
                "tcga_project_id": "TCGA-LUSC",
                "n_samples": 2,
                "mean_log2_tpm": 8.0,
                "mean_within_sample_z": 1.4,
                "other_cohort_mean_log2_tpm": 1.5,
                "pooled_lung_mean_log2_tpm": 4.75,
                "delta_vs_other_cohort": 6.5,
                "delta_vs_pooled_lung": 3.25,
                "abs_delta_rank": 2,
            },
        ]
    ).to_parquet(disease_dir / "lusc_signature.parquet", index=False)

    pd.DataFrame(
        [
            {
                "collection": "hallmark",
                "pathway_name": "HALLMARK_EGFR_SIGNALING",
                "luad_mean_pathway_score": 1.1,
                "lusc_mean_pathway_score": 0.2,
                "luad_median_pathway_score": 1.1,
                "lusc_median_pathway_score": 0.2,
                "luad_std_pathway_score": 0.1,
                "lusc_std_pathway_score": 0.1,
                "luad_n_samples": 2,
                "lusc_n_samples": 2,
                "luad_mean_genes_used": 20,
                "lusc_mean_genes_used": 20,
                "luad_n_genes_in_set": 20,
                "lusc_n_genes_in_set": 20,
                "delta_luad_minus_lusc": 0.9,
            },
            {
                "collection": "hallmark",
                "pathway_name": "HALLMARK_SQUAMOUS_PROGRAM",
                "luad_mean_pathway_score": 0.1,
                "lusc_mean_pathway_score": 1.4,
                "luad_median_pathway_score": 0.1,
                "lusc_median_pathway_score": 1.4,
                "luad_std_pathway_score": 0.1,
                "lusc_std_pathway_score": 0.1,
                "luad_n_samples": 2,
                "lusc_n_samples": 2,
                "luad_mean_genes_used": 20,
                "lusc_mean_genes_used": 20,
                "luad_n_genes_in_set": 20,
                "lusc_n_genes_in_set": 20,
                "delta_luad_minus_lusc": -1.3,
            },
        ]
    ).to_parquet(disease_dir / "pathway_activity.parquet", index=False)

    config_path = tmp_path / "lung_test.yaml"
    config = {
        "project": {
            "name": "lung-pipelin-test",
            "raw_root": "data/raw",
            "interim_root": "data/interim",
            "processed_root": "data/processed",
            "output_root": "outputs",
        },
        "stages": {
            "standardize_tables": {"output_dir": "data/interim/masters"},
            "build_disease_context": {"output_dir": "data/processed/disease_context"},
            "build_model_inputs": {"output_dir": "data/processed/model_inputs"},
            "train_baseline": {"output_dir": "outputs/model_runs"},
            "patient_inference": {"output_dir": "outputs/patient_inference"},
            "rerank_outputs": {"output_dir": "outputs/reports"},
        },
        "model_inputs": {
            "top_signature_genes_per_cohort": 2,
            "top_pathways_per_collection": 2,
            "filter_to_depmap_mapped": False,
            "include_label_aggregate_features": False,
        },
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    cfg = load_config(config_path)
    manifest = run(cfg, dry_run=False)

    output_dir = tmp_path / "data" / "processed" / "model_inputs"
    assert manifest["status"] == "implemented"
    for name in [
        "sample_features.parquet",
        "drug_features.parquet",
        "pair_features.parquet",
        "train_table.parquet",
        "labels_y.parquet",
    ]:
        assert (output_dir / name).exists()

    sample_features = pd.read_parquet(output_dir / "sample_features.parquet")
    drug_features = pd.read_parquet(output_dir / "drug_features.parquet")
    pair_features = pd.read_parquet(output_dir / "pair_features.parquet")
    train_table = pd.read_parquet(output_dir / "train_table.parquet")

    assert sample_features.shape[0] == 2
    assert drug_features.shape[0] == 2
    assert pair_features.shape[0] == 2
    assert train_table.shape[0] == 2
    assert "ctx__sigdelta__egfr" in sample_features.columns
    assert "ctx__hallmark__hallmark_egfr_signaling" in sample_features.columns
    assert "sample__crispr__EGFR" in sample_features.columns
    assert "sample__has_crispr_profile" in sample_features.columns
    assert "drug__target_count" in drug_features.columns
    assert "drug_morgan_0000" in drug_features.columns
    assert "drug_desc_mol_wt" in drug_features.columns
    assert "drug_has_valid_smiles" in drug_features.columns
    assert "sample__mean_label_regression" not in sample_features.columns

    gefitinib_pair = pair_features.loc[pair_features["canonical_drug_id"] == "1"].iloc[0]
    sox2_pair = pair_features.loc[pair_features["canonical_drug_id"] == "2"].iloc[0]
    assert float(gefitinib_pair["pair__mean_target_delta"]) > 0
    assert float(sox2_pair["pair__mean_target_delta"]) > 0
