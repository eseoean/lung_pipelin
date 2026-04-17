from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from lung_pipeline.config import load_config
from lung_pipeline.stages.standardize_tables import run


def test_standardize_tables_builds_expected_outputs(tmp_path: Path) -> None:
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir(parents=True, exist_ok=True)

    gdsc_csv = sources_dir / "GDSC2-dataset.csv"
    pd.DataFrame(
        [
            {
                "DATASET": "GDSC2",
                "COSMIC_ID": 1001,
                "CELL_LINE_NAME": "A549",
                "TCGA_DESC": "LUAD",
                "DRUG_ID": 1,
                "DRUG_NAME": "Gefitinib",
                "LN_IC50": 0.1,
                "PUTATIVE_TARGET": "EGFR",
                "PATHWAY_NAME": "RTK",
                "WEBRELEASE": "Y",
            },
            {
                "DATASET": "GDSC2",
                "COSMIC_ID": 1002,
                "CELL_LINE_NAME": "H1975",
                "TCGA_DESC": "LUSC",
                "DRUG_ID": 2,
                "DRUG_NAME": "Cisplatin",
                "LN_IC50": 0.8,
                "PUTATIVE_TARGET": "DNA",
                "PATHWAY_NAME": "DNA replication",
                "WEBRELEASE": "Y",
            },
            {
                "DATASET": "GDSC2",
                "COSMIC_ID": 9999,
                "CELL_LINE_NAME": "MCF7",
                "TCGA_DESC": "BRCA",
                "DRUG_ID": 3,
                "DRUG_NAME": "Tamoxifen",
                "LN_IC50": 0.3,
                "PUTATIVE_TARGET": "ESR1",
                "PATHWAY_NAME": "Hormone",
                "WEBRELEASE": "Y",
            },
        ]
    ).to_csv(gdsc_csv, index=False)

    compounds_csv = sources_dir / "Compounds-annotation.csv"
    pd.DataFrame(
        [
            {
                "DRUG_ID": 1,
                "DRUG_NAME": "Gefitinib",
                "SCREENING_SITE": "Sanger",
                "SYNONYMS": "Iressa",
                "TARGET": "EGFR",
                "TARGET_PATHWAY": "RTK signaling",
            },
            {
                "DRUG_ID": 2,
                "DRUG_NAME": "Cisplatin",
                "SCREENING_SITE": "Sanger",
                "SYNONYMS": "",
                "TARGET": "DNA",
                "TARGET_PATHWAY": "DNA replication",
            },
        ]
    ).to_csv(compounds_csv, index=False)

    depmap_model_csv = sources_dir / "Model.csv"
    pd.DataFrame(
        [
            {
                "ModelID": "ACH-0001",
                "COSMICID": 1001,
                "CellLineName": "A549",
                "StrippedCellLineName": "A549",
                "CCLEName": "A549_LUNG",
                "OncotreeCode": "LUAD",
                "OncotreePrimaryDisease": "Lung Cancer",
            },
            {
                "ModelID": "ACH-0002",
                "COSMICID": 1002,
                "CellLineName": "H1975",
                "StrippedCellLineName": "H1975",
                "CCLEName": "H1975_LUNG",
                "OncotreeCode": "LUSC",
                "OncotreePrimaryDisease": "Lung Cancer",
            },
        ]
    ).to_csv(depmap_model_csv, index=False)

    exact_drug_catalog = sources_dir / "drug_features_catalog.parquet"
    pd.DataFrame(
        [
            {
                "DRUG_ID": 1,
                "drug_name_norm": "gefitinib",
                "canonical_smiles": "COC1=C(NC2=NC=NC3=CC(OCCCN4CCOCC4)=C(OC)C=C32)C=C(C=C1)Cl",
                "canonical_smiles_raw": "same",
                "match_source": "exact_repo",
                "has_smiles": 1,
            },
            {
                "DRUG_ID": 2,
                "drug_name_norm": "cisplatin",
                "canonical_smiles": "N.N.[Cl-].[Cl-].[Pt+2]",
                "canonical_smiles_raw": "same",
                "match_source": "exact_repo",
                "has_smiles": 1,
            },
        ]
    ).to_parquet(exact_drug_catalog, index=False)

    exact_target_mapping = sources_dir / "drug_target_mapping.parquet"
    pd.DataFrame(
        [
            {"canonical_drug_id": "1", "target_gene_symbol": "EGFR"},
            {"canonical_drug_id": "2", "target_gene_symbol": "DNA"},
        ]
    ).to_parquet(exact_target_mapping, index=False)

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
        "standardization": {
            "cancer_codes": ["LUAD", "LUSC"],
            "binary_quantile": 0.5,
            "cache_dir": ".cache/standardize_tables",
            "sources": {
                "gdsc_dataset": str(gdsc_csv),
                "gdsc_compounds": str(compounds_csv),
                "depmap_model": str(depmap_model_csv),
                "exact_drug_catalog": str(exact_drug_catalog),
                "exact_target_mapping": str(exact_target_mapping),
            },
        },
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    cfg = load_config(config_path)
    manifest = run(cfg, dry_run=False)

    output_dir = tmp_path / "data" / "interim" / "masters"
    assert manifest["status"] == "implemented"
    assert (output_dir / "gdsc_lung_response.parquet").exists()
    assert (output_dir / "depmap_mapping.parquet").exists()
    assert (output_dir / "drug_master.parquet").exists()
    assert (output_dir / "drug_target_mapping.parquet").exists()
    assert (output_dir / "cell_line_master.parquet").exists()
    assert (output_dir / "response_labels.parquet").exists()
    assert (output_dir / "source_crosswalks.json").exists()

    drug_master = pd.read_parquet(output_dir / "drug_master.parquet")
    response_labels = pd.read_parquet(output_dir / "response_labels.parquet")
    cell_master = pd.read_parquet(output_dir / "cell_line_master.parquet")
    crosswalks = json.loads((output_dir / "source_crosswalks.json").read_text(encoding="utf-8"))

    assert drug_master.shape[0] == 2
    assert int(drug_master["has_smiles"].sum()) == 2
    assert response_labels.shape[0] == 2
    assert set(response_labels["model_id"]) == {"ACH-0001", "ACH-0002"}
    assert int(cell_master["is_depmap_mapped"].sum()) == 2
    assert crosswalks["counts"]["mapped_cell_lines"] == 2

