from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd

from lung_pipeline.ipf_model_inputs import build_ipf_model_inputs


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_drugbank_fixture(path: Path) -> None:
    xml = """<?xml version="1.0" encoding="UTF-8"?>
<drugbank xmlns="http://www.drugbank.ca">
  <drug type="small molecule">
    <drugbank-id primary="true">DB0001</drugbank-id>
    <name>Nintedanib</name>
    <groups><group>approved</group></groups>
    <synonyms><synonym>BIBF 1120</synonym></synonyms>
    <calculated-properties>
      <property><kind>SMILES</kind><value>CCO</value></property>
    </calculated-properties>
    <targets>
      <target><polypeptide><gene-name>MMP7</gene-name></polypeptide></target>
      <target><polypeptide><gene-name>COL1A1</gene-name></polypeptide></target>
    </targets>
    <drug-interactions>
      <drug-interaction>
        <drug>Nested drug that should not be parsed</drug>
      </drug-interaction>
    </drug-interactions>
  </drug>
  <drug type="small molecule">
    <drugbank-id primary="true">DB0002</drugbank-id>
    <name>Pirfenidone</name>
    <groups><group>approved</group></groups>
    <targets>
      <target><polypeptide><gene-name>TGFB1</gene-name></polypeptide></target>
    </targets>
  </drug>
</drugbank>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("full database.xml", xml)


def _write_lincs_fixtures(base: Path) -> None:
    pd.DataFrame(
        [
            {
                "pert_id": "BRD-K0001",
                "pert_iname": "Nintedanib",
                "pert_type": "trt_cp",
                "is_touchstone": 1,
                "canonical_smiles": "CCO",
            }
        ]
    ).to_csv(base / "GSE92742_Broad_LINCS_pert_info.txt.gz", sep="\t", index=False)

    pd.DataFrame(
        [
            {"pert_id": "BRD-K0001", "pert_iname": "Nintedanib", "pert_type": "trt_cp", "cell_id": "A375"},
            {"pert_id": "BRD-K0001", "pert_iname": "Nintedanib", "pert_type": "trt_cp", "cell_id": "MCF7"},
        ]
    ).to_csv(base / "GSE92742_Broad_LINCS_sig_info.txt.gz", sep="\t", index=False)

    pd.DataFrame(
        [{"pr_gene_id": 1, "pr_gene_symbol": "MMP7"}]
    ).to_csv(base / "GSE92742_Broad_LINCS_gene_info.txt.gz", sep="\t", index=False)


def _write_signature_inputs(repo_root: Path) -> None:
    docs = repo_root / "docs" / "ipf"

    _write_csv(
        docs / "gse32537_bulk_ipf_vs_control_top_genes.csv",
        [
            {"feature_label": "MMP7", "delta_ipf_vs_control": 2.4},
            {"feature_label": "COL1A1", "delta_ipf_vs_control": 1.9},
        ],
    )
    _write_csv(
        docs / "gse47460_bulk_ipf_vs_control_top_genes.csv",
        [
            {"feature_label": "MMP7", "delta_ipf_vs_control": 1.8},
            {"feature_label": "CXCL14", "delta_ipf_vs_control": 1.4},
        ],
    )
    _write_csv(
        docs / "gse122960_expression_ipf_vs_control_top_genes.csv",
        [
            {"rank": 1, "gene_id": "ENSG1", "gene_name": "MMP7", "log2_fc_ipf_vs_control": 1.7},
            {"rank": 2, "gene_id": "ENSG2", "gene_name": "COL1A1", "log2_fc_ipf_vs_control": 1.5},
        ],
    )
    _write_csv(
        docs / "gse136831_expression_ipf_vs_control_top_genes.csv",
        [
            {
                "manuscript_identity": "AT2",
                "rank_within_manuscript": 1,
                "gene_id": "ENSG1",
                "gene_name": "MMP7",
                "log2_fc_ipf_vs_control": 2.1,
            }
        ],
    )
    _write_csv(
        docs / "gse233844_pbmc_expression_top_genes.csv",
        [
            {
                "comparison": "IPF-Progressive vs Control",
                "rank_within_comparison": 1,
                "gene_id": "ENSG3",
                "gene_name": "MMP7",
                "log2_fc": 1.2,
            }
        ],
    )


def test_build_ipf_model_inputs_from_synthetic_repo(tmp_path: Path) -> None:
    repo_root = tmp_path
    _write_signature_inputs(repo_root)

    knowledge_root = repo_root / "data" / "raw" / "knowledge"
    chembl_root = knowledge_root / "chembl"
    drugbank_root = knowledge_root / "drugbank"
    lincs_root = knowledge_root / "lincs"
    opentargets_root = knowledge_root / "opentargets" / "association_by_overall_indirect"
    chembl_root.mkdir(parents=True, exist_ok=True)
    drugbank_root.mkdir(parents=True, exist_ok=True)
    lincs_root.mkdir(parents=True, exist_ok=True)
    opentargets_root.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [{"chembl_id": "CHEMBL1", "canonical_smiles": "CCO", "standard_inchi": "", "standard_inchi_key": ""}]
    ).to_csv(chembl_root / "chembl_36_chemreps.txt.gz", sep="\t", index=False)
    (chembl_root / "chembl_uniprot_mapping.txt").write_text("# placeholder\n")
    _write_drugbank_fixture(drugbank_root / "drugbank_all_full_database.xml.zip")
    _write_lincs_fixtures(lincs_root)

    summary = build_ipf_model_inputs(
        repo_root=repo_root,
        disease_features_parquet=repo_root / "out/disease_features.parquet",
        drug_features_parquet=repo_root / "out/drug_features.parquet",
        ranking_features_parquet=repo_root / "out/ranking_features.parquet",
        train_table_parquet=repo_root / "out/train_table.parquet",
        pseudo_labels_parquet=repo_root / "out/pseudo_labels.parquet",
        disease_features_csv=repo_root / "docs/ipf/ipf_disease_features.csv",
        drug_features_csv=repo_root / "docs/ipf/ipf_drug_features.csv",
        ranking_features_csv=repo_root / "docs/ipf/ipf_ranking_features.csv",
        train_table_csv=repo_root / "docs/ipf/ipf_train_table.csv",
        pseudo_labels_csv=repo_root / "docs/ipf/ipf_pseudo_labels.csv",
        summary_json=repo_root / "docs/ipf/ipf_model_inputs_summary.json",
    )

    drug_features = pd.read_csv(repo_root / "docs/ipf/ipf_drug_features.csv")
    pseudo_labels = pd.read_csv(repo_root / "docs/ipf/ipf_pseudo_labels.csv")
    disease_features = pd.read_csv(repo_root / "docs/ipf/ipf_disease_features.csv")

    assert summary["disease_gene_count"] >= 3
    assert summary["drug_candidate_count"] == 2
    assert "Nintedanib" == pseudo_labels.iloc[0]["drug_name"]
    assert "Nested drug that should not be parsed" not in drug_features["drug_name"].tolist()

    nintedanib = drug_features.loc[drug_features["drug_name"] == "Nintedanib"].iloc[0]
    assert int(nintedanib["has_lincs_match"]) == 1
    assert int(nintedanib["target_overlap_count"]) == 2
    assert set(disease_features.head(2)["gene_name"]) >= {"MMP7", "COL1A1"}
