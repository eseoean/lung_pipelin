from __future__ import annotations

import pandas as pd
import yaml
from pathlib import Path

from lung_pipeline.config import load_config
from lung_pipeline.stages.build_disease_context import run


def _write_count_file(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("# gene-model: GENCODE v36\n", encoding="utf-8")
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False, mode="a")


def test_build_disease_context_builds_expected_outputs(tmp_path: Path) -> None:
    sources_dir = tmp_path / "sources"
    luad_dir = sources_dir / "luad_counts"
    lusc_dir = sources_dir / "lusc_counts"
    luad_dir.mkdir(parents=True, exist_ok=True)
    lusc_dir.mkdir(parents=True, exist_ok=True)

    _write_count_file(
        luad_dir / "luad_a.rna_seq.augmented_star_gene_counts.tsv",
        [
            {"gene_id": "N_unmapped", "gene_name": None, "gene_type": None, "tpm_unstranded": None},
            {"gene_id": "ENSG1", "gene_name": "EGFR", "gene_type": "protein_coding", "tpm_unstranded": 120.0},
            {"gene_id": "ENSG2", "gene_name": "MUC1", "gene_type": "protein_coding", "tpm_unstranded": 80.0},
            {"gene_id": "ENSG3", "gene_name": "SOX2", "gene_type": "protein_coding", "tpm_unstranded": 5.0},
            {"gene_id": "ENSG4", "gene_name": "TP63", "gene_type": "protein_coding", "tpm_unstranded": 3.0},
            {"gene_id": "ENSG5", "gene_name": "ACTB", "gene_type": "protein_coding", "tpm_unstranded": 40.0},
        ],
    )
    _write_count_file(
        luad_dir / "luad_b.rna_seq.augmented_star_gene_counts.tsv",
        [
            {"gene_id": "ENSG1", "gene_name": "EGFR", "gene_type": "protein_coding", "tpm_unstranded": 90.0},
            {"gene_id": "ENSG2", "gene_name": "MUC1", "gene_type": "protein_coding", "tpm_unstranded": 70.0},
            {"gene_id": "ENSG3", "gene_name": "SOX2", "gene_type": "protein_coding", "tpm_unstranded": 8.0},
            {"gene_id": "ENSG4", "gene_name": "TP63", "gene_type": "protein_coding", "tpm_unstranded": 4.0},
            {"gene_id": "ENSG5", "gene_name": "ACTB", "gene_type": "protein_coding", "tpm_unstranded": 35.0},
        ],
    )
    _write_count_file(
        lusc_dir / "lusc_a.rna_seq.augmented_star_gene_counts.tsv",
        [
            {"gene_id": "ENSG1", "gene_name": "EGFR", "gene_type": "protein_coding", "tpm_unstranded": 8.0},
            {"gene_id": "ENSG2", "gene_name": "MUC1", "gene_type": "protein_coding", "tpm_unstranded": 6.0},
            {"gene_id": "ENSG3", "gene_name": "SOX2", "gene_type": "protein_coding", "tpm_unstranded": 95.0},
            {"gene_id": "ENSG4", "gene_name": "TP63", "gene_type": "protein_coding", "tpm_unstranded": 85.0},
            {"gene_id": "ENSG5", "gene_name": "ACTB", "gene_type": "protein_coding", "tpm_unstranded": 50.0},
        ],
    )
    _write_count_file(
        lusc_dir / "lusc_b.rna_seq.augmented_star_gene_counts.tsv",
        [
            {"gene_id": "ENSG1", "gene_name": "EGFR", "gene_type": "protein_coding", "tpm_unstranded": 10.0},
            {"gene_id": "ENSG2", "gene_name": "MUC1", "gene_type": "protein_coding", "tpm_unstranded": 5.0},
            {"gene_id": "ENSG3", "gene_name": "SOX2", "gene_type": "protein_coding", "tpm_unstranded": 110.0},
            {"gene_id": "ENSG4", "gene_name": "TP63", "gene_type": "protein_coding", "tpm_unstranded": 90.0},
            {"gene_id": "ENSG5", "gene_name": "ACTB", "gene_type": "protein_coding", "tpm_unstranded": 45.0},
        ],
    )

    luad_manifest = sources_dir / "luad_manifest.tsv"
    pd.DataFrame(
        [
            {"id": "luad-file-1", "filename": "luad_a.rna_seq.augmented_star_gene_counts.tsv", "md5": "a", "size": 1},
            {"id": "luad-file-2", "filename": "luad_b.rna_seq.augmented_star_gene_counts.tsv", "md5": "b", "size": 1},
        ]
    ).to_csv(luad_manifest, sep="\t", index=False)
    lusc_manifest = sources_dir / "lusc_manifest.tsv"
    pd.DataFrame(
        [
            {"id": "lusc-file-1", "filename": "lusc_a.rna_seq.augmented_star_gene_counts.tsv", "md5": "a", "size": 1},
            {"id": "lusc-file-2", "filename": "lusc_b.rna_seq.augmented_star_gene_counts.tsv", "md5": "b", "size": 1},
        ]
    ).to_csv(lusc_manifest, sep="\t", index=False)

    hallmark_gmt = sources_dir / "hallmark.gmt"
    hallmark_gmt.write_text(
        "\n".join(
            [
                "HALLMARK_LUAD_SIGNAL\tna\tEGFR\tMUC1",
                "HALLMARK_LUSC_SIGNAL\tna\tSOX2\tTP63",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    oncogenic_gmt = sources_dir / "oncogenic.gmt"
    oncogenic_gmt.write_text(
        "KRAS_SIGNALING_UP\tna\tEGFR\tACTB\n",
        encoding="utf-8",
    )

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
        "disease_context": {
            "cache_dir": ".cache/disease_context",
            "expression_value_column": "tpm_unstranded",
            "gene_types": ["protein_coding"],
            "minimum_pathway_genes": 2,
            "patient_feature_top_k_hallmark": 2,
            "max_files_per_cohort": None,
            "sources": {
                "tcga_luad_counts": str(luad_dir),
                "tcga_lusc_counts": str(lusc_dir),
                "tcga_luad_manifest": str(luad_manifest),
                "tcga_lusc_manifest": str(lusc_manifest),
                "msigdb_hallmark": str(hallmark_gmt),
                "msigdb_oncogenic": str(oncogenic_gmt),
            },
        },
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    cfg = load_config(config_path)
    manifest = run(cfg, dry_run=False)

    output_dir = tmp_path / "data" / "processed" / "disease_context"
    assert manifest["status"] == "implemented"
    assert (output_dir / "luad_signature.parquet").exists()
    assert (output_dir / "lusc_signature.parquet").exists()
    assert (output_dir / "pathway_activity.parquet").exists()
    assert (output_dir / "patient_features.parquet").exists()

    luad_signature = pd.read_parquet(output_dir / "luad_signature.parquet")
    lusc_signature = pd.read_parquet(output_dir / "lusc_signature.parquet")
    pathway_activity = pd.read_parquet(output_dir / "pathway_activity.parquet")
    patient_features = pd.read_parquet(output_dir / "patient_features.parquet")

    egfr_luad = float(luad_signature.loc[luad_signature["gene_symbol"] == "EGFR", "delta_vs_other_cohort"].iloc[0])
    egfr_lusc = float(lusc_signature.loc[lusc_signature["gene_symbol"] == "EGFR", "delta_vs_other_cohort"].iloc[0])
    assert egfr_luad > 0
    assert egfr_lusc < 0

    assert {"HALLMARK_LUAD_SIGNAL", "HALLMARK_LUSC_SIGNAL"} <= set(pathway_activity["pathway_name"])
    assert patient_features.shape[0] == 4
    assert "pathway__hallmark__hallmark_luad_signal" in patient_features.columns
    assert "pathway__hallmark__hallmark_lusc_signal" in patient_features.columns
