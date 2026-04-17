from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from lung_pipeline.config import load_config
from lung_pipeline.stages.build_slim_inputs import run


def test_build_slim_inputs_prunes_feature_blocks(tmp_path: Path) -> None:
    model_input_dir = tmp_path / "data" / "processed" / "model_inputs"
    model_input_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for idx in range(6):
        rows.append(
            {
                "pair_id": f"S{idx % 3}__D{idx // 3}",
                "sample_id": f"S{idx % 3}",
                "canonical_drug_id": f"D{idx // 3}",
                "label_regression": float(idx),
                "label_binary": int(idx % 2 == 0),
                "drug_has_valid_smiles": 1 if idx < 5 else 0,
                "sample__crispr__A": float(idx),
                "sample__crispr__B": float(idx),
                "sample__crispr__C": float(idx % 2),
                "drug_morgan_0000": float(1 if idx < 5 else 0),
                "drug_morgan_0001": float(idx % 2),
                "drug_morgan_0002": 0.0,
                "drug_desc_mw": float(100 + idx),
            }
        )
    pd.DataFrame(rows).to_parquet(model_input_dir / "train_table.parquet", index=False)
    pd.DataFrame(rows)[["pair_id", "sample_id", "canonical_drug_id", "label_regression", "label_binary"]].to_parquet(
        model_input_dir / "labels_y.parquet",
        index=False,
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
            "build_slim_inputs": {"output_dir": "data/processed/model_inputs_slim"},
            "train_baseline": {"output_dir": "outputs/model_runs"},
            "patient_inference": {"output_dir": "outputs/patient_inference"},
            "rerank_outputs": {"output_dir": "outputs/reports"},
        },
        "slim_inputs": {
            "filter_invalid_smiles_rows": True,
            "gene_low_variance_remove_count": 1,
            "morgan_var_threshold": 0.01,
            "correlation_threshold": 0.95,
        },
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    cfg = load_config(config_path)
    manifest = run(cfg, dry_run=False)
    output_dir = tmp_path / "data" / "processed" / "model_inputs_slim"
    summary = json.loads((output_dir / "slim_input_summary.json").read_text())
    slim_table = pd.read_parquet(output_dir / "train_table.parquet")

    assert manifest["status"] == "implemented"
    assert summary["invalid_smiles_rows_removed"] == 1
    assert summary["group_counts_slim"]["gene"] <= 2
    assert summary["group_counts_slim"]["morgan"] <= 1
    assert "sample__crispr__B" not in slim_table.columns
    assert "drug_morgan_0002" not in slim_table.columns
