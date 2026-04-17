from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from lung_pipeline.config import load_config
from lung_pipeline.stages.train_baseline import run


def test_train_baseline_runs_quick_gtex_groupcv_ablation(tmp_path: Path) -> None:
    model_input_dir = tmp_path / "data" / "processed" / "model_inputs"
    model_input_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for drug_idx, drug_id in enumerate(["D1", "D2", "D3"], start=1):
        for sample_idx, sample_id in enumerate(["S1", "S2", "S3"], start=1):
            rows.append(
                {
                    "pair_id": f"{sample_id}__{drug_id}",
                    "sample_id": sample_id,
                    "canonical_drug_id": drug_id,
                    "label_regression": float(drug_idx * 0.5 + sample_idx * 0.1),
                    "label_binary": int(sample_idx % 2 == 0),
                    "numeric_base_1": float(drug_idx + sample_idx),
                    "numeric_base_2": float(drug_idx * sample_idx),
                    "ctx__signature__mean_abs_delta_vs_normal_top": float(sample_idx),
                    "ctx__signormal__egfr": float(sample_idx * 0.5),
                    "pair__mean_target_delta_vs_normal": float(drug_idx * 0.2),
                    "pair__top50_target_hits_vs_normal": float(drug_idx),
                    "pair__cohort": "LUAD" if sample_idx < 3 else "LUSC",
                    "drug_name": f"Drug-{drug_id}",
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
            "train_baseline": {"output_dir": "outputs/model_runs"},
            "patient_inference": {"output_dir": "outputs/patient_inference"},
            "rerank_outputs": {"output_dir": "outputs/reports"},
        },
        "train_baseline": {
            "mode": "quick_gtex_groupcv_ablation",
            "n_splits": 3,
            "random_state": 42,
            "models": ["random_forest", "flat_mlp"],
            "conditions": [
                "no_gtex",
                "sample_summary_only",
                "sample_nonconstant",
                "sample_only",
                "pair_only",
                "pair_plus_summary",
                "pair_plus_top4",
                "pair_plus_nonconstant",
                "full_gtex",
            ],
            "random_forest": {
                "n_estimators": 20,
                "min_samples_leaf": 1,
                "random_state": 42,
                "n_jobs": 1,
            },
            "flat_mlp": {
                "hidden_dims": [16, 8],
                "dropout": 0.0,
                "learning_rate": 0.005,
                "weight_decay": 0.0,
                "batch_size": 4,
                "epochs": 2,
                "random_state": 42,
                "device": "cpu",
            },
        },
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    cfg = load_config(config_path)
    manifest = run(cfg, dry_run=False)

    output_dir = tmp_path / "outputs" / "model_runs"
    assert manifest["status"] == "implemented"
    assert (output_dir / "groupcv_metrics.json").exists()
    assert (output_dir / "randomcv_metrics.json").exists()
    assert (output_dir / "oof_manifest.json").exists()

    metrics = json.loads((output_dir / "groupcv_metrics.json").read_text())
    random_metrics = json.loads((output_dir / "randomcv_metrics.json").read_text())
    expected_conditions = {
        "no_gtex",
        "sample_summary_only",
        "sample_nonconstant",
        "sample_only",
        "pair_only",
        "pair_plus_summary",
        "pair_plus_top4",
        "pair_plus_nonconstant",
        "full_gtex",
    }
    assert set(metrics["models"]) == {"random_forest", "flat_mlp"}
    assert set(metrics["models"]["random_forest"]["conditions"]) == expected_conditions
    assert set(metrics["models"]["flat_mlp"]["conditions"]) == expected_conditions
    assert set(random_metrics["models"]) == {"random_forest", "flat_mlp"}
    assert set(random_metrics["models"]["random_forest"]["conditions"]) == expected_conditions
    assert set(random_metrics["models"]["flat_mlp"]["conditions"]) == expected_conditions
