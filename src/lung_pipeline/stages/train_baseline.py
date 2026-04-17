from __future__ import annotations

import json
from typing import Any

from ..config import repo_root, resolve_repo_path
from ..io import write_json
from ..ipf_train_baseline import run_ipf_train_baselines
from ._common import build_stage_manifest, resolve_stage_inputs, resolve_stage_notes


def run(cfg: dict[str, Any], dry_run: bool = False) -> dict[str, Any]:
    default_notes = [
        "Start with simple baseline models before stacking more context.",
        "Use both GroupCV and random-split evaluation tracks.",
        "Group key defaults to canonical_drug_id from configs/lung.yaml.",
    ]
    manifest = build_stage_manifest(
        cfg,
        "train_baseline",
        resolve_stage_inputs(cfg, "train_baseline", ["model_input_table"]),
        resolve_stage_notes(cfg, "train_baseline", default_notes),
        dry_run,
    )

    is_ipf = str(cfg.get("study", {}).get("disease", "")).upper() == "IPF"
    if dry_run or not is_ipf:
        return manifest

    root = repo_root(cfg)
    outputs = [
        resolve_repo_path(cfg, raw_path)
        for raw_path in cfg["stage_contracts"]["train_baseline"]["outputs"]
    ]
    random_metrics_path, accession_metrics_path, ranking_manifest_path = outputs
    train_table_path = root / "data/processed/model_inputs/train_table.parquet"
    disease_features_path = root / "data/processed/model_inputs/disease_features.parquet"

    required_inputs = {
        "train_table_parquet": str(train_table_path),
        "disease_features_parquet": str(disease_features_path),
    }
    missing_inputs = [path for path in required_inputs.values() if not resolve_repo_path(cfg, path).exists()]
    artifacts: dict[str, Any] = {"required_inputs": required_inputs}
    if missing_inputs:
        artifacts["ipf_train_baseline"] = {"missing_inputs": missing_inputs}
        manifest["status"] = "partial_built_missing_inputs"
        manifest["artifacts"] = artifacts
        write_json(resolve_repo_path(cfg, "outputs/manifests/train_baseline.json"), manifest)
        return manifest

    baseline_manifest = run_ipf_train_baselines(
        train_table_parquet=train_table_path,
        disease_features_parquet=disease_features_path,
        random_metrics_json=random_metrics_path,
        accession_metrics_json=accession_metrics_path,
        ranking_manifest_json=ranking_manifest_path,
        random_oof_csv=root / "outputs/model_runs/randomcv_oof_predictions.csv",
        accession_oof_csv=root / "outputs/model_runs/accessioncv_oof_predictions.csv",
        random_seed=int(cfg.get("study", {}).get("random_seed", 42)),
        n_splits=int(cfg.get("study", {}).get("n_splits", 3)),
    )

    random_metrics = json.loads(random_metrics_path.read_text())
    accession_metrics = json.loads(accession_metrics_path.read_text())
    write_json(root / "docs/ipf/ipf_randomcv_metrics.json", random_metrics)
    write_json(root / "docs/ipf/ipf_accessioncv_metrics.json", accession_metrics)
    write_json(
        root / "docs/ipf/ipf_train_baseline_summary.json",
        {
            "study": "IPF",
            "row_count": baseline_manifest["row_count"],
            "feature_count": baseline_manifest["feature_count"],
            "randomcv_best_model": baseline_manifest["randomcv_best_model"],
            "randomcv_best_spearman": baseline_manifest["randomcv_best_spearman"],
            "accessioncv_best_model": baseline_manifest["accessioncv_best_model"],
            "accessioncv_best_spearman": baseline_manifest["accessioncv_best_spearman"],
            "group_distribution": baseline_manifest["group_distribution"],
            "artifacts": baseline_manifest["artifacts"],
        },
    )

    artifacts["ipf_train_baseline"] = baseline_manifest
    manifest["status"] = "partial_built"
    manifest["artifacts"] = artifacts
    write_json(resolve_repo_path(cfg, "outputs/manifests/train_baseline.json"), manifest)
    write_json(root / "docs/ipf/manifests/train_baseline.json", manifest)
    return manifest
