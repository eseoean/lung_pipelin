from __future__ import annotations

from typing import Any

from ..config import repo_root, resolve_repo_path
from ..io import write_json
from ..ipf_patient_inference import run_ipf_patient_inference

from ._common import build_stage_manifest, resolve_stage_inputs, resolve_stage_notes


def run(cfg: dict[str, Any], dry_run: bool = False) -> dict[str, Any]:
    default_notes = [
        "Patient inference consumes disease-context outputs plus trained models.",
        "TCGA patient-side features are the default inference input family.",
    ]
    manifest = build_stage_manifest(
        cfg,
        "patient_inference",
        resolve_stage_inputs(cfg, "patient_inference", ["patient_features", "trained_models"]),
        resolve_stage_notes(cfg, "patient_inference", default_notes),
        dry_run,
    )

    is_ipf = str(cfg.get("study", {}).get("disease", "")).upper() == "IPF"
    if dry_run or not is_ipf:
        return manifest

    root = repo_root(cfg)
    outputs = [
        resolve_repo_path(cfg, raw_path)
        for raw_path in cfg["stage_contracts"]["patient_inference"]["outputs"]
    ]
    output_parquet, manifest_path = outputs

    required_inputs = {
        "train_table_parquet": str(root / "data/processed/model_inputs/train_table.parquet"),
        "accession_metrics_json": str(root / "docs/ipf/ipf_accessioncv_metrics.json"),
        "random_metrics_json": str(root / "docs/ipf/ipf_randomcv_metrics.json"),
    }
    missing_inputs = [path for path in required_inputs.values() if not resolve_repo_path(cfg, path).exists()]
    artifacts: dict[str, Any] = {"required_inputs": required_inputs}
    if missing_inputs:
        artifacts["ipf_patient_inference"] = {"missing_inputs": missing_inputs}
        manifest["status"] = "partial_built_missing_inputs"
        manifest["artifacts"] = artifacts
        write_json(resolve_repo_path(cfg, "outputs/manifests/patient_inference.json"), manifest)
        write_json(root / "docs/ipf/manifests/patient_inference.json", manifest)
        return manifest

    patient_manifest = run_ipf_patient_inference(
        train_table_parquet=root / "data/processed/model_inputs/train_table.parquet",
        accession_metrics_json=root / "docs/ipf/ipf_accessioncv_metrics.json",
        random_metrics_json=root / "docs/ipf/ipf_randomcv_metrics.json",
        output_parquet=output_parquet,
        manifest_json=manifest_path,
        summary_json=root / "docs/ipf/ipf_patient_inference_summary.json",
        review_csv=root / "docs/ipf/ipf_patient_ranked_candidates.csv",
        random_seed=int(cfg.get("study", {}).get("random_seed", 42)),
    )

    artifacts["ipf_patient_inference"] = patient_manifest
    manifest["status"] = "partial_built"
    manifest["artifacts"] = artifacts
    write_json(resolve_repo_path(cfg, "outputs/manifests/patient_inference.json"), manifest)
    write_json(root / "docs/ipf/manifests/patient_inference.json", manifest)
    return manifest
