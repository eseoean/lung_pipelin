from __future__ import annotations

from typing import Any

from ..config import repo_root, resolve_repo_path
from ..io import write_json
from ..ipf_model_inputs import (
    build_ipf_model_inputs,
    default_model_input_paths,
    default_model_input_signature_paths,
)
from ..registry import dataset_buckets
from ._common import build_stage_manifest, resolve_stage_inputs, resolve_stage_notes


def run(cfg: dict[str, Any], dry_run: bool = False) -> dict[str, Any]:
    buckets = dataset_buckets(cfg)
    default_inputs = (
        buckets.get("supervision", [])
        + buckets.get("cell_context", [])
        + buckets.get("drug_knowledge", [])
        + buckets.get("perturbation_pathway", [])
        + ["disease_context_outputs"]
    )
    default_notes = [
        "Final learning unit is cell line x drug.",
        "Separate outputs into sample X, drug X, pair X, and labels y.",
        "LINCS is marked as very important in the planning doc.",
    ]
    manifest = build_stage_manifest(
        cfg,
        "build_model_inputs",
        resolve_stage_inputs(cfg, "build_model_inputs", default_inputs),
        resolve_stage_notes(cfg, "build_model_inputs", default_notes),
        dry_run,
    )

    is_ipf = str(cfg.get("study", {}).get("disease", "")).upper() == "IPF"
    if dry_run or not is_ipf:
        return manifest

    root = repo_root(cfg)
    knowledge_paths = default_model_input_paths(root)
    signature_paths = default_model_input_signature_paths(root)

    required_inputs = {**knowledge_paths, **signature_paths}
    missing_inputs = [str(path) for path in required_inputs.values() if not path.exists()]

    outputs = [
        resolve_repo_path(cfg, raw_path)
        for raw_path in cfg["stage_contracts"]["build_model_inputs"]["outputs"]
    ]
    disease_features_path, drug_features_path, ranking_features_path, train_table_path, pseudo_labels_path = outputs

    artifacts: dict[str, Any] = {
        "required_inputs": {name: str(path) for name, path in required_inputs.items()},
    }

    if missing_inputs:
        artifacts["ipf_model_inputs"] = {
            "missing_inputs": missing_inputs,
        }
        manifest["status"] = "partial_built_missing_inputs"
        manifest["artifacts"] = artifacts
        write_json(resolve_repo_path(cfg, "outputs/manifests/build_model_inputs.json"), manifest)
        write_json(root / "docs/ipf/manifests/build_model_inputs.json", manifest)
        return manifest

    summary = build_ipf_model_inputs(
        repo_root=root,
        disease_features_parquet=disease_features_path,
        drug_features_parquet=drug_features_path,
        ranking_features_parquet=ranking_features_path,
        train_table_parquet=train_table_path,
        pseudo_labels_parquet=pseudo_labels_path,
        disease_features_csv=root / "docs/ipf/ipf_disease_features.csv",
        drug_features_csv=root / "docs/ipf/ipf_drug_features.csv",
        ranking_features_csv=root / "docs/ipf/ipf_ranking_features.csv",
        train_table_csv=root / "docs/ipf/ipf_train_table.csv",
        pseudo_labels_csv=root / "docs/ipf/ipf_pseudo_labels.csv",
        summary_json=root / "docs/ipf/ipf_model_inputs_summary.json",
    )

    artifacts["ipf_model_inputs"] = {
        "source_mode": "signature_reversal_pseudo_labels",
        "disease_features_parquet": str(disease_features_path),
        "drug_features_parquet": str(drug_features_path),
        "ranking_features_parquet": str(ranking_features_path),
        "train_table_parquet": str(train_table_path),
        "pseudo_labels_parquet": str(pseudo_labels_path),
        "disease_features_csv": str(root / "docs/ipf/ipf_disease_features.csv"),
        "drug_features_csv": str(root / "docs/ipf/ipf_drug_features.csv"),
        "ranking_features_csv": str(root / "docs/ipf/ipf_ranking_features.csv"),
        "train_table_csv": str(root / "docs/ipf/ipf_train_table.csv"),
        "pseudo_labels_csv": str(root / "docs/ipf/ipf_pseudo_labels.csv"),
        "summary_json": str(root / "docs/ipf/ipf_model_inputs_summary.json"),
        **summary,
    }
    manifest["status"] = "partial_built"
    manifest["artifacts"] = artifacts
    write_json(resolve_repo_path(cfg, "outputs/manifests/build_model_inputs.json"), manifest)
    write_json(root / "docs/ipf/manifests/build_model_inputs.json", manifest)
    return manifest
