from __future__ import annotations

from typing import Any

from ..config import repo_root, resolve_repo_path
from ..io import write_json
from ..ipf_rerank import run_ipf_rerank
from ..registry import dataset_buckets
from ._common import build_stage_manifest, resolve_stage_inputs, resolve_stage_notes


def run(cfg: dict[str, Any], dry_run: bool = False) -> dict[str, Any]:
    buckets = dataset_buckets(cfg)
    default_notes = [
        "Fuse model scores with safety and translational evidence.",
        "ClinicalTrials and ADMET live in the final interpretation layer.",
        "SIDER is reserved for external validation or plausibility checks.",
    ]
    manifest = build_stage_manifest(
        cfg,
        "rerank_outputs",
        resolve_stage_inputs(
            cfg,
            "rerank_outputs",
            buckets.get("validation_filter", []) + ["patient_scores"],
        ),
        resolve_stage_notes(cfg, "rerank_outputs", default_notes),
        dry_run,
    )

    is_ipf = str(cfg.get("study", {}).get("disease", "")).upper() == "IPF"
    if dry_run or not is_ipf:
        return manifest

    root = repo_root(cfg)
    outputs = [
        resolve_repo_path(cfg, raw_path)
        for raw_path in cfg["stage_contracts"]["rerank_outputs"]["outputs"]
    ]
    final_ranked_path, translation_report_path, download_gap_report_path = outputs

    required_inputs = {
        "patient_scores_parquet": str(root / "outputs/patient_inference/patient_ranked_candidates.parquet"),
    }
    missing_inputs = [path for path in required_inputs.values() if not resolve_repo_path(cfg, path).exists()]
    artifacts: dict[str, Any] = {
        "required_inputs": required_inputs,
        "local_knowledge_root": str(root / "data/raw/knowledge"),
    }
    if missing_inputs:
        artifacts["ipf_rerank_outputs"] = {"missing_inputs": missing_inputs}
        manifest["status"] = "partial_built_missing_inputs"
        manifest["artifacts"] = artifacts
        write_json(resolve_repo_path(cfg, "outputs/manifests/rerank_outputs.json"), manifest)
        write_json(root / "docs/ipf/manifests/rerank_outputs.json", manifest)
        return manifest

    rerank_manifest = run_ipf_rerank(
        patient_scores_parquet=root / "outputs/patient_inference/patient_ranked_candidates.parquet",
        final_ranked_parquet=final_ranked_path,
        manifest_json=root / "outputs/reports/rerank_manifest.json",
        summary_json=root / "docs/ipf/ipf_rerank_summary.json",
        review_csv=root / "docs/ipf/ipf_final_ranked_candidates.csv",
        translation_report_md=translation_report_path,
        download_gap_report_md=download_gap_report_path,
        knowledge_root=root / "data/raw/knowledge",
    )

    write_json(
        root / "docs/ipf/manifests/rerank_execution_manifest.json",
        rerank_manifest,
    )
    if translation_report_path.exists():
        (root / "docs/ipf/ipf_translation_support_report.md").write_text(translation_report_path.read_text())
    if download_gap_report_path.exists():
        (root / "docs/ipf/ipf_download_gap_report.md").write_text(download_gap_report_path.read_text())

    artifacts["ipf_rerank_outputs"] = rerank_manifest
    manifest["status"] = "partial_built"
    manifest["artifacts"] = artifacts
    write_json(resolve_repo_path(cfg, "outputs/manifests/rerank_outputs.json"), manifest)
    write_json(root / "docs/ipf/manifests/rerank_outputs.json", manifest)
    return manifest
