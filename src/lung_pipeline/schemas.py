from __future__ import annotations

DEFAULT_STAGE_CONTRACTS = {
    "standardize_tables": {
        "description": "Build canonical master tables for drugs, cell lines, labels, and source crosswalks.",
        "outputs": [
            "data/interim/masters/drug_master.parquet",
            "data/interim/masters/cell_line_master.parquet",
            "data/interim/masters/response_labels.parquet",
            "data/interim/masters/source_crosswalks.json",
        ],
    },
    "build_disease_context": {
        "description": "Create LUAD/LUSC disease signatures, pathway activity summaries, and patient-side context tables.",
        "outputs": [
            "data/processed/disease_context/luad_signature.parquet",
            "data/processed/disease_context/lusc_signature.parquet",
            "data/processed/disease_context/pathway_activity.parquet",
            "data/processed/disease_context/patient_features.parquet",
        ],
    },
    "build_model_inputs": {
        "description": "Assemble sample, drug, pair, and training tables for downstream model fitting.",
        "outputs": [
            "data/processed/model_inputs/sample_features.parquet",
            "data/processed/model_inputs/drug_features.parquet",
            "data/processed/model_inputs/pair_features.parquet",
            "data/processed/model_inputs/train_table.parquet",
            "data/processed/model_inputs/labels_y.parquet",
        ],
    },
    "train_baseline": {
        "description": "Run baseline GroupCV and random-split experiments.",
        "outputs": [
            "outputs/model_runs/groupcv_metrics.json",
            "outputs/model_runs/randomcv_metrics.json",
            "outputs/model_runs/oof_manifest.json",
        ],
    },
    "patient_inference": {
        "description": "Score patient-side lung profiles with trained models.",
        "outputs": [
            "outputs/patient_inference/patient_scores.parquet",
            "outputs/patient_inference/patient_manifest.json",
        ],
    },
    "rerank_outputs": {
        "description": "Fuse model scores with biology, safety, and translational filters.",
        "outputs": [
            "outputs/reports/final_ranked_candidates.parquet",
            "outputs/reports/reproducibility_report.md",
            "outputs/reports/clinical_support_report.md",
        ],
    },
}

STAGE_CONTRACTS = DEFAULT_STAGE_CONTRACTS


def get_stage_contract(cfg: dict, stage_name: str) -> dict:
    override = cfg.get("stage_contracts", {}).get(stage_name)
    if override:
        return override
    return DEFAULT_STAGE_CONTRACTS[stage_name]
