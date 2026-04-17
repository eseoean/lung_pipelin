from __future__ import annotations

DATASET_BUCKETS: dict[str, list[str]] = {
    "disease_context": ["tcga_luad", "tcga_lusc", "gtex_lung", "geo_lung", "cptac_lung"],
    "supervision": ["gdsc", "prism"],
    "cell_context": ["depmap", "cosmic"],
    "drug_knowledge": ["drugbank", "chembl", "dgidb", "oncokb", "opentargets"],
    "perturbation_pathway": ["lincs", "string", "msigdb"],
    "validation_filter": ["admet", "siderside", "clinicaltrials"],
}

PIPELINE_PRIORITIES: list[str] = [
    "Build drug master table",
    "Build cell line master table",
    "Build LUAD/LUSC disease signatures",
    "Build final cell line x drug training table",
]

PIPELINE_STEPS: list[str] = [
    "standardize_tables",
    "build_disease_context",
    "build_model_inputs",
    "train_baseline",
    "patient_inference",
    "rerank_outputs",
]

