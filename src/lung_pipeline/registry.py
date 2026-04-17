from __future__ import annotations

DEFAULT_DATASET_BUCKETS: dict[str, list[str]] = {
    "disease_context": ["tcga_luad", "tcga_lusc", "gtex_lung", "geo_lung", "cptac_lung"],
    "supervision": ["gdsc", "prism"],
    "cell_context": ["depmap", "cosmic"],
    "drug_knowledge": ["drugbank", "chembl", "dgidb", "oncokb", "opentargets"],
    "perturbation_pathway": ["lincs", "string", "msigdb"],
    "validation_filter": ["admet", "siderside", "clinicaltrials"],
}

DEFAULT_PIPELINE_PRIORITIES: list[str] = [
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

# Backward-compatible aliases for the original scaffold.
DATASET_BUCKETS = DEFAULT_DATASET_BUCKETS
PIPELINE_PRIORITIES = DEFAULT_PIPELINE_PRIORITIES


def dataset_buckets(cfg: dict | None = None) -> dict[str, list[str]]:
    if cfg is None:
        return DEFAULT_DATASET_BUCKETS
    return cfg.get("dataset_buckets", DEFAULT_DATASET_BUCKETS)


def pipeline_priorities(cfg: dict | None = None) -> list[str]:
    if cfg is None:
        return DEFAULT_PIPELINE_PRIORITIES
    return cfg.get("pipeline_priorities", DEFAULT_PIPELINE_PRIORITIES)
