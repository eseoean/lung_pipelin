# lung_pipelin

Clean-room repository for rebuilding pulmonary disease drug-prioritization pipelines from raw source datasets to model-ready inputs.

This repository is organized around the published project plan:

1. Standardize master tables.
2. Build disease context.
3. Build learning tables.
4. Train baseline models.
5. Run patient inference.
6. Re-rank and interpret outputs.

## Dataset Buckets

- Disease context: `TCGA`, `GTEx`, `GEO`, `CPTAC`
- Supervision labels: `GDSC`, `PRISM`
- Cell-line context: `DepMap`, `COSMIC`
- Drug structure and mechanism: `DrugBank`, `ChEMBL`, `DGIdb`, `OncoKB`, `OpenTargets`
- Perturbation and pathway context: `LINCS`, `STRING`, `MSigDB`
- Validation and filtering: `ADMET`, `SIDER`, `ClinicalTrials.gov`

## Immediate Priorities

- Build a `drug master table`
- Build a `cell line master table`
- Build `LUAD/LUSC disease signatures`
- Build the final `cell line x drug` training table (`X`, `y`)

## IPF Branch Notes

The `ipf` branch adapts the scaffold for idiopathic pulmonary fibrosis (IPF):

- bulk IPF GEO cohorts as the disease axis
- scRNA-seq IPF cohorts as the cell-state axis
- pseudo-label ranking instead of direct `ln(IC50)` supervision
- translational reranking with ADMET, SIDER, and ClinicalTrials

Useful commands on the `ipf` branch:

```bash
make dry-run CONFIG=configs/ipf.yaml
PYTHONPATH=src python scripts/audit_ipf_sources.py
make ipf-download-plan
make ipf-download-geo-small
make ipf-build-cell-reference
```

If the S3 bucket is missing required IPF GEO cohorts, use the download helper to plan and fetch official GEO assets into the local raw-data area:

```bash
make ipf-download-geo
```

## Repository Layout

```text
configs/        Pipeline and dataset configuration
docs/           Project docs and data-role notes
scripts/        Thin stage launchers
src/            Pipeline package
tests/          Lightweight config and registry tests
data/           Local runtime data roots (ignored except .gitkeep)
outputs/        Local runtime outputs (ignored except .gitkeep)
```

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
make dry-run
```

## Stage Commands

```bash
make stage-standardize
make stage-disease
make stage-model-inputs
make stage-train
make stage-patient
make stage-rerank
```

## Notes

- The first version in this repo focuses on a clean pipeline skeleton, contracts, and stage manifests.
- Raw data and generated artifacts are intentionally kept out of git.
- Optional sources like `OncoKB` and licensed `COSMIC` assets are configured but can stay disabled until access is ready.
