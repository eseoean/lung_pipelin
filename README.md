# lung_pipelin

Clean-room repository for rebuilding the lung cancer drug-response pipeline from raw source datasets to model-ready inputs.

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

