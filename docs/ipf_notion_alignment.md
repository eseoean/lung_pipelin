# IPF Notion Alignment

This branch adapts the clean-room lung pipeline scaffold to the public IPF planning page.

## Planning Signals Extracted from the Notion Page

### 1. Disease axis

- Bulk IPF cohorts from GEO:
  - `GSE32537`
  - `GSE47460`
- Normal baseline:
  - `GTEx Lung`
- Recommended optional additions:
  - one or two extra bulk IPF cohorts

### 2. Cell-state axis

- Single-cell cohorts:
  - `GSE122960`
  - `GSE136831`
- Intended role:
  - fibroblast and myofibroblast programs
  - epithelial and macrophage state definition
  - deconvolution reference for bulk cohorts
  - cell-type-aware target interpretation

### 3. External validation axis

- Suggested cohort:
  - `GSE233844`
- Validation goal:
  - reproduce tissue signatures in an external blood/PBMC view
  - confirm immune-side reproducibility

### 4. Learning reformulation

The planning page explicitly recommends moving away from the cancer setting:

- Cancer:
  - `row = cell line x drug`
  - `y = ln(IC50)`
- IPF:
  - primary path = disease signature reversal
  - fallback path = small supervised fibrosis-model labels

The branch therefore centers the IPF pipeline on:

- disease signatures
- scRNA-derived cell-state signatures
- LINCS perturbation signatures
- target, pathway, and network features
- pseudo-labels/ranking scores built from reversal + target relevance + network proximity

## What This Branch Adds

- `configs/ipf.yaml`
- `configs/datasets_ipf.yaml`
- config-driven stage contracts and stage notes
- an S3 availability audit for IPF-required datasets
- dry-run manifests for the IPF pipeline shape
