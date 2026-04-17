# Notion Alignment

The current scaffold was aligned to the published planning document.

## Extracted Top-Level Structure

- Six dataset buckets:
  - disease context
  - supervision labels
  - cell-line context
  - drug structure and mechanism
  - perturbation and pathway
  - validation and interpretation
- Six execution steps:
  - standardize tables
  - build disease context
  - build learning tables
  - train models
  - patient inference
  - rerank and postprocess

## Immediate Priorities Reflected in This Repo

1. Build `drug master table`
2. Build `cell line master table`
3. Build `LUAD/LUSC disease signature`
4. Build final `X`, `y` learning table

## How It Maps to This Repository

- `src/lung_pipeline/stages/standardize_tables.py`
  - drug master
  - cell line master
  - response label normalization
- `src/lung_pipeline/stages/build_disease_context.py`
  - disease signatures
  - pathway activity
  - patient-side features
- `src/lung_pipeline/stages/build_model_inputs.py`
  - sample features
  - drug features
  - pair features
  - labels

