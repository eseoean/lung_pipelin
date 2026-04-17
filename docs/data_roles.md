# Data Roles

## 1. Disease Context

- `TCGA LUAD/LUSC`: disease signatures, subtype context, patient features
- `GTEx`: normal-lung baseline
- `GEO`: external disease-cohort validation
- `CPTAC`: proteomics support and pathway validation

## 2. Learning Labels

- `GDSC`: primary response label source
- `PRISM`: auxiliary response and repurposing support

## 3. Cell-Line Context

- `DepMap`: gene dependency and cell-line feature source
- `COSMIC`: mutation and lineage context

## 4. Drug Knowledge

- `DrugBank` and `ChEMBL`: structure and canonicalization
- `DGIdb`, `OncoKB`, `OpenTargets`: target and mechanism support

## 5. Perturbation and Pathway

- `LINCS`: perturbation signatures
- `STRING`: interaction graph support
- `MSigDB`: pathway activity scoring

## 6. Validation and Filtering

- `ADMET`: feasibility and safety filters
- `SIDER`: side-effect plausibility checks
- `ClinicalTrials.gov`: translational support

## IPF Adaptation

On the `ipf` branch, the disease side changes as follows:

- `GEO bulk IPF cohorts` replace TCGA disease cohorts
- `GEO single-cell IPF cohorts` replace the cancer cell-line-centered view
- `GTEx Lung` remains the normal baseline
- direct cell-line drug-response labels are replaced by reversal-oriented pseudo-labels
