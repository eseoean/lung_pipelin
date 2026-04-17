# IPF Data Roles

## 1. Disease Context

- `GEO bulk IPF cohorts`: disease-vs-normal signal
- `GTEx Lung`: normal baseline
- `GEO external PBMC cohort`: reproducibility and validation

## 2. Cell-State Context

- `GEO single-cell IPF cohorts`: fibrosis-driving cell states, deconvolution, target interpretation

## 3. Drug Knowledge

- `DrugBank` and `ChEMBL`: canonicalization, structures, SMILES
- `OpenTargets`: disease and target context

## 4. Perturbation and Pathway

- `LINCS`: reversal-oriented perturbation signal
- `STRING`: network proximity features
- `MSigDB`: pathway enrichment and fibrosis programs

## 5. Validation and Translation

- `ADMET`: feasibility and safety filters
- `SIDER`: side-effect plausibility checks
- `ClinicalTrials.gov`: translational support

## 6. Transfer-Support Only

- `GDSC`, `DepMap`, `COSMIC`

These remain useful as optional support assets, but not as the natural primary supervision source for IPF.
