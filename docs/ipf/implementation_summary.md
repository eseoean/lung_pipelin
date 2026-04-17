# IPF Branch Work Summary

This branch adapts the lung pipeline scaffold from a lung-cancer response workflow to an idiopathic pulmonary fibrosis (IPF) drug repositioning workflow aligned to the Notion planning page.

## What changed

- Added IPF-specific dataset configuration in [configs/datasets_ipf.yaml](/Users/skku_aws2_18/pre_project/lung_pipelin/configs/datasets_ipf.yaml)
- Added an IPF study configuration in [configs/ipf.yaml](/Users/skku_aws2_18/pre_project/lung_pipelin/configs/ipf.yaml)
- Refactored stage registry and stage contracts so dataset buckets, priorities, stage inputs, and stage notes can be driven by config instead of hardcoded lung-cancer defaults
- Added IPF alignment and data-role docs:
  - [docs/ipf_notion_alignment.md](/Users/skku_aws2_18/pre_project/lung_pipelin/docs/ipf_notion_alignment.md)
  - [docs/ipf_data_roles.md](/Users/skku_aws2_18/pre_project/lung_pipelin/docs/ipf_data_roles.md)
- Added an S3 audit utility for IPF sources:
  - [scripts/audit_ipf_sources.py](/Users/skku_aws2_18/pre_project/lung_pipelin/scripts/audit_ipf_sources.py)
- Added a GEO download helper and workflow note for missing IPF cohorts:
  - [scripts/download_ipf_geo_sources.py](/Users/skku_aws2_18/pre_project/lung_pipelin/scripts/download_ipf_geo_sources.py)
  - [docs/ipf_geo_download_workflow.md](/Users/skku_aws2_18/pre_project/lung_pipelin/docs/ipf_geo_download_workflow.md)
- Added a first real scRNA parser for `GSE136831`:
  - [scripts/build_ipf_cell_state_reference.py](/Users/skku_aws2_18/pre_project/lung_pipelin/scripts/build_ipf_cell_state_reference.py)
  - [src/lung_pipeline/ipf_cell_state.py](/Users/skku_aws2_18/pre_project/lung_pipelin/src/lung_pipeline/ipf_cell_state.py)
- Added an expression-aware sparse-matrix parser for `GSE136831`:
  - [scripts/build_ipf_cell_state_expression_reference.py](/Users/skku_aws2_18/pre_project/lung_pipelin/scripts/build_ipf_cell_state_expression_reference.py)
  - [src/lung_pipeline/ipf_cell_state_expression.py](/Users/skku_aws2_18/pre_project/lung_pipelin/src/lung_pipeline/ipf_cell_state_expression.py)
- Added a GEO sample-metadata parser for `GSE122960`:
  - [scripts/build_ipf_sample_reference.py](/Users/skku_aws2_18/pre_project/lung_pipelin/scripts/build_ipf_sample_reference.py)
  - [src/lung_pipeline/ipf_geo_metadata.py](/Users/skku_aws2_18/pre_project/lung_pipelin/src/lung_pipeline/ipf_geo_metadata.py)
- Added an expression-aware filtered-H5 parser for `GSE122960`:
  - [scripts/build_ipf_sample_expression_reference.py](/Users/skku_aws2_18/pre_project/lung_pipelin/scripts/build_ipf_sample_expression_reference.py)
  - [src/lung_pipeline/ipf_geo_expression.py](/Users/skku_aws2_18/pre_project/lung_pipelin/src/lung_pipeline/ipf_geo_expression.py)
- Added a PBMC external-validation parser for `GSE233844`:
  - [scripts/build_ipf_pbmc_validation_reference.py](/Users/skku_aws2_18/pre_project/lung_pipelin/scripts/build_ipf_pbmc_validation_reference.py)
  - [src/lung_pipeline/ipf_pbmc_validation.py](/Users/skku_aws2_18/pre_project/lung_pipelin/src/lung_pipeline/ipf_pbmc_validation.py)
- Added a first real IPF model-input builder:
  - [scripts/build_ipf_model_inputs.py](/Users/skku_aws2_18/pre_project/lung_pipelin/scripts/build_ipf_model_inputs.py)
  - [src/lung_pipeline/ipf_model_inputs.py](/Users/skku_aws2_18/pre_project/lung_pipelin/src/lung_pipeline/ipf_model_inputs.py)

## Working assumptions from the planning page

- Primary learning setup: disease-signature reversal for IPF rather than cell-line drug-response regression
- Core bulk disease axis:
  - `GSE32537`
  - `GSE47460`
  - `GTEx Lung`
- Core cell-state axis:
  - `GSE122960`
  - `GSE136831`
- External validation axis:
  - `GSE233844`
- Shared knowledge sources reused from the existing bucket:
  - LINCS
  - DrugBank
  - ChEMBL
  - STRING
  - MSigDB
  - Open Targets
  - ADMET
  - SIDER
  - ClinicalTrials

## Audit result

Inventory artifacts were generated under [docs/ipf](/Users/skku_aws2_18/pre_project/lung_pipelin/docs/ipf).

- Available in `s3://say2-4team/Lung_raw/`: `13`
- Missing in bucket and planned for download: `3`

Missing IPF-specific GEO datasets:

- `GSE32537`
- `GSE47460`
- `GSE122960`
- `GSE136831`
- `GSE233844`

See:

- [docs/ipf/ipf_dataset_inventory.md](/Users/skku_aws2_18/pre_project/lung_pipelin/docs/ipf/ipf_dataset_inventory.md)
- [docs/ipf/ipf_missing_downloads.csv](/Users/skku_aws2_18/pre_project/lung_pipelin/docs/ipf/ipf_missing_downloads.csv)

## GEO download status

Planning and download-report artifacts were generated for the missing GEO cohorts.

- [docs/ipf/ipf_geo_download_plan.csv](/Users/skku_aws2_18/pre_project/lung_pipelin/docs/ipf/ipf_geo_download_plan.csv)
- [docs/ipf/ipf_geo_download_report.md](/Users/skku_aws2_18/pre_project/lung_pipelin/docs/ipf/ipf_geo_download_report.md)
- [docs/ipf/ipf_geo_supplementary_queue.csv](/Users/skku_aws2_18/pre_project/lung_pipelin/docs/ipf/ipf_geo_supplementary_queue.csv)

Current local download status:

- Landing pages downloaded for all `5` GEO accessions
- `family.soft.gz` downloaded for all `5` GEO accessions
- `series_matrix.txt.gz` downloaded for `4/5` accessions
- `GSE47460_series_matrix.txt.gz` returned `HTTP 404`
- Small supplementary files downloaded for `7` assets
- Large supplementary payload queue reduced to `0` files
- Large supplementary payloads now present locally for:
  - `GSE32537_RAW.tar`
  - `GSE47460_RAW.tar`
  - `GSE122960_RAW.tar`
  - `GSE136831_RAW.tar`
  - `GSE136831_RawCounts_Sparse.mtx.gz`
  - `GSE233844_RAW.tar`

The downloaded files currently live in the local ignored raw-data path:

- `data/raw/geo/ipf/GSE32537/`
- `data/raw/geo/ipf/GSE47460/`
- `data/raw/geo/ipf/GSE122960/`
- `data/raw/geo/ipf/GSE136831/`
- `data/raw/geo/ipf/GSE233844/`

## Dry-run artifacts

The IPF dry-run manifest and per-stage manifests were saved for review.

- [docs/ipf/ipf_dry_run_manifest.json](/Users/skku_aws2_18/pre_project/lung_pipelin/docs/ipf/ipf_dry_run_manifest.json)
- [docs/ipf/manifests](/Users/skku_aws2_18/pre_project/lung_pipelin/docs/ipf/manifests)

These dry-run outputs confirm the intended IPF stage sequence:

1. Standardize tables and register IPF cohorts
2. Build bulk and scRNA disease context
3. Build reversal-oriented model inputs and pseudo-label tables
4. Train ranking-oriented baselines
5. Score disease signatures
6. Rerank with target, safety, and translation support

## First real disease-context artifacts

Using the downloaded `GSE136831` supplementary files, the branch now builds a metadata-driven scRNA cell-state reference.

- Reviewable table:
  - [docs/ipf/gse136831_cell_state_reference.csv](/Users/skku_aws2_18/pre_project/lung_pipelin/docs/ipf/gse136831_cell_state_reference.csv)
- Summary:
  - [docs/ipf/gse136831_cell_state_reference_summary.json](/Users/skku_aws2_18/pre_project/lung_pipelin/docs/ipf/gse136831_cell_state_reference_summary.json)
- Updated stage manifest:
  - [docs/ipf/manifests/build_disease_context.json](/Users/skku_aws2_18/pre_project/lung_pipelin/docs/ipf/manifests/build_disease_context.json)

Current parser result:

- accession: `GSE136831`
- total cells: `312,928`
- aggregated disease x cell-state rows: `285`
- gene reference rows: `45,947`
- disease distribution:
  - `IPF 147,169`
  - `Control 96,303`
  - `COPD 69,456`

This is intentionally a metadata-driven first pass. Expression-level cell-state signatures still require the remaining large sparse-matrix supplementary payloads.

Using the downloaded `GSE136831_RawCounts_Sparse.mtx.gz`, the branch now also builds an expression-aware cell-state reference and manuscript-level `IPF vs Control` top-gene tables.

- Reviewable group reference:
  - [docs/ipf/gse136831_expression_group_reference.csv](/Users/skku_aws2_18/pre_project/lung_pipelin/docs/ipf/gse136831_expression_group_reference.csv)
- Reviewable top genes:
  - [docs/ipf/gse136831_expression_ipf_vs_control_top_genes.csv](/Users/skku_aws2_18/pre_project/lung_pipelin/docs/ipf/gse136831_expression_ipf_vs_control_top_genes.csv)
- Summary:
  - [docs/ipf/gse136831_expression_reference_summary.json](/Users/skku_aws2_18/pre_project/lung_pipelin/docs/ipf/gse136831_expression_reference_summary.json)
- Updated stage manifest:
  - [docs/ipf/manifests/build_disease_context.json](/Users/skku_aws2_18/pre_project/lung_pipelin/docs/ipf/manifests/build_disease_context.json)

Current expression parser result:

- accession: `GSE136831`
- total cells: `312,928`
- group rows: `285`
- matrix gene rows: `45,947`
- manuscript identities compared in `IPF vs Control`: `38`
- top gene rows exported: `570`

This upgrades `GSE136831` from metadata-only grouping to true pseudobulk-style disease x cell-state signatures anchored to the original sparse count matrix.

Using the downloaded `GSE122960` series-matrix and supplementary file index, the branch now also builds a sample-level scRNA reference for IPF and related ILD cohorts.

- Reviewable table:
  - [docs/ipf/gse122960_sample_reference.csv](/Users/skku_aws2_18/pre_project/lung_pipelin/docs/ipf/gse122960_sample_reference.csv)
- Summary:
  - [docs/ipf/gse122960_sample_reference_summary.json](/Users/skku_aws2_18/pre_project/lung_pipelin/docs/ipf/gse122960_sample_reference_summary.json)
- Updated stage manifest:
  - [docs/ipf/manifests/build_disease_context.json](/Users/skku_aws2_18/pre_project/lung_pipelin/docs/ipf/manifests/build_disease_context.json)

Current parser result:

- accession: `GSE122960`
- sample count: `17`
- disease distribution:
  - `Donor 8`
  - `Idiopathic pulmonary fibrosis 5`
  - `Systemic sclerosis-associated interstitial lung disease 2`
  - `Hypersensitivity pneumonitis 1`
  - `Myositis-associated interstitial lung disease 1`
- disease bucket distribution:
  - `Control 8`
  - `IPF 5`
  - `Other-ILD 4`
- filtered H5 availability: `17/17`
- raw H5 availability: `17/17`

This gives the branch a second real disease-context artifact: `GSE136831` provides cell-state-level scRNA metadata, while `GSE122960` provides cohort/sample-level scRNA context and downloadable raw/filtered H5 pointers.

Using the downloaded `GSE122960_RAW.tar` filtered 10x H5 payloads, the branch now also builds an expression-aware sample summary and an `IPF vs Control` top-gene table.

- Reviewable sample summary:
  - [docs/ipf/gse122960_expression_sample_summary.csv](/Users/skku_aws2_18/pre_project/lung_pipelin/docs/ipf/gse122960_expression_sample_summary.csv)
- Reviewable top genes:
  - [docs/ipf/gse122960_expression_ipf_vs_control_top_genes.csv](/Users/skku_aws2_18/pre_project/lung_pipelin/docs/ipf/gse122960_expression_ipf_vs_control_top_genes.csv)
- Summary:
  - [docs/ipf/gse122960_expression_reference_summary.json](/Users/skku_aws2_18/pre_project/lung_pipelin/docs/ipf/gse122960_expression_reference_summary.json)
- Updated stage manifest:
  - [docs/ipf/manifests/build_disease_context.json](/Users/skku_aws2_18/pre_project/lung_pipelin/docs/ipf/manifests/build_disease_context.json)

Current expression parser result:

- accession: `GSE122960`
- matched filtered H5 files: `17`
- total cells: `80,919`
- median cells per sample: `5,193`
- detected genes in union: `33,694`
- disease bucket distribution:
  - `Control 8`
  - `IPF 5`
  - `Other-ILD 4`
- bucket total UMIs:
  - `Control 325,254,287`
  - `IPF 84,748,890`
  - `Other-ILD 167,902,570`
- top differential genes exported: `200`

This gives the branch a third real disease-context artifact: `GSE122960` now contributes not just metadata, but sample-level pseudobulk expression summaries and a first-pass `IPF vs Control` gene ranking.

Using the downloaded `GSE233844` series matrix and per-sample PBMC sparse matrices, the branch now also builds an external-validation PBMC reference with progression-aware comparisons.

- Reviewable sample reference:
  - [docs/ipf/gse233844_pbmc_sample_reference.csv](/Users/skku_aws2_18/pre_project/lung_pipelin/docs/ipf/gse233844_pbmc_sample_reference.csv)
- Reviewable expression sample summary:
  - [docs/ipf/gse233844_pbmc_expression_sample_summary.csv](/Users/skku_aws2_18/pre_project/lung_pipelin/docs/ipf/gse233844_pbmc_expression_sample_summary.csv)
- Reviewable top genes:
  - [docs/ipf/gse233844_pbmc_expression_top_genes.csv](/Users/skku_aws2_18/pre_project/lung_pipelin/docs/ipf/gse233844_pbmc_expression_top_genes.csv)
- Summaries:
  - [docs/ipf/gse233844_pbmc_sample_reference_summary.json](/Users/skku_aws2_18/pre_project/lung_pipelin/docs/ipf/gse233844_pbmc_sample_reference_summary.json)
  - [docs/ipf/gse233844_pbmc_expression_reference_summary.json](/Users/skku_aws2_18/pre_project/lung_pipelin/docs/ipf/gse233844_pbmc_expression_reference_summary.json)
- Updated stage manifest:
  - [docs/ipf/manifests/build_disease_context.json](/Users/skku_aws2_18/pre_project/lung_pipelin/docs/ipf/manifests/build_disease_context.json)

Current `GSE233844` parser result:

- accession: `GSE233844`
- sample count: `38`
- group distribution:
  - `Progressive IPF 12`
  - `Stable IPF 13`
  - `Control 13`
- total cells: `185,613`
- median cells per sample: `4,925`
- detected genes in union: `26,212`
- comparison count: `3`
  - `IPF-Progressive vs Control`
  - `IPF-Stable vs Control`
  - `IPF-Progressive vs IPF-Stable`
- top gene rows exported: `450`

This gives the branch its first real external-validation scRNA signature set, extending disease-context artifacts beyond discovery cohorts into a progression-aware validation cohort.

Using the downloaded `GSE32537` series matrix and platform table, the branch now also builds a bulk-lung discovery reference with a true `IPF vs Control` expression comparison.

- Reviewable sample reference:
  - [docs/ipf/gse32537_bulk_sample_reference.csv](/Users/skku_aws2_18/pre_project/lung_pipelin/docs/ipf/gse32537_bulk_sample_reference.csv)
- Reviewable expression reference:
  - [docs/ipf/gse32537_bulk_expression_reference.csv](/Users/skku_aws2_18/pre_project/lung_pipelin/docs/ipf/gse32537_bulk_expression_reference.csv)
- Reviewable top genes:
  - [docs/ipf/gse32537_bulk_ipf_vs_control_top_genes.csv](/Users/skku_aws2_18/pre_project/lung_pipelin/docs/ipf/gse32537_bulk_ipf_vs_control_top_genes.csv)
- Summary:
  - [docs/ipf/gse32537_bulk_reference_summary.json](/Users/skku_aws2_18/pre_project/lung_pipelin/docs/ipf/gse32537_bulk_reference_summary.json)

Current `GSE32537` parser result:

- accession: `GSE32537`
- sample count: `217`
- disease bucket distribution:
  - `IPF 119`
  - `Control 50`
  - `Other-IIP 48`
- matrix feature rows: `11,950`
- gene-level rows: `10,992`
- top gene rows exported: `200`

This gives the branch its first real bulk-lung discovery signature and anchors the discovery axis to an interpretable `IPF vs Control` cohort instead of scRNA-only context.

Using the downloaded `GSE47460` family soft, the branch now also builds a bulk-cohort metadata reference spanning control, COPD, IPF, and other ILD subjects.

- Reviewable sample reference:
  - [docs/ipf/gse47460_bulk_sample_reference.csv](/Users/skku_aws2_18/pre_project/lung_pipelin/docs/ipf/gse47460_bulk_sample_reference.csv)
- Summary:
  - [docs/ipf/gse47460_bulk_sample_reference_summary.json](/Users/skku_aws2_18/pre_project/lung_pipelin/docs/ipf/gse47460_bulk_sample_reference_summary.json)

Current `GSE47460` parser result:

- accession: `GSE47460`
- sample count: `582`
- disease state distribution:
  - `Interstitial lung disease 254`
  - `Chronic Obstructive Lung Disease 220`
  - `Control 108`
- disease bucket distribution:
  - `COPD 220`
  - `IPF 160`
  - `Control 108`
  - `Other-ILD 94`

This gives the branch a second real bulk cohort reference and makes the disease-context stage span both discovery bulk cohorts and the scRNA/PBMC validation cohorts described in the planning page.

Using the downloaded `GSE47460_RAW.tar`, the branch now also builds an expression-aware bulk reference and an `IPF vs Control` top-gene table for the same cohort.

- Reviewable expression sample summary:
  - [docs/ipf/gse47460_bulk_expression_reference.csv](/Users/skku_aws2_18/pre_project/lung_pipelin/docs/ipf/gse47460_bulk_expression_reference.csv)
- Reviewable top genes:
  - [docs/ipf/gse47460_bulk_ipf_vs_control_top_genes.csv](/Users/skku_aws2_18/pre_project/lung_pipelin/docs/ipf/gse47460_bulk_ipf_vs_control_top_genes.csv)
- Summary:
  - [docs/ipf/gse47460_bulk_expression_reference_summary.json](/Users/skku_aws2_18/pre_project/lung_pipelin/docs/ipf/gse47460_bulk_expression_reference_summary.json)

Current `GSE47460` expression parser result:

- accession: `GSE47460`
- sample count: `582`
- feature union count: `52,170`
- disease bucket distribution:
  - `COPD 220`
  - `IPF 160`
  - `Control 108`
  - `Other-ILD 94`
- `IPF` samples in comparison: `160`
- `Control` samples in comparison: `108`
- top gene rows exported: `200`

This upgrades `GSE47460` from cohort metadata support to a true bulk expression reference, giving the IPF branch two real bulk discovery axes (`GSE32537`, `GSE47460`) in addition to the scRNA and PBMC validation references.

## First real model-input artifacts

Using the bulk/scRNA/PBMC disease signatures together with downloaded DrugBank and LINCS knowledge sources, the branch now also builds the first reversal-oriented IPF model-input tables.

- Reviewable disease feature table:
  - [docs/ipf/ipf_disease_features.csv](/Users/skku_aws2_18/pre_project/lung_pipelin/docs/ipf/ipf_disease_features.csv)
- Reviewable drug feature table:
  - [docs/ipf/ipf_drug_features.csv](/Users/skku_aws2_18/pre_project/lung_pipelin/docs/ipf/ipf_drug_features.csv)
- Reviewable ranking table:
  - [docs/ipf/ipf_ranking_features.csv](/Users/skku_aws2_18/pre_project/lung_pipelin/docs/ipf/ipf_ranking_features.csv)
- Reviewable train table:
  - [docs/ipf/ipf_train_table.csv](/Users/skku_aws2_18/pre_project/lung_pipelin/docs/ipf/ipf_train_table.csv)
- Reviewable pseudo labels:
  - [docs/ipf/ipf_pseudo_labels.csv](/Users/skku_aws2_18/pre_project/lung_pipelin/docs/ipf/ipf_pseudo_labels.csv)
- Summary:
  - [docs/ipf/ipf_model_inputs_summary.json](/Users/skku_aws2_18/pre_project/lung_pipelin/docs/ipf/ipf_model_inputs_summary.json)
- Updated stage manifest:
  - [docs/ipf/manifests/build_model_inputs.json](/Users/skku_aws2_18/pre_project/lung_pipelin/docs/ipf/manifests/build_model_inputs.json)

Current `build_model_inputs` result:

- disease genes aggregated: `1,085`
- drug candidates retained: `5,292`
- train rows: `5,292`
- pseudo-label rows: `5,292`
- LINCS-matched drugs: `1,750`
- approved drugs: `2,737`
- mean target-overlap count: `0.206`
- max pseudo-label score: `0.80`

Current top-ranked first-pass candidates:

- `Copper`
- `Zinc`
- `Zinc acetate`
- `Zinc chloride`
- `Zinc sulfate, unspecified form`
- `Fostamatinib`
- `Vorinostat`
- `Glutathione`
- `Marimastat`
- `Vitamin A`

This is intentionally a first-pass heuristic ranking. It is already useful as a reproducible input layer, but the ranking still over-prioritizes broad micronutrient / supplement-like agents because the current pseudo label emphasizes target overlap, approval status, and LINCS presence before downstream fibrosis-specific reranking.

## Validation status

- `make test`: passed
- `make ipf-dry-run`: passed
- `scripts/audit_ipf_sources.py --config configs/ipf.yaml`: passed
- `make ipf-download-plan`: passed
- `make ipf-download-geo`: passed
- `make ipf-download-geo-small`: passed
- `make ipf-build-cell-reference`: passed
- `make ipf-build-cell-expression`: passed
- `make ipf-build-sample-reference`: passed
- `make ipf-build-sample-expression`: passed
- `make ipf-build-pbmc-validation`: passed
- `make ipf-build-bulk-references`: passed
- `make ipf-build-model-inputs`: passed

## Next recommended steps

1. Promote the local GEO downloads into a stable shared raw-data location and update `configs/datasets_ipf.yaml` if the final landing path changes
2. Refine pseudo-label scoring so broad ions/supplements do not dominate the ranking ahead of fibrosis-relevant targeted agents
3. Add explicit target/pathway/network fusion using ChEMBL/Open Targets/STRING rather than keeping them as downloaded-but-not-yet-scored sources
4. Move from input construction into `train_baseline` with IPF-specific reversal/ranking objectives and accession-aware validation
