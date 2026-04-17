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
- Supplementary payload queue identified for `14` candidate GEO files

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

## Validation status

- `make test`: passed
- `make ipf-dry-run`: passed
- `scripts/audit_ipf_sources.py --config configs/ipf.yaml`: passed
- `make ipf-download-plan`: passed
- `make ipf-download-geo`: passed

## Next recommended steps

1. Download the remaining supplementary payloads from `docs/ipf/ipf_geo_supplementary_queue.csv`
2. Promote the local GEO downloads into a stable shared raw-data location and update `configs/datasets_ipf.yaml` if the final landing path changes
3. Replace dry-run placeholders with real builders for:
   - bulk IPF signatures
   - scRNA cell-state references
   - pseudo-label generation
   - reversal and network scoring
4. Add accession-aware validation once multiple IPF cohorts are ingested
