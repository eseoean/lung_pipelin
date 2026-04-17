# IPF GEO Download Workflow

This note documents how to fetch the IPF-specific GEO cohorts that were not present in `s3://say2-4team/Lung_raw/geo/`.

## Target accessions

- `GSE32537`
- `GSE47460`
- `GSE122960`
- `GSE136831`
- `GSE233844`

## Local landing path

Downloaded assets land under:

```text
data/raw/geo/ipf/<ACCESSION>/
```

This directory is intentionally ignored by git.

## Commands

Generate a URL and file-layout plan only:

```bash
make ipf-download-plan
```

Attempt to download GEO landing pages, family soft files, series matrices, and supplementary index pages:

```bash
make ipf-download-geo
```

If you also want supplementary payload files linked from the GEO `suppl/` directory:

```bash
PYTHONPATH=src python3 scripts/download_ipf_geo_sources.py \
  --config configs/ipf.yaml \
  --download-supplementary
```

## Generated planning and reporting artifacts

The script writes planning and reporting files to `docs/ipf/`:

- `ipf_geo_download_plan.json`
- `ipf_geo_download_plan.csv`
- `ipf_geo_download_report.json`
- `ipf_geo_download_report.md`

## Notes

- Bulk GEO accessions are expected to expose `series_matrix` and `family.soft` assets more reliably.
- scRNA GEO accessions often store count matrices or object bundles in the supplementary directory; the helper therefore downloads the supplementary index page by default even when the full payload download is skipped.
- The repository keeps download logic and manifests in git, but not the raw downloaded cohort files themselves.
