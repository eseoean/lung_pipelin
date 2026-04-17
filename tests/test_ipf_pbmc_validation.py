from __future__ import annotations

import csv
import gzip
import json
import tarfile
from pathlib import Path

import pandas as pd

from lung_pipeline.ipf_pbmc_validation import (
    build_gse233844_pbmc_expression_reference,
    build_gse233844_pbmc_sample_reference,
)


def _write_series_matrix(path: Path) -> None:
    lines = [
        '!Sample_title\t"Progressive IPF patient 1"\t"Control Subject 27"',
        '!Sample_geo_accession\t"GSM7437857"\t"GSM7437882"',
        '!Sample_source_name_ch1\t"PBMCs"\t"PBMCs"',
        '!Sample_characteristics_ch1\t"cell type: PBMCs"\t"cell type: PBMCs"',
        '!Sample_characteristics_ch1\t"group assignment: Progressive IPF"\t"group assignment: Control"',
        '!Sample_characteristics_ch1\t"age (years): 71"\t"age (years): 69"',
        '!Sample_characteristics_ch1\t"sex: Male"\t"sex: Female"',
        '!Sample_characteristics_ch1\t"race/ethnicity: White"\t"race/ethnicity: Asian"',
        '!Sample_characteristics_ch1\t"smoking status: Former"\t"smoking status: Never"',
        '!Sample_characteristics_ch1\t"pack years: 25"\t"pack years: 0"',
        '!Sample_characteristics_ch1\t"DLCO (% predicted): 42"\t"DLCO (% predicted): 91"',
        '!Sample_characteristics_ch1\t"FVC (% predicted): 58"\t"FVC (% predicted): 99"',
        '!Sample_supplementary_file_1\t"ftp://x/GSM7437857_P01_barcodes.tsv.gz"\t"ftp://x/GSM7437882_C27_barcodes.tsv.gz"',
        '!Sample_supplementary_file_2\t"ftp://x/GSM7437857_P01_features.tsv.gz"\t"ftp://x/GSM7437882_C27_features.tsv.gz"',
        '!Sample_supplementary_file_3\t"ftp://x/GSM7437857_P01_matrix.mtx.gz"\t"ftp://x/GSM7437882_C27_matrix.mtx.gz"',
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as fh:
        fh.write("\n".join(lines) + "\n")


def _write_filelist(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["Name", "Size"], delimiter="\t")
        writer.writeheader()
        for name in [
            "GSM7437857_P01_barcodes.tsv.gz",
            "GSM7437857_P01_features.tsv.gz",
            "GSM7437857_P01_matrix.mtx.gz",
            "GSM7437882_C27_barcodes.tsv.gz",
            "GSM7437882_C27_features.tsv.gz",
            "GSM7437882_C27_matrix.mtx.gz",
        ]:
            writer.writerow({"Name": name, "Size": "1"})


def _write_gzip_text(path: Path, text: str) -> None:
    with gzip.open(path, "wt") as fh:
        fh.write(text)


def _write_raw_tar(path: Path) -> None:
    tmp = path.parent / "tmp_contents"
    tmp.mkdir(parents=True, exist_ok=True)
    files = {
        "GSM7437857_P01_barcodes.tsv.gz": "cell1\ncell2\n",
        "GSM7437882_C27_barcodes.tsv.gz": "cell1\ncell2\n",
        "GSM7437857_P01_features.tsv.gz": "ENSG1\tGENE_A\tGene Expression\nENSG2\tGENE_B\tGene Expression\nENSG3\tGENE_C\tGene Expression\n",
        "GSM7437882_C27_features.tsv.gz": "ENSG1\tGENE_A\tGene Expression\nENSG2\tGENE_B\tGene Expression\nENSG3\tGENE_C\tGene Expression\n",
        "GSM7437857_P01_matrix.mtx.gz": "%%MatrixMarket matrix coordinate integer general\n%metadata_json: {\"format_version\": 2}\n3 2 4\n1 1 8\n2 1 2\n2 2 5\n3 2 3\n",
        "GSM7437882_C27_matrix.mtx.gz": "%%MatrixMarket matrix coordinate integer general\n%metadata_json: {\"format_version\": 2}\n3 2 4\n1 1 1\n2 1 1\n2 2 1\n3 2 1\n",
    }
    members: list[Path] = []
    for name, text in files.items():
        file_path = tmp / name
        _write_gzip_text(file_path, text)
        members.append(file_path)
    with tarfile.open(path, "w") as tf:
        for member in members:
            tf.add(member, arcname=member.name)


def test_build_gse233844_pbmc_references(tmp_path: Path) -> None:
    series_matrix_path = tmp_path / "GSE233844_series_matrix.txt.gz"
    filelist_path = tmp_path / "filelist.txt"
    raw_tar_path = tmp_path / "GSE233844_RAW.tar"
    sample_parquet = tmp_path / "pbmc_sample_reference.parquet"
    sample_csv = tmp_path / "pbmc_sample_reference.csv"
    sample_summary_json = tmp_path / "pbmc_sample_reference_summary.json"
    expression_parquet = tmp_path / "pbmc_expression_reference.parquet"
    expression_csv = tmp_path / "pbmc_expression_sample_summary.csv"
    top_genes_csv = tmp_path / "pbmc_expression_top_genes.csv"
    expression_summary_json = tmp_path / "pbmc_expression_reference_summary.json"

    _write_series_matrix(series_matrix_path)
    _write_filelist(filelist_path)
    _write_raw_tar(raw_tar_path)

    sample_summary = build_gse233844_pbmc_sample_reference(
        series_matrix_path=series_matrix_path,
        filelist_path=filelist_path,
        output_parquet=sample_parquet,
        output_csv=sample_csv,
        summary_json=sample_summary_json,
    )
    assert sample_summary["sample_count"] == 2
    assert sample_summary["group_bucket_distribution"]["IPF-Progressive"] == 1
    assert sample_summary["group_bucket_distribution"]["Control"] == 1

    expression_summary = build_gse233844_pbmc_expression_reference(
        sample_reference_path=sample_csv,
        raw_tar_path=raw_tar_path,
        output_parquet=expression_parquet,
        output_csv=expression_csv,
        top_genes_csv=top_genes_csv,
        summary_json=expression_summary_json,
        top_gene_limit_per_comparison=2,
    )
    assert expression_summary["sample_count"] == 2
    assert expression_summary["comparison_count"] == 1
    assert expression_summary["top_gene_rows_exported"] == 2

    sample_frame = pd.read_parquet(sample_parquet)
    assert set(sample_frame["group_bucket"]) == {"IPF-Progressive", "Control"}

    expression_frame = pd.read_parquet(expression_parquet)
    assert expression_frame["cell_count"].sum() == 4

    top_genes = pd.read_csv(top_genes_csv)
    assert set(top_genes["comparison"]) == {"IPF-Progressive vs Control"}
    assert len(top_genes) == 2

    saved_summary = json.loads(expression_summary_json.read_text())
    assert saved_summary["matched_matrix_count"] == 2
    assert saved_summary["group_bucket_distribution"]["IPF-Progressive"] == 1
