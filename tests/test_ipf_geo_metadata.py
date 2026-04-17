from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

import pandas as pd

from lung_pipeline.ipf_geo_metadata import build_gse122960_sample_reference


def _write_gzip(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as fh:
        fh.write(text)


def test_build_gse122960_sample_reference(tmp_path: Path) -> None:
    series_path = tmp_path / "GSE122960_series_matrix.txt.gz"
    filelist_path = tmp_path / "filelist.txt"
    output_parquet = tmp_path / "sample_reference.parquet"
    output_csv = tmp_path / "sample_reference.csv"
    summary_json = tmp_path / "sample_reference_summary.json"

    _write_gzip(
        series_path,
        "\n".join(
            [
                '!Sample_title\t"Donor_01"\t"IPF_01"',
                '!Sample_geo_accession\t"GSM1"\t"GSM2"',
                '!Sample_source_name_ch1\t"Lung"\t"Lung"',
                '!Sample_organism_ch1\t"Homo sapiens"\t"Homo sapiens"',
                '!Sample_characteristics_ch1\t"disease condition: Donor"\t"disease condition: Idiopathic pulmonary fibrosis"',
                '!Sample_library_strategy\t"RNA-Seq"\t"RNA-Seq"',
                '!Sample_supplementary_file_1\t"ftp://x/GSM1_filtered.h5"\t"ftp://x/GSM2_filtered.h5"',
                '!Sample_supplementary_file_2\t"ftp://x/GSM1_raw.h5"\t"ftp://x/GSM2_raw.h5"',
            ]
        ),
    )

    with filelist_path.open("w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["#Archive/File", "Name", "Time", "Size", "Type"])
        writer.writerow(["File", "GSM1_filtered.h5", "", "", "H5"])
        writer.writerow(["File", "GSM1_raw.h5", "", "", "H5"])
        writer.writerow(["File", "GSM2_filtered.h5", "", "", "H5"])
        writer.writerow(["File", "GSM2_raw.h5", "", "", "H5"])

    summary = build_gse122960_sample_reference(
        series_matrix_path=series_path,
        filelist_path=filelist_path,
        output_parquet=output_parquet,
        output_csv=output_csv,
        summary_json=summary_json,
    )

    assert summary["sample_count"] == 2
    assert summary["filtered_h5_available_count"] == 2
    assert summary["raw_h5_available_count"] == 2

    frame = pd.read_parquet(output_parquet)
    assert set(frame["disease_bucket"]) == {"Control", "IPF"}
    assert set(frame["gsm_accession"]) == {"GSM1", "GSM2"}

    saved_summary = json.loads(summary_json.read_text())
    assert saved_summary["disease_distribution"]["Donor"] == 1
