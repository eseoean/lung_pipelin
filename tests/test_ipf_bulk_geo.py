from __future__ import annotations

import gzip
import io
import json
import tarfile
from pathlib import Path

import pandas as pd

from lung_pipeline.ipf_bulk_geo import (
    build_gse32537_bulk_reference,
    build_gse47460_bulk_expression_reference,
    build_gse47460_bulk_sample_reference,
)


def _write_gse32537_series_matrix(path: Path) -> None:
    text = """!Sample_title\t"Case 1"\t"Control 1"
!Sample_geo_accession\t"GSM1"\t"GSM2"
!Sample_source_name_ch1\t"Lung tissue"\t"Lung tissue"
!Sample_characteristics_ch1\t"age: 70"\t"age: 60"
!Sample_characteristics_ch1\t"gender: Female"\t"gender: Male"
!Sample_characteristics_ch1\t"final diagnosis: IPF/UIP"\t"final diagnosis: control"
!Sample_characteristics_ch1\t"repository: LTRC"\t"repository: LTRC"
!Sample_characteristics_ch1\t"tissue source: R Lower"\t"tissue source: R Lower"
!Sample_characteristics_ch1\t"preservative: Frozen"\t"preservative: Frozen"
!Sample_characteristics_ch1\t"rin: 8.2"\t"rin: 7.5"
!Sample_characteristics_ch1\t"smoking status: former"\t"smoking status: nonsmoker"
!Sample_characteristics_ch1\t"quit how many years ago: 10"\t"quit how many years ago: nonsmoker"
!Sample_characteristics_ch1\t"pack years: 15"\t"pack years: 0"
!Sample_characteristics_ch1\t"st. george's total score: 45"\t"st. george's total score: 12"
!Sample_characteristics_ch1\t"fvc pre-bronchodilator % predicted: 58"\t"fvc pre-bronchodilator % predicted: 98"
!Sample_characteristics_ch1\t"dlco % predicted: 42"\t"dlco % predicted: 96"
!series_matrix_table_begin
"ID_REF"\t"GSM1"\t"GSM2"
1\t10\t2
2\t1\t9
!series_matrix_table_end
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as fh:
        fh.write(text)


def _write_gse32537_family_soft(path: Path) -> None:
    text = """^PLATFORM = GPL6244
!platform_table_begin
ID\tGB_LIST\tSPOT_ID\tseqname\tRANGE_GB\tRANGE_STRAND\tRANGE_START\tRANGE_STOP\ttotal_probes\tgene_assignment\tmrna_assignment\tcategory
1\t\tprobe1\tchr1\tNC_000001.10\t+\t1\t2\t1\tNM_000001 // GENE_A // desc // loc // 1\tassign\tmain
2\t\tprobe2\tchr1\tNC_000001.10\t+\t3\t4\t1\tNM_000002 // GENE_B // desc // loc // 2\tassign\tmain
!platform_table_end
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as fh:
        fh.write(text)


def _write_gse47460_family_soft(path: Path) -> None:
    text = """^SAMPLE = GSM1149948
!Sample_title = LT000842RU_CTRL
!Sample_source_name_ch1 = Flash frozen whole lung
!Sample_characteristics_ch1 = disease state: Control
!Sample_characteristics_ch1 = Sex: 1-Male
!Sample_characteristics_ch1 = age: 75
!Sample_characteristics_ch1 = smoker?: 2-Ever (>100)
!Sample_characteristics_ch1 = %predicted fev1 (pre-bd): 96
!Sample_characteristics_ch1 = %predicted fvc (pre-bd): 97
!Sample_characteristics_ch1 = %predicted dlco: 78
!Sample_characteristics_ch1 = ild subtype:
^SAMPLE = GSM1149950
!Sample_title = LT001600RL_ILD
!Sample_source_name_ch1 = Flash frozen whole lung
!Sample_characteristics_ch1 = disease state: ILD
!Sample_characteristics_ch1 = Sex: 2-Female
!Sample_characteristics_ch1 = age: 61
!Sample_characteristics_ch1 = smoker?: 1-Never
!Sample_characteristics_ch1 = %predicted fev1 (pre-bd): 70
!Sample_characteristics_ch1 = %predicted fvc (pre-bd): 68
!Sample_characteristics_ch1 = %predicted dlco: 45
!Sample_characteristics_ch1 = ild subtype: IPF
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as fh:
        fh.write(text)


def _write_gse47460_raw_tar(path: Path) -> None:
    sample_a = """FEATURES\tFeatureNum\tControlType\tProbeName\tSystematicName\tgProcessedSignal
DATA\t1\t0\tA_1\tNM_001\t100
DATA\t2\t0\tA_2\tNM_002\t20
DATA\t3\t1\tCTRL\tCTRL\t500
"""
    sample_b = """FEATURES\tFeatureNum\tControlType\tProbeName\tSystematicName\tgProcessedSignal
DATA\t1\t0\tA_1\tNM_001\t10
DATA\t2\t0\tA_2\tNM_002\t200
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w") as tf:
        payloads = {
            "GSM1149948_LT000842RU_CTRL.txt.gz": sample_a,
            "GSM1149950_LT001600RL_ILD.txt.gz": sample_b,
        }
        for name, text in payloads.items():
            raw = gzip.compress(text.encode())
            info = tarfile.TarInfo(name=name)
            info.size = len(raw)
            tf.addfile(info, io.BytesIO(raw))


def test_build_gse32537_bulk_reference(tmp_path: Path) -> None:
    series_matrix = tmp_path / "GSE32537_series_matrix.txt.gz"
    family_soft = tmp_path / "GSE32537_family.soft.gz"
    _write_gse32537_series_matrix(series_matrix)
    _write_gse32537_family_soft(family_soft)

    sample_parquet = tmp_path / "sample.parquet"
    sample_csv = tmp_path / "sample.csv"
    expr_parquet = tmp_path / "expr.parquet"
    expr_csv = tmp_path / "expr.csv"
    top_csv = tmp_path / "top.csv"
    summary_json = tmp_path / "summary.json"

    summary = build_gse32537_bulk_reference(
        series_matrix_path=series_matrix,
        family_soft_path=family_soft,
        sample_reference_parquet=sample_parquet,
        sample_reference_csv=sample_csv,
        expression_summary_parquet=expr_parquet,
        expression_summary_csv=expr_csv,
        top_genes_csv=top_csv,
        summary_json=summary_json,
        top_gene_limit=2,
    )

    assert summary["sample_count"] == 2
    assert summary["ipf_sample_count"] == 1
    assert summary["control_sample_count"] == 1
    top = pd.read_csv(top_csv)
    assert len(top) == 2
    assert "delta_ipf_vs_control" in top.columns
    saved = json.loads(summary_json.read_text())
    assert saved["disease_bucket_distribution"]["IPF"] == 1
    assert saved["disease_bucket_distribution"]["Control"] == 1


def test_build_gse47460_bulk_sample_reference(tmp_path: Path) -> None:
    family_soft = tmp_path / "GSE47460_family.soft.gz"
    _write_gse47460_family_soft(family_soft)

    out_parquet = tmp_path / "gse47460.parquet"
    out_csv = tmp_path / "gse47460.csv"
    summary_json = tmp_path / "gse47460_summary.json"

    summary = build_gse47460_bulk_sample_reference(
        family_soft_path=family_soft,
        output_parquet=out_parquet,
        output_csv=out_csv,
        summary_json=summary_json,
    )

    assert summary["sample_count"] == 2
    assert summary["disease_bucket_distribution"]["Control"] == 1
    assert summary["disease_bucket_distribution"]["IPF"] == 1


def test_build_gse47460_bulk_expression_reference(tmp_path: Path) -> None:
    family_soft = tmp_path / "GSE47460_family.soft.gz"
    _write_gse47460_family_soft(family_soft)
    _write_gse47460_raw_tar(tmp_path / "supplementary" / "GSE47460_RAW.tar")

    sample_csv = tmp_path / "gse47460.csv"
    build_gse47460_bulk_sample_reference(
        family_soft_path=family_soft,
        output_parquet=tmp_path / "gse47460.parquet",
        output_csv=sample_csv,
        summary_json=tmp_path / "gse47460_summary.json",
    )

    summary = build_gse47460_bulk_expression_reference(
        sample_reference_path=sample_csv,
        raw_tar_path=tmp_path / "supplementary" / "GSE47460_RAW.tar",
        output_parquet=tmp_path / "expr.parquet",
        output_csv=tmp_path / "expr.csv",
        top_genes_csv=tmp_path / "top.csv",
        summary_json=tmp_path / "expr_summary.json",
        top_gene_limit=2,
    )

    assert summary["sample_count"] == 2
    assert summary["ipf_sample_count"] == 1
    assert summary["control_sample_count"] == 1
    top = pd.read_csv(tmp_path / "top.csv")
    assert len(top) == 2
    assert set(top["feature_label"]) == {"NM_001", "NM_002"}
