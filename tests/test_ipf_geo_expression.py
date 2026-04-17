from __future__ import annotations

import json
import tarfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from lung_pipeline.ipf_geo_expression import build_gse122960_expression_reference


def _write_mock_10x_h5(path: Path, data: list[int], indices: list[int], indptr: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        grp = h5.create_group("GRCh38")
        grp.create_dataset("barcodes", data=np.array([b"cell1", b"cell2"]))
        grp.create_dataset("data", data=np.array(data, dtype=np.int32))
        grp.create_dataset("indices", data=np.array(indices, dtype=np.int32))
        grp.create_dataset("indptr", data=np.array(indptr, dtype=np.int32))
        grp.create_dataset("shape", data=np.array([3, 2], dtype=np.int32))
        grp.create_dataset("genes", data=np.array([b"ENSG1", b"ENSG2", b"ENSG3"]))
        grp.create_dataset("gene_names", data=np.array([b"GENE_A", b"GENE_B", b"GENE_C"]))


def test_build_gse122960_expression_reference(tmp_path: Path) -> None:
    sample_reference_path = tmp_path / "gse122960_sample_reference.csv"
    raw_tar_path = tmp_path / "GSE122960_RAW.tar"
    output_parquet = tmp_path / "expression_reference.parquet"
    output_csv = tmp_path / "expression_reference.csv"
    top_genes_csv = tmp_path / "top_genes.csv"
    summary_json = tmp_path / "expression_reference_summary.json"

    sample_reference = pd.DataFrame(
        [
            {
                "accession": "GSE122960",
                "gsm_accession": "GSM1",
                "sample_title": "Donor_01",
                "disease_condition": "Donor",
                "disease_bucket": "Control",
                "filtered_h5_name": "GSM1_Donor_filtered_gene_bc_matrices_h5.h5",
            },
            {
                "accession": "GSE122960",
                "gsm_accession": "GSM2",
                "sample_title": "IPF_01",
                "disease_condition": "Idiopathic pulmonary fibrosis",
                "disease_bucket": "IPF",
                "filtered_h5_name": "GSM2_IPF_filtered_gene_bc_matrices_h5.h5",
            },
        ]
    )
    sample_reference.to_csv(sample_reference_path, index=False)

    donor_h5 = tmp_path / "GSM1_Donor_filtered_gene_bc_matrices_h5.h5"
    ipf_h5 = tmp_path / "GSM2_IPF_filtered_gene_bc_matrices_h5.h5"
    _write_mock_10x_h5(donor_h5, data=[1, 1, 1, 1], indices=[0, 1, 1, 2], indptr=[0, 2, 4])
    _write_mock_10x_h5(ipf_h5, data=[2, 5, 7, 2], indices=[0, 1, 1, 2], indptr=[0, 2, 4])

    with tarfile.open(raw_tar_path, "w") as tf:
        tf.add(donor_h5, arcname=donor_h5.name)
        tf.add(ipf_h5, arcname=ipf_h5.name)

    summary = build_gse122960_expression_reference(
        sample_reference_path=sample_reference_path,
        raw_tar_path=raw_tar_path,
        output_parquet=output_parquet,
        output_csv=output_csv,
        top_genes_csv=top_genes_csv,
        summary_json=summary_json,
        top_gene_limit=3,
    )

    assert summary["sample_count"] == 2
    assert summary["total_cells"] == 4
    assert summary["top_gene_count"] == 3

    frame = pd.read_parquet(output_parquet)
    assert set(frame["disease_bucket"]) == {"Control", "IPF"}
    assert frame["cell_count"].sum() == 4

    top_genes = pd.read_csv(top_genes_csv)
    assert len(top_genes) == 3
    assert "log2_fc_ipf_vs_control" in top_genes.columns

    saved_summary = json.loads(summary_json.read_text())
    assert saved_summary["disease_bucket_distribution"]["Control"] == 1
    assert saved_summary["disease_bucket_distribution"]["IPF"] == 1
