from __future__ import annotations

import gzip
import json
from pathlib import Path

import pandas as pd

from lung_pipeline.ipf_cell_state_expression import build_gse136831_expression_reference


def _write_gzip(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as fh:
        fh.write(text)


def test_build_gse136831_expression_reference(tmp_path: Path) -> None:
    metadata_path = tmp_path / "metadata.txt.gz"
    gene_ids_path = tmp_path / "genes.txt.gz"
    barcodes_path = tmp_path / "barcodes.txt.gz"
    matrix_path = tmp_path / "matrix.mtx.gz"
    output_parquet = tmp_path / "expression_reference.parquet"
    output_csv = tmp_path / "expression_reference.csv"
    top_genes_csv = tmp_path / "top_genes.csv"
    summary_json = tmp_path / "summary.json"

    _write_gzip(
        metadata_path,
        "\n".join(
            [
                "\t".join(
                    [
                        "CellBarcode_Identity",
                        "nUMI",
                        "nGene",
                        "CellType_Category",
                        "Manuscript_Identity",
                        "Subclass_Cell_Identity",
                        "Disease_Identity",
                        "Subject_Identity",
                        "Library_Identity",
                    ]
                ),
                "\t".join(["ctrl_1", "10", "2", "Myeloid", "Macrophage", "Macrophage", "Control", "S1", "L1"]),
                "\t".join(["ctrl_2", "9", "2", "Myeloid", "Macrophage", "Macrophage", "Control", "S1", "L1"]),
                "\t".join(["ipf_1", "20", "2", "Myeloid", "Macrophage", "Macrophage", "IPF", "S2", "L2"]),
                "\t".join(["ipf_2", "18", "2", "Myeloid", "Macrophage", "Macrophage", "IPF", "S2", "L2"]),
            ]
        ),
    )

    _write_gzip(
        gene_ids_path,
        "\n".join(
            [
                '"Ensembl_GeneID"\t"HGNC_EnsemblAlt_GeneID"',
                '"ENSG1"\t"GENE_A"',
                '"ENSG2"\t"GENE_B"',
                '"ENSG3"\t"GENE_C"',
            ]
        ),
    )
    _write_gzip(barcodes_path, "ctrl_1\nctrl_2\nipf_1\nipf_2\n")
    _write_gzip(
        matrix_path,
        "\n".join(
            [
                "%%MatrixMarket matrix coordinate integer general",
                "3 4 8",
                "1 1 1",
                "2 1 1",
                "1 2 1",
                "2 2 1",
                "1 3 10",
                "3 3 1",
                "1 4 8",
                "3 4 1",
            ]
        ),
    )

    summary = build_gse136831_expression_reference(
        metadata_path=metadata_path,
        gene_ids_path=gene_ids_path,
        barcodes_path=barcodes_path,
        matrix_path=matrix_path,
        output_parquet=output_parquet,
        output_csv=output_csv,
        top_genes_csv=top_genes_csv,
        summary_json=summary_json,
        top_gene_limit_per_manuscript=2,
    )

    assert summary["total_cells"] == 4
    assert summary["group_rows"] == 2
    assert summary["manuscripts_compared_ipf_vs_control"] == 1

    frame = pd.read_parquet(output_parquet)
    assert set(frame["disease_identity"]) == {"Control", "IPF"}
    assert frame["total_umis"].sum() == 24

    top_genes = pd.read_csv(top_genes_csv)
    assert len(top_genes) == 2
    assert top_genes.iloc[0]["manuscript_identity"] == "Macrophage"

    saved_summary = json.loads(summary_json.read_text())
    assert saved_summary["disease_distribution"]["Control"] == 2
    assert saved_summary["disease_distribution"]["IPF"] == 2
