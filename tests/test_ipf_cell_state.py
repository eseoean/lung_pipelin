from __future__ import annotations

import gzip
import json
from pathlib import Path

import pandas as pd

from lung_pipeline.ipf_cell_state import build_gse136831_cell_state_reference


def _write_gzip(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as fh:
        fh.write(text)


def test_build_gse136831_cell_state_reference(tmp_path: Path) -> None:
    metadata_path = tmp_path / "metadata.txt.gz"
    gene_ids_path = tmp_path / "genes.txt.gz"
    output_parquet = tmp_path / "reference.parquet"
    output_csv = tmp_path / "reference.csv"
    summary_json = tmp_path / "summary.json"

    _write_gzip(
        metadata_path,
        "\n".join(
            [
                '"CellBarcode_Identity"\t"nUMI"\t"nGene"\t"CellType_Category"\t"Manuscript_Identity"\t"Subclass_Cell_Identity"\t"Disease_Identity"\t"Subject_Identity"\t"Library_Identity"',
                '"A1"\t100\t50\t"Myeloid"\t"Macrophage"\t"Alveolar"\t"Control"\t"S1"\t"L1"',
                '"A2"\t200\t70\t"Myeloid"\t"Macrophage"\t"Alveolar"\t"Control"\t"S1"\t"L1"',
                '"B1"\t300\t90\t"Mesenchymal"\t"Myofibroblast"\t"Myofibroblast"\t"IPF"\t"S2"\t"L2"',
            ]
        ),
    )
    _write_gzip(
        gene_ids_path,
        "\n".join(
            [
                '"Ensembl_GeneID"\t"HGNC_EnsemblAlt_GeneID"',
                '"ENSG00000000003"\t"TSPAN6"',
                '"ENSG00000000005"\t"TNMD"',
            ]
        ),
    )

    summary = build_gse136831_cell_state_reference(
        metadata_path=metadata_path,
        gene_ids_path=gene_ids_path,
        output_parquet=output_parquet,
        output_csv=output_csv,
        summary_json=summary_json,
    )

    assert output_parquet.exists()
    assert output_csv.exists()
    assert summary_json.exists()
    assert summary["total_cells"] == 3
    assert summary["cell_state_rows"] == 2
    assert summary["gene_reference_rows"] == 2

    frame = pd.read_parquet(output_parquet)
    assert set(frame["disease_identity"]) == {"Control", "IPF"}
    assert "cell_count" in frame.columns

    saved_summary = json.loads(summary_json.read_text())
    assert saved_summary["top_subclass_distribution"]["Alveolar"] == 2
