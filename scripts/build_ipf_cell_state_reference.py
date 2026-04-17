from __future__ import annotations

import argparse
from pathlib import Path

from lung_pipeline.config import load_config, repo_root
from lung_pipeline.ipf_cell_state import (
    build_gse136831_cell_state_reference,
    default_gse136831_paths,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a metadata-driven cell-state reference from GSE136831 supplementary files."
    )
    parser.add_argument(
        "--config",
        default="configs/ipf.yaml",
        help="Path to the IPF config file.",
    )
    parser.add_argument(
        "--metadata-path",
        default=None,
        help="Override path for the GSE136831 metadata table.",
    )
    parser.add_argument(
        "--gene-ids-path",
        default=None,
        help="Override path for the GSE136831 gene ID table.",
    )
    parser.add_argument(
        "--output-parquet",
        default="data/processed/disease_context/ipf_cell_state_reference.parquet",
        help="Output parquet path for the aggregated reference table.",
    )
    parser.add_argument(
        "--output-csv",
        default="docs/ipf/gse136831_cell_state_reference.csv",
        help="Output CSV path for a reviewable aggregated reference table.",
    )
    parser.add_argument(
        "--summary-json",
        default="docs/ipf/gse136831_cell_state_reference_summary.json",
        help="Output JSON path for the parser summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    root = repo_root(cfg)

    defaults = default_gse136831_paths(root)
    metadata_path = Path(args.metadata_path) if args.metadata_path else defaults["metadata"]
    gene_ids_path = Path(args.gene_ids_path) if args.gene_ids_path else defaults["gene_ids"]
    output_parquet = root / args.output_parquet
    output_csv = root / args.output_csv
    summary_json = root / args.summary_json

    summary = build_gse136831_cell_state_reference(
        metadata_path=metadata_path,
        gene_ids_path=gene_ids_path,
        output_parquet=output_parquet,
        output_csv=output_csv,
        summary_json=summary_json,
    )
    print(summary)


if __name__ == "__main__":
    main()
