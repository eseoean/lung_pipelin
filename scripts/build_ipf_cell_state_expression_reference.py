from __future__ import annotations

import argparse
from pathlib import Path

from lung_pipeline.config import load_config, repo_root
from lung_pipeline.ipf_cell_state_expression import (
    build_gse136831_expression_reference,
    default_gse136831_expression_paths,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an expression-aware GSE136831 cell-state reference from sparse raw counts."
    )
    parser.add_argument("--config", default="configs/ipf.yaml", help="Path to the IPF config file.")
    parser.add_argument("--metadata-path", default=None, help="Override path for GSE136831 metadata.")
    parser.add_argument("--gene-ids-path", default=None, help="Override path for GSE136831 gene map.")
    parser.add_argument("--barcodes-path", default=None, help="Override path for GSE136831 barcode list.")
    parser.add_argument("--matrix-path", default=None, help="Override path for GSE136831 sparse matrix.")
    parser.add_argument(
        "--output-parquet",
        default="data/processed/disease_context/ipf_gse136831_expression_reference.parquet",
        help="Output parquet path for the expression-aware group reference.",
    )
    parser.add_argument(
        "--output-csv",
        default="docs/ipf/gse136831_expression_group_reference.csv",
        help="Output CSV path for the expression-aware group reference.",
    )
    parser.add_argument(
        "--top-genes-csv",
        default="docs/ipf/gse136831_expression_ipf_vs_control_top_genes.csv",
        help="Output CSV path for manuscript-level IPF vs Control top genes.",
    )
    parser.add_argument(
        "--summary-json",
        default="docs/ipf/gse136831_expression_reference_summary.json",
        help="Output JSON path for the expression reference summary.",
    )
    parser.add_argument(
        "--top-gene-limit-per-manuscript",
        type=int,
        default=15,
        help="Top genes to export for each manuscript identity with both IPF and Control cells.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    root = repo_root(cfg)
    defaults = default_gse136831_expression_paths(root)

    summary = build_gse136831_expression_reference(
        metadata_path=Path(args.metadata_path) if args.metadata_path else defaults["metadata"],
        gene_ids_path=Path(args.gene_ids_path) if args.gene_ids_path else defaults["gene_ids"],
        barcodes_path=Path(args.barcodes_path) if args.barcodes_path else defaults["barcodes"],
        matrix_path=Path(args.matrix_path) if args.matrix_path else defaults["raw_counts_matrix"],
        output_parquet=root / args.output_parquet,
        output_csv=root / args.output_csv,
        top_genes_csv=root / args.top_genes_csv,
        summary_json=root / args.summary_json,
        top_gene_limit_per_manuscript=args.top_gene_limit_per_manuscript,
    )
    print(summary)


if __name__ == "__main__":
    main()
