from __future__ import annotations

import argparse
from pathlib import Path

from lung_pipeline.config import load_config, repo_root
from lung_pipeline.ipf_pbmc_validation import (
    build_gse233844_pbmc_expression_reference,
    build_gse233844_pbmc_sample_reference,
    default_gse233844_paths,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build GSE233844 PBMC metadata and expression references for IPF external validation."
    )
    parser.add_argument("--config", default="configs/ipf.yaml", help="Path to the IPF config file.")
    parser.add_argument("--series-matrix-path", default=None, help="Override path for the GEO series matrix.")
    parser.add_argument("--filelist-path", default=None, help="Override path for the supplementary file list.")
    parser.add_argument("--raw-tar-path", default=None, help="Override path for the supplementary RAW tar.")
    parser.add_argument(
        "--sample-reference-parquet",
        default="data/processed/disease_context/ipf_gse233844_pbmc_sample_reference.parquet",
        help="Output parquet path for the PBMC sample reference.",
    )
    parser.add_argument(
        "--sample-reference-csv",
        default="docs/ipf/gse233844_pbmc_sample_reference.csv",
        help="Output CSV path for the PBMC sample reference.",
    )
    parser.add_argument(
        "--sample-summary-json",
        default="docs/ipf/gse233844_pbmc_sample_reference_summary.json",
        help="Output JSON path for the PBMC sample reference summary.",
    )
    parser.add_argument(
        "--expression-reference-parquet",
        default="data/processed/disease_context/ipf_gse233844_pbmc_expression_reference.parquet",
        help="Output parquet path for the PBMC expression reference.",
    )
    parser.add_argument(
        "--expression-summary-csv",
        default="docs/ipf/gse233844_pbmc_expression_sample_summary.csv",
        help="Output CSV path for the PBMC expression sample summary.",
    )
    parser.add_argument(
        "--top-genes-csv",
        default="docs/ipf/gse233844_pbmc_expression_top_genes.csv",
        help="Output CSV path for PBMC comparison top genes.",
    )
    parser.add_argument(
        "--expression-summary-json",
        default="docs/ipf/gse233844_pbmc_expression_reference_summary.json",
        help="Output JSON path for the PBMC expression reference summary.",
    )
    parser.add_argument(
        "--top-gene-limit-per-comparison",
        type=int,
        default=150,
        help="Top genes to export for each PBMC comparison.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    root = repo_root(cfg)
    defaults = default_gse233844_paths(root)
    sample_reference_csv_path = root / args.sample_reference_csv

    sample_summary = build_gse233844_pbmc_sample_reference(
        series_matrix_path=Path(args.series_matrix_path) if args.series_matrix_path else defaults["series_matrix"],
        filelist_path=Path(args.filelist_path) if args.filelist_path else defaults["filelist"],
        output_parquet=root / args.sample_reference_parquet,
        output_csv=sample_reference_csv_path,
        summary_json=root / args.sample_summary_json,
    )

    expression_summary = build_gse233844_pbmc_expression_reference(
        sample_reference_path=sample_reference_csv_path,
        raw_tar_path=Path(args.raw_tar_path) if args.raw_tar_path else defaults["raw_tar"],
        output_parquet=root / args.expression_reference_parquet,
        output_csv=root / args.expression_summary_csv,
        top_genes_csv=root / args.top_genes_csv,
        summary_json=root / args.expression_summary_json,
        top_gene_limit_per_comparison=args.top_gene_limit_per_comparison,
    )

    print({"sample_reference": sample_summary, "expression_reference": expression_summary})


if __name__ == "__main__":
    main()
