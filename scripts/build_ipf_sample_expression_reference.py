from __future__ import annotations

import argparse
from pathlib import Path

from lung_pipeline.config import load_config, repo_root
from lung_pipeline.ipf_geo_expression import (
    build_gse122960_expression_reference,
    default_gse122960_expression_paths,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an expression-aware GSE122960 reference from filtered 10x H5 files."
    )
    parser.add_argument(
        "--config",
        default="configs/ipf.yaml",
        help="Path to the IPF config file.",
    )
    parser.add_argument(
        "--sample-reference-path",
        default=None,
        help="Override path for the GSE122960 sample reference table.",
    )
    parser.add_argument(
        "--raw-tar-path",
        default=None,
        help="Override path for the GSE122960 RAW tar archive.",
    )
    parser.add_argument(
        "--output-parquet",
        default="data/processed/disease_context/ipf_gse122960_expression_reference.parquet",
        help="Output parquet path for the expression-aware sample summary.",
    )
    parser.add_argument(
        "--output-csv",
        default="docs/ipf/gse122960_expression_sample_summary.csv",
        help="Output CSV path for the reviewable expression sample summary.",
    )
    parser.add_argument(
        "--top-genes-csv",
        default="docs/ipf/gse122960_expression_ipf_vs_control_top_genes.csv",
        help="Output CSV path for IPF-vs-Control top genes.",
    )
    parser.add_argument(
        "--summary-json",
        default="docs/ipf/gse122960_expression_reference_summary.json",
        help="Output JSON path for the expression reference summary.",
    )
    parser.add_argument(
        "--top-gene-limit",
        type=int,
        default=200,
        help="Number of top differential genes to keep in the review CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    root = repo_root(cfg)
    defaults = default_gse122960_expression_paths(root)

    sample_reference_path = (
        Path(args.sample_reference_path)
        if args.sample_reference_path
        else defaults["sample_reference"]
    )
    raw_tar_path = Path(args.raw_tar_path) if args.raw_tar_path else defaults["raw_tar"]

    summary = build_gse122960_expression_reference(
        sample_reference_path=sample_reference_path,
        raw_tar_path=raw_tar_path,
        output_parquet=root / args.output_parquet,
        output_csv=root / args.output_csv,
        top_genes_csv=root / args.top_genes_csv,
        summary_json=root / args.summary_json,
        top_gene_limit=args.top_gene_limit,
    )
    print(summary)


if __name__ == "__main__":
    main()
