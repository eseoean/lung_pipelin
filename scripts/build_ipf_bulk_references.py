from __future__ import annotations

import argparse
from pathlib import Path

from lung_pipeline.config import load_config, repo_root
from lung_pipeline.ipf_bulk_geo import (
    build_gse32537_bulk_reference,
    build_gse47460_bulk_sample_reference,
    default_gse32537_paths,
    default_gse47460_paths,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build IPF bulk GEO references for GSE32537 and GSE47460."
    )
    parser.add_argument("--config", default="configs/ipf.yaml", help="Path to the IPF config file.")
    parser.add_argument(
        "--gse32537-top-gene-limit",
        type=int,
        default=200,
        help="Top genes to export for GSE32537 IPF vs Control comparison.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    root = repo_root(cfg)

    gse32537_paths = default_gse32537_paths(root)
    gse47460_paths = default_gse47460_paths(root)

    gse32537_summary = build_gse32537_bulk_reference(
        series_matrix_path=gse32537_paths["series_matrix"],
        family_soft_path=gse32537_paths["family_soft"],
        sample_reference_parquet=root / "data/processed/disease_context/ipf_gse32537_bulk_sample_reference.parquet",
        sample_reference_csv=root / "docs/ipf/gse32537_bulk_sample_reference.csv",
        expression_summary_parquet=root / "data/processed/disease_context/ipf_gse32537_bulk_expression_reference.parquet",
        expression_summary_csv=root / "docs/ipf/gse32537_bulk_expression_reference.csv",
        top_genes_csv=root / "docs/ipf/gse32537_bulk_ipf_vs_control_top_genes.csv",
        summary_json=root / "docs/ipf/gse32537_bulk_reference_summary.json",
        top_gene_limit=args.gse32537_top_gene_limit,
    )

    gse47460_summary = build_gse47460_bulk_sample_reference(
        family_soft_path=gse47460_paths["family_soft"],
        output_parquet=root / "data/processed/disease_context/ipf_gse47460_bulk_sample_reference.parquet",
        output_csv=root / "docs/ipf/gse47460_bulk_sample_reference.csv",
        summary_json=root / "docs/ipf/gse47460_bulk_sample_reference_summary.json",
    )

    print({"gse32537_bulk_reference": gse32537_summary, "gse47460_bulk_sample_reference": gse47460_summary})


if __name__ == "__main__":
    main()
