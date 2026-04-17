from __future__ import annotations

import argparse
from pathlib import Path

from lung_pipeline.config import load_config, repo_root
from lung_pipeline.ipf_geo_metadata import (
    build_gse122960_sample_reference,
    default_gse122960_paths,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a sample-level reference table from GSE122960 GEO metadata."
    )
    parser.add_argument(
        "--config",
        default="configs/ipf.yaml",
        help="Path to the IPF config file.",
    )
    parser.add_argument(
        "--series-matrix-path",
        default=None,
        help="Override path for the GSE122960 series matrix.",
    )
    parser.add_argument(
        "--filelist-path",
        default=None,
        help="Override path for the GSE122960 supplementary filelist.",
    )
    parser.add_argument(
        "--output-parquet",
        default="data/processed/disease_context/ipf_gse122960_sample_reference.parquet",
        help="Output parquet path for the sample reference table.",
    )
    parser.add_argument(
        "--output-csv",
        default="docs/ipf/gse122960_sample_reference.csv",
        help="Output CSV path for the reviewable sample reference table.",
    )
    parser.add_argument(
        "--summary-json",
        default="docs/ipf/gse122960_sample_reference_summary.json",
        help="Output JSON path for the parser summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    root = repo_root(cfg)
    defaults = default_gse122960_paths(root)

    series_matrix_path = (
        Path(args.series_matrix_path) if args.series_matrix_path else defaults["series_matrix"]
    )
    filelist_path = Path(args.filelist_path) if args.filelist_path else defaults["filelist"]

    summary = build_gse122960_sample_reference(
        series_matrix_path=series_matrix_path,
        filelist_path=filelist_path,
        output_parquet=root / args.output_parquet,
        output_csv=root / args.output_csv,
        summary_json=root / args.summary_json,
    )
    print(summary)


if __name__ == "__main__":
    main()
