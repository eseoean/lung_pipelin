from __future__ import annotations

import argparse

from lung_pipeline.config import load_config
from lung_pipeline.stages.build_model_inputs import run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build IPF reversal-oriented model inputs and pseudo-label tables."
    )
    parser.add_argument(
        "--config",
        default="configs/ipf.yaml",
        help="Path to the IPF config file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    summary = run(cfg, dry_run=False)
    print(summary)


if __name__ == "__main__":
    main()
