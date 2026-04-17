from __future__ import annotations

import argparse
import json
from typing import Callable

from .config import load_config
from .registry import PIPELINE_STEPS
from .stages import (
    build_disease_context,
    build_model_inputs,
    patient_inference,
    rerank_outputs,
    standardize_tables,
    train_baseline,
)

RUNNERS: dict[str, Callable] = {
    "standardize_tables": standardize_tables.run,
    "build_disease_context": build_disease_context.run,
    "build_model_inputs": build_model_inputs.run,
    "train_baseline": train_baseline.run,
    "patient_inference": patient_inference.run,
    "rerank_outputs": rerank_outputs.run,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lung pipeline stage runner")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument(
        "--stage",
        default="all",
        choices=["all", *PIPELINE_STEPS],
        help="Stage to run",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write stage manifests without marking stage directories ready",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    stages = PIPELINE_STEPS if args.stage == "all" else [args.stage]
    results = [RUNNERS[name](cfg, dry_run=args.dry_run) for name in stages]
    print(json.dumps({"config": args.config, "stages": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

