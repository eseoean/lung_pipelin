from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import resolve_repo_path, stage_output_dir
from ..io import ensure_dir, utc_now_iso, write_json
from ..schemas import STAGE_CONTRACTS


def build_stage_manifest(
    cfg: dict[str, Any],
    stage_name: str,
    inputs: list[str],
    notes: list[str],
    dry_run: bool,
) -> dict[str, Any]:
    contract = STAGE_CONTRACTS[stage_name]
    output_dir = ensure_dir(stage_output_dir(cfg, stage_name))
    manifest = {
        "stage": stage_name,
        "description": contract["description"],
        "status": "dry_run" if dry_run else "scaffold_ready",
        "generated_at": utc_now_iso(),
        "output_dir": str(output_dir),
        "inputs": inputs,
        "outputs": [str(resolve_repo_path(cfg, item)) for item in contract["outputs"]],
        "notes": notes,
    }
    manifest_path = resolve_repo_path(cfg, f"outputs/manifests/{stage_name}.json")
    write_json(manifest_path, manifest)
    if not dry_run:
        marker = Path(output_dir) / "STAGE_READY.txt"
        marker.write_text(
            "This stage scaffold is ready. Port the implementation into this module.\n"
        )
    return manifest

