from __future__ import annotations

import re
from pathlib import Path

from .source_io import ensure_local_copy


def load_gmt_sets(
    source: str | Path,
    cache_dir: Path,
) -> dict[str, list[str]]:
    path = ensure_local_copy(source, cache_dir)
    pathways: dict[str, list[str]] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        pathway_name, _description, *genes = parts
        deduped = sorted({gene.strip().upper() for gene in genes if gene.strip()})
        if deduped:
            pathways[pathway_name] = deduped
    return pathways


def safe_feature_name(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
    return cleaned or "pathway"
