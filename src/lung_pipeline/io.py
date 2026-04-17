from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

