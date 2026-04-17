from __future__ import annotations

import subprocess
import time
from pathlib import Path


def ensure_local_copy(uri_or_path: str | Path, cache_dir: Path) -> Path:
    value = str(uri_or_path).strip()
    if not value:
        raise ValueError("Empty source path/URI is not allowed.")
    if value.startswith("s3://"):
        cache_dir.mkdir(parents=True, exist_ok=True)
        local_path = cache_dir / Path(value).name
        if local_path.exists() and local_path.stat().st_size > 0:
            return local_path
        last_error: subprocess.CalledProcessError | None = None
        for attempt in range(4):
            try:
                subprocess.run(["aws", "s3", "cp", value, str(local_path)], check=True)
                if local_path.exists() and local_path.stat().st_size > 0:
                    return local_path
            except subprocess.CalledProcessError as exc:
                last_error = exc
                if local_path.exists():
                    local_path.unlink()
                if attempt < 3:
                    time.sleep(2 * (attempt + 1))
        if last_error is not None:
            raise last_error
        return local_path
    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Source file not found: {path}")
    return path


def maybe_local_copy(uri_or_path: str | Path | None, cache_dir: Path) -> Path | None:
    if uri_or_path is None:
        return None
    value = str(uri_or_path).strip()
    if not value:
        return None
    try:
        return ensure_local_copy(value, cache_dir)
    except FileNotFoundError:
        return None
