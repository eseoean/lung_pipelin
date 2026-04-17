from __future__ import annotations

import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from lung_pipeline.config import load_config, repo_root


def run_aws_ls(uri: str) -> list[str]:
    cmd = ["aws", "s3", "ls", uri, "--recursive"]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        return []
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def status_for_source(name: str, meta: dict, lines: list[str]) -> tuple[str, list[str], str]:
    required_accessions = meta.get("required_accessions", [])
    required_files = meta.get("required_files", [])
    if required_accessions:
        matched = [line for line in lines if any(acc in line for acc in required_accessions)]
        missing = [acc for acc in required_accessions if not any(acc in line for line in lines)]
        if len(matched) == len(required_accessions):
            return "available", matched, ""
        if matched:
            return "partial", matched, f"Missing accessions: {', '.join(missing)}"
        return "missing_in_bucket", [], f"Missing accessions: {', '.join(required_accessions)}"
    if required_files:
        matched = [line for line in lines if any(name_ in line for name_ in required_files)]
        missing = [name_ for name_ in required_files if not any(name_ in line for line in lines)]
        if len(matched) == len(required_files):
            return "available", matched, ""
        if matched:
            return "partial", matched, f"Missing files: {', '.join(missing)}"
        return "missing_in_bucket", [], f"Missing files: {', '.join(required_files)}"
    if lines:
        return "available", lines[:5], ""
    return "missing_prefix", [], "Prefix listed in config but no objects found."


def main() -> None:
    cfg = load_config(Path("configs/ipf.yaml"))
    root = repo_root(cfg)
    out_dir = root / "docs" / "ipf"
    out_dir.mkdir(parents=True, exist_ok=True)

    data_sources = cfg["data_sources"]
    planned_downloads = cfg.get("planned_external_downloads", {})

    rows: list[dict] = []
    missing_rows: list[dict] = []

    for source_name, meta in data_sources.items():
        uri = meta["uri"]
        lines = run_aws_ls(uri)
        status, matched, detail = status_for_source(source_name, meta, lines)
        row = {
            "source_name": source_name,
            "uri": uri,
            "enabled": meta.get("enabled", False),
            "status": status,
            "required_accessions": ",".join(meta.get("required_accessions", [])),
            "required_files": ",".join(meta.get("required_files", [])),
            "matched_count": len(matched),
            "detail": detail or meta.get("note", ""),
            "example_matches": matched[:3],
        }
        rows.append(row)

        if status != "available":
            required_accessions = meta.get("required_accessions", [])
            if required_accessions:
                for accession in required_accessions:
                    missing_rows.append(
                        {
                            "source_name": source_name,
                            "accession_or_file": accession,
                            "status": "available" if accession in "".join(matched) else "missing",
                            "external_url": planned_downloads.get(accession, {}).get("url", ""),
                        }
                    )
            else:
                missing_rows.append(
                    {
                        "source_name": source_name,
                        "accession_or_file": meta.get("required_files", ["bucket contents"])[0],
                        "status": status,
                        "external_url": "",
                    }
                )

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "study": cfg["study"]["name"],
        "objective": cfg["study"]["objective"],
        "bucket_root": "s3://say2-4team/Lung_raw/",
        "sources": rows,
        "missing_downloads": missing_rows,
    }

    (out_dir / "ipf_dataset_inventory.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )

    with (out_dir / "ipf_dataset_inventory.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "source_name",
                "uri",
                "enabled",
                "status",
                "required_accessions",
                "required_files",
                "matched_count",
                "detail",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in writer.fieldnames})

    with (out_dir / "ipf_missing_downloads.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["source_name", "accession_or_file", "status", "external_url"],
        )
        writer.writeheader()
        writer.writerows(missing_rows)

    available = [row for row in rows if row["status"] == "available"]
    partial = [row for row in rows if row["status"] == "partial"]
    missing = [row for row in rows if row["status"] not in {"available", "partial"}]

    md_lines = [
        "# IPF Dataset Inventory",
        "",
        f"- Generated at: `{summary['generated_at']}`",
        f"- Study: `{summary['study']}`",
        f"- Objective: `{summary['objective']}`",
        f"- Bucket root: `{summary['bucket_root']}`",
        "",
        "## Availability Summary",
        "",
        f"- Available sources: `{len(available)}`",
        f"- Partial sources: `{len(partial)}`",
        f"- Missing sources: `{len(missing)}`",
        "",
        "## Available in Bucket",
        "",
    ]
    for row in available:
        md_lines.append(f"- `{row['source_name']}` -> `{row['uri']}`")

    md_lines.extend(["", "## Partial / Missing", ""])
    for row in partial + missing:
        md_lines.append(f"- `{row['source_name']}` -> `{row['status']}`: {row['detail']}")

    md_lines.extend(["", "## Required External Downloads", ""])
    for row in missing_rows:
        if row["status"] == "missing":
            url = row["external_url"] or "TBD"
            md_lines.append(f"- `{row['source_name']}` / `{row['accession_or_file']}` -> {url}")

    (out_dir / "ipf_dataset_inventory.md").write_text("\n".join(md_lines) + "\n")


if __name__ == "__main__":
    main()
