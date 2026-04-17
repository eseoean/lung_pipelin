from __future__ import annotations

import argparse
import csv
import json
import re
import ssl
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from lung_pipeline.config import load_config, repo_root


GEO_HTML_URL = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={accession}"
GEO_SERIES_SOFT_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/{series_stub}/{accession}/soft/"
    "{accession}_family.soft.gz"
)
GEO_SERIES_MATRIX_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/{series_stub}/{accession}/matrix/"
    "{accession}_series_matrix.txt.gz"
)
GEO_SUPPL_INDEX_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/{series_stub}/{accession}/suppl/"
)


class HrefParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        attr_map = dict(attrs)
        href = attr_map.get("href")
        if href:
            self.hrefs.append(href)


@dataclass
class DownloadRow:
    accession: str
    category: str
    asset_type: str
    url: str
    relative_path: str
    status: str
    detail: str = ""


def _series_stub(accession: str) -> str:
    match = re.fullmatch(r"GSE(\d+)", accession)
    if not match:
        raise ValueError(f"Unsupported GEO accession: {accession}")
    digits = match.group(1)
    return f"GSE{digits[:-3]}nnn"


def _fetch(url: str, timeout: int = 60, verify_ssl: bool = True) -> bytes:
    req = Request(url, headers={"User-Agent": "lung-pipelin-ipf/0.1"})
    context = None if verify_ssl else ssl._create_unverified_context()
    with urlopen(req, timeout=timeout, context=context) as resp:
        return resp.read()


def _head_content_length(url: str, timeout: int = 60, verify_ssl: bool = True) -> int | None:
    req = Request(
        url,
        headers={"User-Agent": "lung-pipelin-ipf/0.1"},
        method="HEAD",
    )
    context = None if verify_ssl else ssl._create_unverified_context()
    with urlopen(req, timeout=timeout, context=context) as resp:
        header = resp.headers.get("Content-Length")
        return int(header) if header and header.isdigit() else None


def _safe_fetch(url: str) -> tuple[str, bytes | None, str]:
    try:
        return "available", _fetch(url), ""
    except URLError as exc:
        detail = str(exc.reason)
        if "CERTIFICATE_VERIFY_FAILED" in detail:
            try:
                return "available", _fetch(url, verify_ssl=False), "downloaded with unverified SSL context"
            except HTTPError as retry_exc:
                return "http_error", None, f"HTTP {retry_exc.code}"
            except URLError as retry_exc:
                return "network_error", None, str(retry_exc.reason)
            except Exception as retry_exc:  # pragma: no cover - defensive
                return "error", None, str(retry_exc)
        return "network_error", None, detail
    except HTTPError as exc:
        return "http_error", None, f"HTTP {exc.code}"
    except Exception as exc:  # pragma: no cover - defensive
        return "error", None, str(exc)


def _safe_head_content_length(url: str) -> tuple[str, int | None, str]:
    try:
        return "available", _head_content_length(url), ""
    except URLError as exc:
        detail = str(exc.reason)
        if "CERTIFICATE_VERIFY_FAILED" in detail:
            try:
                return "available", _head_content_length(url, verify_ssl=False), "head with unverified SSL context"
            except HTTPError as retry_exc:
                return "http_error", None, f"HTTP {retry_exc.code}"
            except URLError as retry_exc:
                return "network_error", None, str(retry_exc.reason)
            except Exception as retry_exc:  # pragma: no cover - defensive
                return "error", None, str(retry_exc)
        return "network_error", None, detail
    except HTTPError as exc:
        return "http_error", None, f"HTTP {exc.code}"
    except Exception as exc:  # pragma: no cover - defensive
        return "error", None, str(exc)


def build_base_rows(cfg: dict) -> list[DownloadRow]:
    rows: list[DownloadRow] = []
    for accession, meta in cfg.get("planned_external_downloads", {}).items():
        category = meta.get("category", "geo")
        series_stub = _series_stub(accession)
        rows.extend(
            [
                DownloadRow(
                    accession=accession,
                    category=category,
                    asset_type="landing_page",
                    url=GEO_HTML_URL.format(accession=accession),
                    relative_path=f"{accession}/{accession}_landing_page.html",
                    status="planned",
                ),
                DownloadRow(
                    accession=accession,
                    category=category,
                    asset_type="family_soft",
                    url=GEO_SERIES_SOFT_URL.format(
                        series_stub=series_stub,
                        accession=accession,
                    ),
                    relative_path=f"{accession}/{accession}_family.soft.gz",
                    status="planned",
                ),
                DownloadRow(
                    accession=accession,
                    category=category,
                    asset_type="series_matrix",
                    url=GEO_SERIES_MATRIX_URL.format(
                        series_stub=series_stub,
                        accession=accession,
                    ),
                    relative_path=f"{accession}/{accession}_series_matrix.txt.gz",
                    status="planned",
                ),
                DownloadRow(
                    accession=accession,
                    category=category,
                    asset_type="supplementary_index",
                    url=GEO_SUPPL_INDEX_URL.format(
                        series_stub=series_stub,
                        accession=accession,
                    ),
                    relative_path=f"{accession}/{accession}_supplementary_index.html",
                    status="planned",
                ),
            ]
        )
    return rows


def write_plan(rows: Iterable[DownloadRow], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_list = [asdict(row) for row in rows]
    (out_dir / "ipf_geo_download_plan.json").write_text(
        json.dumps(rows_list, ensure_ascii=False, indent=2)
    )
    with (out_dir / "ipf_geo_download_plan.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "accession",
                "category",
                "asset_type",
                "url",
                "relative_path",
                "status",
                "detail",
            ],
        )
        writer.writeheader()
        writer.writerows(rows_list)


def write_report(rows: Iterable[DownloadRow], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_list = [asdict(row) for row in rows]
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rows": rows_list,
    }
    (out_dir / "ipf_geo_download_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2)
    )

    available = [row for row in rows_list if row["status"] == "downloaded"]
    skipped = [row for row in rows_list if row["status"] == "exists"]
    supplementary_queue = [
        row for row in rows_list if row["status"] == "planned_supplementary"
    ]
    missing = [
        row
        for row in rows_list
        if row["status"] not in {"downloaded", "exists", "planned_supplementary"}
    ]

    with (out_dir / "ipf_geo_supplementary_queue.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "accession",
                "category",
                "asset_type",
                "url",
                "relative_path",
                "status",
                "detail",
            ],
        )
        writer.writeheader()
        writer.writerows(supplementary_queue)

    md = [
        "# IPF GEO Download Report",
        "",
        f"- Generated at: `{report['generated_at']}`",
        f"- Downloaded assets: `{len(available)}`",
        f"- Already present: `{len(skipped)}`",
        f"- Supplementary queue: `{len(supplementary_queue)}`",
        f"- Unavailable or failed: `{len(missing)}`",
        "",
        "## Downloaded",
        "",
    ]
    for row in available:
        md.append(
            f"- `{row['accession']}` / `{row['asset_type']}` -> `{row['relative_path']}`"
        )

    md.extend(["", "## Supplementary Queue", ""])
    for row in supplementary_queue:
        md.append(
            f"- `{row['accession']}` -> `{row['url']}`"
        )

    md.extend(["", "## Unavailable or Failed", ""])
    for row in missing:
        md.append(
            f"- `{row['accession']}` / `{row['asset_type']}` -> "
            f"`{row['status']}` {row['detail']}".rstrip()
        )

    (out_dir / "ipf_geo_download_report.md").write_text("\n".join(md) + "\n")


def download_rows(rows: list[DownloadRow], data_root: Path) -> list[DownloadRow]:
    result: list[DownloadRow] = []
    for row in rows:
        dest = data_root / row.relative_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            result.append(
                DownloadRow(**{**asdict(row), "status": "exists", "detail": "already present"})
            )
            continue

        status, payload, detail = _safe_fetch(row.url)
        if payload is None:
            result.append(DownloadRow(**{**asdict(row), "status": status, "detail": detail}))
            continue

        dest.write_bytes(payload)
        result.append(DownloadRow(**{**asdict(row), "status": "downloaded"}))
    return result


def enrich_with_supplementary_links(rows: list[DownloadRow], data_root: Path) -> list[DownloadRow]:
    enriched = list(rows)
    for row in rows:
        if row.asset_type != "supplementary_index":
            continue
        index_path = data_root / row.relative_path
        if not index_path.exists():
            continue
        parser = HrefParser()
        parser.feed(index_path.read_text(errors="ignore"))
        extra_hrefs = [
            href
            for href in parser.hrefs
            if href not in {"../"}
            and not href.endswith("/")
            and not href.startswith("http://")
            and not href.startswith("https://")
        ]
        for href in extra_hrefs:
            url = row.url + href
            relative_path = f"{row.accession}/supplementary/{href}"
            supp_row = DownloadRow(
                accession=row.accession,
                category=row.category,
                asset_type="supplementary_file",
                url=url,
                relative_path=relative_path,
                status="planned_supplementary",
            )
            enriched.append(supp_row)
    return enriched


def materialize_supplementary_rows(
    rows: list[DownloadRow],
    data_root: Path,
    supplementary_max_bytes: int | None,
) -> list[DownloadRow]:
    materialized: list[DownloadRow] = []
    for row in rows:
        if row.status != "planned_supplementary":
            materialized.append(row)
            continue

        dest = data_root / row.relative_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            materialized.append(
                DownloadRow(**{**asdict(row), "status": "exists", "detail": "already present"})
            )
            continue

        if supplementary_max_bytes is None:
            payload_status, payload, payload_detail = _safe_fetch(row.url)
            if payload is None:
                materialized.append(
                    DownloadRow(
                        **{
                            **asdict(row),
                            "status": payload_status,
                            "detail": payload_detail,
                        }
                    )
                )
                continue
            dest.write_bytes(payload)
            materialized.append(
                DownloadRow(**{**asdict(row), "status": "downloaded", "detail": f"{len(payload)} bytes"})
            )
            continue

        status, size, detail = _safe_head_content_length(row.url)
        if size is None:
            materialized.append(DownloadRow(**{**asdict(row), "status": status, "detail": detail}))
            continue
        if size <= supplementary_max_bytes:
            payload_status, payload, payload_detail = _safe_fetch(row.url)
            if payload is None:
                materialized.append(
                    DownloadRow(
                        **{
                            **asdict(row),
                            "status": payload_status,
                            "detail": payload_detail,
                        }
                    )
                )
                continue
            dest.write_bytes(payload)
            materialized.append(
                DownloadRow(**{**asdict(row), "status": "downloaded", "detail": f"{size} bytes"})
            )
        else:
            materialized.append(
                DownloadRow(
                    **{
                        **asdict(row),
                        "status": "planned_supplementary",
                        "detail": f"{size} bytes exceeds threshold {supplementary_max_bytes}",
                    }
                )
            )
    return materialized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan or download missing GEO IPF cohorts.")
    parser.add_argument(
        "--config",
        default="configs/ipf.yaml",
        help="Path to the IPF config file.",
    )
    parser.add_argument(
        "--data-root",
        default="data/raw/geo/ipf",
        help="Local landing directory for downloaded GEO assets.",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Only generate the plan files without attempting downloads.",
    )
    parser.add_argument(
        "--download-supplementary",
        action="store_true",
        help="After saving supplementary index pages, also fetch linked supplementary files.",
    )
    parser.add_argument(
        "--supplementary-max-bytes",
        type=int,
        default=None,
        help="Only download supplementary files up to this byte threshold.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    root = repo_root(cfg)
    docs_dir = root / "docs" / "ipf"
    data_root = root / args.data_root

    planned_rows = build_base_rows(cfg)
    write_plan(planned_rows, docs_dir)

    if args.plan_only:
        return

    downloaded = download_rows(planned_rows, data_root)
    enriched = enrich_with_supplementary_links(downloaded, data_root)
    if args.download_supplementary:
        enriched = materialize_supplementary_rows(
            enriched,
            data_root,
            supplementary_max_bytes=args.supplementary_max_bytes,
        )
    write_report(enriched, docs_dir)


if __name__ == "__main__":
    main()
