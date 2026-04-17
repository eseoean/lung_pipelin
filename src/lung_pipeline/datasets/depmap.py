from __future__ import annotations

import re

import pandas as pd


def norm_name(value: object) -> str:
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def name_variants(value: object) -> list[str]:
    base = norm_name(value)
    if not base:
        return []
    variants = [base]
    if base.startswith("NCI"):
        variants.append(base[3:])
    elif re.fullmatch(r"h\d+", base):
        variants.append("nci" + base)
    if base.startswith("hcc") and "gr" in base:
        variants.append(base.replace("gr", ""))
    uniq: list[str] = []
    for item in variants:
        if item and item not in uniq:
            uniq.append(item)
    return uniq


def normalize_cosmic_id(value: object) -> str:
    raw = str(value).strip()
    if not raw or raw.lower() in {"nan", "none", "<na>", "-666"}:
        return ""
    try:
        numeric = int(float(raw))
    except (TypeError, ValueError):
        digits = "".join(ch for ch in raw if ch.isdigit())
        if not digits:
            return ""
        numeric = int(digits)
    return str(numeric) if numeric > 0 else ""


def parse_gene_col(column_name: str) -> str:
    match = re.match(r"^(.+?)\s*\(\d+\)$", str(column_name))
    if match:
        return match.group(1).strip()
    return str(column_name).strip()


def build_depmap_mapping(model_df: pd.DataFrame, gdsc_cell_df: pd.DataFrame) -> pd.DataFrame:
    alias_rows: list[dict[str, object]] = []
    for _, row in model_df.iterrows():
        model_id = str(row["ModelID"]).strip()
        depmap_cosmic_id = normalize_cosmic_id(row.get("COSMICID", ""))
        sources = [
            ("CellLineName", row.get("CellLineName", "")),
            ("StrippedCellLineName", row.get("StrippedCellLineName", "")),
            ("CCLENamePrefix", str(row.get("CCLEName", "")).split("_")[0]),
        ]
        for source, value in sources:
            for alias in name_variants(value):
                alias_rows.append(
                    {
                        "ModelID": model_id,
                        "alias": alias,
                        "alias_source": source,
                        "CellLineName": row.get("CellLineName", ""),
                        "StrippedCellLineName": row.get("StrippedCellLineName", ""),
                        "OncotreeCode": row.get("OncotreeCode", ""),
                        "OncotreePrimaryDisease": row.get("OncotreePrimaryDisease", ""),
                        "depmap_cosmic_id": depmap_cosmic_id,
                    }
                )
    alias_df = pd.DataFrame(alias_rows).drop_duplicates(subset=["ModelID", "alias", "alias_source"])
    source_rank = {"CellLineName": 0, "StrippedCellLineName": 1, "CCLENamePrefix": 2}

    cosmic_df = (
        model_df[
            [
                "ModelID",
                "CellLineName",
                "StrippedCellLineName",
                "OncotreeCode",
                "OncotreePrimaryDisease",
                "COSMICID",
            ]
        ]
        .copy()
        .rename(columns={"COSMICID": "depmap_cosmic_id"})
    )
    cosmic_df["depmap_cosmic_id"] = cosmic_df["depmap_cosmic_id"].map(normalize_cosmic_id)
    cosmic_df = cosmic_df[cosmic_df["depmap_cosmic_id"] != ""].drop_duplicates(
        subset=["ModelID", "depmap_cosmic_id"]
    )

    mapping_rows: list[dict[str, object]] = []
    gdsc_cell_df = gdsc_cell_df.copy()
    gdsc_cell_df["cell_line_name"] = gdsc_cell_df["cell_line_name"].astype(str)
    gdsc_cell_df["gdsc_cosmic_id"] = gdsc_cell_df["COSMIC_ID"].map(normalize_cosmic_id)
    gdsc_cell_df = gdsc_cell_df.drop_duplicates(subset=["cell_line_name"]).sort_values("cell_line_name")

    for _, gdsc_row in gdsc_cell_df.iterrows():
        cell_line = str(gdsc_row["cell_line_name"]).strip()
        gdsc_cosmic_id = str(gdsc_row.get("gdsc_cosmic_id", "")).strip()

        if gdsc_cosmic_id:
            cosmic_matches = cosmic_df[cosmic_df["depmap_cosmic_id"] == gdsc_cosmic_id].copy()
            if not cosmic_matches.empty:
                alias_candidates = []
                for query_rank, alias in enumerate(name_variants(cell_line)):
                    matched = alias_df[
                        (alias_df["alias"] == alias)
                        & (alias_df["ModelID"].astype(str).isin(set(cosmic_matches["ModelID"].astype(str))))
                    ].copy()
                    if matched.empty:
                        continue
                    matched["query_alias"] = alias
                    matched["query_rank"] = query_rank
                    matched["source_rank"] = matched["alias_source"].map(source_rank).fillna(99)
                    alias_candidates.append(matched)

                if alias_candidates:
                    candidate_df = pd.concat(alias_candidates, ignore_index=True).sort_values(
                        ["query_rank", "source_rank", "ModelID"]
                    )
                    best = candidate_df.iloc[0]
                    mapping_rows.append(
                        {
                            "cell_line_name": cell_line,
                            "gdsc_cosmic_id": gdsc_cosmic_id,
                            "ModelID": best["ModelID"],
                            "depmap_cosmic_id": best["depmap_cosmic_id"],
                            "depmap_cell_line_name": best["CellLineName"],
                            "depmap_stripped_name": best["StrippedCellLineName"],
                            "depmap_oncotree_code": best["OncotreeCode"],
                            "depmap_primary_disease": best["OncotreePrimaryDisease"],
                            "matched_alias": best["query_alias"],
                            "matched_alias_source": best["alias_source"],
                            "mapping_rule": (
                                "exact_cosmic_id + "
                                f"gdsc:{best['query_alias']} -> depmap:{best['alias_source']}"
                            ),
                        }
                    )
                    continue

                best = cosmic_matches.sort_values(["ModelID"]).iloc[0]
                mapping_rows.append(
                    {
                        "cell_line_name": cell_line,
                        "gdsc_cosmic_id": gdsc_cosmic_id,
                        "ModelID": best["ModelID"],
                        "depmap_cosmic_id": best["depmap_cosmic_id"],
                        "depmap_cell_line_name": best["CellLineName"],
                        "depmap_stripped_name": best["StrippedCellLineName"],
                        "depmap_oncotree_code": best["OncotreeCode"],
                        "depmap_primary_disease": best["OncotreePrimaryDisease"],
                        "matched_alias": "",
                        "matched_alias_source": "",
                        "mapping_rule": "exact_cosmic_id",
                    }
                )
                continue

        candidates = []
        for query_rank, alias in enumerate(name_variants(cell_line)):
            matched = alias_df[alias_df["alias"] == alias].copy()
            if matched.empty:
                continue
            matched["query_alias"] = alias
            matched["query_rank"] = query_rank
            matched["source_rank"] = matched["alias_source"].map(source_rank).fillna(99)
            candidates.append(matched)

        if not candidates:
            mapping_rows.append(
                {
                    "cell_line_name": cell_line,
                    "gdsc_cosmic_id": gdsc_cosmic_id,
                    "ModelID": "",
                    "depmap_cosmic_id": "",
                    "depmap_cell_line_name": "",
                    "depmap_stripped_name": "",
                    "depmap_oncotree_code": "",
                    "depmap_primary_disease": "",
                    "matched_alias": "",
                    "matched_alias_source": "",
                    "mapping_rule": "unmatched",
                }
            )
            continue

        candidate_df = pd.concat(candidates, ignore_index=True).sort_values(
            ["query_rank", "source_rank", "ModelID"]
        )
        best = candidate_df.iloc[0]
        mapping_rows.append(
            {
                "cell_line_name": cell_line,
                "gdsc_cosmic_id": gdsc_cosmic_id,
                "ModelID": best["ModelID"],
                "depmap_cosmic_id": best["depmap_cosmic_id"],
                "depmap_cell_line_name": best["CellLineName"],
                "depmap_stripped_name": best["StrippedCellLineName"],
                "depmap_oncotree_code": best["OncotreeCode"],
                "depmap_primary_disease": best["OncotreePrimaryDisease"],
                "matched_alias": best["query_alias"],
                "matched_alias_source": best["alias_source"],
                "mapping_rule": f"gdsc:{best['query_alias']} -> depmap:{best['alias_source']}",
            }
        )
    return pd.DataFrame(mapping_rows)


def build_cell_line_master(gdsc_labels: pd.DataFrame, mapping_df: pd.DataFrame) -> pd.DataFrame:
    base = (
        gdsc_labels.groupby("cell_line_name", as_index=False)
        .agg(
            gdsc_cosmic_id=("COSMIC_ID", lambda s: next((normalize_cosmic_id(v) for v in s if normalize_cosmic_id(v)), "")),
            tcga_desc_values=("TCGA_DESC", lambda s: sorted({str(v) for v in s if str(v).strip()})),
            gdsc_versions=("gdsc_version", lambda s: sorted({str(v) for v in s if str(v).strip()})),
        )
        .rename(columns={"cell_line_name": "sample_id"})
    )
    base["cell_line_name"] = base["sample_id"]
    merged = base.merge(mapping_df, on="cell_line_name", how="left")
    merged["gdsc_cosmic_id"] = (
        merged.get("gdsc_cosmic_id_x", "").fillna("").astype(str).where(
            merged.get("gdsc_cosmic_id_x", "").fillna("").astype(str) != "",
            merged.get("gdsc_cosmic_id_y", "").fillna("").astype(str),
        )
    )
    merged["model_id"] = merged["ModelID"].fillna("").astype(str)
    merged["is_depmap_mapped"] = merged["model_id"].ne("").astype(int)
    keep = [
        "sample_id",
        "cell_line_name",
        "gdsc_cosmic_id",
        "tcga_desc_values",
        "gdsc_versions",
        "model_id",
        "depmap_cosmic_id",
        "depmap_cell_line_name",
        "depmap_stripped_name",
        "depmap_oncotree_code",
        "depmap_primary_disease",
        "matched_alias",
        "matched_alias_source",
        "mapping_rule",
        "is_depmap_mapped",
    ]
    return merged[keep].reset_index(drop=True)


def build_depmap_crispr_long(
    crispr_csv: str,
    mapping_df: pd.DataFrame,
    chunksize: int = 32,
) -> pd.DataFrame:
    valid_map = mapping_df[mapping_df["ModelID"].astype(str) != ""].copy()
    model_to_cell = dict(zip(valid_map["ModelID"].astype(str), valid_map["cell_line_name"].astype(str)))
    wanted_model_ids = set(model_to_cell)

    selected_chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(crispr_csv, chunksize=chunksize, low_memory=False):
        id_col = str(chunk.columns[0])
        subset = chunk[chunk[id_col].astype(str).isin(wanted_model_ids)].copy()
        if not subset.empty:
            selected_chunks.append(subset)

    if not selected_chunks:
        return pd.DataFrame(columns=["cell_line_name", "gene_name", "dependency"])

    wide = pd.concat(selected_chunks, ignore_index=True)
    id_col = str(wide.columns[0])
    gene_cols = [col for col in wide.columns if col != id_col]

    long_df = wide.melt(id_vars=[id_col], value_vars=gene_cols, var_name="gene_col", value_name="dependency")
    long_df["gene_name"] = long_df["gene_col"].map(parse_gene_col)
    long_df["cell_line_name"] = long_df[id_col].astype(str).map(model_to_cell)
    long_df = long_df.dropna(subset=["cell_line_name", "dependency"]).copy()
    long_df["dependency"] = pd.to_numeric(long_df["dependency"], errors="coerce")
    long_df = long_df.dropna(subset=["dependency"])
    return long_df[["cell_line_name", "gene_name", "dependency"]].reset_index(drop=True)


def build_sample_crispr_wide(depmap_long: pd.DataFrame) -> pd.DataFrame:
    if depmap_long.empty:
        return pd.DataFrame(columns=["sample_id"])

    wide = depmap_long.pivot_table(
        index="cell_line_name",
        columns="gene_name",
        values="dependency",
        aggfunc="mean",
    )
    wide.columns = [f"sample__crispr__{str(col)}" for col in wide.columns]
    wide = wide.reset_index().rename(columns={"cell_line_name": "sample_id"})
    numeric_cols = [col for col in wide.columns if col != "sample_id"]
    if numeric_cols:
        medians = wide[numeric_cols].median(numeric_only=True)
        wide[numeric_cols] = wide[numeric_cols].fillna(medians).fillna(0.0)
    return wide
