from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ..config import resolve_repo_path, stage_output_dir
from ..datasets.depmap import build_cell_line_master, build_depmap_mapping
from ..datasets.drug_knowledge import (
    build_chembl_lookups,
    build_drug_catalog,
    build_drugbank_curated_lookups,
    build_lincs_smiles_lookup,
    build_target_mapping,
    overlay_exact_drug_catalog,
)
from ..datasets.gdsc import build_gdsc_response_table, build_response_labels
from ..datasets.source_io import ensure_local_copy, maybe_local_copy
from ..io import ensure_dir, write_json
from ..registry import DATASET_BUCKETS, PIPELINE_PRIORITIES
from ._common import build_stage_manifest


def _load_optional_sources(cache_dir: Path, standardization_cfg: dict[str, Any]) -> dict[str, Path | None]:
    sources = standardization_cfg.get("sources", {})
    return {
        "lincs_pert_info_primary": maybe_local_copy(sources.get("lincs_pert_info_primary"), cache_dir),
        "lincs_pert_info_secondary": maybe_local_copy(sources.get("lincs_pert_info_secondary"), cache_dir),
        "drugbank_master": maybe_local_copy(sources.get("drugbank_master"), cache_dir),
        "drugbank_synonym": maybe_local_copy(sources.get("drugbank_synonym"), cache_dir),
        "chembl_master": maybe_local_copy(sources.get("chembl_master"), cache_dir),
        "exact_drug_catalog": maybe_local_copy(sources.get("exact_drug_catalog"), cache_dir),
        "exact_target_mapping": maybe_local_copy(sources.get("exact_target_mapping"), cache_dir),
    }


def _build_source_crosswalks(
    *,
    sources: dict[str, Any],
    gdsc_labels: pd.DataFrame,
    mapping_df: pd.DataFrame,
    drug_master: pd.DataFrame,
    target_mapping: pd.DataFrame,
    optional_paths: dict[str, Path | None],
) -> dict[str, Any]:
    return {
        "source_paths": {key: str(value) for key, value in sources.items()},
        "optional_local_sources": {key: str(value) if value else "" for key, value in optional_paths.items()},
        "counts": {
            "gdsc_rows": int(gdsc_labels.shape[0]),
            "gdsc_unique_cell_lines": int(gdsc_labels["cell_line_name"].nunique()),
            "gdsc_unique_drugs": int(gdsc_labels["DRUG_ID"].nunique()),
            "mapped_cell_lines": int((mapping_df["ModelID"].astype(str) != "").sum()),
            "drug_master_rows": int(drug_master.shape[0]),
            "drug_master_has_smiles": int(drug_master["has_smiles"].sum()),
            "drug_target_mapping_rows": int(target_mapping.shape[0]),
        },
        "mapping_rule_breakdown": {
            str(key): int(value)
            for key, value in mapping_df["mapping_rule"].value_counts(dropna=False).to_dict().items()
        },
        "drug_match_source_breakdown": {
            str(key): int(value)
            for key, value in drug_master["match_source"].value_counts(dropna=False).to_dict().items()
        },
        "top_priorities": PIPELINE_PRIORITIES,
    }


def run(cfg: dict[str, Any], dry_run: bool = False) -> dict[str, Any]:
    standardization_cfg = cfg.get("standardization", {})
    sources = standardization_cfg.get("sources", {})
    inputs = DATASET_BUCKETS["supervision"] + DATASET_BUCKETS["cell_context"] + DATASET_BUCKETS["drug_knowledge"]
    notes = [
        "Priority: build drug master table.",
        "Priority: build cell line master table.",
        "Expected joins: GDSC/PRISM to DrugBank/ChEMBL and GDSC to DepMap/COSMIC.",
        f"Top repo priorities: {', '.join(PIPELINE_PRIORITIES[:2])}.",
    ]
    if dry_run:
        return build_stage_manifest(
            cfg,
            "standardize_tables",
            inputs,
            notes,
            dry_run,
            status="dry_run",
            extra={"implementation_state": "stage contract and source plan ready"},
        )

    output_dir = ensure_dir(stage_output_dir(cfg, "standardize_tables"))
    cache_dir = ensure_dir(resolve_repo_path(cfg, standardization_cfg.get("cache_dir", ".cache/standardize_tables")))

    gdsc_dataset = ensure_local_copy(sources["gdsc_dataset"], cache_dir)
    gdsc_compounds = ensure_local_copy(sources["gdsc_compounds"], cache_dir)
    depmap_model = ensure_local_copy(sources["depmap_model"], cache_dir)
    optional_paths = _load_optional_sources(cache_dir, standardization_cfg)

    cancer_codes = list(standardization_cfg.get("cancer_codes", ["LUAD", "LUSC"]))
    binary_quantile = float(standardization_cfg.get("binary_quantile", 0.3))

    gdsc_labels = build_gdsc_response_table(str(gdsc_dataset), cancer_codes)
    compounds_df = pd.read_csv(gdsc_compounds, low_memory=False)
    model_df = pd.read_csv(depmap_model, low_memory=False)

    gdsc_mapping_input = gdsc_labels[["cell_line_name", "COSMIC_ID"]].drop_duplicates(subset=["cell_line_name"])
    mapping_df = build_depmap_mapping(model_df, gdsc_mapping_input)
    cell_line_master = build_cell_line_master(gdsc_labels, mapping_df)

    response_labels = build_response_labels(gdsc_labels, binary_quantile).merge(
        cell_line_master[["sample_id", "model_id", "is_depmap_mapped", "gdsc_cosmic_id"]],
        on="sample_id",
        how="left",
    )

    smiles_sources = []
    for key in ["lincs_pert_info_primary", "lincs_pert_info_secondary"]:
        path = optional_paths.get(key)
        if path:
            smiles_sources.extend(build_lincs_smiles_lookup([str(path)]))
    if optional_paths.get("chembl_master"):
        smiles_sources.extend(build_chembl_lookups(str(optional_paths["chembl_master"])))
    if optional_paths.get("drugbank_master") and optional_paths.get("drugbank_synonym"):
        smiles_sources.extend(
            build_drugbank_curated_lookups(
                str(optional_paths["drugbank_master"]),
                str(optional_paths["drugbank_synonym"]),
            )
        )

    drug_master = build_drug_catalog(gdsc_labels, compounds_df, smiles_sources)
    if optional_paths.get("exact_drug_catalog"):
        exact_catalog = pd.read_parquet(optional_paths["exact_drug_catalog"])
        drug_master = overlay_exact_drug_catalog(drug_master, exact_catalog)

    if optional_paths.get("exact_target_mapping"):
        target_mapping = pd.read_parquet(optional_paths["exact_target_mapping"]).copy()
        target_mapping["canonical_drug_id"] = target_mapping["canonical_drug_id"].astype(str).str.strip()
        target_mapping["target_gene_symbol"] = target_mapping["target_gene_symbol"].astype(str).str.strip()
        target_mapping = target_mapping.drop_duplicates().reset_index(drop=True)
    else:
        target_mapping = build_target_mapping(drug_master)

    gdsc_labels.to_parquet(output_dir / "gdsc_lung_response.parquet", index=False)
    mapping_df.to_parquet(output_dir / "depmap_mapping.parquet", index=False)
    drug_master.to_parquet(output_dir / "drug_master.parquet", index=False)
    target_mapping.to_parquet(output_dir / "drug_target_mapping.parquet", index=False)
    cell_line_master.to_parquet(output_dir / "cell_line_master.parquet", index=False)
    response_labels.to_parquet(output_dir / "response_labels.parquet", index=False)

    crosswalks = _build_source_crosswalks(
        sources=sources,
        gdsc_labels=gdsc_labels,
        mapping_df=mapping_df,
        drug_master=drug_master,
        target_mapping=target_mapping,
        optional_paths=optional_paths,
    )
    write_json(output_dir / "source_crosswalks.json", crosswalks)

    notes = notes + [
        f"GDSC lung rows: {gdsc_labels.shape[0]}",
        f"Mapped cell lines: {(mapping_df['ModelID'].astype(str) != '').sum()}",
        f"Drug master rows: {drug_master.shape[0]}",
    ]
    return build_stage_manifest(
        cfg,
        "standardize_tables",
        inputs,
        notes,
        dry_run=False,
        status="implemented",
        extra={
            "implementation_state": "step1_actual_tables_built",
            "row_counts": crosswalks["counts"],
        },
    )
