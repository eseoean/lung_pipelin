from __future__ import annotations

from typing import Any

import pandas as pd

from ..config import resolve_repo_path, stage_output_dir
from ..datasets.drug_knowledge import build_drug_structure_features
from ..registry import DATASET_BUCKETS
from ..io import ensure_dir, write_json
from ._common import build_stage_manifest

DEFAULT_MODEL_INPUT_SETTINGS = {
    "top_signature_genes_per_cohort": 12,
    "top_pathways_per_collection": 8,
    "filter_to_depmap_mapped": False,
    "include_label_aggregate_features": False,
    "include_sample_crispr_features": True,
    "include_drug_structure_features": True,
    "drug_fingerprint_radius": 2,
    "drug_fingerprint_nbits": 2048,
}


def run(cfg: dict[str, Any], dry_run: bool = False) -> dict[str, Any]:
    inputs = (
        DATASET_BUCKETS["supervision"]
        + DATASET_BUCKETS["cell_context"]
        + DATASET_BUCKETS["drug_knowledge"]
        + DATASET_BUCKETS["perturbation_pathway"]
        + ["disease_context_outputs"]
    )
    notes = [
        "Final learning unit is cell line x drug.",
        "Separate outputs into sample X, drug X, pair X, and labels y.",
        "LINCS is marked as very important in the planning doc.",
    ]
    settings = _stage_settings(cfg)
    if dry_run:
        return build_stage_manifest(
            cfg,
            "build_model_inputs",
            inputs,
            notes,
            dry_run,
            status="dry_run",
            extra={
                "implementation_state": "masters + disease context model-input builder ready",
                "top_signature_genes_per_cohort": settings["top_signature_genes_per_cohort"],
                "top_pathways_per_collection": settings["top_pathways_per_collection"],
                "filter_to_depmap_mapped": settings["filter_to_depmap_mapped"],
                "include_label_aggregate_features": settings["include_label_aggregate_features"],
                "include_sample_crispr_features": settings["include_sample_crispr_features"],
                "include_drug_structure_features": settings["include_drug_structure_features"],
            },
        )

    masters_dir = stage_output_dir(cfg, "standardize_tables")
    disease_dir = stage_output_dir(cfg, "build_disease_context")
    output_dir = ensure_dir(stage_output_dir(cfg, "build_model_inputs"))

    response_labels = pd.read_parquet(masters_dir / "response_labels.parquet")
    cell_line_master = pd.read_parquet(masters_dir / "cell_line_master.parquet")
    drug_master = pd.read_parquet(masters_dir / "drug_master.parquet")
    target_mapping = pd.read_parquet(masters_dir / "drug_target_mapping.parquet")
    luad_signature = pd.read_parquet(disease_dir / "luad_signature.parquet")
    lusc_signature = pd.read_parquet(disease_dir / "lusc_signature.parquet")
    pathway_activity = pd.read_parquet(disease_dir / "pathway_activity.parquet")
    sample_crispr_path = masters_dir / "sample_crispr_wide.parquet"
    sample_crispr_wide = pd.read_parquet(sample_crispr_path) if sample_crispr_path.exists() else None

    labels_y = _prepare_labels_y(response_labels, settings["filter_to_depmap_mapped"])
    sample_cohorts = _build_sample_cohort_map(labels_y, cell_line_master)
    sample_features = _build_sample_features(
        labels_y=labels_y,
        cell_line_master=cell_line_master,
        sample_cohorts=sample_cohorts,
        sample_crispr_wide=sample_crispr_wide,
        luad_signature=luad_signature,
        lusc_signature=lusc_signature,
        pathway_activity=pathway_activity,
        top_signature_genes_per_cohort=settings["top_signature_genes_per_cohort"],
        top_pathways_per_collection=settings["top_pathways_per_collection"],
        include_label_aggregate_features=settings["include_label_aggregate_features"],
        include_sample_crispr_features=settings["include_sample_crispr_features"],
    )
    drug_features = _build_drug_features(
        drug_master,
        target_mapping,
        include_drug_structure_features=settings["include_drug_structure_features"],
        drug_fingerprint_radius=settings["drug_fingerprint_radius"],
        drug_fingerprint_nbits=settings["drug_fingerprint_nbits"],
    )
    pair_features = _build_pair_features(
        labels_y=labels_y,
        sample_cohorts=sample_cohorts,
        target_mapping=target_mapping,
        luad_signature=luad_signature,
        lusc_signature=lusc_signature,
    )
    train_table = _build_train_table(labels_y, sample_features, drug_features, pair_features)

    sample_features.to_parquet(output_dir / "sample_features.parquet", index=False)
    drug_features.to_parquet(output_dir / "drug_features.parquet", index=False)
    pair_features.to_parquet(output_dir / "pair_features.parquet", index=False)
    train_table.to_parquet(output_dir / "train_table.parquet", index=False)
    labels_y.to_parquet(output_dir / "labels_y.parquet", index=False)

    write_json(
        output_dir / "model_input_summary.json",
        {
            "settings": settings,
            "row_counts": {
                "labels_y": int(labels_y.shape[0]),
                "sample_features": int(sample_features.shape[0]),
                "drug_features": int(drug_features.shape[0]),
                "pair_features": int(pair_features.shape[0]),
                "train_table": int(train_table.shape[0]),
            },
            "column_counts": {
                "sample_features": int(sample_features.shape[1]),
                "drug_features": int(drug_features.shape[1]),
                "pair_features": int(pair_features.shape[1]),
                "train_table": int(train_table.shape[1]),
            },
            "cohort_breakdown": {
                str(key): int(value)
                for key, value in sample_cohorts["cohort"].value_counts(dropna=False).to_dict().items()
            },
        },
    )

    notes = notes + [
        f"Labels rows: {labels_y.shape[0]}",
        f"Sample features rows: {sample_features.shape[0]}",
        f"Drug features rows: {drug_features.shape[0]}",
        f"Pair features rows: {pair_features.shape[0]}",
    ]
    return build_stage_manifest(
        cfg,
        "build_model_inputs",
        inputs,
        notes,
        dry_run=False,
        status="implemented",
        extra={
            "implementation_state": "step3_model_inputs_built",
            "row_counts": {
                "labels_y": int(labels_y.shape[0]),
                "sample_features": int(sample_features.shape[0]),
                "drug_features": int(drug_features.shape[0]),
                "pair_features": int(pair_features.shape[0]),
                "train_table": int(train_table.shape[0]),
            },
        },
    )


def _stage_settings(cfg: dict[str, Any]) -> dict[str, Any]:
    section = cfg.get("model_inputs", {})
    settings = {**DEFAULT_MODEL_INPUT_SETTINGS, **section}
    settings["top_signature_genes_per_cohort"] = int(settings["top_signature_genes_per_cohort"])
    settings["top_pathways_per_collection"] = int(settings["top_pathways_per_collection"])
    settings["filter_to_depmap_mapped"] = bool(settings["filter_to_depmap_mapped"])
    settings["include_label_aggregate_features"] = bool(settings["include_label_aggregate_features"])
    settings["include_sample_crispr_features"] = bool(settings["include_sample_crispr_features"])
    settings["include_drug_structure_features"] = bool(settings["include_drug_structure_features"])
    settings["drug_fingerprint_radius"] = int(settings["drug_fingerprint_radius"])
    settings["drug_fingerprint_nbits"] = int(settings["drug_fingerprint_nbits"])
    return settings


def _prepare_labels_y(
    response_labels: pd.DataFrame,
    filter_to_depmap_mapped: bool,
) -> pd.DataFrame:
    labels = response_labels.copy()
    if filter_to_depmap_mapped:
        labels = labels[labels["is_depmap_mapped"] == 1].copy()
    labels["sample_id"] = labels["sample_id"].astype(str).str.strip()
    labels["canonical_drug_id"] = labels["canonical_drug_id"].astype(str).str.strip()
    labels["pair_id"] = labels["sample_id"] + "__" + labels["canonical_drug_id"]
    return labels.reset_index(drop=True)


def _build_sample_cohort_map(
    labels_y: pd.DataFrame,
    cell_line_master: pd.DataFrame,
) -> pd.DataFrame:
    labels_mode = (
        labels_y.groupby("sample_id")["TCGA_DESC"]
        .agg(lambda values: values.dropna().astype(str).mode().iloc[0] if not values.dropna().empty else "")
        .reset_index()
        .rename(columns={"TCGA_DESC": "cohort"})
    )
    sample_base = cell_line_master[
        [
            "sample_id",
            "cell_line_name",
            "model_id",
            "is_depmap_mapped",
            "depmap_oncotree_code",
            "depmap_primary_disease",
        ]
    ].drop_duplicates("sample_id")
    sample_cohorts = sample_base.merge(labels_mode, on="sample_id", how="left")
    sample_cohorts["cohort"] = sample_cohorts["cohort"].fillna("").astype(str).str.strip()
    fallback = sample_cohorts["depmap_oncotree_code"].fillna("").astype(str).str.strip()
    sample_cohorts.loc[sample_cohorts["cohort"] == "", "cohort"] = fallback[sample_cohorts["cohort"] == ""]
    return sample_cohorts


def _build_sample_features(
    *,
    labels_y: pd.DataFrame,
    cell_line_master: pd.DataFrame,
    sample_cohorts: pd.DataFrame,
    sample_crispr_wide: pd.DataFrame | None,
    luad_signature: pd.DataFrame,
    lusc_signature: pd.DataFrame,
    pathway_activity: pd.DataFrame,
    top_signature_genes_per_cohort: int,
    top_pathways_per_collection: int,
    include_label_aggregate_features: bool,
    include_sample_crispr_features: bool,
) -> pd.DataFrame:
    if include_label_aggregate_features:
        label_summary = (
            labels_y.groupby("sample_id")
            .agg(
                sample__n_pairs=("pair_id", "size"),
                sample__n_unique_drugs=("canonical_drug_id", "nunique"),
                sample__mean_label_regression=("label_regression", "mean"),
                sample__std_label_regression=("label_regression", "std"),
                sample__sensitive_fraction=("label_binary", "mean"),
            )
            .reset_index()
        )
        label_summary["sample__std_label_regression"] = (
            label_summary["sample__std_label_regression"].fillna(0.0)
        )
        base = sample_cohorts.merge(label_summary, on="sample_id", how="left")
    else:
        base = sample_cohorts.copy()
    base["sample__is_depmap_mapped"] = pd.to_numeric(base["is_depmap_mapped"], errors="coerce").fillna(0).astype(int)
    base["sample__is_luad"] = (base["cohort"] == "LUAD").astype(int)
    base["sample__is_lusc"] = (base["cohort"] == "LUSC").astype(int)

    top_signature_features = _build_signature_context_features(
        luad_signature=luad_signature,
        lusc_signature=lusc_signature,
        top_signature_genes_per_cohort=top_signature_genes_per_cohort,
    )
    top_pathway_features = _build_pathway_context_features(
        pathway_activity=pathway_activity,
        top_pathways_per_collection=top_pathways_per_collection,
    )

    context_by_cohort = {}
    for cohort_name in ["LUAD", "LUSC"]:
        combined = {"cohort": cohort_name}
        combined.update(top_signature_features.get(cohort_name, {}))
        combined.update(top_pathway_features.get(cohort_name, {}))
        context_by_cohort[cohort_name] = combined
    context_df = pd.DataFrame(context_by_cohort.values())

    sample_features = base.merge(context_df, on="cohort", how="left")
    sample_features["sample__has_crispr_profile"] = 0
    if include_sample_crispr_features and sample_crispr_wide is not None and not sample_crispr_wide.empty:
        crispr_features = sample_crispr_wide.copy()
        crispr_features["sample_id"] = crispr_features["sample_id"].astype(str).str.strip()
        crispr_columns = [column for column in crispr_features.columns if column.startswith("sample__crispr__")]
        if crispr_columns:
            crispr_features["sample__has_crispr_profile"] = (
                crispr_features[crispr_columns].notna().any(axis=1).astype(int)
            )
            medians = crispr_features[crispr_columns].median(numeric_only=True)
            crispr_features[crispr_columns] = crispr_features[crispr_columns].fillna(medians).fillna(0.0)
            sample_features = sample_features.merge(
                crispr_features[["sample_id", "sample__has_crispr_profile", *crispr_columns]],
                on="sample_id",
                how="left",
                suffixes=("", "_crispr"),
            )
            if "sample__has_crispr_profile_crispr" in sample_features.columns:
                sample_features["sample__has_crispr_profile"] = (
                    pd.to_numeric(sample_features["sample__has_crispr_profile_crispr"], errors="coerce")
                    .fillna(pd.to_numeric(sample_features["sample__has_crispr_profile"], errors="coerce"))
                    .fillna(0)
                    .astype(int)
                )
                sample_features = sample_features.drop(columns=["sample__has_crispr_profile_crispr"])
            else:
                sample_features["sample__has_crispr_profile"] = (
                    pd.to_numeric(sample_features["sample__has_crispr_profile"], errors="coerce").fillna(0).astype(int)
                )
            for column in crispr_columns:
                sample_features[column] = pd.to_numeric(sample_features[column], errors="coerce")
            fill_values = sample_features[crispr_columns].median(numeric_only=True)
            sample_features[crispr_columns] = sample_features[crispr_columns].fillna(fill_values).fillna(0.0)
    keep_columns = [
        "sample_id",
        "cell_line_name",
        "model_id",
        "cohort",
        "depmap_oncotree_code",
        "depmap_primary_disease",
        "sample__is_depmap_mapped",
        "sample__is_luad",
        "sample__is_lusc",
        "sample__has_crispr_profile",
    ] + [
        column
        for column in sample_features.columns
        if column.startswith("ctx__") or column.startswith("sample__crispr__")
    ]
    if include_label_aggregate_features:
        keep_columns.extend(
            [
                "sample__n_pairs",
                "sample__n_unique_drugs",
                "sample__mean_label_regression",
                "sample__std_label_regression",
                "sample__sensitive_fraction",
            ]
        )
    return sample_features[keep_columns].sort_values("sample_id").reset_index(drop=True)


def _build_signature_context_features(
    *,
    luad_signature: pd.DataFrame,
    lusc_signature: pd.DataFrame,
    top_signature_genes_per_cohort: int,
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    signatures = {"LUAD": luad_signature, "LUSC": lusc_signature}
    for cohort_name, signature_df in signatures.items():
        top_df = signature_df.nsmallest(top_signature_genes_per_cohort, "abs_delta_rank")
        features: dict[str, float] = {
            "ctx__signature__mean_abs_delta_top": float(top_df["delta_vs_other_cohort"].abs().mean()),
            "ctx__signature__mean_expression_top": float(top_df["mean_log2_tpm"].mean()),
        }
        for _, row in top_df.iterrows():
            gene = str(row["gene_symbol"]).strip().lower()
            features[f"ctx__sigdelta__{gene}"] = float(row["delta_vs_other_cohort"])
            features[f"ctx__sigexpr__{gene}"] = float(row["mean_log2_tpm"])
        if {"normal_abs_delta_rank", "delta_vs_normal_lung"} <= set(signature_df.columns):
            top_normal_df = signature_df.nsmallest(top_signature_genes_per_cohort, "normal_abs_delta_rank")
            features["ctx__signature__mean_abs_delta_vs_normal_top"] = float(
                top_normal_df["delta_vs_normal_lung"].abs().mean()
            )
            for _, row in top_normal_df.iterrows():
                gene = str(row["gene_symbol"]).strip().lower()
                features[f"ctx__signormal__{gene}"] = float(row["delta_vs_normal_lung"])
        result[cohort_name] = features
    return result


def _build_pathway_context_features(
    *,
    pathway_activity: pd.DataFrame,
    top_pathways_per_collection: int,
) -> dict[str, dict[str, float]]:
    result = {"LUAD": {}, "LUSC": {}}
    if pathway_activity.empty:
        return result

    for collection, df in pathway_activity.groupby("collection"):
        top_df = df.assign(abs_delta=df["delta_luad_minus_lusc"].abs()).nlargest(
            top_pathways_per_collection,
            "abs_delta",
        )
        for _, row in top_df.iterrows():
            pathway_key = _safe_feature_name(str(row["pathway_name"]))
            result["LUAD"][f"ctx__{collection}__{pathway_key}"] = float(row.get("luad_mean_pathway_score", 0.0))
            result["LUSC"][f"ctx__{collection}__{pathway_key}"] = float(row.get("lusc_mean_pathway_score", 0.0))
    return result


def _build_drug_features(
    drug_master: pd.DataFrame,
    target_mapping: pd.DataFrame,
    *,
    include_drug_structure_features: bool,
    drug_fingerprint_radius: int,
    drug_fingerprint_nbits: int,
) -> pd.DataFrame:
    target_summary = (
        target_mapping.assign(
            canonical_drug_id=target_mapping["canonical_drug_id"].astype(str).str.strip(),
            target_gene_symbol=target_mapping["target_gene_symbol"].astype(str).str.strip(),
        )
        .groupby("canonical_drug_id")
        .agg(
            drug__target_count=("target_gene_symbol", "nunique"),
            drug__target_list=("target_gene_symbol", lambda values: "|".join(sorted({v for v in values if v}))),
        )
        .reset_index()
    )
    features = drug_master.copy()
    features["canonical_drug_id"] = features["canonical_drug_id"].astype(str).str.strip()
    features = features.merge(target_summary, on="canonical_drug_id", how="left")
    features["drug__target_count"] = pd.to_numeric(features["drug__target_count"], errors="coerce").fillna(0).astype(int)
    features["drug__target_list"] = features["drug__target_list"].fillna("")
    features["drug__has_target_mapping"] = (features["drug__target_count"] > 0).astype(int)
    features["drug__has_smiles"] = pd.to_numeric(features["has_smiles"], errors="coerce").fillna(0).astype(int)
    features["drug__smiles_length"] = features["canonical_smiles"].fillna("").astype(str).str.len()
    features["drug__synonym_count"] = (
        features["synonyms"].fillna("").astype(str).apply(lambda value: len([x for x in value.split(",") if x.strip()]))
    )
    structure_columns: list[str] = []
    if include_drug_structure_features:
        structure_df = build_drug_structure_features(
            features[["canonical_drug_id", "canonical_smiles"]],
            radius=drug_fingerprint_radius,
            nbits=drug_fingerprint_nbits,
        )
        features = features.merge(structure_df, on="canonical_drug_id", how="left")
        structure_columns = [
            "drug_has_valid_smiles",
            *[f"drug_morgan_{bit_idx:04d}" for bit_idx in range(drug_fingerprint_nbits)],
            "drug_desc_mol_wt",
            "drug_desc_logp",
            "drug_desc_tpsa",
            "drug_desc_hbd",
            "drug_desc_hba",
            "drug_desc_rot_bonds",
            "drug_desc_ring_count",
            "drug_desc_heavy_atoms",
            "drug_desc_frac_csp3",
        ]
    keep_columns = [
        "canonical_drug_id",
        "DRUG_ID",
        "drug_name",
        "drug_name_norm",
        "match_source",
        "target_pathway",
        "drug__has_smiles",
        "drug__smiles_length",
        "drug__target_count",
        "drug__has_target_mapping",
        "drug__synonym_count",
        "drug__target_list",
    ] + [column for column in structure_columns if column in features.columns]
    return features[keep_columns].sort_values("canonical_drug_id").reset_index(drop=True)


def _build_pair_features(
    *,
    labels_y: pd.DataFrame,
    sample_cohorts: pd.DataFrame,
    target_mapping: pd.DataFrame,
    luad_signature: pd.DataFrame,
    lusc_signature: pd.DataFrame,
) -> pd.DataFrame:
    cohort_lookup = sample_cohorts.set_index("sample_id")["cohort"].to_dict()
    target_lookup = (
        target_mapping.assign(
            canonical_drug_id=target_mapping["canonical_drug_id"].astype(str).str.strip(),
            target_gene_symbol=target_mapping["target_gene_symbol"].astype(str).str.strip().str.upper(),
        )
        .groupby("canonical_drug_id")["target_gene_symbol"]
        .agg(lambda values: sorted({value for value in values if value}))
        .to_dict()
    )

    signature_lookups = {
        "LUAD": _make_signature_lookup(luad_signature),
        "LUSC": _make_signature_lookup(lusc_signature),
    }

    rows = []
    for _, row in labels_y[["pair_id", "sample_id", "canonical_drug_id"]].iterrows():
        sample_id = str(row["sample_id"])
        drug_id = str(row["canonical_drug_id"])
        cohort = str(cohort_lookup.get(sample_id, "")).strip()
        targets = target_lookup.get(drug_id, [])
        lookup = signature_lookups.get(cohort, {})
        stats = _summarize_targets_against_signature(targets, lookup)
        rows.append(
            {
                "pair_id": row["pair_id"],
                "sample_id": sample_id,
                "canonical_drug_id": drug_id,
                "pair__cohort": cohort,
                **stats,
            }
        )
    return pd.DataFrame(rows).sort_values("pair_id").reset_index(drop=True)


def _make_signature_lookup(signature_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    lookup: dict[str, dict[str, float]] = {}
    for _, row in signature_df.iterrows():
        gene = str(row["gene_symbol"]).strip().upper()
        lookup[gene] = {
            "mean_log2_tpm": float(row["mean_log2_tpm"]),
            "delta_vs_other_cohort": float(row["delta_vs_other_cohort"]),
            "abs_delta_rank": float(row["abs_delta_rank"]),
            "delta_vs_normal_lung": float(row.get("delta_vs_normal_lung", 0.0)),
            "normal_abs_delta_rank": float(row.get("normal_abs_delta_rank", 0.0)),
        }
    return lookup


def _summarize_targets_against_signature(
    targets: list[str],
    signature_lookup: dict[str, dict[str, float]],
) -> dict[str, float]:
    matched = [signature_lookup[target] for target in targets if target in signature_lookup]
    if not targets:
        return {
            "pair__target_count": 0.0,
            "pair__matched_target_count": 0.0,
            "pair__target_match_fraction": 0.0,
            "pair__mean_target_expression": 0.0,
            "pair__mean_target_delta": 0.0,
            "pair__mean_target_delta_vs_normal": 0.0,
            "pair__max_target_delta": 0.0,
            "pair__min_target_delta": 0.0,
            "pair__max_target_delta_vs_normal": 0.0,
            "pair__min_target_delta_vs_normal": 0.0,
            "pair__top50_target_hits": 0.0,
            "pair__top200_target_hits": 0.0,
            "pair__top50_target_hits_vs_normal": 0.0,
            "pair__top200_target_hits_vs_normal": 0.0,
        }
    if not matched:
        return {
            "pair__target_count": float(len(targets)),
            "pair__matched_target_count": 0.0,
            "pair__target_match_fraction": 0.0,
            "pair__mean_target_expression": 0.0,
            "pair__mean_target_delta": 0.0,
            "pair__mean_target_delta_vs_normal": 0.0,
            "pair__max_target_delta": 0.0,
            "pair__min_target_delta": 0.0,
            "pair__max_target_delta_vs_normal": 0.0,
            "pair__min_target_delta_vs_normal": 0.0,
            "pair__top50_target_hits": 0.0,
            "pair__top200_target_hits": 0.0,
            "pair__top50_target_hits_vs_normal": 0.0,
            "pair__top200_target_hits_vs_normal": 0.0,
        }

    deltas = [item["delta_vs_other_cohort"] for item in matched]
    normal_deltas = [item["delta_vs_normal_lung"] for item in matched]
    expressions = [item["mean_log2_tpm"] for item in matched]
    ranks = [item["abs_delta_rank"] for item in matched]
    normal_ranks = [item["normal_abs_delta_rank"] for item in matched if item["normal_abs_delta_rank"] > 0]
    return {
        "pair__target_count": float(len(targets)),
        "pair__matched_target_count": float(len(matched)),
        "pair__target_match_fraction": float(len(matched)) / float(len(targets)),
        "pair__mean_target_expression": float(sum(expressions) / len(expressions)),
        "pair__mean_target_delta": float(sum(deltas) / len(deltas)),
        "pair__mean_target_delta_vs_normal": float(sum(normal_deltas) / len(normal_deltas)),
        "pair__max_target_delta": float(max(deltas)),
        "pair__min_target_delta": float(min(deltas)),
        "pair__max_target_delta_vs_normal": float(max(normal_deltas)),
        "pair__min_target_delta_vs_normal": float(min(normal_deltas)),
        "pair__top50_target_hits": float(sum(rank <= 50 for rank in ranks)),
        "pair__top200_target_hits": float(sum(rank <= 200 for rank in ranks)),
        "pair__top50_target_hits_vs_normal": float(sum(rank <= 50 for rank in normal_ranks)),
        "pair__top200_target_hits_vs_normal": float(sum(rank <= 200 for rank in normal_ranks)),
    }


def _build_train_table(
    labels_y: pd.DataFrame,
    sample_features: pd.DataFrame,
    drug_features: pd.DataFrame,
    pair_features: pd.DataFrame,
) -> pd.DataFrame:
    sample_feature_columns = [
        column
        for column in sample_features.columns
        if column not in {"sample_id", "cell_line_name", "model_id"}
    ]
    drug_feature_columns = [
        column
        for column in drug_features.columns
        if column not in {"canonical_drug_id", "DRUG_ID", "drug_name"}
    ]
    pair_feature_columns = [
        column
        for column in pair_features.columns
        if column not in {"pair_id", "sample_id", "canonical_drug_id"}
    ]

    train_table = labels_y.merge(
        sample_features[["sample_id", *sample_feature_columns]],
        on="sample_id",
        how="left",
    ).merge(
        drug_features[["canonical_drug_id", *drug_feature_columns]],
        on="canonical_drug_id",
        how="left",
    ).merge(
        pair_features[["pair_id", "sample_id", "canonical_drug_id", *pair_feature_columns]],
        on=["pair_id", "sample_id", "canonical_drug_id"],
        how="left",
    )
    return train_table.sort_values("pair_id").reset_index(drop=True)


def _safe_feature_name(value: str) -> str:
    chars = []
    for char in value.strip().lower():
        chars.append(char if char.isalnum() else "_")
    cleaned = "".join(chars)
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_")
