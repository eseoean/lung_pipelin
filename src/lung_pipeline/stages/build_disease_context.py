from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ..config import resolve_repo_path, stage_output_dir
from ..datasets.gtex import load_gtex_lung_reference
from ..datasets.msigdb import load_gmt_sets, safe_feature_name
from ..datasets.tcga import load_count_entries, read_expression_profile
from ..io import ensure_dir, write_json
from ..registry import DATASET_BUCKETS
from ._common import build_stage_manifest

DEFAULT_DISEASE_CONTEXT_SOURCES = {
    "tcga_luad_counts": "s3://say2-4team/Lung_raw/tcga_luad_rna_star_counts/",
    "tcga_lusc_counts": "s3://say2-4team/Lung_raw/tcga_lusc_rna_star_counts/",
    "tcga_luad_manifest": "s3://say2-4team/Lung_raw/manifests/luad_manifest_rna_only.tsv",
    "tcga_lusc_manifest": "s3://say2-4team/Lung_raw/manifests/lusc_manifest_rna_only.tsv",
    "msigdb_hallmark": "s3://say2-4team/Lung_raw/msigdb/h.all.v2026.1.Hs.symbols.gmt",
    "msigdb_oncogenic": "s3://say2-4team/Lung_raw/msigdb/c6.all.v2026.1.Hs.symbols.gmt",
    "gtex_lung_tpm": "s3://say2-4team/Lung_raw/gtex/v8/rna-seq/tpms-by-tissue/gene_tpm_2017-06-05_v8_lung.gct.gz",
    "gtex_sample_attributes": "s3://say2-4team/Lung_raw/gtex/GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt",
    "gtex_subject_phenotypes": "s3://say2-4team/Lung_raw/gtex/GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS.txt",
}

COHORTS = {
    "LUAD": {
        "project_id": "TCGA-LUAD",
        "counts_source_key": "tcga_luad_counts",
        "manifest_source_key": "tcga_luad_manifest",
    },
    "LUSC": {
        "project_id": "TCGA-LUSC",
        "counts_source_key": "tcga_lusc_counts",
        "manifest_source_key": "tcga_lusc_manifest",
    },
}


def _stage_settings(cfg: dict[str, Any]) -> dict[str, Any]:
    section = cfg.get("disease_context", {})
    sources = {**DEFAULT_DISEASE_CONTEXT_SOURCES, **section.get("sources", {})}
    return {
        "cache_dir": resolve_repo_path(cfg, section.get("cache_dir", ".cache/disease_context")),
        "sources": sources,
        "expression_value_column": section.get("expression_value_column", "tpm_unstranded"),
        "gene_types": section.get("gene_types", ["protein_coding"]),
        "minimum_pathway_genes": int(section.get("minimum_pathway_genes", 10)),
        "patient_feature_top_k_hallmark": int(section.get("patient_feature_top_k_hallmark", 25)),
        "max_files_per_cohort": section.get("max_files_per_cohort"),
    }


def run(cfg: dict[str, Any], dry_run: bool = False) -> dict[str, Any]:
    inputs = DATASET_BUCKETS["disease_context"] + ["msigdb"]
    notes = [
        "Build LUAD/LUSC disease signatures first.",
        "Use TCGA RNA STAR counts as the primary cohort expression source.",
        "Use GTEx only when a normal-lung expression matrix is available; annotation-only snapshots are treated as optional metadata.",
        "Treat CPTAC as downstream validation support before direct training features.",
    ]
    settings = _stage_settings(cfg)
    if dry_run:
        return build_stage_manifest(
            cfg,
            "build_disease_context",
            inputs,
            notes,
            dry_run,
            status="dry_run",
            extra={
                "implementation_state": "TCGA + MSigDB disease-context builder ready",
                "expression_value_column": settings["expression_value_column"],
                "minimum_pathway_genes": settings["minimum_pathway_genes"],
                "patient_feature_top_k_hallmark": settings["patient_feature_top_k_hallmark"],
            },
        )

    output_dir = ensure_dir(stage_output_dir(cfg, "build_disease_context"))
    cache_dir = ensure_dir(Path(settings["cache_dir"]))
    msigdb_dir = ensure_dir(cache_dir / "msigdb")

    pathway_sets = _load_pathway_sets(settings["sources"], msigdb_dir)
    gtex_reference, gtex_stats = _load_gtex_reference(settings["sources"], cache_dir)
    signatures: dict[str, pd.DataFrame] = {}
    signature_counts: dict[str, int] = {}
    patient_feature_frames: list[pd.DataFrame] = []
    pathway_score_frames: list[pd.DataFrame] = []
    cohort_stats: dict[str, Any] = {}

    for cohort_name, cohort_cfg in COHORTS.items():
        cohort_cache = ensure_dir(cache_dir / cohort_name.lower())
        entries = load_count_entries(
            settings["sources"][cohort_cfg["counts_source_key"]],
            cohort_cache,
            manifest_source=settings["sources"].get(cohort_cfg["manifest_source_key"]),
            max_files=settings["max_files_per_cohort"],
        )
        if not entries:
            raise ValueError(f"No TCGA RNA count files found for cohort {cohort_name}")

        signature_df, sample_features_df, pathway_scores_df, stats = _build_cohort_context(
            cohort_name=cohort_name,
            project_id=cohort_cfg["project_id"],
            entries=entries,
            cache_dir=cohort_cache,
            expression_value_column=settings["expression_value_column"],
            gene_types=settings["gene_types"],
            pathway_sets=pathway_sets,
            minimum_pathway_genes=settings["minimum_pathway_genes"],
        )
        signatures[cohort_name] = signature_df
        signature_counts[cohort_name] = int(stats["n_samples"])
        patient_feature_frames.append(sample_features_df)
        if not pathway_scores_df.empty:
            pathway_score_frames.append(pathway_scores_df)
        cohort_stats[cohort_name] = stats

    signatures = _attach_signature_deltas(signatures, signature_counts, gtex_reference)
    pathway_scores = (
        pd.concat(pathway_score_frames, ignore_index=True)
        if pathway_score_frames
        else pd.DataFrame(
            columns=[
                "sample_id",
                "cohort",
                "tcga_project_id",
                "collection",
                "pathway_name",
                "n_genes_in_set",
                "n_genes_used",
                "pathway_score",
            ]
        )
    )
    pathway_activity = _summarize_pathway_activity(pathway_scores)
    patient_features = _build_patient_features(
        pd.concat(patient_feature_frames, ignore_index=True),
        pathway_scores,
        top_k=settings["patient_feature_top_k_hallmark"],
    )

    luad_path = output_dir / "luad_signature.parquet"
    lusc_path = output_dir / "lusc_signature.parquet"
    pathway_path = output_dir / "pathway_activity.parquet"
    patient_path = output_dir / "patient_features.parquet"
    signatures["LUAD"].to_parquet(luad_path, index=False)
    signatures["LUSC"].to_parquet(lusc_path, index=False)
    pathway_activity.to_parquet(pathway_path, index=False)
    patient_features.to_parquet(patient_path, index=False)

    write_json(
        output_dir / "disease_context_summary.json",
        {
            "cohort_stats": cohort_stats,
            "gtex_lung_reference_stats": gtex_stats,
            "pathway_collections": {
                name: {"n_pathways": len(sets)}
                for name, sets in pathway_sets.items()
            },
            "outputs": {
                "luad_signature_rows": int(signatures["LUAD"].shape[0]),
                "lusc_signature_rows": int(signatures["LUSC"].shape[0]),
                "pathway_activity_rows": int(pathway_activity.shape[0]),
                "patient_features_rows": int(patient_features.shape[0]),
                "patient_features_columns": int(patient_features.shape[1]),
            },
        },
    )

    return build_stage_manifest(
        cfg,
        "build_disease_context",
        inputs,
        notes,
        dry_run,
        status="implemented",
        extra={
            "implementation_state": "TCGA + MSigDB disease-context outputs built",
            "cohort_stats": cohort_stats,
            "gtex_lung_reference_stats": gtex_stats,
            "pathway_collections": {
                name: len(sets) for name, sets in pathway_sets.items()
            },
            "patient_feature_columns": int(patient_features.shape[1]),
        },
    )


def _load_pathway_sets(
    sources: dict[str, Any],
    cache_dir: Path,
) -> dict[str, dict[str, list[str]]]:
    collections: dict[str, dict[str, list[str]]] = {}
    hallmark_source = sources.get("msigdb_hallmark")
    if hallmark_source:
        collections["hallmark"] = load_gmt_sets(hallmark_source, cache_dir)
    oncogenic_source = sources.get("msigdb_oncogenic")
    if oncogenic_source:
        collections["oncogenic"] = load_gmt_sets(oncogenic_source, cache_dir)
    return collections


def _load_gtex_reference(
    sources: dict[str, Any],
    cache_dir: Path,
) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    gtex_source = sources.get("gtex_lung_tpm")
    if not gtex_source:
        return None, {"status": "missing_source"}
    reference, stats = load_gtex_lung_reference(gtex_source, cache_dir / "gtex")
    return reference, {"status": "loaded", **stats}


def _build_cohort_context(
    *,
    cohort_name: str,
    project_id: str,
    entries: list[dict[str, Any]],
    cache_dir: Path,
    expression_value_column: str,
    gene_types: list[str],
    pathway_sets: dict[str, dict[str, list[str]]],
    minimum_pathway_genes: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    gene_sum: pd.Series | None = None
    gene_z_sum: pd.Series | None = None
    sample_rows: list[dict[str, Any]] = []
    pathway_rows: list[dict[str, Any]] = []
    n_samples = 0

    for entry in entries:
        expression, stats = read_expression_profile(
            entry["source_path"],
            cache_dir,
            value_column=expression_value_column,
            gene_types=gene_types,
        )
        if expression.empty:
            continue
        n_samples += 1
        sample_mean = float(expression.mean())
        sample_std = float(expression.std(ddof=0))
        if sample_std <= 0.0:
            sample_std = 1.0
        z_scores = (expression - sample_mean) / sample_std

        gene_sum = expression if gene_sum is None else gene_sum.add(expression, fill_value=0.0)
        gene_z_sum = z_scores if gene_z_sum is None else gene_z_sum.add(z_scores, fill_value=0.0)

        sample_rows.append(
            {
                "sample_id": entry["sample_id"],
                "cohort": cohort_name,
                "tcga_project_id": project_id,
                "source_filename": entry["filename"],
                "gdc_file_id": entry["gdc_file_id"],
                **stats,
            }
        )
        for collection_name, pathways in pathway_sets.items():
            for pathway_name, genes in pathways.items():
                values = z_scores.reindex(genes).dropna()
                if int(values.shape[0]) < minimum_pathway_genes:
                    continue
                pathway_rows.append(
                    {
                        "sample_id": entry["sample_id"],
                        "cohort": cohort_name,
                        "tcga_project_id": project_id,
                        "collection": collection_name,
                        "pathway_name": pathway_name,
                        "n_genes_in_set": len(genes),
                        "n_genes_used": int(values.shape[0]),
                        "pathway_score": float(values.mean()),
                    }
                )

    if n_samples == 0 or gene_sum is None or gene_z_sum is None:
        raise ValueError(f"No valid expression profiles were processed for cohort {cohort_name}")

    signature = pd.DataFrame(
        {
            "gene_symbol": gene_sum.index,
            "cohort": cohort_name,
            "tcga_project_id": project_id,
            "n_samples": n_samples,
            "mean_log2_tpm": gene_sum / n_samples,
            "mean_within_sample_z": gene_z_sum / n_samples,
        }
    ).reset_index(drop=True)
    signature = signature.sort_values("mean_log2_tpm", ascending=False).reset_index(drop=True)

    sample_features = pd.DataFrame(sample_rows)
    pathway_scores = pd.DataFrame(pathway_rows)
    stats = {
        "n_samples": n_samples,
        "n_signature_genes": int(signature.shape[0]),
        "n_pathway_rows": int(pathway_scores.shape[0]),
        "mean_detected_genes": float(sample_features["expr__detected_genes"].mean()),
        "mean_log2_tpm": float(sample_features["expr__mean_log2_tpm"].mean()),
    }
    return signature, sample_features, pathway_scores, stats


def _attach_signature_deltas(
    signatures: dict[str, pd.DataFrame],
    sample_counts: dict[str, int],
    gtex_reference: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    merged = None
    for cohort_name, df in signatures.items():
        subset = df[["gene_symbol", "mean_log2_tpm"]].rename(
            columns={"mean_log2_tpm": f"{cohort_name.lower()}_mean_log2_tpm"}
        )
        merged = subset if merged is None else merged.merge(subset, on="gene_symbol", how="outer")
    if merged is None:
        return signatures

    weighted_total = 0.0
    total_samples = float(sum(sample_counts.values()))
    for cohort_name, n_samples in sample_counts.items():
        weighted_total += merged[f"{cohort_name.lower()}_mean_log2_tpm"].fillna(0.0) * float(n_samples)
    merged["pooled_lung_mean_log2_tpm"] = weighted_total / total_samples
    if gtex_reference is not None and not gtex_reference.empty:
        normal_lookup = gtex_reference[["gene_symbol", "normal_lung_mean_log2_tpm"]].drop_duplicates("gene_symbol")
        merged = merged.merge(normal_lookup, on="gene_symbol", how="left")
    else:
        merged["normal_lung_mean_log2_tpm"] = 0.0
    merged["normal_lung_mean_log2_tpm"] = merged["normal_lung_mean_log2_tpm"].fillna(0.0)

    cohorts = list(signatures)
    for cohort_name, df in signatures.items():
        other = next(name for name in cohorts if name != cohort_name)
        lookup = merged.set_index("gene_symbol")
        df["other_cohort_mean_log2_tpm"] = lookup.loc[df["gene_symbol"], f"{other.lower()}_mean_log2_tpm"].to_numpy()
        df["pooled_lung_mean_log2_tpm"] = lookup.loc[df["gene_symbol"], "pooled_lung_mean_log2_tpm"].to_numpy()
        df["normal_lung_mean_log2_tpm"] = lookup.loc[df["gene_symbol"], "normal_lung_mean_log2_tpm"].to_numpy()
        df["delta_vs_other_cohort"] = df["mean_log2_tpm"] - df["other_cohort_mean_log2_tpm"]
        df["delta_vs_pooled_lung"] = df["mean_log2_tpm"] - df["pooled_lung_mean_log2_tpm"]
        df["delta_vs_normal_lung"] = df["mean_log2_tpm"] - df["normal_lung_mean_log2_tpm"]
        df["abs_delta_rank"] = (
            df["delta_vs_other_cohort"].abs().rank(method="dense", ascending=False).astype(int)
        )
        df["normal_abs_delta_rank"] = (
            df["delta_vs_normal_lung"].abs().rank(method="dense", ascending=False).astype(int)
        )
        signatures[cohort_name] = df.sort_values(
            ["abs_delta_rank", "normal_abs_delta_rank", "mean_log2_tpm"],
            ascending=[True, True, False],
        ).reset_index(drop=True)
    return signatures


def _summarize_pathway_activity(pathway_scores: pd.DataFrame) -> pd.DataFrame:
    if pathway_scores.empty:
        return pd.DataFrame(
            columns=[
                "collection",
                "pathway_name",
                "luad_mean_pathway_score",
                "lusc_mean_pathway_score",
                "delta_luad_minus_lusc",
            ]
        )

    grouped = (
        pathway_scores.groupby(["collection", "pathway_name", "cohort"])
        .agg(
            mean_pathway_score=("pathway_score", "mean"),
            median_pathway_score=("pathway_score", "median"),
            std_pathway_score=("pathway_score", "std"),
            n_samples=("sample_id", "nunique"),
            mean_genes_used=("n_genes_used", "mean"),
            n_genes_in_set=("n_genes_in_set", "max"),
        )
        .unstack("cohort")
    )
    grouped.columns = [
        f"{cohort.lower()}_{metric}" for metric, cohort in grouped.columns.to_flat_index()
    ]
    grouped = grouped.reset_index()
    if "luad_mean_pathway_score" not in grouped.columns:
        grouped["luad_mean_pathway_score"] = 0.0
    if "lusc_mean_pathway_score" not in grouped.columns:
        grouped["lusc_mean_pathway_score"] = 0.0
    grouped["delta_luad_minus_lusc"] = (
        grouped["luad_mean_pathway_score"] - grouped["lusc_mean_pathway_score"]
    )
    return grouped.sort_values(
        ["collection", "delta_luad_minus_lusc"],
        ascending=[True, False],
    ).reset_index(drop=True)


def _build_patient_features(
    sample_features: pd.DataFrame,
    pathway_scores: pd.DataFrame,
    *,
    top_k: int,
) -> pd.DataFrame:
    if pathway_scores.empty:
        return sample_features.sort_values(["cohort", "sample_id"]).reset_index(drop=True)

    hallmark = pathway_scores[pathway_scores["collection"] == "hallmark"].copy()
    if hallmark.empty:
        return sample_features.sort_values(["cohort", "sample_id"]).reset_index(drop=True)

    ranking = (
        hallmark.groupby(["pathway_name", "cohort"])["pathway_score"]
        .mean()
        .unstack("cohort")
        .fillna(0.0)
    )
    ranking["abs_delta"] = (ranking.get("LUAD", 0.0) - ranking.get("LUSC", 0.0)).abs()
    top_pathways = ranking.sort_values("abs_delta", ascending=False).head(top_k).index.tolist()

    wide = (
        hallmark[hallmark["pathway_name"].isin(top_pathways)]
        .pivot_table(
            index=["sample_id", "cohort", "tcga_project_id"],
            columns="pathway_name",
            values="pathway_score",
            aggfunc="mean",
        )
        .reset_index()
    )
    renamed = {}
    for column in wide.columns:
        if column in {"sample_id", "cohort", "tcga_project_id"}:
            continue
        renamed[column] = f"pathway__hallmark__{safe_feature_name(str(column))}"
    wide = wide.rename(columns=renamed)

    merged = sample_features.merge(
        wide,
        on=["sample_id", "cohort", "tcga_project_id"],
        how="left",
    )
    return merged.sort_values(["cohort", "sample_id"]).reset_index(drop=True)
