from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..config import resolve_repo_path, stage_output_dir
from ..io import ensure_dir, write_json
from ._common import build_stage_manifest

DEFAULT_SLIM_INPUT_SETTINGS = {
    "input_dir": "",
    "filter_invalid_smiles_rows": True,
    "gene_low_variance_remove_count": 13719,
    "morgan_var_threshold": 0.01,
    "correlation_threshold": 0.95,
}


def run(cfg: dict[str, Any], dry_run: bool = False) -> dict[str, Any]:
    notes = [
        "Start from the rich train_table and prune only the high-dimensional sample/drug blocks.",
        "Mimic the old exact-slim recipe: invalid-smiles filtering, gene variance pruning, Morgan variance pruning, then correlation pruning.",
        "Keep non-CRISPR and non-Morgan context blocks unchanged for fair comparison.",
    ]
    settings = _stage_settings(cfg)
    input_dir = (
        resolve_repo_path(cfg, settings["input_dir"])
        if settings["input_dir"]
        else stage_output_dir(cfg, "build_model_inputs")
    )
    if dry_run:
        return build_stage_manifest(
            cfg,
            "build_slim_inputs",
            ["build_model_inputs_outputs"],
            notes,
            dry_run,
            status="dry_run",
            extra={
                "input_dir": str(input_dir),
                "filter_invalid_smiles_rows": settings["filter_invalid_smiles_rows"],
                "gene_low_variance_remove_count": settings["gene_low_variance_remove_count"],
                "morgan_var_threshold": settings["morgan_var_threshold"],
                "correlation_threshold": settings["correlation_threshold"],
            },
        )

    output_dir = ensure_dir(stage_output_dir(cfg, "build_slim_inputs"))
    train_table = pd.read_parquet(input_dir / "train_table.parquet")
    labels_y = pd.read_parquet(input_dir / "labels_y.parquet")

    slim_table, slim_labels, summary = _build_slim_tables(
        train_table=train_table,
        labels_y=labels_y,
        settings=settings,
    )

    slim_table.to_parquet(output_dir / "train_table.parquet", index=False)
    slim_labels.to_parquet(output_dir / "labels_y.parquet", index=False)
    write_json(output_dir / "slim_input_summary.json", summary)

    notes = notes + [
        f"Input rows: {summary['input_shape'][0]}",
        f"Slim rows: {summary['slim_shape'][0]}",
        f"Slim columns: {summary['slim_shape'][1]}",
        f"CRISPR columns kept: {summary['group_counts_slim']['gene']}",
        f"Morgan columns kept: {summary['group_counts_slim']['morgan']}",
    ]
    return build_stage_manifest(
        cfg,
        "build_slim_inputs",
        ["build_model_inputs_outputs"],
        notes,
        dry_run=False,
        status="implemented",
        extra=summary,
    )


def _stage_settings(cfg: dict[str, Any]) -> dict[str, Any]:
    section = cfg.get("slim_inputs", {})
    settings = {**DEFAULT_SLIM_INPUT_SETTINGS, **section}
    settings["input_dir"] = str(settings["input_dir"])
    settings["filter_invalid_smiles_rows"] = bool(settings["filter_invalid_smiles_rows"])
    settings["gene_low_variance_remove_count"] = int(settings["gene_low_variance_remove_count"])
    settings["morgan_var_threshold"] = float(settings["morgan_var_threshold"])
    settings["correlation_threshold"] = float(settings["correlation_threshold"])
    return settings


def _build_slim_tables(
    *,
    train_table: pd.DataFrame,
    labels_y: pd.DataFrame,
    settings: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    filtered = train_table.copy()
    invalid_smiles_rows = 0

    gene_cols = [column for column in filtered.columns if column.startswith("sample__crispr__")]
    morgan_cols = [column for column in filtered.columns if column.startswith("drug_morgan_")]

    if settings["filter_invalid_smiles_rows"] and "drug_has_valid_smiles" in filtered.columns and morgan_cols:
        morgan_sum = filtered[morgan_cols].abs().sum(axis=1)
        invalid_mask = (pd.to_numeric(filtered["drug_has_valid_smiles"], errors="coerce").fillna(0) <= 0) | (
            morgan_sum <= 0
        )
        invalid_smiles_rows = int(invalid_mask.sum())
        filtered = filtered.loc[~invalid_mask].reset_index(drop=True)

    gene_cols = [column for column in filtered.columns if column.startswith("sample__crispr__")]
    morgan_cols = [column for column in filtered.columns if column.startswith("drug_morgan_")]

    gene_keep, gene_lowvar_drop, gene_corr_drop = _prune_gene_columns(
        filtered,
        gene_cols,
        remove_count=settings["gene_low_variance_remove_count"],
        corr_threshold=settings["correlation_threshold"],
    )
    morgan_keep, morgan_lowvar_drop, morgan_corr_drop = _prune_morgan_columns(
        filtered,
        morgan_cols,
        var_threshold=settings["morgan_var_threshold"],
        corr_threshold=settings["correlation_threshold"],
    )

    drop_cols = set(gene_lowvar_drop) | set(gene_corr_drop) | set(morgan_lowvar_drop) | set(morgan_corr_drop)
    slim_table = filtered[[column for column in filtered.columns if column not in drop_cols]].copy()
    if "pair_id" in labels_y.columns:
        slim_labels = labels_y[labels_y["pair_id"].isin(set(slim_table["pair_id"]))].reset_index(drop=True)
    else:
        slim_labels = labels_y.copy()

    summary = {
        "input_shape": [int(train_table.shape[0]), int(train_table.shape[1])],
        "slim_shape": [int(slim_table.shape[0]), int(slim_table.shape[1])],
        "invalid_smiles_rows_removed": int(invalid_smiles_rows),
        "group_counts_input": {
            "gene": int(len(gene_cols)),
            "morgan": int(len(morgan_cols)),
        },
        "group_counts_slim": {
            "gene": int(len(gene_keep)),
            "morgan": int(len(morgan_keep)),
        },
        "feature_selection": {
            "gene_low_variance_removed": int(len(gene_lowvar_drop)),
            "gene_corr_removed": int(len(gene_corr_drop)),
            "morgan_low_variance_removed": int(len(morgan_lowvar_drop)),
            "morgan_corr_removed": int(len(morgan_corr_drop)),
            "gene_low_variance_remove_count": int(settings["gene_low_variance_remove_count"]),
            "morgan_var_threshold": float(settings["morgan_var_threshold"]),
            "correlation_threshold": float(settings["correlation_threshold"]),
        },
    }
    return slim_table, slim_labels, summary


def _prune_gene_columns(
    frame: pd.DataFrame,
    columns: list[str],
    *,
    remove_count: int,
    corr_threshold: float,
) -> tuple[list[str], list[str], list[str]]:
    if not columns:
        return [], [], []
    gene = frame[columns].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(np.float32)
    if gene.shape[1] <= 1:
        return columns, [], []
    remove_count = min(remove_count, max(gene.shape[1] - 1, 0))
    gene_var = gene.var(axis=0)
    lowvar_drop = gene_var.sort_values(kind="mergesort").index[:remove_count].tolist()
    gene_after_lowvar = gene.drop(columns=lowvar_drop)
    gene_after_corr, corr_drop = _correlation_prune(gene_after_lowvar, corr_threshold)
    return list(gene_after_corr.columns), lowvar_drop, corr_drop


def _prune_morgan_columns(
    frame: pd.DataFrame,
    columns: list[str],
    *,
    var_threshold: float,
    corr_threshold: float,
) -> tuple[list[str], list[str], list[str]]:
    if not columns:
        return [], [], []
    morgan = frame[columns].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(np.float32)
    if morgan.shape[1] <= 1:
        return columns, [], []
    morgan_var = morgan.var(axis=0)
    lowvar_drop = morgan_var.index[morgan_var < var_threshold].tolist()
    morgan_after_lowvar = morgan.drop(columns=lowvar_drop)
    morgan_after_corr, corr_drop = _correlation_prune(morgan_after_lowvar, corr_threshold)
    return list(morgan_after_corr.columns), lowvar_drop, corr_drop


def _correlation_prune(frame: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, list[str]]:
    if frame.shape[1] <= 1:
        return frame.copy(), []
    corr = frame.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop_cols = [column for column in upper.columns if (upper[column] > threshold).any()]
    kept = frame.drop(columns=drop_cols)
    return kept, drop_cols
