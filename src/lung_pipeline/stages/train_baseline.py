from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold

from ..config import stage_output_dir
from ..io import ensure_dir, write_json
from ._common import build_stage_manifest

DEFAULT_TRAIN_BASELINE_SETTINGS = {
    "mode": "quick_gtex_groupcv_ablation",
    "target_column": "label_regression",
    "group_column": "canonical_drug_id",
    "n_splits": 3,
    "conditions": ["no_gtex", "sample_only", "pair_only", "full_gtex"],
    "random_forest": {
        "n_estimators": 240,
        "max_depth": None,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "n_jobs": -1,
        "random_state": 42,
    },
}

GTEX_SAMPLE_COLUMNS = {
    "ctx__signature__mean_abs_delta_vs_normal_top",
}


def run(cfg: dict[str, Any], dry_run: bool = False) -> dict[str, Any]:
    notes = [
        "Start with simple baseline models before stacking more context.",
        "Use both GroupCV and random-split evaluation tracks.",
        "Group key defaults to canonical_drug_id from configs/lung.yaml.",
    ]
    settings = _stage_settings(cfg)
    if dry_run:
        return build_stage_manifest(
            cfg,
            "train_baseline",
            ["model_input_table"],
            notes,
            dry_run,
            status="dry_run",
            extra={
                "mode": settings["mode"],
                "conditions": settings["conditions"],
                "n_splits": settings["n_splits"],
            },
        )

    if settings["mode"] != "quick_gtex_groupcv_ablation":
        raise ValueError(f"Unsupported train_baseline mode: {settings['mode']}")

    model_input_dir = stage_output_dir(cfg, "build_model_inputs")
    output_dir = ensure_dir(stage_output_dir(cfg, "train_baseline"))
    oof_dir = ensure_dir(output_dir / "oof")

    train_table = pd.read_parquet(model_input_dir / "train_table.parquet")
    groupcv_results, oof_manifest = _run_quick_gtex_groupcv_ablation(
        train_table=train_table,
        settings=settings,
        oof_dir=oof_dir,
    )

    write_json(output_dir / "groupcv_metrics.json", groupcv_results)
    write_json(
        output_dir / "randomcv_metrics.json",
        {
            "status": "skipped",
            "reason": "quick_gtex_groupcv_ablation_mode_runs_groupcv_only",
        },
    )
    write_json(output_dir / "oof_manifest.json", oof_manifest)

    best_condition = max(
        groupcv_results["conditions"].items(),
        key=lambda item: item[1]["spearman"],
    )

    return build_stage_manifest(
        cfg,
        "train_baseline",
        ["model_input_table"],
        notes + [
            "Executed quick GTEx ablation with RandomForest only.",
            f"Conditions: {', '.join(settings['conditions'])}",
            f"Best GroupCV condition: {best_condition[0]} ({best_condition[1]['spearman']:.4f})",
        ],
        dry_run=False,
        status="implemented",
        extra={
            "mode": settings["mode"],
            "model": "RandomForestRegressor",
            "conditions": settings["conditions"],
            "best_condition": {
                "name": best_condition[0],
                "spearman": best_condition[1]["spearman"],
                "rmse": best_condition[1]["rmse"],
            },
        },
    )


def _stage_settings(cfg: dict[str, Any]) -> dict[str, Any]:
    section = cfg.get("train_baseline", {})
    settings = {**DEFAULT_TRAIN_BASELINE_SETTINGS, **section}
    settings["random_forest"] = {
        **DEFAULT_TRAIN_BASELINE_SETTINGS["random_forest"],
        **section.get("random_forest", {}),
    }
    settings["n_splits"] = int(settings["n_splits"])
    settings["conditions"] = [str(value) for value in settings["conditions"]]
    return settings


def _run_quick_gtex_groupcv_ablation(
    *,
    train_table: pd.DataFrame,
    settings: dict[str, Any],
    oof_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    target_column = settings["target_column"]
    group_column = settings["group_column"]

    y = pd.to_numeric(train_table[target_column], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    groups = train_table[group_column].astype(str).to_numpy()
    pair_ids = train_table["pair_id"].astype(str).to_numpy()

    numeric_columns = _numeric_feature_columns(train_table, exclude={target_column, "label_binary"})
    splitter = GroupKFold(n_splits=settings["n_splits"])
    split_indices = list(splitter.split(train_table, y, groups))

    results: dict[str, Any] = {
        "mode": settings["mode"],
        "model": "RandomForestRegressor",
        "n_rows": int(train_table.shape[0]),
        "n_numeric_columns_full": int(len(numeric_columns)),
        "conditions": {},
    }
    oof_manifest: dict[str, Any] = {"files": []}

    for condition in settings["conditions"]:
        feature_columns = _feature_columns_for_condition(numeric_columns, condition)
        X = train_table[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
        oof = np.zeros(train_table.shape[0], dtype=float)
        fold_metrics: list[dict[str, Any]] = []

        for fold_idx, (train_idx, valid_idx) in enumerate(split_indices, start=1):
            model = RandomForestRegressor(**settings["random_forest"])
            model.fit(X[train_idx], y[train_idx])
            preds = model.predict(X[valid_idx])
            oof[valid_idx] = preds

            fold_metrics.append(
                {
                    "fold": fold_idx,
                    "n_train": int(len(train_idx)),
                    "n_valid": int(len(valid_idx)),
                    "spearman": _spearman(y[valid_idx], preds),
                    "rmse": _rmse(y[valid_idx], preds),
                }
            )

        overall = {
            "n_features": int(len(feature_columns)),
            "feature_columns": feature_columns,
            "spearman": _spearman(y, oof),
            "rmse": _rmse(y, oof),
            "fold_metrics": fold_metrics,
        }
        results["conditions"][condition] = overall

        oof_path = oof_dir / f"gtex_ablation_groupcv_randomforest_{condition}.parquet"
        pd.DataFrame(
            {
                "pair_id": pair_ids,
                "group": groups,
                "y_true": y,
                "y_pred": oof,
                "condition": condition,
            }
        ).to_parquet(oof_path, index=False)
        oof_manifest["files"].append(
            {
                "condition": condition,
                "path": str(oof_path),
                "n_features": int(len(feature_columns)),
            }
        )

    return results, oof_manifest


def _numeric_feature_columns(train_table: pd.DataFrame, *, exclude: set[str]) -> list[str]:
    blocked = {
        "pair_id",
        "sample_id",
        "canonical_drug_id",
        "DRUG_ID",
        "cell_line_name",
        "drug_name",
        "gdsc_version",
        "putative_target",
        "pathway_name",
        "WEBRELEASE",
        "label_main_type",
        "label_aux_type",
        "model_id",
        "gdsc_cosmic_id",
        "cohort",
        "depmap_oncotree_code",
        "depmap_primary_disease",
        "match_source",
        "target_pathway",
        "drug__target_list",
        "pair__cohort",
        "label_main",
        "label_aux",
        "binary_threshold",
        "is_depmap_mapped",
    } | set(exclude)
    return [
        column
        for column in train_table.columns
        if column not in blocked and pd.api.types.is_numeric_dtype(train_table[column])
    ]


def _feature_columns_for_condition(columns: list[str], condition: str) -> list[str]:
    if condition not in {"no_gtex", "sample_only", "pair_only", "full_gtex"}:
        raise ValueError(f"Unsupported GTEx ablation condition: {condition}")

    sample_gtex_columns = [
        column
        for column in columns
        if column in GTEX_SAMPLE_COLUMNS or column.startswith("ctx__signormal__")
    ]
    pair_gtex_columns = [column for column in columns if "_vs_normal" in column]

    selected = list(columns)
    if condition == "no_gtex":
        selected = [c for c in selected if c not in sample_gtex_columns and c not in pair_gtex_columns]
    elif condition == "sample_only":
        selected = [c for c in selected if c not in pair_gtex_columns]
    elif condition == "pair_only":
        selected = [c for c in selected if c not in sample_gtex_columns]
    return selected


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    series_true = pd.Series(y_true)
    series_pred = pd.Series(y_pred)
    value = series_true.corr(series_pred, method="spearman")
    if pd.isna(value):
        return 0.0
    return float(value)
