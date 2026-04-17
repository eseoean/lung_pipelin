from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..config import stage_output_dir
from ..io import ensure_dir, write_json
from ._common import build_stage_manifest

DEFAULT_TRAIN_BASELINE_SETTINGS = {
    "mode": "quick_gtex_groupcv_ablation",
    "target_column": "label_regression",
    "group_column": "canonical_drug_id",
    "n_splits": 3,
    "conditions": ["no_gtex", "sample_only", "pair_only", "full_gtex"],
    "models": ["random_forest", "flat_mlp"],
    "random_forest": {
        "n_estimators": 240,
        "max_depth": None,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "n_jobs": -1,
        "random_state": 42,
    },
    "flat_mlp": {
        "hidden_dims": [128, 64],
        "dropout": 0.1,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "batch_size": 512,
        "epochs": 18,
        "random_state": 42,
        "device": "auto",
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
                "models": settings["models"],
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

    best = _best_model_condition(groupcv_results)

    return build_stage_manifest(
        cfg,
        "train_baseline",
        ["model_input_table"],
        notes + [
            f"Executed quick GTEx ablation with models: {', '.join(settings['models'])}.",
            f"Conditions: {', '.join(settings['conditions'])}",
            f"Best GroupCV result: {best['model']} / {best['condition']} ({best['spearman']:.4f})",
        ],
        dry_run=False,
        status="implemented",
        extra={
            "mode": settings["mode"],
            "models": settings["models"],
            "conditions": settings["conditions"],
            "best_result": best,
        },
    )


def _stage_settings(cfg: dict[str, Any]) -> dict[str, Any]:
    section = cfg.get("train_baseline", {})
    settings = {**DEFAULT_TRAIN_BASELINE_SETTINGS, **section}
    settings["random_forest"] = {
        **DEFAULT_TRAIN_BASELINE_SETTINGS["random_forest"],
        **section.get("random_forest", {}),
    }
    settings["flat_mlp"] = {
        **DEFAULT_TRAIN_BASELINE_SETTINGS["flat_mlp"],
        **section.get("flat_mlp", {}),
    }
    settings["n_splits"] = int(settings["n_splits"])
    settings["conditions"] = [str(value) for value in settings["conditions"]]
    settings["models"] = [str(value) for value in settings["models"]]
    settings["flat_mlp"]["hidden_dims"] = [int(v) for v in settings["flat_mlp"]["hidden_dims"]]
    settings["flat_mlp"]["batch_size"] = int(settings["flat_mlp"]["batch_size"])
    settings["flat_mlp"]["epochs"] = int(settings["flat_mlp"]["epochs"])
    settings["flat_mlp"]["random_state"] = int(settings["flat_mlp"]["random_state"])
    settings["flat_mlp"]["learning_rate"] = float(settings["flat_mlp"]["learning_rate"])
    settings["flat_mlp"]["weight_decay"] = float(settings["flat_mlp"]["weight_decay"])
    settings["flat_mlp"]["dropout"] = float(settings["flat_mlp"]["dropout"])
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

    model_results: dict[str, Any] = {}
    oof_manifest: dict[str, Any] = {"files": []}

    for model_name in settings["models"]:
        model_results[model_name] = {
            "n_rows": int(train_table.shape[0]),
            "n_numeric_columns_full": int(len(numeric_columns)),
            "conditions": {},
        }
        for condition in settings["conditions"]:
            feature_columns = _feature_columns_for_condition(numeric_columns, condition)
            X = train_table[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
            if model_name == "random_forest":
                metrics, oof = _run_random_forest_groupcv(
                    X=X,
                    y=y,
                    split_indices=split_indices,
                    settings=settings["random_forest"],
                )
            elif model_name == "flat_mlp":
                metrics, oof = _run_flat_mlp_groupcv(
                    X=X,
                    y=y,
                    split_indices=split_indices,
                    settings=settings["flat_mlp"],
                )
            else:
                raise ValueError(f"Unsupported quick ablation model: {model_name}")

            metrics["n_features"] = int(len(feature_columns))
            metrics["feature_columns"] = feature_columns
            model_results[model_name]["conditions"][condition] = metrics

            oof_path = oof_dir / f"gtex_ablation_groupcv_{model_name}_{condition}.parquet"
            pd.DataFrame(
                {
                    "pair_id": pair_ids,
                    "group": groups,
                    "y_true": y,
                    "y_pred": oof,
                    "condition": condition,
                    "model": model_name,
                }
            ).to_parquet(oof_path, index=False)
            oof_manifest["files"].append(
                {
                    "model": model_name,
                    "condition": condition,
                    "path": str(oof_path),
                    "n_features": int(len(feature_columns)),
                }
            )

    return {
        "mode": settings["mode"],
        "models": model_results,
    }, oof_manifest


def _run_random_forest_groupcv(
    *,
    X: np.ndarray,
    y: np.ndarray,
    split_indices: list[tuple[np.ndarray, np.ndarray]],
    settings: dict[str, Any],
) -> tuple[dict[str, Any], np.ndarray]:
    oof = np.zeros(X.shape[0], dtype=float)
    fold_metrics: list[dict[str, Any]] = []
    for fold_idx, (train_idx, valid_idx) in enumerate(split_indices, start=1):
        model = RandomForestRegressor(**settings)
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
    return {
        "spearman": _spearman(y, oof),
        "rmse": _rmse(y, oof),
        "fold_metrics": fold_metrics,
    }, oof


def _run_flat_mlp_groupcv(
    *,
    X: np.ndarray,
    y: np.ndarray,
    split_indices: list[tuple[np.ndarray, np.ndarray]],
    settings: dict[str, Any],
) -> tuple[dict[str, Any], np.ndarray]:
    device = _resolve_torch_device(settings["device"])
    oof = np.zeros(X.shape[0], dtype=float)
    fold_metrics: list[dict[str, Any]] = []

    for fold_idx, (train_idx, valid_idx) in enumerate(split_indices, start=1):
        preds = _fit_predict_flat_mlp(
            X_train=X[train_idx],
            y_train=y[train_idx],
            X_valid=X[valid_idx],
            settings=settings,
            device=device,
            fold_seed=settings["random_state"] + fold_idx,
        )
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

    return {
        "device": str(device),
        "spearman": _spearman(y, oof),
        "rmse": _rmse(y, oof),
        "fold_metrics": fold_metrics,
    }, oof


def _fit_predict_flat_mlp(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    settings: dict[str, Any],
    device: torch.device,
    fold_seed: int,
) -> np.ndarray:
    torch.manual_seed(fold_seed)
    np.random.seed(fold_seed)

    x_mean = X_train.mean(axis=0, keepdims=True)
    x_std = X_train.std(axis=0, keepdims=True)
    x_std[x_std < 1e-6] = 1.0
    X_train_scaled = ((X_train - x_mean) / x_std).astype(np.float32)
    X_valid_scaled = ((X_valid - x_mean) / x_std).astype(np.float32)

    y_mean = float(y_train.mean())
    y_std = float(y_train.std())
    if y_std < 1e-6:
        y_std = 1.0
    y_train_scaled = ((y_train - y_mean) / y_std).astype(np.float32)

    train_dataset = TensorDataset(
        torch.from_numpy(X_train_scaled),
        torch.from_numpy(y_train_scaled).unsqueeze(1),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=settings["batch_size"],
        shuffle=True,
    )

    model = _FlatMLP(
        input_dim=X_train.shape[1],
        hidden_dims=settings["hidden_dims"],
        dropout=settings["dropout"],
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings["learning_rate"],
        weight_decay=settings["weight_decay"],
    )
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(settings["epochs"]):
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        preds_scaled = model(torch.from_numpy(X_valid_scaled).to(device)).squeeze(1).cpu().numpy()
    return preds_scaled.astype(float) * y_std + y_mean


class _FlatMLP(nn.Module):
    def __init__(self, *, input_dim: int, hidden_dims: list[int], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def _resolve_torch_device(device_name: str) -> torch.device:
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "mps":
        return torch.device("mps")
    if device_name == "cuda":
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _best_model_condition(groupcv_results: dict[str, Any]) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for model_name, model_payload in groupcv_results["models"].items():
        for condition, metrics in model_payload["conditions"].items():
            candidate = {
                "model": model_name,
                "condition": condition,
                "spearman": metrics["spearman"],
                "rmse": metrics["rmse"],
            }
            if best is None or candidate["spearman"] > best["spearman"]:
                best = candidate
    if best is None:
        raise ValueError("No GroupCV results were produced.")
    return best


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
