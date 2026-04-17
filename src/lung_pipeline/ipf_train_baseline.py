from __future__ import annotations

import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, ndcg_score, r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .io import write_json


EstimatorFactory = Callable[[int], Any]


def _safe_corr(y_true: np.ndarray, y_pred: np.ndarray, method: str) -> float:
    if len(y_true) == 0:
        return 0.0
    if np.allclose(y_true, y_true[0]) or np.allclose(y_pred, y_pred[0]):
        return 0.0
    true_series = pd.Series(y_true)
    pred_series = pd.Series(y_pred)
    value = true_series.corr(pred_series, method=method)
    if pd.isna(value):
        return 0.0
    return float(value)


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, *, top_k: int = 20) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    k = min(top_k, len(y_true))
    actual_top = set(np.argsort(-y_true)[:k].tolist())
    predicted_top = set(np.argsort(-y_pred)[:k].tolist())
    return {
        "spearman": _safe_corr(y_true, y_pred, "spearman"),
        "pearson": _safe_corr(y_true, y_pred, "pearson"),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "ndcg_at_20": float(ndcg_score(y_true.reshape(1, -1), y_pred.reshape(1, -1), k=k)),
        "top20_overlap": float(len(actual_top & predicted_top) / max(1, k)),
    }


def _feature_columns(frame: pd.DataFrame) -> list[str]:
    numeric_columns = frame.select_dtypes(include=[np.number, bool]).columns.tolist()
    excluded = {"pseudo_label_score", "pseudo_label_rank"}
    return [col for col in numeric_columns if col not in excluded]


def _model_factories() -> dict[str, EstimatorFactory]:
    return {
        "DummyMean": lambda seed: DummyRegressor(strategy="mean"),
        "Ridge": lambda seed: Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", Ridge(alpha=1.0, random_state=seed)),
            ]
        ),
        "RandomForest": lambda seed: RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=3,
            random_state=seed,
            n_jobs=-1,
        ),
        "HistGradientBoosting": lambda seed: HistGradientBoostingRegressor(
            max_depth=6,
            learning_rate=0.05,
            max_iter=300,
            random_state=seed,
        ),
    }


def _dominant_accession_groups(
    train_table: pd.DataFrame,
    disease_features: pd.DataFrame,
) -> tuple[pd.Series, dict[str, int]]:
    gene_to_sources: dict[str, list[str]] = {}
    for row in disease_features.itertuples(index=False):
        gene_name = str(getattr(row, "gene_name", "") or "")
        evidence_sources = str(getattr(row, "evidence_sources", "") or "")
        gene_to_sources[gene_name] = [token for token in evidence_sources.split("|") if token]

    groups: list[str] = []
    for overlap_genes in train_table["target_overlap_genes"].fillna("").astype(str):
        votes: Counter[str] = Counter()
        for gene in overlap_genes.split("|"):
            if not gene:
                continue
            for accession in gene_to_sources.get(gene, []):
                votes[accession] += 1
        groups.append(votes.most_common(1)[0][0] if votes else "NO_OVERLAP")

    series = pd.Series(groups, index=train_table.index, name="dominant_accession")
    return series, {key: int(value) for key, value in series.value_counts().to_dict().items()}


def _evaluate_split(
    frame: pd.DataFrame,
    *,
    feature_columns: list[str],
    y_column: str,
    split_name: str,
    splitter: Any,
    seed: int,
    groups: pd.Series | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    x = frame[feature_columns].fillna(0.0).astype(float).to_numpy()
    y = frame[y_column].astype(float).to_numpy()
    oof = frame[["drugbank_id", "drug_name", y_column]].copy()
    if groups is not None:
        oof["group_key"] = groups.astype(str).values

    model_results: list[dict[str, Any]] = []
    model_oof_predictions: dict[str, np.ndarray] = {}

    split_iter = splitter.split(x, y, groups) if groups is not None else splitter.split(x, y)
    splits = list(split_iter)

    for model_name, factory in _model_factories().items():
        predictions = np.zeros(len(frame), dtype=float)
        fold_metrics: list[dict[str, float]] = []
        start = time.time()

        for fold_idx, (train_idx, valid_idx) in enumerate(splits):
            estimator = factory(seed + fold_idx)
            estimator.fit(x[train_idx], y[train_idx])
            fold_pred = np.asarray(estimator.predict(x[valid_idx]), dtype=float)
            predictions[valid_idx] = fold_pred
            fold_metric = _compute_metrics(y[valid_idx], fold_pred)
            fold_metric["fold"] = fold_idx
            fold_metrics.append(fold_metric)

        metrics = _compute_metrics(y, predictions)
        metrics["elapsed_sec"] = float(time.time() - start)
        metrics["folds"] = len(splits)
        model_results.append(
            {
                "model_name": model_name,
                **metrics,
                "fold_metrics": fold_metrics,
                "top10_predicted_drugs": (
                    frame.assign(_pred=predictions)
                    .sort_values("_pred", ascending=False)["drug_name"]
                    .head(10)
                    .tolist()
                ),
            }
        )
        model_oof_predictions[model_name] = predictions
        oof[f"pred_{model_name}"] = predictions

    model_results = sorted(
        model_results,
        key=lambda item: (
            item["spearman"],
            item["ndcg_at_20"],
            item["pearson"],
        ),
        reverse=True,
    )

    return (
        {
            "split_name": split_name,
            "row_count": int(len(frame)),
            "feature_count": int(len(feature_columns)),
            "feature_columns": feature_columns,
            "best_model": model_results[0]["model_name"],
            "models": model_results,
        },
        oof,
    )


def run_ipf_train_baselines(
    *,
    train_table_parquet: Path,
    disease_features_parquet: Path,
    random_metrics_json: Path,
    accession_metrics_json: Path,
    ranking_manifest_json: Path,
    random_oof_csv: Path,
    accession_oof_csv: Path,
    random_seed: int = 42,
    n_splits: int = 3,
) -> dict[str, Any]:
    train_table = pd.read_parquet(train_table_parquet)
    disease_features = pd.read_parquet(disease_features_parquet)
    feature_columns = _feature_columns(train_table)
    if not feature_columns:
        raise ValueError("No numeric feature columns available for baseline training.")

    random_splitter = KFold(
        n_splits=min(n_splits, len(train_table)),
        shuffle=True,
        random_state=random_seed,
    )
    random_metrics, random_oof = _evaluate_split(
        train_table,
        feature_columns=feature_columns,
        y_column="pseudo_label_score",
        split_name="randomcv",
        splitter=random_splitter,
        seed=random_seed,
    )

    accession_groups, accession_distribution = _dominant_accession_groups(train_table, disease_features)
    unique_groups = accession_groups.nunique()
    accession_splits = min(n_splits, unique_groups)
    if accession_splits < 2:
        raise ValueError("Accession-aware baseline requires at least two accession groups.")
    accession_splitter = GroupKFold(n_splits=accession_splits)
    accession_metrics, accession_oof = _evaluate_split(
        train_table.assign(dominant_accession=accession_groups),
        feature_columns=feature_columns,
        y_column="pseudo_label_score",
        split_name="accessioncv",
        splitter=accession_splitter,
        seed=random_seed,
        groups=accession_groups,
    )
    accession_metrics["group_distribution"] = accession_distribution

    random_oof_csv.parent.mkdir(parents=True, exist_ok=True)
    accession_oof_csv.parent.mkdir(parents=True, exist_ok=True)
    random_oof.to_csv(random_oof_csv, index=False)
    accession_oof.to_csv(accession_oof_csv, index=False)
    write_json(random_metrics_json, random_metrics)
    write_json(accession_metrics_json, accession_metrics)

    manifest = {
        "stage": "train_baseline",
        "study": "IPF",
        "objective": "disease_signature_reversal",
        "label_source": "pseudo_label_score",
        "row_count": int(len(train_table)),
        "feature_count": int(len(feature_columns)),
        "random_seed": int(random_seed),
        "requested_splits": int(n_splits),
        "randomcv_best_model": random_metrics["best_model"],
        "accessioncv_best_model": accession_metrics["best_model"],
        "randomcv_best_spearman": float(random_metrics["models"][0]["spearman"]),
        "accessioncv_best_spearman": float(accession_metrics["models"][0]["spearman"]),
        "group_distribution": accession_distribution,
        "artifacts": {
            "random_metrics_json": str(random_metrics_json),
            "accession_metrics_json": str(accession_metrics_json),
            "random_oof_csv": str(random_oof_csv),
            "accession_oof_csv": str(accession_oof_csv),
            "train_table_parquet": str(train_table_parquet),
            "disease_features_parquet": str(disease_features_parquet),
        },
    }
    write_json(ranking_manifest_json, manifest)
    return manifest
