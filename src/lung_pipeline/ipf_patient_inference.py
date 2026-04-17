from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .io import write_json
from .ipf_train_baseline import create_ipf_baseline_estimator, feature_columns_for_baseline


def _load_best_model(metrics_path: Path) -> tuple[str, dict[str, Any]]:
    metrics = json.loads(metrics_path.read_text())
    best = metrics["models"][0]
    return best["model_name"], metrics


def run_ipf_patient_inference(
    *,
    train_table_parquet: Path,
    accession_metrics_json: Path,
    random_metrics_json: Path,
    output_parquet: Path,
    manifest_json: Path,
    summary_json: Path,
    review_csv: Path,
    random_seed: int = 42,
) -> dict[str, Any]:
    train_table = pd.read_parquet(train_table_parquet)
    feature_columns = feature_columns_for_baseline(train_table)
    x = train_table[feature_columns].fillna(0.0).astype(float)
    y = train_table["pseudo_label_score"].astype(float)

    accession_model_name, accession_metrics = _load_best_model(accession_metrics_json)
    random_model_name, random_metrics = _load_best_model(random_metrics_json)

    accession_model = create_ipf_baseline_estimator(accession_model_name, random_seed)
    accession_model.fit(x, y)
    accession_pred = accession_model.predict(x)

    random_model = create_ipf_baseline_estimator(random_model_name, random_seed)
    random_model.fit(x, y)
    random_pred = random_model.predict(x)

    patient_scores = train_table.copy()
    patient_scores["pred_accessioncv"] = accession_pred
    patient_scores["pred_randomcv"] = random_pred
    patient_scores["consensus_model_score"] = (patient_scores["pred_accessioncv"] + patient_scores["pred_randomcv"]) / 2.0
    patient_scores["consensus_model_rank"] = (
        patient_scores["consensus_model_score"].rank(method="first", ascending=False).astype(int)
    )
    patient_scores = patient_scores.sort_values(
        ["consensus_model_score", "pseudo_label_score", "fibrosis_priority_overlap_count", "is_approved"],
        ascending=False,
    ).reset_index(drop=True)
    patient_scores["consensus_model_rank"] = range(1, len(patient_scores) + 1)

    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    review_csv.parent.mkdir(parents=True, exist_ok=True)
    patient_scores.to_parquet(output_parquet, index=False)
    patient_scores.to_csv(review_csv, index=False)

    summary = {
        "study": "IPF",
        "row_count": int(len(patient_scores)),
        "feature_count": int(len(feature_columns)),
        "best_accession_model": accession_model_name,
        "best_random_model": random_model_name,
        "best_accession_spearman": float(accession_metrics["models"][0]["spearman"]),
        "best_random_spearman": float(random_metrics["models"][0]["spearman"]),
        "top_ranked_drugs": patient_scores["drug_name"].head(10).tolist(),
        "output_parquet": str(output_parquet),
        "review_csv": str(review_csv),
    }
    write_json(summary_json, summary)

    manifest = {
        "stage": "patient_inference",
        "study": "IPF",
        "objective": "disease_signature_reversal",
        "selected_models": {
            "accessioncv": accession_model_name,
            "randomcv": random_model_name,
        },
        "row_count": int(len(patient_scores)),
        "feature_columns": feature_columns,
        "artifacts": {
            "train_table_parquet": str(train_table_parquet),
            "accession_metrics_json": str(accession_metrics_json),
            "random_metrics_json": str(random_metrics_json),
            "output_parquet": str(output_parquet),
            "review_csv": str(review_csv),
            "summary_json": str(summary_json),
        },
    }
    write_json(manifest_json, manifest)
    return manifest
