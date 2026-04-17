from __future__ import annotations

import pandas as pd


def build_gdsc_response_table(gdsc_csv: str, cancer_codes: list[str]) -> pd.DataFrame:
    usecols = [
        "DATASET",
        "COSMIC_ID",
        "CELL_LINE_NAME",
        "TCGA_DESC",
        "DRUG_ID",
        "DRUG_NAME",
        "LN_IC50",
        "PUTATIVE_TARGET",
        "PATHWAY_NAME",
        "WEBRELEASE",
    ]
    df = pd.read_csv(gdsc_csv, usecols=usecols, low_memory=False)
    filtered = df[df["TCGA_DESC"].astype(str).isin(cancer_codes)].copy()
    out = pd.DataFrame(
        {
            "gdsc_version": filtered["DATASET"].astype(str).str.strip(),
            "COSMIC_ID": pd.to_numeric(filtered["COSMIC_ID"], errors="coerce").astype("Int64"),
            "cell_line_name": filtered["CELL_LINE_NAME"].astype(str).str.strip(),
            "DRUG_ID": pd.to_numeric(filtered["DRUG_ID"], errors="coerce").astype("Int64"),
            "drug_name": filtered["DRUG_NAME"].astype(str).str.strip(),
            "ln_IC50": pd.to_numeric(filtered["LN_IC50"], errors="coerce"),
            "TCGA_DESC": filtered["TCGA_DESC"].astype(str).str.strip(),
            "putative_target": filtered["PUTATIVE_TARGET"].fillna("").astype(str).str.strip(),
            "pathway_name": filtered["PATHWAY_NAME"].fillna("").astype(str).str.strip(),
            "WEBRELEASE": filtered["WEBRELEASE"].fillna("").astype(str).str.strip(),
        }
    )
    out = out.dropna(subset=["DRUG_ID", "ln_IC50"]).copy()
    out["DRUG_ID"] = out["DRUG_ID"].astype(int)
    return out.reset_index(drop=True)


def build_response_labels(gdsc_labels: pd.DataFrame, binary_quantile: float) -> pd.DataFrame:
    labels = gdsc_labels.copy()
    threshold = float(labels["ln_IC50"].quantile(binary_quantile))
    labels["canonical_drug_id"] = labels["DRUG_ID"].astype(int).astype(str)
    labels["sample_id"] = labels["cell_line_name"].astype(str).str.strip()
    labels["label_regression"] = labels["ln_IC50"]
    labels["label_binary"] = (labels["label_regression"] <= threshold).astype(int)
    labels["label_main"] = labels["label_regression"]
    labels["label_aux"] = labels["label_binary"]
    labels["label_main_type"] = "regression"
    labels["label_aux_type"] = "binary"
    labels["binary_threshold"] = threshold
    keep = [
        "sample_id",
        "cell_line_name",
        "canonical_drug_id",
        "DRUG_ID",
        "drug_name",
        "TCGA_DESC",
        "gdsc_version",
        "putative_target",
        "pathway_name",
        "WEBRELEASE",
        "label_regression",
        "label_binary",
        "label_main",
        "label_aux",
        "label_main_type",
        "label_aux_type",
        "binary_threshold",
    ]
    return labels[keep].reset_index(drop=True)

