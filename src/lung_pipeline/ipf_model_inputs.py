from __future__ import annotations

import json
import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


DRUGBANK_NS = {"db": "http://www.drugbank.ca"}
FIBROSIS_PRIORITY_GENES = {
    "CCL2",
    "COL1A1",
    "COL1A2",
    "COL3A1",
    "CXCL14",
    "FN1",
    "LUM",
    "MET",
    "MMP1",
    "MMP7",
    "MMP9",
    "PDGFRA",
    "PDGFRB",
    "SERPINE1",
    "SFRP2",
    "SPP1",
    "TGFBR1",
    "TGFBR2",
    "TNC",
}
MINERAL_KEYWORDS = (
    "calcium",
    "copper",
    "ferric",
    "ferrous",
    "iron",
    "nitrous acid",
    "zinc",
)


def _normalize_name(value: str) -> str:
    value = str(value or "").strip().lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _coerce_gene_name(value: str) -> str:
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "-666", "---"}:
        return ""
    return text.upper()


def _has_carbon(smiles: str) -> bool:
    text = str(smiles or "")
    return "C" in text or "c" in text


def default_model_input_paths(repo_root: Path) -> dict[str, Path]:
    base = repo_root / "data" / "raw" / "knowledge"
    return {
        "drugbank_zip": base / "drugbank" / "drugbank_all_full_database.xml.zip",
        "chembl_chemreps": base / "chembl" / "chembl_36_chemreps.txt.gz",
        "chembl_uniprot_mapping": base / "chembl" / "chembl_uniprot_mapping.txt",
        "lincs_pert_info": base / "lincs" / "GSE92742_Broad_LINCS_pert_info.txt.gz",
        "lincs_sig_info": base / "lincs" / "GSE92742_Broad_LINCS_sig_info.txt.gz",
        "lincs_gene_info": base / "lincs" / "GSE92742_Broad_LINCS_gene_info.txt.gz",
        "opentargets_indirect": base / "opentargets" / "association_by_overall_indirect",
    }


def default_model_input_signature_paths(repo_root: Path) -> dict[str, Path]:
    docs_root = repo_root / "docs" / "ipf"
    return {
        "gse32537_bulk_top_genes": docs_root / "gse32537_bulk_ipf_vs_control_top_genes.csv",
        "gse47460_bulk_top_genes": docs_root / "gse47460_bulk_ipf_vs_control_top_genes.csv",
        "gse122960_scrna_top_genes": docs_root / "gse122960_expression_ipf_vs_control_top_genes.csv",
        "gse136831_scrna_top_genes": docs_root / "gse136831_expression_ipf_vs_control_top_genes.csv",
        "gse233844_pbmc_top_genes": docs_root / "gse233844_pbmc_expression_top_genes.csv",
    }


def _load_signature_rows(repo_root: Path) -> pd.DataFrame:
    records: list[dict[str, Any]] = []

    signature_paths = default_model_input_signature_paths(repo_root)

    bulk_32537 = pd.read_csv(signature_paths["gse32537_bulk_top_genes"])
    for idx, row in bulk_32537.iterrows():
        gene_name = _coerce_gene_name(row["feature_label"])
        if not gene_name:
            continue
        effect = float(row["delta_ipf_vs_control"])
        records.append(
            {
                "source_accession": "GSE32537",
                "context_type": "bulk_discovery",
                "comparison": "IPF_vs_Control",
                "gene_name": gene_name,
                "gene_id": gene_name,
                "effect_size": effect,
                "direction": "up_in_ipf" if effect > 0 else "down_in_ipf",
                "rank": int(idx + 1),
                "weight": abs(effect),
            }
        )

    bulk_47460 = pd.read_csv(signature_paths["gse47460_bulk_top_genes"])
    for idx, row in bulk_47460.iterrows():
        gene_name = _coerce_gene_name(row["feature_label"])
        if not gene_name:
            continue
        effect = float(row["delta_ipf_vs_control"])
        records.append(
            {
                "source_accession": "GSE47460",
                "context_type": "bulk_discovery",
                "comparison": "IPF_vs_Control",
                "gene_name": gene_name,
                "gene_id": gene_name,
                "effect_size": effect,
                "direction": "up_in_ipf" if effect > 0 else "down_in_ipf",
                "rank": int(idx + 1),
                "weight": abs(effect),
            }
        )

    scrna_122960 = pd.read_csv(signature_paths["gse122960_scrna_top_genes"])
    for _, row in scrna_122960.iterrows():
        gene_name = _coerce_gene_name(row["gene_name"])
        if not gene_name:
            continue
        effect = float(row["log2_fc_ipf_vs_control"])
        records.append(
            {
                "source_accession": "GSE122960",
                "context_type": "scrna_sample",
                "comparison": "IPF_vs_Control",
                "gene_name": gene_name,
                "gene_id": row["gene_id"],
                "effect_size": effect,
                "direction": "up_in_ipf" if effect > 0 else "down_in_ipf",
                "rank": int(row["rank"]),
                "weight": abs(effect),
            }
        )

    scrna_136831 = pd.read_csv(signature_paths["gse136831_scrna_top_genes"])
    for _, row in scrna_136831.iterrows():
        gene_name = _coerce_gene_name(row["gene_name"])
        if not gene_name:
            continue
        effect = float(row["log2_fc_ipf_vs_control"])
        records.append(
            {
                "source_accession": "GSE136831",
                "context_type": "scrna_cell_state",
                "comparison": f"{row['manuscript_identity']}:IPF_vs_Control",
                "gene_name": gene_name,
                "gene_id": row["gene_id"],
                "effect_size": effect,
                "direction": "up_in_ipf" if effect > 0 else "down_in_ipf",
                "rank": int(row["rank_within_manuscript"]),
                "weight": abs(effect),
            }
        )

    pbmc_233844 = pd.read_csv(signature_paths["gse233844_pbmc_top_genes"])
    pbmc_233844 = pbmc_233844[pbmc_233844["comparison"].astype(str).str.endswith("vs Control")]
    for _, row in pbmc_233844.iterrows():
        gene_name = _coerce_gene_name(row["gene_name"])
        if not gene_name:
            continue
        effect = float(row["log2_fc"])
        records.append(
            {
                "source_accession": "GSE233844",
                "context_type": "pbmc_validation",
                "comparison": row["comparison"],
                "gene_name": gene_name,
                "gene_id": row["gene_id"],
                "effect_size": effect,
                "direction": "up_in_ipf" if effect > 0 else "down_in_ipf",
                "rank": int(row["rank_within_comparison"]),
                "weight": abs(effect),
            }
        )

    return pd.DataFrame(records)


def build_disease_features(
    repo_root: Path,
    output_parquet: Path,
    output_csv: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    signature_rows = _load_signature_rows(repo_root)
    aggregated = (
        signature_rows.groupby("gene_name", as_index=False)
        .agg(
            source_count=("source_accession", "nunique"),
            evidence_rows=("source_accession", "size"),
            up_count=("direction", lambda s: int((s == "up_in_ipf").sum())),
            down_count=("direction", lambda s: int((s == "down_in_ipf").sum())),
            mean_signed_score=("effect_size", "mean"),
            max_abs_score=("weight", "max"),
            total_weight=("weight", "sum"),
            evidence_sources=("source_accession", lambda s: "|".join(sorted(set(map(str, s))))),
            comparisons=("comparison", lambda s: "|".join(sorted(set(map(str, s))))),
        )
        .sort_values(["source_count", "total_weight", "max_abs_score"], ascending=[False, False, False])
        .reset_index(drop=True)
    )
    aggregated["gene_rank"] = np.arange(1, len(aggregated) + 1)

    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    aggregated.to_parquet(output_parquet, index=False)
    if output_csv:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        aggregated.to_csv(output_csv, index=False)

    gene_weights = dict(
        zip(
            aggregated["gene_name"],
            aggregated["source_count"] * aggregated["max_abs_score"],
        )
    )
    return aggregated, gene_weights


def _iter_drugbank_drugs(zip_path: Path) -> Iterable[dict[str, Any]]:
    with zipfile.ZipFile(zip_path) as zf:
        member_name = zf.namelist()[0]
        with zf.open(member_name) as fh:
            for _, elem in ET.iterparse(fh, events=("end",)):
                if elem.tag != "{http://www.drugbank.ca}drug" or "type" not in elem.attrib:
                    continue

                drugbank_id = ""
                for id_elem in elem.findall("db:drugbank-id", DRUGBANK_NS):
                    if id_elem.attrib.get("primary") == "true":
                        drugbank_id = (id_elem.text or "").strip()
                        break

                name = (elem.findtext("db:name", default="", namespaces=DRUGBANK_NS) or "").strip()
                groups = [
                    (group.text or "").strip()
                    for group in elem.findall("db:groups/db:group", DRUGBANK_NS)
                    if (group.text or "").strip()
                ]
                synonyms = [
                    (syn.text or "").strip()
                    for syn in elem.findall("db:synonyms/db:synonym", DRUGBANK_NS)
                    if (syn.text or "").strip()
                ]
                smiles = ""
                for prop in elem.findall("db:calculated-properties/db:property", DRUGBANK_NS):
                    kind = (prop.findtext("db:kind", default="", namespaces=DRUGBANK_NS) or "").strip()
                    if kind == "SMILES":
                        smiles = (prop.findtext("db:value", default="", namespaces=DRUGBANK_NS) or "").strip()
                        break

                target_genes = sorted(
                    {
                        _coerce_gene_name(
                            target.findtext("db:polypeptide/db:gene-name", default="", namespaces=DRUGBANK_NS)
                        )
                        for target in elem.findall("db:targets/db:target", DRUGBANK_NS)
                    }
                    - {""}
                )

                yield {
                    "drugbank_id": drugbank_id,
                    "drug_name": name,
                    "normalized_name": _normalize_name(name),
                    "drug_type": elem.attrib.get("type", ""),
                    "groups": "|".join(groups),
                    "is_approved": int("approved" in groups),
                    "is_investigational": int("investigational" in groups),
                    "synonyms": "|".join(synonyms[:20]),
                    "target_genes": "|".join(target_genes),
                    "target_gene_count": len(target_genes),
                    "drugbank_smiles": smiles,
                }
                elem.clear()


def _load_lincs_lookup(pert_info_path: Path, sig_info_path: Path) -> pd.DataFrame:
    pert = pd.read_csv(
        pert_info_path,
        sep="\t",
        usecols=["pert_id", "pert_iname", "pert_type", "is_touchstone", "canonical_smiles"],
    )
    pert = pert[pert["pert_type"] == "trt_cp"].copy()
    pert["normalized_name"] = pert["pert_iname"].map(_normalize_name)

    sig = pd.read_csv(sig_info_path, sep="\t", usecols=["pert_id", "pert_iname", "pert_type", "cell_id"])
    sig = sig[sig["pert_type"] == "trt_cp"].copy()
    sig_counts = sig.groupby("pert_id", as_index=False).agg(
        lincs_signature_count=("cell_id", "size"),
        lincs_cell_count=("cell_id", "nunique"),
    )

    merged = pert.merge(sig_counts, how="left", on="pert_id")
    merged["lincs_signature_count"] = merged["lincs_signature_count"].fillna(0).astype(int)
    merged["lincs_cell_count"] = merged["lincs_cell_count"].fillna(0).astype(int)
    merged = merged.sort_values(
        ["is_touchstone", "lincs_signature_count", "lincs_cell_count"],
        ascending=[False, False, False],
    )
    return merged.drop_duplicates("normalized_name")


def build_drug_features(
    repo_root: Path,
    disease_gene_weights: dict[str, float],
    output_parquet: Path,
    output_csv: Path | None = None,
) -> pd.DataFrame:
    paths = default_model_input_paths(repo_root)
    lincs_lookup = _load_lincs_lookup(paths["lincs_pert_info"], paths["lincs_sig_info"])
    lincs_by_name = lincs_lookup.set_index("normalized_name").to_dict(orient="index")

    rows = []
    for drug in _iter_drugbank_drugs(paths["drugbank_zip"]):
        synonyms = [s for s in str(drug["synonyms"]).split("|") if s]
        candidate_names = [drug["drug_name"], *synonyms]
        lincs_match = None
        for candidate in candidate_names:
            normalized = _normalize_name(candidate)
            if normalized and normalized in lincs_by_name:
                lincs_match = lincs_by_name[normalized]
                break

        target_genes = [gene for gene in str(drug["target_genes"]).split("|") if gene]
        overlap_genes = sorted({gene for gene in target_genes if gene in disease_gene_weights})
        overlap_score = float(sum(disease_gene_weights[g] for g in overlap_genes))

        lincs_smiles = ""
        if lincs_match is not None:
            smiles = str(lincs_match.get("canonical_smiles", "") or "").strip()
            if smiles not in {"", "-666", "nan"}:
                lincs_smiles = smiles

        rows.append(
            {
                **drug,
                "has_lincs_match": int(lincs_match is not None),
                "lincs_pert_id": "" if lincs_match is None else str(lincs_match.get("pert_id", "")),
                "lincs_pert_iname": "" if lincs_match is None else str(lincs_match.get("pert_iname", "")),
                "lincs_signature_count": 0 if lincs_match is None else int(lincs_match.get("lincs_signature_count", 0)),
                "lincs_cell_count": 0 if lincs_match is None else int(lincs_match.get("lincs_cell_count", 0)),
                "lincs_canonical_smiles": lincs_smiles,
                "canonical_smiles": drug["drugbank_smiles"] or lincs_smiles,
                "target_overlap_genes": "|".join(overlap_genes),
                "target_overlap_count": len(overlap_genes),
                "target_overlap_weight": overlap_score,
            }
        )

    drug_features = pd.DataFrame(rows)
    drug_features = drug_features[
        (
            (drug_features["is_approved"] == 1)
            | (drug_features["is_investigational"] == 1)
            | (drug_features["has_lincs_match"] == 1)
        )
        & ((drug_features["target_gene_count"] > 0) | (drug_features["has_lincs_match"] == 1))
    ].copy()
    drug_features = drug_features.sort_values(
        ["target_overlap_weight", "has_lincs_match", "is_approved", "target_gene_count"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    drug_features.to_parquet(output_parquet, index=False)
    if output_csv:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        drug_features.to_csv(output_csv, index=False)
    return drug_features


def build_ranking_inputs(
    drug_features: pd.DataFrame,
    disease_features: pd.DataFrame,
    ranking_output_parquet: Path,
    train_output_parquet: Path,
    pseudo_label_output_parquet: Path,
    ranking_output_csv: Path | None = None,
    train_output_csv: Path | None = None,
    pseudo_label_output_csv: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ranking = drug_features.copy()

    if len(ranking) == 0:
        raise ValueError("No drug features were generated for ranking inputs.")

    overlap_max = ranking["target_overlap_weight"].max() or 1.0
    sig_max = ranking["lincs_signature_count"].max() or 1.0
    overlap_count_max = ranking["target_overlap_count"].max() or 1.0
    density = ranking["target_overlap_count"] / ranking["target_gene_count"].clip(lower=1)
    density_max = density.max() or 1.0

    ranking["target_overlap_norm"] = ranking["target_overlap_weight"] / overlap_max
    ranking["lincs_support_norm"] = np.log1p(ranking["lincs_signature_count"]) / np.log1p(sig_max)
    ranking["target_overlap_count_norm"] = np.log1p(ranking["target_overlap_count"]) / np.log1p(overlap_count_max)
    ranking["target_overlap_density"] = density
    ranking["target_overlap_density_norm"] = density / density_max
    ranking["specific_overlap_norm"] = (
        ranking["target_overlap_density_norm"] * ranking["target_overlap_count_norm"]
    )
    ranking["approval_bonus"] = ranking["is_approved"].astype(float)
    ranking["smiles_bonus"] = ranking["canonical_smiles"].astype(str).str.len().gt(0).astype(float)
    ranking["target_overlap_genes_list"] = ranking["target_overlap_genes"].fillna("").astype(str).str.split("|")
    ranking["fibrosis_priority_overlap_count"] = ranking["target_overlap_genes_list"].map(
        lambda genes: sum(1 for gene in genes if gene in FIBROSIS_PRIORITY_GENES)
    )
    fibrosis_max = ranking["fibrosis_priority_overlap_count"].max() or 1.0
    ranking["fibrosis_priority_overlap_norm"] = (
        ranking["fibrosis_priority_overlap_count"] / fibrosis_max
    )
    ranking["broad_target_penalty"] = np.where(
        (ranking["target_gene_count"] >= 100) & (ranking["has_lincs_match"] == 0),
        np.minimum((ranking["target_gene_count"] - 100) / 150.0, 1.0),
        0.0,
    )
    ranking["mineral_keyword_penalty"] = ranking["drug_name"].fillna("").astype(str).str.lower().map(
        lambda name: float(any(keyword in name for keyword in MINERAL_KEYWORDS))
    )
    ranking["inorganic_like_penalty"] = ranking["canonical_smiles"].fillna("").astype(str).map(
        lambda smiles: float((not _has_carbon(smiles)) and bool(str(smiles).strip()))
    )
    ranking["broad_chemistry_penalty"] = np.maximum(
        ranking["mineral_keyword_penalty"],
        np.where(
            (ranking["inorganic_like_penalty"] > 0) & (ranking["has_lincs_match"] == 0),
            1.0,
            0.0,
        ),
    )
    ranking["pseudo_label_score"] = (
        0.25 * ranking["target_overlap_norm"]
        + 0.20 * ranking["specific_overlap_norm"]
        + 0.15 * ranking["target_overlap_count_norm"]
        + 0.15 * ranking["lincs_support_norm"]
        + 0.15 * ranking["fibrosis_priority_overlap_norm"]
        + 0.10 * ranking["approval_bonus"]
        + 0.05 * ranking["smiles_bonus"]
        - 0.15 * ranking["broad_target_penalty"]
        - 0.25 * ranking["broad_chemistry_penalty"]
    )
    ranking["pseudo_label_score"] = ranking["pseudo_label_score"].clip(lower=0.0)
    ranking = ranking.sort_values(
        [
            "pseudo_label_score",
            "fibrosis_priority_overlap_count",
            "lincs_support_norm",
            "target_overlap_weight",
            "has_lincs_match",
            "is_approved",
        ],
        ascending=[False, False, False, False, False, False],
    ).reset_index(drop=True)
    ranking["pseudo_label_rank"] = np.arange(1, len(ranking) + 1)
    ranking = ranking.drop(columns=["target_overlap_genes_list"])

    disease_summary = {
        "disease_gene_panel_size": int(len(disease_features)),
        "disease_gene_panel_top10": "|".join(disease_features.head(10)["gene_name"].tolist()),
    }
    train_table = ranking.copy()
    for key, value in disease_summary.items():
        train_table[key] = value
    train_table["label_source"] = "pseudo_label"
    train_table["study_objective"] = "disease_signature_reversal"

    pseudo_labels = ranking[
        [
            "drugbank_id",
            "drug_name",
            "pseudo_label_score",
            "pseudo_label_rank",
            "target_overlap_count",
            "target_overlap_weight",
            "has_lincs_match",
            "is_approved",
        ]
    ].copy()

    ranking_output_parquet.parent.mkdir(parents=True, exist_ok=True)
    ranking.to_parquet(ranking_output_parquet, index=False)
    train_output_parquet.parent.mkdir(parents=True, exist_ok=True)
    train_table.to_parquet(train_output_parquet, index=False)
    pseudo_label_output_parquet.parent.mkdir(parents=True, exist_ok=True)
    pseudo_labels.to_parquet(pseudo_label_output_parquet, index=False)

    if ranking_output_csv:
        ranking_output_csv.parent.mkdir(parents=True, exist_ok=True)
        ranking.to_csv(ranking_output_csv, index=False)
    if train_output_csv:
        train_output_csv.parent.mkdir(parents=True, exist_ok=True)
        train_table.to_csv(train_output_csv, index=False)
    if pseudo_label_output_csv:
        pseudo_label_output_csv.parent.mkdir(parents=True, exist_ok=True)
        pseudo_labels.to_csv(pseudo_label_output_csv, index=False)

    return ranking, train_table, pseudo_labels


def build_ipf_model_inputs(
    repo_root: Path,
    disease_features_parquet: Path,
    drug_features_parquet: Path,
    ranking_features_parquet: Path,
    train_table_parquet: Path,
    pseudo_labels_parquet: Path,
    disease_features_csv: Path | None = None,
    drug_features_csv: Path | None = None,
    ranking_features_csv: Path | None = None,
    train_table_csv: Path | None = None,
    pseudo_labels_csv: Path | None = None,
    summary_json: Path | None = None,
) -> dict[str, Any]:
    disease_features, disease_gene_weights = build_disease_features(
        repo_root=repo_root,
        output_parquet=disease_features_parquet,
        output_csv=disease_features_csv,
    )
    drug_features = build_drug_features(
        repo_root=repo_root,
        disease_gene_weights=disease_gene_weights,
        output_parquet=drug_features_parquet,
        output_csv=drug_features_csv,
    )
    ranking_features, train_table, pseudo_labels = build_ranking_inputs(
        drug_features=drug_features,
        disease_features=disease_features,
        ranking_output_parquet=ranking_features_parquet,
        train_output_parquet=train_table_parquet,
        pseudo_label_output_parquet=pseudo_labels_parquet,
        ranking_output_csv=ranking_features_csv,
        train_output_csv=train_table_csv,
        pseudo_label_output_csv=pseudo_labels_csv,
    )

    summary = {
        "study": "IPF",
        "objective": "disease_signature_reversal",
        "disease_gene_count": int(len(disease_features)),
        "drug_candidate_count": int(len(drug_features)),
        "train_row_count": int(len(train_table)),
        "pseudo_label_count": int(len(pseudo_labels)),
        "top_ranked_drugs": pseudo_labels.head(10)["drug_name"].tolist(),
        "lincs_matched_drugs": int(drug_features["has_lincs_match"].sum()),
        "approved_drugs": int(drug_features["is_approved"].sum()),
        "mean_target_overlap": float(drug_features["target_overlap_count"].mean()),
        "max_pseudo_label_score": float(pseudo_labels["pseudo_label_score"].max()),
    }
    if summary_json:
        summary_json.parent.mkdir(parents=True, exist_ok=True)
        summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary
