from __future__ import annotations

import re
from typing import Iterable

import pandas as pd

try:
    from rdkit import Chem  # type: ignore
    from rdkit import RDLogger  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    Chem = None
    RDLogger = None

if RDLogger is not None:  # pragma: no cover - logging side effect
    RDLogger.DisableLog("rdApp.*")


def norm_name(value: object) -> str:
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def normalize_smiles(value: object) -> str:
    raw = str(value).strip()
    if not raw or raw.lower() in {"nan", "none", "<na>", "-666", "restricted"}:
        return ""
    if Chem is None:
        return raw
    mol = Chem.MolFromSmiles(raw)
    if mol is None:
        return ""
    return Chem.MolToSmiles(mol, canonical=True)


def clean_name_variants(value: object) -> list[str]:
    raw = str(value).strip()
    variants = [raw]
    cleaned = re.sub(r"\s*\(\d+\s*[umµ]?m\)\s*$", "", raw, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"\s*\([+-]\)\s*$", "", cleaned).strip()
    cleaned = re.sub(r"\s+-\d*HCl$", "", cleaned, flags=re.IGNORECASE).strip()
    for suffix in [
        " mesylate",
        " maleate",
        " hydrochloride",
        " dihydrochloride",
        " malate",
        " sodium",
        " potassium",
        " phosphate",
        " sulfate",
        " acetate",
        " citrate",
        " fumarate",
        " succinate",
        " tartrate",
        " tosylate",
        " besylate",
        " bromide",
        " malonate",
        " hydrate",
        " monohydrate",
    ]:
        cleaned = re.sub(rf"{re.escape(suffix)}$", "", cleaned, flags=re.IGNORECASE).strip()
    if cleaned and cleaned not in variants:
        variants.append(cleaned)
    uniq: list[str] = []
    for item in variants:
        if item and item not in uniq:
            uniq.append(item)
    return uniq


def build_candidate_names(drug_name: object, synonyms: object) -> list[str]:
    candidates: list[str] = []
    for raw_name in clean_name_variants(drug_name):
        candidates.append(raw_name)
    for synonym in str(synonyms).split(","):
        synonym = synonym.strip()
        if synonym:
            candidates.extend(clean_name_variants(synonym))
    uniq: list[str] = []
    for item in candidates:
        if item and item not in uniq:
            uniq.append(item)
    return uniq


def load_lincs_pert_info(pert_info_path: str) -> pd.DataFrame:
    pert = pd.read_csv(pert_info_path, sep="\t", compression="gzip", low_memory=False)
    pert = pert[pert["pert_type"] == "trt_cp"].copy()
    pert["canonical_smiles"] = pert["canonical_smiles"].map(normalize_smiles)
    keep_cols = [col for col in ["pert_iname", "canonical_smiles"] if col in pert.columns]
    return pert[keep_cols].drop_duplicates().reset_index(drop=True)


def build_lincs_smiles_lookup(pert_info_paths: Iterable[str]) -> list[tuple[str, dict[str, str]]]:
    sources: list[tuple[str, dict[str, str]]] = []
    for path in pert_info_paths:
        pert = load_lincs_pert_info(path)
        lookup: dict[str, str] = {}
        for _, row in pert.dropna().iterrows():
            smiles = normalize_smiles(row["canonical_smiles"])
            key = norm_name(row["pert_iname"])
            if key and smiles and key not in lookup:
                lookup[key] = smiles
        source_name = str(path).split("/")[-1].replace(".txt.gz", "").replace(".gz", "")
        sources.append((source_name, lookup))
    return sources


def build_drugbank_curated_lookups(master_path: str, synonym_path: str) -> list[tuple[str, dict[str, str]]]:
    master = pd.read_parquet(master_path, columns=["drugbank_id", "name", "smiles"])
    master["canonical_smiles"] = master["smiles"].map(normalize_smiles)
    master = master[(master["canonical_smiles"] != "") & master["name"].notna()].copy()

    name_lookup: dict[str, str] = {}
    for _, row in master[["name", "canonical_smiles"]].iterrows():
        key = norm_name(row["name"])
        if key and key not in name_lookup:
            name_lookup[key] = row["canonical_smiles"]

    synonym = pd.read_parquet(synonym_path, columns=["drugbank_id", "synonym"])
    synonym = synonym.merge(master[["drugbank_id", "canonical_smiles"]], on="drugbank_id", how="inner")
    synonym_lookup: dict[str, str] = {}
    for _, row in synonym[["synonym", "canonical_smiles"]].dropna().iterrows():
        key = norm_name(row["synonym"])
        if key and key not in synonym_lookup:
            synonym_lookup[key] = row["canonical_smiles"]
    return [
        ("drugbank_curated_name", name_lookup),
        ("drugbank_curated_synonym", synonym_lookup),
    ]


def build_chembl_lookups(chembl_path: str) -> list[tuple[str, dict[str, str]]]:
    chembl = pd.read_parquet(chembl_path, columns=["chembl_id", "pref_name", "canonical_smiles"])
    chembl = chembl.dropna(subset=["pref_name", "canonical_smiles"]).copy()
    chembl["canonical_smiles"] = chembl["canonical_smiles"].map(normalize_smiles)
    chembl = chembl[chembl["canonical_smiles"] != ""].copy()
    pref_lookup: dict[str, str] = {}
    for _, row in chembl[["pref_name", "canonical_smiles"]].iterrows():
        key = norm_name(row["pref_name"])
        if key and key not in pref_lookup:
            pref_lookup[key] = row["canonical_smiles"]
    return [("chembl_pref_name", pref_lookup)]


def build_drug_catalog(
    gdsc_labels: pd.DataFrame,
    compounds_df: pd.DataFrame,
    smiles_sources: list[tuple[str, dict[str, str]]],
) -> pd.DataFrame:
    lung_drugs = gdsc_labels[["DRUG_ID", "drug_name"]].drop_duplicates("DRUG_ID").copy()
    compounds = compounds_df.rename(columns={"DRUG_NAME": "compound_drug_name"}).copy()
    merged = lung_drugs.merge(compounds, on="DRUG_ID", how="left")

    rows: list[dict[str, object]] = []
    for _, row in merged.iterrows():
        candidate_names = build_candidate_names(row["drug_name"], row.get("SYNONYMS", ""))
        match_source = "unmatched"
        smiles = ""
        matched_candidate_name = ""
        for source_name, lookup in smiles_sources:
            for candidate in candidate_names:
                key = norm_name(candidate)
                if key in lookup:
                    smiles = lookup[key]
                    match_source = source_name
                    matched_candidate_name = candidate
                    break
            if smiles:
                break
        rows.append(
            {
                "canonical_drug_id": str(int(row["DRUG_ID"])),
                "DRUG_ID": int(row["DRUG_ID"]),
                "drug_name": row["drug_name"],
                "drug_name_norm": norm_name(row["drug_name"]),
                "canonical_smiles": smiles,
                "canonical_smiles_raw": smiles,
                "match_source": match_source,
                "matched_candidate_name": matched_candidate_name,
                "has_smiles": int(bool(smiles)),
                "screening_site": row.get("SCREENING_SITE", ""),
                "synonyms": row.get("SYNONYMS", ""),
                "target": row.get("TARGET", ""),
                "target_pathway": row.get("TARGET_PATHWAY", ""),
            }
        )
    return pd.DataFrame(rows)


def overlay_exact_drug_catalog(drug_master: pd.DataFrame, exact_catalog: pd.DataFrame) -> pd.DataFrame:
    exact = exact_catalog.copy()
    exact["canonical_drug_id"] = exact["DRUG_ID"].astype(str).str.strip()
    exact["drug_name_norm"] = exact["drug_name_norm"].fillna("").astype(str)
    exact["canonical_smiles"] = exact["canonical_smiles"].fillna("").astype(str)
    exact["canonical_smiles_raw"] = exact.get("canonical_smiles_raw", exact["canonical_smiles"]).fillna("").astype(str)
    exact["match_source"] = exact["match_source"].fillna("unmatched").astype(str)
    exact["has_smiles"] = pd.to_numeric(exact["has_smiles"], errors="coerce").fillna(0).astype(int)
    exact = exact[
        [
            "canonical_drug_id",
            "drug_name_norm",
            "canonical_smiles",
            "canonical_smiles_raw",
            "match_source",
            "has_smiles",
        ]
    ].drop_duplicates(subset=["canonical_drug_id"])

    merged = drug_master.drop(
        columns=["drug_name_norm", "canonical_smiles", "canonical_smiles_raw", "match_source", "has_smiles"],
        errors="ignore",
    ).merge(exact, on="canonical_drug_id", how="left")
    merged["drug_name_norm"] = merged["drug_name_norm"].fillna(merged["drug_name"].map(norm_name))
    merged["canonical_smiles"] = merged["canonical_smiles"].fillna("")
    merged["canonical_smiles_raw"] = merged["canonical_smiles_raw"].fillna(merged["canonical_smiles"])
    merged["match_source"] = merged["match_source"].fillna("unmatched")
    merged["has_smiles"] = pd.to_numeric(merged["has_smiles"], errors="coerce").fillna(0).astype(int)
    merged["matched_candidate_name"] = merged.get("matched_candidate_name", "").fillna("")
    return merged


def build_target_mapping(drug_catalog: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for _, row in drug_catalog.iterrows():
        raw = str(row.get("target", "")).strip()
        if not raw or raw.lower() in {"nan", "none"}:
            continue
        pieces = re.split(r"[,;/|]+", raw)
        for piece in pieces:
            token = piece.strip().upper()
            token = re.sub(r"\s+", "", token)
            token = token.replace("(S)", "").replace("(M)", "")
            if not token:
                continue
            rows.append(
                {
                    "canonical_drug_id": str(row["canonical_drug_id"]),
                    "target_gene_symbol": token,
                }
            )
    return pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)
