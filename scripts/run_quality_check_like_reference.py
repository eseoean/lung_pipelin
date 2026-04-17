#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold


MORGAN_BITS = 2048
LOW_DENSITY_THRESHOLD = 0.02
MW_OUTLIER_LOW = 100.0
MW_OUTLIER_HIGH = 1000.0

GRADE_MAP = {
    "chembl_norm": "A_chembl",
    "drugbank_name": "B_drugbank_name",
    "drugbank_synonym": "C_drugbank_synonym_fuzzy",
    "drugbank_fuzzy": "C_drugbank_synonym_fuzzy",
    "pubchem_api": "D_pubchem_api",
    "unmatched": "NA_unmatched",
}


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Create a QC report shaped like quality_check_results.json")
    parser.add_argument(
        "--drug-master",
        default=str(repo_root / "data/interim/masters/drug_master.parquet"),
        help="Path to drug_master.parquet",
    )
    parser.add_argument(
        "--labels-y",
        default=str(repo_root / "data/processed/model_inputs_smoke/labels_y.parquet"),
        help="Path to labels_y.parquet",
    )
    parser.add_argument(
        "--train-table",
        default=str(repo_root / "data/processed/model_inputs_smoke/train_table.parquet"),
        help="Path to train_table.parquet",
    )
    parser.add_argument(
        "--output",
        default=str(repo_root / "outputs/reports/quality_check_results_like_reference.json"),
        help="Output JSON path",
    )
    return parser.parse_args()


def grade_of(match_source: str) -> str:
    return GRADE_MAP.get(str(match_source), f"OTHER_{match_source}")


def safe_rate(numerator: int | float, denominator: int | float) -> float:
    if not denominator:
        return 0.0
    return float(numerator) / float(denominator)


def main() -> None:
    args = parse_args()
    drug_master = pd.read_parquet(args.drug_master).copy()
    labels_y = pd.read_parquet(args.labels_y) if Path(args.labels_y).exists() else None
    train_table = pd.read_parquet(args.train_table) if Path(args.train_table).exists() else None

    qc_rows = []
    for row in drug_master.itertuples(index=False):
        smiles = "" if pd.isna(row.canonical_smiles) else str(row.canonical_smiles).strip()
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        parsed = mol is not None
        density = None
        on_bits = None
        mw = None
        scaffold_ok = False
        if parsed:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=MORGAN_BITS)
            on_bits = int(fp.GetNumOnBits())
            density = on_bits / float(MORGAN_BITS)
            mw = float(Descriptors.MolWt(mol))
            try:
                scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                scaffold_ok = scaffold is not None and scaffold.GetNumAtoms() >= 0
            except Exception:
                scaffold_ok = False
        qc_rows.append(
            {
                "canonical_drug_id": str(row.canonical_drug_id),
                "drug_name": str(row.drug_name),
                "match_source": str(row.match_source),
                "grade": grade_of(str(row.match_source)),
                "has_smiles": int(row.has_smiles),
                "parsed": parsed,
                "bit_density": density,
                "on_bits": on_bits,
                "mw": mw,
                "scaffold_ok": scaffold_ok,
            }
        )

    qc_df = pd.DataFrame(qc_rows)
    total_drugs = int(qc_df.shape[0])
    parse_success = int(qc_df["parsed"].sum())
    smiles_count = int(qc_df["has_smiles"].sum())
    scaffold_success = int(qc_df["scaffold_ok"].sum())
    grade_counts = {
        key: int(value)
        for key, value in qc_df["grade"].value_counts(dropna=False).to_dict().items()
    }
    trusted_usable = int(
        qc_df["grade"].isin(["A_chembl", "B_drugbank_name", "C_drugbank_synonym_fuzzy"]).sum()
    )
    ab_grade = int(qc_df["grade"].isin(["A_chembl", "B_drugbank_name"]).sum())
    all_zero = int((qc_df["on_bits"].fillna(0) == 0).sum())
    low_density_mask = qc_df["bit_density"].fillna(0.0) < LOW_DENSITY_THRESHOLD
    low_density_count = int(low_density_mask.sum())
    low_density_non_na_count = int((low_density_mask & (qc_df["grade"] != "NA_unmatched")).sum())
    mw_outlier = int(
        (
            (qc_df["mw"].fillna(0.0) < MW_OUTLIER_LOW)
            | (qc_df["mw"].fillna(0.0) > MW_OUTLIER_HIGH)
        ).sum()
        - qc_df["mw"].isna().sum()
    )
    density_values = qc_df["bit_density"].dropna()

    conflict_count = 0
    conflict_count += int(((qc_df["has_smiles"] == 1) & (~qc_df["parsed"])).sum())
    conflict_count += int(((qc_df["grade"] == "NA_unmatched") & (qc_df["parsed"])).sum())

    features_cleaned_rows = int(train_table.shape[0]) if train_table is not None else None
    original_rows = int(labels_y.shape[0]) if labels_y is not None else None
    removed_rows = (
        int(original_rows - features_cleaned_rows)
        if original_rows is not None and features_cleaned_rows is not None
        else None
    )

    report = {
        "validation_date": datetime.now().date().isoformat(),
        "total_drugs": total_drugs,
        "지표_정의": {
            "final_usable_rate": "A+B+C 등급 약물 비율 (A=chembl_norm, B=drugbank_name, C=drugbank_synonym/drugbank_fuzzy)",
            "bit_density_low_flag": f"Morgan({MORGAN_BITS}) bit density < {LOW_DENSITY_THRESHOLD:.0%}",
            "mw_outlier": f"RDKit MolWt < {MW_OUTLIER_LOW:.0f} or > {MW_OUTLIER_HIGH:.0f}",
        },
        "summary": {
            "smiles_보유": smiles_count,
            "rdkit_파싱_성공": parse_success,
            "raw_parse_success_rate": round(safe_rate(parse_success, total_drugs), 3),
            "final_usable_rate": round(safe_rate(trusted_usable, total_drugs), 3),
            "bit_density_low_flag": low_density_count,
            "mw_outlier": mw_outlier,
            "ab_grade_ratio": round(safe_rate(ab_grade, total_drugs), 3),
            "scaffold_success": scaffold_success,
        },
        "qc_결과": {
            "final_usable_rate_95pct": f"{'PASS' if safe_rate(trusted_usable, total_drugs) >= 0.95 else 'FAIL'} ({safe_rate(trusted_usable, total_drugs):.1%})",
            "ab_grade_60pct": f"{'PASS' if safe_rate(ab_grade, total_drugs) >= 0.60 else 'FAIL'} ({safe_rate(ab_grade, total_drugs):.1%})",
            "no_conflicts": "PASS" if conflict_count == 0 else f"FAIL ({conflict_count})",
        },
        "주요_문제": {
            "1": f"Bit density < 2% 플래그: {low_density_count}개 약물 (정보량 부족)",
            "2": f"Final usable SMILES rate: {safe_rate(trusted_usable, total_drugs):.1%} (목표 95% 미달)",
            "3": f"A+B 등급 비율: {safe_rate(ab_grade, total_drugs):.1%} (목표 60% 미달)",
        },
        "등급_분포": {
            "A_chembl": grade_counts.get("A_chembl", 0),
            "B_drugbank_name": grade_counts.get("B_drugbank_name", 0),
            "C_drugbank_synonym_fuzzy": grade_counts.get("C_drugbank_synonym_fuzzy", 0),
            "D_pubchem_api": grade_counts.get("D_pubchem_api", 0),
            "NA_unmatched": grade_counts.get("NA_unmatched", 0),
        },
        "fingerprint_품질": {
            "morgan_bits": MORGAN_BITS,
            "bit_density_평균": round(float(density_values.mean()), 4),
            "bit_density_중앙값": round(float(density_values.median()), 4),
            "bit_density_p5": round(float(density_values.quantile(0.05)), 4),
            "bit_density_p95": round(float(density_values.quantile(0.95)), 4),
            "all_zero_fingerprint": all_zero,
            "low_density_count": low_density_count,
        },
        "최종_판정": {
            "na_drugs_count": grade_counts.get("NA_unmatched", 0),
            "na_drugs_treatment": "현재 build_model_inputs에서는 유지, 구조 기반 strict FE에서는 제외 후보",
            "low_density_not_na_count": low_density_non_na_count,
            "low_density_treatment": "현재는 유지, structural_limitation 태그 후보",
            "features_cleaned_rows": features_cleaned_rows,
            "removed_rows": removed_rows,
        },
    }

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output_path)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
