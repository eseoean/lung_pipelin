# IPF Translation Support Report

This report fuses currently available local evidence only:

- patient-inference consensus model score
- pseudo-label reversal score
- fibrosis-priority target overlap
- LINCS perturbation support
- approval / investigational status
- broad-target and broad-chemistry penalties

## Top 20 reranked candidates

| final_rerank_rank | drug_name | final_rerank_score | consensus_model_score | pseudo_label_score | target_overlap_count | fibrosis_priority_overlap_count | has_lincs_match | is_approved | support_note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Fostamatinib | 0.7862 | 0.5600 | 0.5742 | 11 | 2 | 1 | 1 | approved, LINCS, fibrosis_targets=2 |
| 2 | Tivozanib | 0.6799 | 0.4385 | 0.4348 | 2 | 2 | 1 | 1 | approved, LINCS, fibrosis_targets=2 |
| 3 | Crizotinib | 0.6331 | 0.4045 | 0.4044 | 2 | 1 | 1 | 1 | approved, LINCS, fibrosis_targets=1 |
| 4 | Sunitinib | 0.6237 | 0.4021 | 0.3985 | 2 | 2 | 1 | 1 | approved, LINCS, fibrosis_targets=2 |
| 5 | Ilomastat | 0.6174 | 0.4115 | 0.4220 | 5 | 2 | 1 | 0 | LINCS, fibrosis_targets=2 |
| 6 | Marimastat | 0.6161 | 0.4283 | 0.4486 | 7 | 3 | 0 | 0 | investigational, fibrosis_targets=3 |
| 7 | Curcumin | 0.6152 | 0.4054 | 0.4103 | 4 | 1 | 1 | 1 | approved, LINCS, fibrosis_targets=1 |
| 8 | PR-15 | 0.6050 | 0.3808 | 0.3863 | 3 | 3 | 0 | 0 | investigational, fibrosis_targets=3 |
| 9 | Dasatinib | 0.5895 | 0.3830 | 0.3846 | 3 | 1 | 1 | 1 | approved, LINCS, fibrosis_targets=1 |
| 10 | Troglitazone | 0.5759 | 0.3584 | 0.3558 | 1 | 1 | 1 | 1 | approved, LINCS, fibrosis_targets=1 |
| 11 | Collagenase clostridium histolyticum | 0.5686 | 0.3557 | 0.3624 | 2 | 2 | 0 | 1 | approved, fibrosis_targets=2 |
| 12 | Nintedanib | 0.5660 | 0.3615 | 0.3617 | 2 | 1 | 1 | 1 | approved, LINCS, fibrosis_targets=1 |
| 13 | Imatinib | 0.5517 | 0.3432 | 0.3414 | 1 | 1 | 1 | 1 | approved, LINCS, fibrosis_targets=1 |
| 14 | Pazopanib | 0.5433 | 0.3378 | 0.3360 | 1 | 1 | 1 | 1 | approved, LINCS, fibrosis_targets=1 |
| 15 | Sorafenib | 0.5422 | 0.3371 | 0.3353 | 1 | 1 | 1 | 1 | approved, LINCS, fibrosis_targets=1 |
| 16 | Parecoxib | 0.5371 | 0.3384 | 0.3403 | 2 | 0 | 1 | 1 | approved, LINCS |
| 17 | Clove oil | 0.5333 | 0.3350 | 0.3462 | 3 | 1 | 0 | 1 | approved, fibrosis_targets=1 |
| 18 | Danazol | 0.5267 | 0.3310 | 0.3335 | 1 | 1 | 1 | 1 | approved, LINCS, fibrosis_targets=1 |
| 19 | Maraviroc | 0.5224 | 0.3193 | 0.3187 | 1 | 0 | 1 | 1 | approved, LINCS |
| 20 | Vorinostat | 0.5154 | 0.2999 | 0.3000 | 0 | 0 | 1 | 1 | approved, LINCS |

