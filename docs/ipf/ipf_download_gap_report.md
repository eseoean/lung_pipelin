# IPF Download Gap Report

Current reranking used only locally available evidence. Missing or partially blocked sources are listed below.

| source | status | path | purpose |
| --- | --- | --- | --- |
| admet | missing | /Users/skku_aws2_18/pre_project/lung_pipelin/data/raw/knowledge/admet | Safety and pharmacokinetic filtering |
| siderside | missing | /Users/skku_aws2_18/pre_project/lung_pipelin/data/raw/knowledge/siderside | Side-effect plausibility checks |
| clinicaltrials | missing | /Users/skku_aws2_18/pre_project/lung_pipelin/data/raw/knowledge/clinicaltrials | Trial-stage translation support |
| string | missing | /Users/skku_aws2_18/pre_project/lung_pipelin/data/raw/knowledge/string | Network proximity / PPI support |
| opentargets | available_schema_gap | /Users/skku_aws2_18/pre_project/lung_pipelin/data/raw/knowledge/opentargets/association_by_overall_indirect | Disease-target association scores are present, but a stable ENSG-to-symbol mapping is still missing in the local IPF branch. |

