CONFIG ?= configs/lung.yaml
PYTHON ?= python3
PYTHONPATH := src

.PHONY: dry-run stage-standardize stage-disease stage-model-inputs stage-train stage-patient stage-rerank ipf-audit ipf-dry-run ipf-download-plan ipf-download-geo ipf-download-geo-small ipf-build-cell-reference ipf-build-cell-expression ipf-build-sample-reference ipf-build-sample-expression ipf-build-pbmc-validation ipf-build-bulk-references ipf-build-model-inputs ipf-train-baseline test

dry-run:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m lung_pipeline.cli --config $(CONFIG) --stage all --dry-run

stage-standardize:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/run_01_standardize_tables.py --config $(CONFIG)

stage-disease:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/run_02_build_disease_context.py --config $(CONFIG)

stage-model-inputs:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/run_03_build_model_inputs.py --config $(CONFIG)

stage-train:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/run_04_train_baseline.py --config $(CONFIG)

stage-patient:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/run_05_patient_inference.py --config $(CONFIG)

stage-rerank:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/run_06_rerank_outputs.py --config $(CONFIG)

ipf-audit:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/audit_ipf_sources.py --config configs/ipf.yaml

ipf-dry-run:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m lung_pipeline.cli --config configs/ipf.yaml --stage all --dry-run

ipf-download-plan:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/download_ipf_geo_sources.py --config configs/ipf.yaml --plan-only

ipf-download-geo:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/download_ipf_geo_sources.py --config configs/ipf.yaml

ipf-download-geo-small:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/download_ipf_geo_sources.py --config configs/ipf.yaml --download-supplementary --supplementary-max-bytes 50000000

ipf-build-cell-reference:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/build_ipf_cell_state_reference.py --config configs/ipf.yaml

ipf-build-cell-expression:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/build_ipf_cell_state_expression_reference.py --config configs/ipf.yaml

ipf-build-sample-reference:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/build_ipf_sample_reference.py --config configs/ipf.yaml

ipf-build-sample-expression:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/build_ipf_sample_expression_reference.py --config configs/ipf.yaml

ipf-build-pbmc-validation:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/build_ipf_pbmc_validation_reference.py --config configs/ipf.yaml

ipf-build-bulk-references:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/build_ipf_bulk_references.py --config configs/ipf.yaml

ipf-build-model-inputs:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/build_ipf_model_inputs.py --config configs/ipf.yaml

ipf-train-baseline:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/run_04_train_baseline.py --config configs/ipf.yaml

test:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest
