CONFIG ?= configs/lung.yaml
PYTHON ?= python3
PYTHONPATH := src

.PHONY: dry-run stage-standardize stage-disease stage-model-inputs stage-train stage-patient stage-rerank ipf-audit ipf-dry-run ipf-download-plan ipf-download-geo test

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

test:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest
