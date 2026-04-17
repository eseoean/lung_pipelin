CONFIG ?= configs/lung.yaml
PYTHONPATH := src

.PHONY: dry-run stage-standardize stage-disease stage-model-inputs stage-train stage-patient stage-rerank test

dry-run:
	PYTHONPATH=$(PYTHONPATH) python -m lung_pipeline.cli --config $(CONFIG) --stage all --dry-run

stage-standardize:
	PYTHONPATH=$(PYTHONPATH) python scripts/run_01_standardize_tables.py --config $(CONFIG)

stage-disease:
	PYTHONPATH=$(PYTHONPATH) python scripts/run_02_build_disease_context.py --config $(CONFIG)

stage-model-inputs:
	PYTHONPATH=$(PYTHONPATH) python scripts/run_03_build_model_inputs.py --config $(CONFIG)

stage-train:
	PYTHONPATH=$(PYTHONPATH) python scripts/run_04_train_baseline.py --config $(CONFIG)

stage-patient:
	PYTHONPATH=$(PYTHONPATH) python scripts/run_05_patient_inference.py --config $(CONFIG)

stage-rerank:
	PYTHONPATH=$(PYTHONPATH) python scripts/run_06_rerank_outputs.py --config $(CONFIG)

test:
	PYTHONPATH=$(PYTHONPATH) pytest

