.PHONY: help setup data validate baseline features train train-fast mlflow test lint fmt cov clean

PY := .venv/bin/python
UV := uv

help:  ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n",$$1,$$2}'

setup:  ## Create the venv and install dependencies
	$(UV) venv --python 3.12
	$(UV) pip install -e ".[dev]"

data:  ## Generate the synthetic dataset
	$(PY) -m src.cli data

validate:  ## Enforce every Pandera contract
	$(PY) -m src.cli validate

baseline:  ## Train + evaluate the Phase 1 logistic baseline
	$(PY) -m src.cli baseline

features:  ## Build the feature matrix and report its shape
	$(PY) -m src.cli features

train:  ## Phase 2: full pipeline, 100 Optuna trials, calibrated champion
	$(PY) -m src.cli train

train-fast:  ## Same pipeline, 12 trials — smoke test only, do not report these numbers
	$(PY) -m src.cli train --fast

mlflow:  ## Open the MLflow UI against the local run store
	.venv/bin/mlflow ui --backend-store-uri sqlite:///artifacts/mlflow.db

test:  ## Run the test suite
	$(PY) -m pytest tests -q

cov:  ## Run tests with a coverage report
	$(PY) -m pytest tests -q --cov=src --cov-report=term-missing

lint:  ## Lint and type-check
	$(PY) -m ruff check src tests
	$(PY) -m ruff format --check src tests

fmt:  ## Auto-format
	$(PY) -m ruff format src tests
	$(PY) -m ruff check --fix src tests

clean:  ## Remove generated artifacts and caches
	rm -rf artifacts/* data/synthetic/*.parquet data/feature_spec.yaml .pytest_cache .ruff_cache .coverage
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
