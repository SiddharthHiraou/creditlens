.PHONY: help setup data validate baseline train test lint fmt cov clean

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

train: baseline  ## Alias for the current champion training path (Phase 1: baseline)

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
	rm -rf artifacts/* data/synthetic/*.parquet .pytest_cache .ruff_cache .coverage
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
