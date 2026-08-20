.PHONY: help setup data validate baseline features train train-fast audit explain warm-cache serve loadtest up down mlflow test lint fmt cov clean

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

audit:  ## Phase 3: SHAP, ECOA reason codes, counterfactuals, fairness, mitigation
	$(PY) -m src.cli audit

explain:  ## Adverse action reasons + counterfactual levers for one declined applicant
	$(PY) -m src.cli explain

warm-cache:  ## Precompute applicant history features into the serving cache
	$(PY) -m src.api.warm_cache

serve:  ## Run the API (http://localhost:8000/docs)
	.venv/bin/uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 1

loadtest:  ## Headless Locust run: 50 users, 60s, against a running server
	.venv/bin/locust -f loadtest/locustfile.py --headless -u 50 -r 10 -t 60s \
		--host http://localhost:8000 --csv artifacts/loadtest

up:  ## Bring up API, Postgres, Redis and MLflow with docker compose
	docker compose -f infra/docker-compose.yml up --build

down:  ## Tear the stack down
	docker compose -f infra/docker-compose.yml down -v

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
