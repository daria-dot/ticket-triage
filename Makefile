.PHONY: help install lint test up down ingest train serve clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'

install:  ## Install dependencies and pre-commit hooks
	python -m venv .venv
	.venv/bin/pip install -e ".[dev]"
	.venv/bin/pre-commit install

lint:  ## Run linters and type checker
	.venv/bin/ruff check src tests
	.venv/bin/mypy src

test:  ## Run the test suite
	.venv/bin/pytest

up:  ## Start local stack (Postgres, MLflow)
	docker compose up -d

down:  ## Stop local stack
	docker compose down

ingest:  ## Pull issues from the GitHub API
	.venv/bin/python -m triage.ingest

train:  ## Train and log a model run
	.venv/bin/python -m triage.models.train

serve:  ## Run the API locally
	.venv/bin/uvicorn triage.api.main:app --reload --port 8000

clean:  ## Remove caches and build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache
