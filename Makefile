.PHONY: help install lint test up down db-init ingest train export-dataset train-cloud serve clean \
	infra-up endpoint-up infra-down infra-apply infra-down-all infra-status

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'

install:  ## Install dependencies and pre-commit hooks
	python -m venv .venv
	.venv/bin/pip install -e ".[dev,training]"
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

db-init:  ## Apply the DB schema and feature views (idempotent)
	docker compose exec -T postgres psql -U $${POSTGRES_USER:-triage} -d $${POSTGRES_DB:-triage} -f - < sql/schema.sql
	docker compose exec -T postgres psql -U $${POSTGRES_USER:-triage} -d $${POSTGRES_DB:-triage} -f - < sql/views/issues_only.sql
	docker compose exec -T postgres psql -U $${POSTGRES_USER:-triage} -d $${POSTGRES_DB:-triage} -f - < sql/views/label_category_map.sql
	docker compose exec -T postgres psql -U $${POSTGRES_USER:-triage} -d $${POSTGRES_DB:-triage} -f - < sql/views/issue_categories.sql

ingest: db-init  ## Pull issues from the GitHub API
	.venv/bin/python -m triage.ingest

train:  ## Train and log the TF-IDF baseline
	.venv/bin/python -m triage.models.train

export-dataset:  ## Export the training frame to S3 for cloud training
	.venv/bin/python -m triage.models.export_dataset --bucket $$(cd terraform && terraform output -raw artifacts_bucket)

train-cloud:  ## Train the embedding model on a SageMaker spot GPU (~5c per run)
	.venv/bin/python -m triage.models.launch_training --bucket $$(cd terraform && terraform output -raw artifacts_bucket)

train-cloud-tfidf:  ## Fit the TF-IDF baseline and its thresholds on a spot CPU (~10c per run)
	.venv/bin/python -m triage.models.launch_tfidf_training --bucket $$(cd terraform && terraform output -raw artifacts_bucket)

serve:  ## Run the API locally
	.venv/bin/uvicorn triage.api.main:app --reload --port 8000

clean:  ## Remove caches and build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache

infra-up:  ## Provision the database (bills ~$0.02/hour once up)
	cd terraform && terraform apply -var=create_rds=true

endpoint-up:  ## Provision the SageMaker endpoint (bills ~$0.12/hour once up)
	cd terraform && terraform apply -var=create_sagemaker=true

infra-down:  ## Remove everything billable, keeping free resources (VPC, ECR, IAM, S3)
	cd terraform && terraform apply -var=create_rds=false -var=create_sagemaker=false

infra-apply:  ## Apply config changes without provisioning anything billable
	cd terraform && terraform apply

infra-down-all:  ## Destroy everything, including ECR images and CI's OIDC role
	cd terraform && terraform destroy

infra-status:  ## List anything currently billable, to catch what was left running
	@printf 'RDS instances:      '
	@aws rds describe-db-instances --output text \
		--query "DBInstances[].[DBInstanceIdentifier,DBInstanceStatus,DBInstanceClass]" \
		| grep . || echo "none"
	@printf 'SageMaker endpoints: '
	@aws sagemaker list-endpoints --output text \
		--query "Endpoints[].[EndpointName,EndpointStatus]" \
		| grep . || echo "none"
	@printf 'Month-to-date spend: '
	@aws budgets describe-budgets --output text \
		--account-id $$(aws sts get-caller-identity --query Account --output text) \
		--query "Budgets[].[CalculatedSpend.ActualSpend.Amount,BudgetLimit.Amount]" \
		| awk '{printf "$$%s of $$%s budget\n", $$1, $$2}'
