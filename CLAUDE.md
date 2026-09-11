# Ticket Triage — Project Context

Drop this at the repo root. Claude Code reads it automatically at the start of every session.

## What this is

An end-to-end ML service that classifies incoming support tickets by category,
priority, and owning team. Training data comes from the GitHub Issues API, using
maintainer-applied labels as ground truth.

This is a portfolio project. The engineering around the model matters more than
the model. A reviewer should be able to read the repo and see: live data
ingestion, reproducible training, a deployed API, infrastructure as code, CI/CD,
and drift monitoring.

## Stack

| Layer | Choice |
|---|---|
| Language | Python 3.11 |
| Data store | Postgres 16 |
| Modelling | scikit-learn → sentence-transformers → DistilBERT |
| Tracking | MLflow |
| API | FastAPI + Pydantic |
| Container | Docker, multi-stage, non-root |
| Registry | Amazon ECR |
| Deploy | SageMaker real-time endpoint |
| IaC | Terraform |
| CI/CD | GitHub Actions |
| Monitoring | Evidently, reports to S3 |

## Build order

Do not skip ahead. Each phase must run before the next begins.

### Phase 1 — Data
- Ingestion script hitting the GitHub Issues API across 5–6 large repos
- Raw responses land in Postgres as JSONB, unmodified
- Feature engineering lives in SQL views, not Python
- Scheduled re-pull so new issues arrive continuously
- Exit criteria: `make ingest` populates the database, re-runnable without duplicates

### Phase 2 — Baseline model
- TF-IDF + logistic regression, multi-label
- Per-label precision and recall. No single accuracy number.
- MLflow logs params, metrics, and the model artifact
- Exit criteria: a registered model in MLflow with documented metrics

### Phase 3 — Service
- FastAPI: `/predict`, `/health`, `/metrics`
- Pydantic request and response schemas, explicit validation errors
- Every prediction logged to Postgres with input hash, output, model version, latency
- Exit criteria: `docker compose up` gives a working local stack

### Phase 4 — Container and CI
- Multi-stage Dockerfile, slim base, non-root user, pinned dependencies
- GitHub Actions: lint, type check, unit tests, data validation, build, push to ECR
- Exit criteria: merge to main produces a tagged image in ECR

### Phase 5 — Infrastructure
- Terraform: VPC, ECR, RDS, SageMaker endpoint, IAM roles, S3 bucket
- Remote state in S3 with DynamoDB locking
- `plan` runs on PR, `apply` is manual
- Exit criteria: `terraform apply` builds the whole environment from nothing

### Phase 6 — Model improvement
- Sentence embeddings + classifier
- Then DistilBERT fine-tune if the gain justifies it
- Comparison arm: zero-shot LLM on cost per 1000 tickets, latency, F1
- Exit criteria: a results table comparing all approaches on the same test set

### Phase 7 — Monitoring
- Weekly job comparing live feature distributions against the training set
- Evidently report written to S3, served as a static page
- Alert threshold documented and justified
- Exit criteria: a drift report a stranger can interpret

## Conventions

- Trunk-based. Short-lived feature branches, squash merge.
- Every config value from environment variables. No hardcoded paths or secrets.
- `make` targets for every common operation: `ingest`, `train`, `serve`, `test`, `deploy`
- Tests: unit tests on transformation logic, one integration test hitting the live API
- Type hints everywhere. `ruff` and `mypy` in CI.
- Docstrings explain *why*, not *what*

## Working agreement for Claude Code

Read this before generating anything.

1. **Do not make architectural decisions.** If a choice affects the design —
   model family, schema shape, deployment topology, library selection — stop and
   present the options with trade-offs. Wait for a decision.

2. **Explain before implementing.** For any non-trivial file, describe the
   approach in a few sentences first.

3. **Small commits.** One logical change per commit. Do not generate five files
   at once.

4. **No silent fallbacks.** Never wrap something in try/except and return a
   default to make an error disappear. Fail loudly.

5. **Flag anything that costs money.** Any Terraform resource or AWS call that
   incurs charges gets called out explicitly before it is written.

6. **When asked why, answer honestly.** If a choice was arbitrary, say it was
   arbitrary rather than inventing a justification.

## AWS cost guardrails

- Billing budget alert set before the first resource is created
- SageMaker endpoints are billed per second whether or not they are called
- `terraform destroy` after every session that provisioned something
- Weekly check of the billing console for orphaned NAT gateways and load balancers

## README requirements

The README is the deliverable most reviewers actually read. It needs:

- One-paragraph problem statement
- Architecture diagram
- The metrics table, including the baseline
- Decisions and their reasoning, especially the ones that went against the obvious choice
- What is deliberately not built, and why
- Local setup that works from a clean clone
