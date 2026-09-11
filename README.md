# Ticket Triage

Classifies incoming support tickets by category, priority, and owning team.
Trained on GitHub Issues with maintainer-applied labels as ground truth.

**Status:** in development — Phase 1 (data ingestion)

## Architecture

_Diagram goes here once Phase 3 is done._

## Results

| Approach | Macro F1 | P95 latency | Cost / 1k |
|---|---|---|---|
| TF-IDF + LogReg | — | — | — |

## Local setup

```bash
cp .env.example .env     # fill in GITHUB_TOKEN
make install
make up
make ingest
```

## Decisions

_Record decisions here as you make them, with reasoning._

## Not built, deliberately

_What you scoped out and why._
