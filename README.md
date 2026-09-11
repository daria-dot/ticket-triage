# Ticket Triage

Classifies incoming support tickets by category, priority, and owning team.
Trained on GitHub Issues with maintainer-applied labels as ground truth.

**Status:** in development — Phase 1 (data ingestion) complete, Phase 2 (baseline model) next

Training data comes from six actively-maintained repos with structured issue
labels: `huggingface/transformers`, `pandas-dev/pandas`,
`scikit-learn/scikit-learn`, `microsoft/vscode`, `kubernetes/kubernetes`,
`rust-lang/rust`.

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

- **raw_issues is a single upsert table**, keyed on `(repo, issue_number)`, not
  an append-only event log. Re-ingesting the same issue overwrites its row
  instead of duplicating it, which directly satisfies the "re-runnable
  without duplicates" requirement and keeps things simple. The trade-off:
  it only holds the latest label state, not a history of label changes —
  acceptable for now since nothing downstream needs that history yet.
- **Ingestion pulls the raw GitHub Issues API response as-is, PRs included.**
  The issues endpoint returns pull requests too; separating "real issue" from
  "PR" is feature engineering and belongs in a SQL view (`sql/views/`), not in
  the ingestion script.
- **Re-pulls use GitHub's `since` filter**, based on the max `updated_at`
  already stored per repo, so a re-run only fetches what changed instead of
  refetching everything.
- **The scheduled workflow (`.github/workflows/ingest.yml`) runs against a
  throwaway Postgres service container**, not a persistent database — there
  is no deployed database until Phase 5 (RDS). Until then it functions as a
  regression check that ingestion still works against the live API on a
  schedule, not as real continuous data accumulation. Requires an
  `INGEST_GITHUB_TOKEN` repo secret to be added manually (Settings → Secrets
  and variables → Actions) — not something committed to the repo.

## Not built, deliberately

- **No historical label-change tracking.** raw_issues stores current state
  only. If drift monitoring later needs "how did this label change over
  time," that's an append-only log added in Phase 7, not now.
- **No real scheduled ingestion into a persistent store yet.** That needs
  Phase 5's RDS instance to exist first.
