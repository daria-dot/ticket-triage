# Ticket Triage

Classifies incoming support tickets into categories — bug, feature, docs,
question, duplicate — so they can be routed without a human reading each one
first. Training data comes from GitHub Issues across four actively-maintained
repos, using maintainer-applied labels as ground truth, which gives a few
hundred thousand real, messily-written tickets that someone already took the
trouble to label.

The engineering around the model is the point here, not the model. What follows
is live data ingestion, reproducible training, a deployed endpoint,
infrastructure as code, CI/CD, and the reasoning behind the choices — including
the ones that went against the obvious answer.

**Status:** Phases 1–6 complete — data, baseline model, service, container and
CI, infrastructure, model comparison — plus fitted decision thresholds. Drift
monitoring is deliberately not built; see the end of this file for why.

Training data: `huggingface/transformers`, `pandas-dev/pandas`,
`scikit-learn/scikit-learn`, `microsoft/vscode` — 467,491 issues ingested.

## Architecture

```mermaid
flowchart TB
    subgraph ingest ["Data"]
        GH["GitHub Issues API"] -->|"raw JSONB, unmodified"| PG[("Postgres<br/>raw_issues")]
        PG -->|"feature engineering in SQL"| V["views<br/>issues_only<br/>label_category_map<br/>issue_categories"]
    end

    subgraph train ["Training"]
        V --> T["TF-IDF + one-vs-rest<br/>logistic regression"]
        T -->|"params, per-label metrics,<br/>model artifact"| ML[("MLflow registry")]
    end

    subgraph serve ["Serving"]
        ML -->|"model.tar.gz"| S3[("S3 artifacts")]
        API["FastAPI<br/>predict, health, metrics"] -->|"hash, text, output,<br/>version, latency"| PG
        API -->|"INFERENCE_BACKEND=sagemaker"| SM["SageMaker endpoint<br/>ping, invocations"]
        S3 --> SM
        ECR[("ECR")] --> SM
    end

    subgraph ci ["CI/CD"]
        GHA["GitHub Actions"] -->|"lint, types, tests, build"| ECR
        GHA -.->|"OIDC, no stored keys"| AWS["AWS"]
    end
```

`INFERENCE_BACKEND` decides where a prediction is computed: in this process
against the registered model, or by calling the deployed endpoint. Either way
the API does the logging, because the endpoint cannot — a model server has no
database to write to. So a prediction served from AWS is recorded exactly like
one served locally, and the deployed path stays observable.

## Results

Every number below is on the identical held-out test set — 62,331 issues,
assigned by hashing each issue's identity rather than drawn at random, so the
comparison is like-for-like. Per-label precision and recall stay the primary
reading; macro F1 exists only to give the comparison a single ordering.

| Category | TF-IDF | bge-small-en-v1.5 |
|---|---|---|
| bug | **0.573** | 0.511 |
| feature | **0.558** | 0.541 |
| docs | **0.686** | 0.591 |
| question | **0.207** | 0.180 |
| duplicate | **0.301** | 0.281 |
| **macro** | **0.465** | **0.421** |

Both arms fit one threshold per label on the same validation split under the
same rule, so the comparison is of approaches rather than of operating points.
Counting words wins on every label.

### The threshold was doing more damage than the model

The first version of these numbers came from `predict()`, which cuts every label
at 0.5. That is a scikit-learn default, not a decision anyone made — and with
`class_weight="balanced"` over labels running from 19.8% positive (bug) down to
1.3% (docs), it sits nowhere near the F1-optimal point.

The tell was in the shape of the results rather than their size: they ordered
*exactly* by label rarity. That is the signature of one wrong cut applied to
five different distributions, not of five independent modelling failures. The
serving path never applied 0.5 at all — it returns probabilities — so the
metrics were describing an operating point that production did not use.

Fitting one threshold per label on a held-out validation split, model and
features untouched:

| Category | F1 at 0.5 | F1 fitted | |
|---|---|---|---|
| bug | 0.564 | 0.573 | +0.009 |
| feature | 0.530 | 0.558 | +0.028 |
| docs | 0.403 | **0.686** | **+0.283** |
| question | 0.164 | 0.207 | +0.043 |
| duplicate | 0.294 | 0.301 | +0.007 |
| **macro** | **0.391** | **0.465** | **+0.074** |

Carving out the validation split cost a tenth of the training data, so the
retrained model was re-scored at 0.5 as a control: macro F1 0.391 against the
0.389 originally published. Losing those rows changed nothing measurable, which
is what makes the remaining gain attributable to the thresholds rather than to
anything else moving at the same time.

`docs` needed 0.947 and `question` 0.841. Those labels were not being classified
badly; they were being asked the wrong question.

**What this costs, and why it is a choice rather than a free win.** Maximising
F1 per label is a default, and on `question` it trades recall 0.615 → 0.238 to
buy precision 0.095 → 0.183. F1 rose; a triage system that must not drop
questions on the floor got worse. F1-optimal is not product-optimal, and until
somebody prices a missed ticket against a misrouted one, any choice here is
provisional. Pinning a recall floor per label and maximising precision under it
is the same fitting code with a different objective.

Precision remains poor on `question` and `duplicate` in absolute terms.
`question` is the hardest of the five — "is this a question" is far more
semantic than lexical, and TF-IDF sees only words.

### Word counting beat sentence embeddings, and not because of truncation

Two encoders have now been tried against the same classifier, hyperparameters,
split and threshold rule. Both lose, and the second one is what makes the
result interesting.

| Approach | Context | Dims | Macro F1 at 0.5 |
|---|---|---|---|
| TF-IDF, 20k features + bigrams | whole document | 20,000 | **0.391** |
| `all-MiniLM-L6-v2` | 256 tokens | 384 | 0.324 |
| `bge-small-en-v1.5` | 512 tokens | 384 | 0.328 |

The first version of this section named truncation as the leading suspect:
MiniLM stops at 256 tokens where the average issue runs nearer 440, while
TF-IDF reads everything. It was a good hypothesis and it is wrong. Doubling the
context window to 512 tokens, on a newer and stronger encoder, moved macro F1
by **0.004** — noise. Whatever MiniLM was cutting off, the classifier was not
going to use it.

What survives:

- **The task is more lexical than semantic.** "feature request", stack traces,
  `duplicate of #123`, `DOC:` and `QST:` title prefixes are literal strings. A
  bigram index over 20,000 features matches them directly; an embedding of the
  whole document dilutes them into a summary of what the issue is *about*,
  which is a different question from how it should be *routed*.
- **Compression.** 384 dimensions against 20,000 features discards a lot when
  surface form carries the signal. Both encoders output 384 dimensions, so this
  one is untested rather than confirmed — it is the remaining candidate.
- **Domain.** Both models are trained on general web text, not developer
  writing. Also untested.

What no longer survives is the caveat this section used to carry. "Embeddings
lose" rested on a single weak encoder until a stronger, longer-context one
reproduced the result almost exactly. Two independent encoders agreeing, 0.004
apart, is a finding rather than a suspicion.

One honest wrinkle: the MiniLM number predates the validation split, so it was
trained on 251,146 rows against bge-small's 219,701. That gap is worth about
0.002 on TF-IDF, which was measured directly — far smaller than the difference
being claimed, but it is not nothing and the two encoder numbers are not quite
like-for-like.

The cost of knowing this was 878 billable seconds of spot GPU, roughly twenty
cents.

### On the model that was not fine-tuned

The build order calls for a DistilBERT fine-tune "if the gain justifies it".
The gain from embeddings was negative twice, and the second run removed the one
explanation under which a heavier model might have reversed it: if truncation
were the problem, more context would have helped, and 512 tokens did not. A
fine-tune would be a third pass at a representation the evidence says is the
wrong one for this task. That is the criterion doing its job rather than a step
skipped.

## Local setup

From a clean clone:

```bash
cp .env.example .env
```

Fill in `GITHUB_TOKEN` (a personal access token; read-only public scope is
enough), then:

```bash
make install
```

```bash
make up
```

```bash
make ingest
```

Ingestion takes roughly 90 minutes for all four repos and is re-runnable
without duplicates — a second run only fetches what changed. Then:

```bash
make train
```

Model training that needs a GPU runs on AWS rather than locally — embedding the
corpus takes about 40 minutes on an M1 laptop and roughly 12 on a spot GPU, for
about eight cents:

```bash
make export-dataset
```

```bash
make train-cloud
```

```bash
make serve
```

`make test` runs the suite; the integration tests need `make up` first and use a
separate `triage_test` database so they can never touch ingested data.

## AWS and cost

Infrastructure is Terraform, with remote state in S3 and DynamoDB locking.
Everything billable is gated off by default, so an apply made for an unrelated
change provisions nothing:

```bash
make infra-status
```

```bash
make infra-up
```

```bash
make endpoint-up
```

```bash
make infra-down
```

`infra-status` lists anything currently billable alongside month-to-date spend.
RDS runs about $0.02/hour and the SageMaker endpoint about $0.12/hour, so both
are created on demand and destroyed after. A budget alert at $20 is the backstop
for when that discipline fails.

The permanent footprint is a few cents a month: container images in ECR, the
model artifact and Terraform state in S3.

## Decisions

**Feature engineering lives in SQL views, not Python.** `issues_only` filters out
pull requests, `label_category_map` maps each repo's own label vocabulary onto a
shared taxonomy, and `issue_categories` joins them into one row per issue with a
boolean per category. Keeping this in SQL means the training data is defined in
one place that can be queried and inspected directly, rather than reconstructed
by running a script.

**The label map is a curated allowlist, not keyword matching.** Component tags,
workflow labels and engagement labels are deliberately left unmapped. An issue
with none of the category labels becomes a valid all-zero training example — a
real negative — rather than a synthetic "other" class invented by us.

**raw_issues is a single upsert table** keyed on `(repo, issue_number)`, not an
append-only log. Re-ingesting overwrites, which satisfies "re-runnable without
duplicates" directly. The trade-off is that it holds only current label state,
not a history of label changes.

**Ingestion stores the raw API response as-is, pull requests included.** The
issues endpoint returns PRs too, and deciding what counts as a real issue is
feature engineering — so it belongs in a view, not in the ingestion script.

**Prediction logs keep the raw text, not just a hash.** The spec's minimum was a
hash, which is the privacy-conscious default for real customer tickets. These
tickets are already-public GitHub issues, so that argument doesn't apply, and
Phase 7's drift monitoring cannot compare feature distributions against a
one-way hash. Keeping the text now avoids a schema change later that would
leave all earlier predictions useless for comparison.

**The API pins an exact model version** via `MODEL_VERSION` rather than tracking
a "latest" or an MLflow alias. Serving what a config file names is auditable and
reproducible; serving whatever was registered most recently means an untested
training run becomes live on the next restart, with no deploy and no record.

**CI authenticates to AWS through GitHub's OIDC provider**, assuming a role
scoped to `main` on this one repository, so there are no long-lived AWS keys in
repository secrets. This did not work first time: GitHub embeds immutable
numeric owner and repository IDs in the OIDC subject claim
(`repo:owner@186747603/name@1365824571:ref:refs/heads/main`) so that deleting a
repo and recreating it under the same name cannot inherit its trust. CloudTrail
showed the real claim; the policy now accepts both forms.

**The API logs predictions, the endpoint does not.** Giving the model server a
database would put application concerns inside the thing whose only job is
turning text into numbers, so inference is routed through the API instead and
the log covers both paths identically. Verified against a live endpoint: a
prediction served from AWS took 398ms against roughly 8ms in-process — the
round trip — and landed in Postgres with its input text intact.

There is deliberately no fallback between the backends. An unreachable endpoint
surfaces as an error, because quietly answering from a different model than the
caller believes they are using is worse than failing.

**The split is three ways, and `val` was carved out of train rather than test.**
Fitting a decision threshold is fitting a parameter, so it needs data the model
has not seen — and taking that from `test` would leak, destroying the
comparability the split exists to provide. `val` is buckets 20–29, previously
train; `test` stays at buckets 0–19, the same 62,331 issues as before, verified
by per-label count. Every number published before this change stays directly
comparable to every number after it. A test asserts that property rather than
trusting it.

**The train/test split is hashed from issue identity, not drawn at random.**
`train_test_split(random_state=...)` fixes which *positions* land in the test
set, and a `SELECT` without `ORDER BY` promises nothing about which row sits at
a given position — so the same seed could quietly evaluate different issues
after a vacuum or a parallel scan. Comparing approaches means the split has to
be pinned to the issues themselves, so it lives in the view as a hash of
`(repo, issue_number)`. New issues land on a side without disturbing existing
ones.

**GPU training runs on SageMaker, not the laptop.** Embedding the corpus takes
about 40 minutes locally on an 8GB M1 and around 12 on a spot GPU for eight
cents, with the machine left usable throughout. Managed spot gave a 67%
discount on the run that produced the numbers above: 331 billable seconds
against 1,002 of wall clock.

**Training dependencies are an optional extra, not runtime ones.** torch and the
transformer weights would add gigabytes to a serving image that only answers
HTTP requests. If an embedding model ever wins on merit, paying that becomes a
deliberate choice.

**No NAT gateway.** It would cost about $32/month for outbound access nothing
here needs. RDS only requires a subnet group spanning two availability zones,
and the SageMaker endpoint isn't VPC-attached because it authenticates by IAM
rather than network position — so it gains nothing from sitting inside the VPC.

**The RDS master password is generated by Terraform**, not stored in tfvars or
Secrets Manager. It ends up in state either way, this requires no manual
handling, and Secrets Manager would add about $0.40/month for a database not
meant to outlive a working session.

**Training jobs run on prebuilt images, and the pin that looks safest is not.**
The TF-IDF job first went to SageMaker's scikit-learn image with
`scikit-learn>=1.5` pinned, so the pickled pipeline would match the version the
API unpickles it under. That image is Python 3.9 carrying numpy 1.x; the pin
pulled numpy 2.x in beside compiled extensions built against 1.x, and the job
died at import with `numpy._core.multiarray failed to import` before reading a
row. A pin intended to prevent a version mismatch produced a worse one.

It now runs on the PyTorch py312 image the embedding job already proved out,
installing nothing, because the value of a prebuilt image is that its pinned
world agrees with itself. The instance type selects the CPU build, so a job with
no tensors in it still gets no GPU. Both library versions are recorded in
`metrics.json` rather than inferred from which image happened to run — which is
how the remaining 1.8.0-against-1.9.1 gap between training and serving is
visible at all.

**Postgres runs with `statement_timeout` and `temp_file_limit` set.** An
unbounded aggregation query once spilled temp files until the host disk hit
120MB free and the machine crashed. Those two settings make a runaway query fail
loudly with a clear error rather than taking the machine down with it.

**Real-time inference, reluctantly.** Serverless is the better fit — it scales to
zero, so an idle demo endpoint costs nothing — but serverless endpoints cannot
be created in this AWS account. Creation fails with a bare "Request to service
failed", produces no container logs, and fails identically with no model
artifact attached. Ruled out before concluding it was AWS's problem: the image
was pulled from ECR and run exactly as SageMaker runs it and served correctly;
every IAM action simulates as allowed; the manifest is a single amd64 v2
manifest rather than the multi-arch list that breaks some AWS services; quotas
are non-zero; startup takes under two seconds. The same image, artifact and role
then created a real-time endpoint first time and returned correct predictions.

## Not built, deliberately

**Priority and owning-team classification.** The original framing covered
category, priority and team. Only category is built, because GitHub labels give
reliable ground truth for category and nothing equivalent for the other two —
inventing labels for them would produce a model that scores well against its own
assumptions and means nothing.

**A zero-shot LLM comparison arm.** The build order asks for one, measured on
cost per thousand tickets against latency and F1. It needs an API key that
isn't set up, and the question it answers — is training our own model worth it
versus calling someone else's — is worth answering properly or not at all,
rather than with a half-run.

**A DistilBERT fine-tune.** Conditional on embeddings showing promise, which
they didn't. See the results section.

**Drift monitoring.** Not built, and not for lack of time. Drift monitoring
compares live prediction distributions against the training set, and there is
no live traffic to compare: the endpoint is created on demand and destroyed
after, and the prediction log holds three rows, all of them mine. A weekly job
watching an empty table would be decoration.

The groundwork is done rather than skipped. Predictions are logged with their
raw text specifically so distributions can be compared later, the hashed split
means "the training distribution" is a precisely defined set rather than a
moving target, and routing inference through the API means a deployed
prediction is recorded like any other. What is missing is traffic, which is a
reason to wait rather than a thing to fake.

**Continuous accumulation into RDS.** The security group is scoped to a single
IP, so GitHub Actions runners can't reach it, and the instance is destroyed
between sessions. The scheduled ingest workflow runs against a throwaway
container as a regression check that ingestion still works against the live API.

**Authentication on the API.** There's nothing to protect yet and no users. The
SageMaker endpoint is IAM-authenticated by default; the FastAPI service is local
only.

**Historical label-change tracking.** `raw_issues` stores current state only. If
drift monitoring later needs label histories, that's an append-only log added
when it's actually needed.
