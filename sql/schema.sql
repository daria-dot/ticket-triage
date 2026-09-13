-- Raw GitHub Issues API responses, unmodified. Feature engineering happens in
-- sql/views/, not here and not in Python.
CREATE TABLE IF NOT EXISTS raw_issues (
    repo TEXT NOT NULL,
    issue_number INTEGER NOT NULL,
    payload JSONB NOT NULL,
    fetched_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (repo, issue_number)
);

-- Lets the ingest script find "updated_at of the newest issue we have per repo"
-- so re-pulls can use the GitHub API's `since` filter instead of refetching everything.
CREATE INDEX IF NOT EXISTS idx_raw_issues_repo_updated_at
    ON raw_issues (repo, (payload ->> 'updated_at'));

-- Every /predict call. input_text is the exact title+body sent to the model,
-- not just its hash -- kept so a prediction can be re-examined later and bad
-- ones debugged. Safe to retain here since the source tickets are
-- already-public GitHub issues, not private customer data.
CREATE TABLE IF NOT EXISTS predictions (
    id BIGSERIAL PRIMARY KEY,
    input_hash TEXT NOT NULL,
    input_text TEXT NOT NULL,
    output JSONB NOT NULL,
    model_version TEXT NOT NULL,
    latency_ms DOUBLE PRECISION NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions (created_at);
