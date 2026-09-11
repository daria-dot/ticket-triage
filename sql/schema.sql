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
