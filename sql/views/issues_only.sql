-- The GitHub Issues API returns pull requests too (they carry a
-- "pull_request" key that plain issues never have). Ticket triage is about
-- issues, not PRs, so every downstream view builds on this one instead of
-- raw_issues directly.
CREATE OR REPLACE VIEW issues_only AS
SELECT *
FROM raw_issues
WHERE payload -> 'pull_request' IS NULL;
