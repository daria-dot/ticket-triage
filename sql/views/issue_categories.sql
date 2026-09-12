-- One row per issue, with a boolean per shared category. LEFT JOINs so an
-- issue with no mapped label still gets a row (all-false), a real negative
-- example for every category rather than being dropped from training data.
--
-- `split` is derived from a hash of the issue's identity, not drawn at random
-- in Python. train_test_split(random_state=...) only fixes which *positions*
-- land in the test set, and SELECT without ORDER BY makes no promise about
-- which row sits at a given position -- so the same seed could silently
-- evaluate different issues after a vacuum or a parallel scan. Hashing the
-- identity pins a given issue to a given side permanently, which is what makes
-- comparing TF-IDF against embeddings against anything later an honest
-- comparison. New issues land on a side without disturbing existing ones.
CREATE OR REPLACE VIEW issue_categories AS
SELECT
    io.repo,
    io.issue_number,
    io.payload ->> 'title' AS title,
    io.payload ->> 'body' AS body,
    coalesce(bool_or(m.category = 'bug'), false) AS is_bug,
    coalesce(bool_or(m.category = 'feature'), false) AS is_feature,
    coalesce(bool_or(m.category = 'docs'), false) AS is_docs,
    coalesce(bool_or(m.category = 'question'), false) AS is_question,
    coalesce(bool_or(m.category = 'duplicate'), false) AS is_duplicate,
    -- Appended rather than placed with the other identity columns: CREATE OR
    -- REPLACE VIEW can only add columns at the end, and keeping this file
    -- replaceable is what makes `make db-init` idempotent.
    CASE
        WHEN ('x' || substr(md5(io.repo || '#' || io.issue_number::text), 1, 6))::bit(24)::int % 100 < 20
            THEN 'test'
        ELSE 'train'
    END AS split
FROM issues_only io
LEFT JOIN LATERAL jsonb_array_elements(io.payload -> 'labels') AS label ON true
LEFT JOIN label_category_map m
    ON m.repo = io.repo AND m.raw_label = label ->> 'name'
GROUP BY io.repo, io.issue_number, io.payload ->> 'title', io.payload ->> 'body';
