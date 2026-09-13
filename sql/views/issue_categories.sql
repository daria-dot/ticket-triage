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
--
-- Three ways, not two. Choosing a decision threshold is fitting a parameter,
-- so it needs data the model has not seen; tuning it on `test` would leak and
-- destroy exactly the comparability this hashed split exists to protect.
-- `val` is carved out of what was previously train (buckets 20-29), leaving
-- `test` at buckets 0-19 -- bit-for-bit the same 62,331 issues as before, so
-- every number already published stays directly comparable. Training loses
-- about a tenth of its rows, which is the price of being able to quote an
-- honest threshold.
--
-- The bucket is computed once in the subquery rather than repeated per branch:
-- three copies of an md5 expression is three chances for them to drift apart,
-- and a split that disagrees with itself would be invisible until it had
-- already poisoned a comparison.
CREATE OR REPLACE VIEW issue_categories AS
SELECT
    repo,
    issue_number,
    title,
    body,
    is_bug,
    is_feature,
    is_docs,
    is_question,
    is_duplicate,
    -- Appended rather than placed with the other identity columns: CREATE OR
    -- REPLACE VIEW can only add columns at the end, and keeping this file
    -- replaceable is what makes `make db-init` idempotent.
    CASE
        WHEN bucket < 20 THEN 'test'
        WHEN bucket < 30 THEN 'val'
        ELSE 'train'
    END AS split
FROM (
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
        ('x' || substr(md5(io.repo || '#' || io.issue_number::text), 1, 6))::bit(24)::int % 100
            AS bucket
    FROM issues_only io
    LEFT JOIN LATERAL jsonb_array_elements(io.payload -> 'labels') AS label ON true
    LEFT JOIN label_category_map m
        ON m.repo = io.repo AND m.raw_label = label ->> 'name'
    GROUP BY io.repo, io.issue_number, io.payload ->> 'title', io.payload ->> 'body'
) buckets;
