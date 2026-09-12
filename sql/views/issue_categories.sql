-- One row per issue, with a boolean per shared category. LEFT JOINs so an
-- issue with no mapped label still gets a row (all-false), a real negative
-- example for every category rather than being dropped from training data.
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
    coalesce(bool_or(m.category = 'duplicate'), false) AS is_duplicate
FROM issues_only io
LEFT JOIN LATERAL jsonb_array_elements(io.payload -> 'labels') AS label ON true
LEFT JOIN label_category_map m
    ON m.repo = io.repo AND m.raw_label = label ->> 'name'
GROUP BY io.repo, io.issue_number, io.payload ->> 'title', io.payload ->> 'body';
