-- Maps each repo's own label vocabulary onto a small, shared category
-- vocabulary: bug, feature, docs, question, duplicate.
--
-- This is a deliberately curated allowlist, not a keyword match. Component
-- tags (e.g. scikit-learn's module:*, pandas's Groupby/Datetime, vscode's
-- terminal/git), workflow/status labels (wontfix, Stale, Needs Triage,
-- Waiting for Reviewer), and engagement labels (good first issue, help
-- wanted) are deliberately left unmapped -- an issue with none of these
-- category labels is a valid all-zero multi-label training example, not
-- a synthetic "other" ground truth invented by us.
--
-- Built from the actual label distribution across the ~495k issues ingested
-- in Phase 1 (see README Decisions). Revisit if a re-ingest surfaces label
-- vocabulary this doesn't cover yet.
CREATE OR REPLACE VIEW label_category_map AS
SELECT * FROM (VALUES
    -- huggingface/transformers
    ('huggingface/transformers', 'bug',                    'bug'),
    ('huggingface/transformers', 'Feature request',         'feature'),
    ('huggingface/transformers', 'New model',                'feature'),
    ('huggingface/transformers', 'model card',              'docs'),
    ('huggingface/transformers', 'Documentation',            'docs'),
    ('huggingface/transformers', 'Discussion',               'question'),
    ('huggingface/transformers', 'Usage',                    'question'),
    ('huggingface/transformers', 'Need more information',    'question'),

    -- pandas-dev/pandas
    ('pandas-dev/pandas', 'Bug',              'bug'),
    ('pandas-dev/pandas', 'Regression',        'bug'),
    ('pandas-dev/pandas', 'Docs',              'docs'),
    ('pandas-dev/pandas', 'Enhancement',       'feature'),
    ('pandas-dev/pandas', 'Usage Question',    'question'),
    ('pandas-dev/pandas', 'Duplicate Report',  'duplicate'),

    -- scikit-learn/scikit-learn
    ('scikit-learn/scikit-learn', 'Bug',           'bug'),
    ('scikit-learn/scikit-learn', 'New Feature',   'feature'),
    ('scikit-learn/scikit-learn', 'Enhancement',   'feature'),
    ('scikit-learn/scikit-learn', 'Documentation', 'docs'),

    -- microsoft/vscode
    ('microsoft/vscode', 'bug',             'bug'),
    ('microsoft/vscode', 'feature-request', 'feature'),
    ('microsoft/vscode', '*duplicate',      'duplicate'),
    ('microsoft/vscode', '*question',       'question')
) AS t(repo, raw_label, category);
