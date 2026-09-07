-- SQLite queries used to validate the bounded report datasets.
-- The Python validation step loads the named CSV/JSON inputs into in-memory
-- tables before executing these statements.

-- benchmark_summary
SELECT
    total_wall_seconds / 3600.0 AS wall_hours,
    accuracy,
    macro_f1,
    weighted_f1,
    total_model_fit_count AS fit_total,
    train_rows,
    test_rows
FROM benchmark_summary;

-- model_log
WITH runtimes AS (
    SELECT
        model_name AS model,
        SUM(elapsed_seconds) / 3600.0 AS hours
    FROM model_log
    GROUP BY model_name
), ranked AS (
    SELECT
        model,
        hours,
        ROW_NUMBER() OVER (ORDER BY hours DESC) AS runtime_rank,
        SUM(hours) OVER () AS all_model_hours
    FROM runtimes
)
SELECT model, hours, hours / all_model_hours AS share
FROM ranked
WHERE runtime_rank <= 6
UNION ALL
SELECT 'Other 11 models', SUM(hours), SUM(hours) / MAX(all_model_hours)
FROM ranked
WHERE runtime_rank > 6;

-- predictions
SELECT
    iteration,
    true_label,
    pred_label,
    p_std,
    agreement_rate
FROM predictions;

-- capacity_replay
SELECT *
FROM capacity_replay;

-- memory_model
SELECT *
FROM memory_model;
