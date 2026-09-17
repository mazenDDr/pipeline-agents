| model | mode | VRAM GiB | RSS GiB | TTFT 6k s | prefill tok/s | decode tok/s | plan valid (prompt / enforced) | plan sound (prompt / enforced) | code pass [95% CI] | s / call | tokens / call | truncated |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3.5-4b | no-think | 3.3 | 6.9 | 1.52 | 3999 | 93 | 1.00 / 1.00 | 1.00 / 0.88 | 0.67 [0.42, 0.92] (n=12) | 6.7 | 637 | 0 |
| qwen3.5-4b | think | 3.3 | 8.8 | 1.49 | 4055 | 95 | 1.00 / 0.75 | 1.00 / 0.75 | 0.50 [0.25, 0.75] (n=12) | 27.6 | 2667 | 2 |

Code pass rate per task:

| model / mode | cv_pipeline | dedupe_normalized | filtered_topk | impute_skewed | int_coded_categorical | leaky_column | mixed_dates | padded_keys | percentile_clip | sentinel_missing | skewed_regression | unseen_category |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3.5-4b (no-think) | 0.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 | 0.0 | 1.0 | 1.0 | 1.0 | 0.0 |
| qwen3.5-4b (think) | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 1.0 | 0.0 | 0.0 |
