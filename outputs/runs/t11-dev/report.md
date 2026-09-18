# t11-dev

Means over tasks of the mean over seeds; 95% intervals from a nested bootstrap (tasks, then seeds). Error cells count as failures.

| arm | cells | success rate | revisions | model calls | shadow $ | seconds | status |
|---|---|---|---|---|---|---|---|
| baseline | 18 | 0.78 [0.50, 1.00] | 0.5 [0.0, 1.4] | 1.5 [1.0, 2.4] | 0.0012 [0.0006, 0.0022] | 74.7 [36.5, 139.6] | succeeded 17, failed 1 |
| multi-r2 | 18 | 0.61 [0.28, 0.89] | 2.1 [0.8, 3.8] | 20.3 [13.0, 29.4] | 0.0078 [0.0044, 0.0121] | 278.0 [159.3, 426.4] | succeeded 17, failed 1 |
| multi | 18 | 0.61 [0.28, 0.89] | 2.1 [0.8, 3.8] | 20.3 [13.0, 29.4] | 0.0078 [0.0044, 0.0121] | 277.5 [159.4, 425.6] | succeeded 17, failed 1 |

## Per task (successes / cells, and how the others failed)

| task | baseline | multi-r2 | multi |
|---|---|---|---|
| adult-1-income | 2/3 (failed 1) | 1/3 (score 2) | 1/3 (score 2) |
| adult-2-education | 3/3 | 3/3 | 3/3 |
| air-1-co-daily | 3/3 | 3/3 | 3/3 |
| air-2-co-estimate | 2/3 (score 1) | 1/3 (clean_rerun 1, score 1) | 1/3 (clean_rerun 1, score 1) |
| bike-1-forecast | 1/3 (score 2) | 1/3 (score 2) | 1/3 (score 2) |
| bike-2-weather | 3/3 | 2/3 (answer 1) | 2/3 (answer 1) |

## Paired differences (a - b, same task and seed)

| a | b | metric | difference [95% CI] | pairs | real? |
|---|---|---|---|---|---|
| baseline | multi-r2 | success | 0.17 [-0.06, 0.44] | 18 | no |
| baseline | multi-r2 | revisions | -1.6 [-3.2, -0.6] | 18 | yes |
| baseline | multi-r2 | model_calls | -18.8 [-27.4, -11.9] | 18 | yes |
| baseline | multi-r2 | shadow_usd | -0.0066 [-0.0105, -0.0038] | 18 | yes |
| baseline | multi-r2 | seconds | -203.4 [-325.0, -120.2] | 18 | yes |
| baseline | multi | success | 0.17 [-0.06, 0.44] | 18 | no |
| baseline | multi | revisions | -1.6 [-3.2, -0.6] | 18 | yes |
| baseline | multi | model_calls | -18.8 [-27.4, -11.9] | 18 | yes |
| baseline | multi | shadow_usd | -0.0066 [-0.0105, -0.0038] | 18 | yes |
| baseline | multi | seconds | -202.8 [-323.4, -119.8] | 18 | yes |
| multi-r2 | multi | success | 0.00 [0.00, 0.00] | 18 | no |
| multi-r2 | multi | revisions | 0.0 [0.0, 0.0] | 18 | no |
| multi-r2 | multi | model_calls | 0.0 [0.0, 0.0] | 18 | no |
| multi-r2 | multi | shadow_usd | 0.0000 [0.0000, 0.0000] | 18 | no |
| multi-r2 | multi | seconds | 0.6 [-0.3, 1.7] | 18 | no |
