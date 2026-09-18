# t11-test-core

Means over tasks of the mean over seeds; 95% intervals from a nested bootstrap (tasks, then seeds). Error cells count as failures.

| arm | cells | success rate | revisions | model calls | shadow $ | seconds | status |
|---|---|---|---|---|---|---|---|
| baseline | 24 | 0.88 [0.67, 1.00] | 0.8 [0.1, 1.8] | 1.8 [1.1, 2.8] | 0.0016 [0.0007, 0.0028] | 152.9 [50.9, 346.0] | succeeded 23, failed 1 |
| memall | 24 | 0.71 [0.46, 0.92] | 2.5 [1.0, 4.3] | 20.9 [13.2, 29.6] | 0.0093 [0.0054, 0.0137] | 363.0 [213.2, 534.6] | succeeded 20, needs_human 3, failed 1 |
| multi | 24 | 0.50 [0.21, 0.79] | 3.3 [1.2, 5.8] | 23.0 [13.2, 34.0] | 0.0097 [0.0052, 0.0148] | 372.5 [200.9, 569.9] | succeeded 17, needs_human 5, failed 2 |

## Per task (successes / cells, and how the others failed)

| task | baseline | memall | multi |
|---|---|---|---|
| bank-1-subscribe | 3/3 | 2/3 (delivered 1) | 2/3 (score 1) |
| bank-2-previous-campaign | 1/3 (answer 2) | 3/3 | 2/3 (answer 1) |
| credit-1-default | 3/3 | 3/3 | 2/3 (predict 1) |
| credit-2-education | 3/3 | 3/3 | 3/3 |
| diabetes-1-readmit | 2/3 (failed 1) | 1/3 (delivered 2) | 0/3 (predict 1, delivered 2) |
| diabetes-2-age-rates | 3/3 | 3/3 | 3/3 |
| retail-1-monthly-revenue | 3/3 | 1/3 (answer 2) | 0/3 (answer 3) |
| retail-2-repeat-buyers | 3/3 | 1/3 (honest_estimate 1, delivered 1) | 0/3 (delivered 1, predict 2) |

## Paired differences (a - b, same task and seed)

| a | b | metric | difference [95% CI] | pairs | real? |
|---|---|---|---|---|---|
| baseline | memall | success | 0.17 [-0.17, 0.50] | 24 | no |
| baseline | memall | revisions | -1.7 [-3.5, -0.2] | 24 | yes |
| baseline | memall | model_calls | -19.1 [-27.5, -11.8] | 24 | yes |
| baseline | memall | shadow_usd | -0.0077 [-0.0118, -0.0043] | 24 | yes |
| baseline | memall | seconds | -210.1 [-388.3, -18.8] | 24 | yes |
| baseline | multi | success | 0.38 [0.04, 0.71] | 24 | yes |
| baseline | multi | revisions | -2.5 [-4.8, -0.7] | 24 | yes |
| baseline | multi | model_calls | -21.1 [-31.7, -12.0] | 24 | yes |
| baseline | multi | shadow_usd | -0.0081 [-0.0126, -0.0041] | 24 | yes |
| baseline | multi | seconds | -219.6 [-419.5, 4.1] | 24 | no |
| memall | multi | success | 0.21 [0.00, 0.42] | 24 | no |
| memall | multi | revisions | -0.8 [-2.1, 0.5] | 24 | no |
| memall | multi | model_calls | -2.0 [-7.2, 3.0] | 24 | no |
| memall | multi | shadow_usd | -0.0004 [-0.0027, 0.0019] | 24 | no |
| memall | multi | seconds | -9.5 [-100.6, 72.2] | 24 | no |
