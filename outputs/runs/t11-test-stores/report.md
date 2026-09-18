# t11-test-stores

Means over tasks of the mean over seeds; 95% intervals from a nested bootstrap (tasks, then seeds). Error cells count as failures.

| arm | cells | success rate | revisions | model calls | shadow $ | seconds | status |
|---|---|---|---|---|---|---|---|
| memepi | 24 | 0.62 [0.38, 0.88] | 2.7 [1.1, 4.4] | 21.7 [12.9, 30.7] | 0.0094 [0.0050, 0.0138] | 360.3 [199.3, 522.5] | succeeded 20, needs_human 4 |
| memproc | 24 | 0.46 [0.17, 0.75] | 2.7 [0.9, 4.8] | 20.5 [12.4, 29.8] | 0.0095 [0.0050, 0.0145] | 390.9 [208.3, 595.8] | succeeded 18, needs_human 4, failed 2 |
| memsem | 24 | 0.58 [0.29, 0.88] | 2.7 [1.1, 4.5] | 21.2 [13.1, 30.3] | 0.0091 [0.0053, 0.0133] | 359.1 [208.7, 523.2] | succeeded 19, needs_human 3, failed 2 |

## Per task (successes / cells, and how the others failed)

| task | memepi | memproc | memsem |
|---|---|---|---|
| bank-1-subscribe | 2/3 (score 1) | 2/3 (delivered 1) | 2/3 (score 1) |
| bank-2-previous-campaign | 2/3 (answer 1) | 2/3 (answer 1) | 2/3 (answer 1) |
| credit-1-default | 2/3 (delivered 1) | 1/3 (delivered 1, score 1) | 2/3 (predict 1) |
| credit-2-education | 3/3 | 3/3 | 3/3 |
| diabetes-1-readmit | 1/3 (delivered 1, score 1) | 0/3 (delivered 2, score 1) | 0/3 (predict 1, delivered 2) |
| diabetes-2-age-rates | 3/3 | 3/3 | 3/3 |
| retail-1-monthly-revenue | 0/3 (answer 3) | 0/3 (answer 3) | 0/3 (answer 3) |
| retail-2-repeat-buyers | 2/3 (delivered 1) | 0/3 (honest_estimate 1, predict 2) | 2/3 (score 1) |

## Paired differences (a - b, same task and seed)

| a | b | metric | difference [95% CI] | pairs | real? |
|---|---|---|---|---|---|
| memepi | memproc | success | 0.17 [-0.08, 0.42] | 24 | no |
| memepi | memproc | revisions | 0.0 [-1.4, 1.2] | 24 | no |
| memepi | memproc | model_calls | 1.1 [-5.5, 7.0] | 24 | no |
| memepi | memproc | shadow_usd | -0.0002 [-0.0036, 0.0025] | 24 | no |
| memepi | memproc | seconds | -30.6 [-161.4, 62.8] | 24 | no |
| memepi | memsem | success | 0.04 [-0.17, 0.25] | 24 | no |
| memepi | memsem | revisions | 0.0 [-1.3, 1.3] | 24 | no |
| memepi | memsem | model_calls | 0.4 [-4.8, 6.2] | 24 | no |
| memepi | memsem | shadow_usd | 0.0003 [-0.0022, 0.0029] | 24 | no |
| memepi | memsem | seconds | 1.2 [-95.7, 95.7] | 24 | no |
| memproc | memsem | success | -0.12 [-0.38, 0.12] | 24 | no |
| memproc | memsem | revisions | 0.0 [-1.5, 1.5] | 24 | no |
| memproc | memsem | model_calls | -0.7 [-7.5, 5.5] | 24 | no |
| memproc | memsem | shadow_usd | 0.0005 [-0.0026, 0.0036] | 24 | no |
| memproc | memsem | seconds | 31.8 [-76.4, 149.1] | 24 | no |
