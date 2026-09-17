# t9-dev

Means over tasks of the mean over seeds; 95% intervals from a nested bootstrap (tasks, then seeds). Error cells count as failures.

| arm | cells | success rate | revisions | model calls | shadow $ | seconds | status |
|---|---|---|---|---|---|---|---|
| baseline | 18 | 0.72 [0.44, 1.00] | 0.9 [0.0, 2.4] | 1.9 [1.0, 3.4] | 0.0015 [0.0006, 0.0031] | 91.9 [35.3, 185.0] | failed 2, succeeded 16 |
| multi | 18 | 0.44 [0.11, 0.83] | 3.0 [1.3, 4.7] | 24.9 [14.9, 35.4] | 0.0096 [0.0054, 0.0142] | 340.3 [185.7, 508.0] | needs_human 6, succeeded 12 |

## Per task (successes / cells, and how the others failed)

| task | baseline | multi |
|---|---|---|
| adult-1-income | 1/3 (predict 2) | 0/3 (predict 3) |
| adult-2-education | 3/3 | 3/3 |
| air-1-co-daily | 3/3 | 3/3 |
| air-2-co-estimate | 2/3 (score 1) | 0/3 (delivered 2, predict 1) |
| bike-1-forecast | 1/3 (score 2) | 0/3 (score 3) |
| bike-2-weather | 3/3 | 2/3 (answer 1) |

## Paired differences (a - b, same task and seed)

| a | b | metric | difference [95% CI] | pairs | real? |
|---|---|---|---|---|---|
| baseline | multi | success | 0.28 [0.06, 0.56] | 18 | yes |
| baseline | multi | revisions | -2.1 [-3.6, -0.8] | 18 | yes |
| baseline | multi | model_calls | -23.0 [-32.6, -13.9] | 18 | yes |
| baseline | multi | shadow_usd | -0.0081 [-0.0116, -0.0048] | 18 | yes |
| baseline | multi | seconds | -248.4 [-363.1, -143.3] | 18 | yes |
