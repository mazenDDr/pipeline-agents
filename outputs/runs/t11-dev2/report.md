# t11-dev2

Means over tasks of the mean over seeds; 95% intervals from a nested bootstrap (tasks, then seeds). Error cells count as failures.

| arm | cells | success rate | revisions | model calls | shadow $ | seconds | status |
|---|---|---|---|---|---|---|---|
| multi | 18 | 0.61 [0.33, 0.89] | 2.3 [1.0, 4.2] | 21.9 [14.3, 31.3] | 0.0084 [0.0048, 0.0129] | 303.6 [175.3, 459.0] | succeeded 17, needs_human 1 |

## Per task (successes / cells, and how the others failed)

| task | multi |
|---|---|
| adult-1-income | 1/3 (score 2) |
| adult-2-education | 3/3 |
| air-1-co-daily | 2/3 (delivered 1) |
| air-2-co-estimate | 2/3 (score 1) |
| bike-1-forecast | 1/3 (score 2) |
| bike-2-weather | 2/3 (answer 1) |

## Paired differences (a - b, same task and seed)

| a | b | metric | difference [95% CI] | pairs | real? |
|---|---|---|---|---|---|
