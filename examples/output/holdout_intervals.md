# Held-out year, 95% intervals

PR-AUC: moving-block bootstrap (10-day blocks, 500 reps). Detection rate: exact binomial (Clopper-Pearson) on only a handful of events. False alarms: exact Poisson.

| model | pr_auc | pr_auc_lo | pr_auc_hi | detection_rate | det_lo | det_hi | fa_per_year | fa_lo | fa_hi |
|---|---|---|---|---|---|---|---|---|---|
| rules | 0.475 | 0.282 | 0.666 | 1.000 | 0.398 | 1.000 | 7.984 | 3.447 | 15.731 |
| robust_z | 0.604 | 0.396 | 0.773 | 1.000 | 0.398 | 1.000 | 7.984 | 3.447 | 15.731 |
| isolation_forest | 0.643 | 0.410 | 0.809 | 1.000 | 0.398 | 1.000 | 3.992 | 1.088 | 10.221 |
| autoencoder | 0.685 | 0.464 | 0.856 | 1.000 | 0.398 | 1.000 | 4.990 | 1.620 | 11.644 |
| gradient_boosting | 0.937 | 0.826 | 0.995 | 1.000 | 0.398 | 1.000 | 1.996 | 0.242 | 7.210 |
| random_forest | 0.937 | 0.851 | 0.988 | 1.000 | 0.398 | 1.000 | 1.996 | 0.242 | 7.210 |
| rules_fixed_policy | 0.475 | 0.282 | 0.666 | 0.750 | 0.194 | 0.994 | 4.990 | 1.620 | 11.644 |
