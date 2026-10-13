# Held-out year across 8 generator seeds

Mean and standard deviation across seeds of the held-out metrics. Thresholds re-selected per seed.

| model | pr_auc_mean | pr_auc_std | detection_rate_mean | detection_rate_std | mean_lead_days_mean | mean_lead_days_std | fa_per_year_mean | fa_per_year_std | dq_events_alerted_mean | dq_events_alerted_std |
|---|---|---|---|---|---|---|---|---|---|---|
| autoencoder | 0.676 | 0.049 | 0.969 | 0.088 | 7.573 | 4.917 | 6.237 | 1.578 | 4.750 | 0.707 |
| gradient_boosting | 0.921 | 0.043 | 0.906 | 0.129 | 10.792 | 1.350 | 1.996 | 1.067 | 0.000 | 0.000 |
| isolation_forest | 0.598 | 0.032 | 1.000 | 0.000 | 8.875 | 2.248 | 6.611 | 2.498 | 3.250 | 0.463 |
| random_forest | 0.960 | 0.022 | 1.000 | 0.000 | 10.406 | 1.706 | 1.123 | 1.639 | 0.000 | 0.000 |
| robust_z | 0.577 | 0.036 | 0.969 | 0.088 | 9.198 | 3.246 | 7.734 | 2.371 | 3.250 | 0.463 |
| rules | 0.454 | 0.044 | 0.875 | 0.134 | 7.781 | 4.611 | 10.478 | 3.064 | 3.250 | 0.463 |
| rules_fixed_policy | 0.454 | 0.044 | 0.594 | 0.229 | 4.792 | 6.761 | 6.112 | 2.096 | 3.000 | 0.535 |
