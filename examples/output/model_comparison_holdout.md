# Model comparison, holdout

Assumed costs (stylized): missed liquidity event = 100, false-alarm week = 1. Thresholds are chosen on out-of-fold scores only.

| model | threshold | pr_auc | precision | recall | events_detected | events_total | detection_rate | mean_lead_days | fa_per_year | dq_events_alerted | dq_events_total | cost |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rules | 0.805 | 0.475 | 0.690 | 0.185 | 4 | 4 | 1.000 | 3.750 | 7.984 | 3 | 5 | 8.000 |
| robust_z | 5.029 | 0.604 | 0.810 | 0.315 | 4 | 4 | 1.000 | 8.500 | 7.984 | 3 | 5 | 8.000 |
| isolation_forest | 0.455 | 0.643 | 0.756 | 0.315 | 4 | 4 | 1.000 | 10.000 | 3.992 | 3 | 5 | 4.000 |
| autoencoder | 1.704 | 0.685 | 0.800 | 0.370 | 4 | 4 | 1.000 | 10.250 | 4.990 | 5 | 5 | 5.000 |
| gradient_boosting | 0.594 | 0.937 | 0.906 | 0.889 | 4 | 4 | 1.000 | 13.250 | 1.996 | 0 | 5 | 2.000 |
| random_forest | 0.617 | 0.937 | 0.933 | 0.648 | 4 | 4 | 1.000 | 9.250 | 1.996 | 0 | 5 | 2.000 |
| rules_fixed_policy | 1.000 | 0.475 | 0.643 | 0.083 | 3 | 4 | 0.750 | 3.000 | 4.990 | 3 | 5 | 105.000 |
