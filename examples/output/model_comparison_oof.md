# Model comparison, oof

Assumed costs (stylized): missed liquidity event = 100, false-alarm week = 1. Thresholds are chosen on out-of-fold scores only.

| model | threshold | pr_auc | precision | recall | events_detected | events_total | detection_rate | mean_lead_days | fa_per_year | dq_events_alerted | dq_events_total | cost |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rules | 0.805 | 0.343 | 0.471 | 0.110 | 7 | 7 | 1.000 | 4.000 | 9.591 | 7 | 9 | 24.000 |
| robust_z | 5.029 | 0.439 | 0.590 | 0.211 | 7 | 7 | 1.000 | 2.857 | 9.990 | 6 | 9 | 25.000 |
| isolation_forest | 0.455 | 0.521 | 0.652 | 0.266 | 7 | 7 | 1.000 | 10.000 | 6.394 | 7 | 9 | 16.000 |
| autoencoder | 1.704 | 0.574 | 0.682 | 0.275 | 7 | 7 | 1.000 | 11.000 | 5.994 | 9 | 9 | 15.000 |
| gradient_boosting | 0.594 | 0.746 | 0.784 | 0.399 | 7 | 7 | 1.000 | 8.571 | 3.597 | 0 | 9 | 9.000 |
| random_forest | 0.617 | 0.846 | 0.973 | 0.335 | 7 | 7 | 1.000 | 6.714 | 0.000 | 0 | 9 | 0.000 |
| rules_fixed_policy | 1.000 | 0.343 | 0.375 | 0.041 | 5 | 7 | 0.714 | 2.800 | 5.595 | 5 | 9 | 214.000 |
