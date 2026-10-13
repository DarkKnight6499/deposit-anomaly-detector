# Gradient boosting, held-out year, feature groups removed

Thresholds re-selected on out-of-fold scores for each variant.

| features | n_features | pr_auc | detection_rate | mean_lead_days | fa_per_year | dq_events_alerted | dq_events_total |
|---|---|---|---|---|---|---|---|
| all features | 42 | 0.937 | 1.000 | 13.250 | 1.996 | 0 | 5 |
| without ops and concentration signals | 38 | 0.936 | 1.000 | 13.750 | 2.994 | 0 | 5 |
| without market and rate-spread signals | 33 | 0.844 | 1.000 | 9.000 | 1.996 | 0 | 5 |
| flows only (no market, no ops) | 29 | 0.830 | 1.000 | 8.750 | 1.996 | 0 | 5 |
