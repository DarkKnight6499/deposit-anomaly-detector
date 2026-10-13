# Detection by event type, holdout

| model | event_type | events | alerted | rate | mean_lead_days |
|---|---|---|---|---|---|
| rules | data_glitch | 3 | 3 | 1.00 | -1.00 |
| rules | flight_to_quality | 1 | 1 | 1.00 | 1.00 |
| rules | idiosyncratic_run | 2 | 2 | 1.00 | 4.00 |
| rules | payment_outage | 2 | 0 | 0.00 |  |
| rules | slow_bleed | 1 | 1 | 1.00 | 6.00 |
| robust_z | data_glitch | 3 | 3 | 1.00 | -1.00 |
| robust_z | flight_to_quality | 1 | 1 | 1.00 | 1.00 |
| robust_z | idiosyncratic_run | 2 | 2 | 1.00 | 5.00 |
| robust_z | payment_outage | 2 | 0 | 0.00 |  |
| robust_z | slow_bleed | 1 | 1 | 1.00 | 23.00 |
| isolation_forest | data_glitch | 3 | 3 | 1.00 | 0.00 |
| isolation_forest | flight_to_quality | 1 | 1 | 1.00 | 7.00 |
| isolation_forest | idiosyncratic_run | 2 | 2 | 1.00 | 5.00 |
| isolation_forest | payment_outage | 2 | 0 | 0.00 |  |
| isolation_forest | slow_bleed | 1 | 1 | 1.00 | 23.00 |
| autoencoder | data_glitch | 3 | 3 | 1.00 | 0.00 |
| autoencoder | flight_to_quality | 1 | 1 | 1.00 | 6.00 |
| autoencoder | idiosyncratic_run | 2 | 2 | 1.00 | 6.50 |
| autoencoder | payment_outage | 2 | 2 | 1.00 | 0.00 |
| autoencoder | slow_bleed | 1 | 1 | 1.00 | 22.00 |
| gradient_boosting | data_glitch | 3 | 0 | 0.00 |  |
| gradient_boosting | flight_to_quality | 1 | 1 | 1.00 | 6.00 |
| gradient_boosting | idiosyncratic_run | 2 | 2 | 1.00 | 7.00 |
| gradient_boosting | payment_outage | 2 | 0 | 0.00 |  |
| gradient_boosting | slow_bleed | 1 | 1 | 1.00 | 33.00 |
| random_forest | data_glitch | 3 | 0 | 0.00 |  |
| random_forest | flight_to_quality | 1 | 1 | 1.00 | 1.00 |
| random_forest | idiosyncratic_run | 2 | 2 | 1.00 | 5.00 |
| random_forest | payment_outage | 2 | 0 | 0.00 |  |
| random_forest | slow_bleed | 1 | 1 | 1.00 | 26.00 |
| rules_fixed_policy | data_glitch | 3 | 3 | 1.00 | -1.00 |
| rules_fixed_policy | flight_to_quality | 1 | 0 | 0.00 |  |
| rules_fixed_policy | idiosyncratic_run | 2 | 2 | 1.00 | 1.50 |
| rules_fixed_policy | payment_outage | 2 | 0 | 0.00 |  |
| rules_fixed_policy | slow_bleed | 1 | 1 | 1.00 | 6.00 |
