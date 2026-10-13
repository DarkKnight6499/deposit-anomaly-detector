# Detection by event type, oof

| model | event_type | events | alerted | rate | mean_lead_days |
|---|---|---|---|---|---|
| rules | data_glitch | 5 | 5 | 1.00 | -1.00 |
| rules | flight_to_quality | 2 | 2 | 1.00 | 6.00 |
| rules | idiosyncratic_run | 3 | 3 | 1.00 | 3.33 |
| rules | payment_outage | 4 | 2 | 0.50 | -2.50 |
| rules | slow_bleed | 2 | 2 | 1.00 | 3.00 |
| robust_z | data_glitch | 5 | 5 | 1.00 | -1.00 |
| robust_z | flight_to_quality | 2 | 2 | 1.00 | 5.50 |
| robust_z | idiosyncratic_run | 3 | 3 | 1.00 | 4.67 |
| robust_z | payment_outage | 4 | 1 | 0.25 | -2.00 |
| robust_z | slow_bleed | 2 | 2 | 1.00 | -2.50 |
| isolation_forest | data_glitch | 5 | 5 | 1.00 | 0.00 |
| isolation_forest | flight_to_quality | 2 | 2 | 1.00 | 6.00 |
| isolation_forest | idiosyncratic_run | 3 | 3 | 1.00 | 5.33 |
| isolation_forest | payment_outage | 4 | 2 | 0.50 | -2.00 |
| isolation_forest | slow_bleed | 2 | 2 | 1.00 | 21.00 |
| autoencoder | data_glitch | 5 | 5 | 1.00 | 0.00 |
| autoencoder | flight_to_quality | 2 | 2 | 1.00 | 8.00 |
| autoencoder | idiosyncratic_run | 3 | 3 | 1.00 | 6.33 |
| autoencoder | payment_outage | 4 | 4 | 1.00 | -0.50 |
| autoencoder | slow_bleed | 2 | 2 | 1.00 | 21.00 |
| gradient_boosting | data_glitch | 5 | 0 | 0.00 |  |
| gradient_boosting | flight_to_quality | 2 | 2 | 1.00 | 5.50 |
| gradient_boosting | idiosyncratic_run | 3 | 3 | 1.00 | 7.00 |
| gradient_boosting | payment_outage | 4 | 0 | 0.00 |  |
| gradient_boosting | slow_bleed | 2 | 2 | 1.00 | 14.00 |
| random_forest | data_glitch | 5 | 0 | 0.00 |  |
| random_forest | flight_to_quality | 2 | 2 | 1.00 | 5.50 |
| random_forest | idiosyncratic_run | 3 | 3 | 1.00 | 4.00 |
| random_forest | payment_outage | 4 | 0 | 0.00 |  |
| random_forest | slow_bleed | 2 | 2 | 1.00 | 12.00 |
| rules_fixed_policy | data_glitch | 5 | 5 | 1.00 | -1.00 |
| rules_fixed_policy | flight_to_quality | 2 | 2 | 1.00 | 5.50 |
| rules_fixed_policy | idiosyncratic_run | 3 | 2 | 0.67 | 2.50 |
| rules_fixed_policy | payment_outage | 4 | 0 | 0.00 |  |
| rules_fixed_policy | slow_bleed | 2 | 1 | 0.50 | -2.00 |
