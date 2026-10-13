# Permutation importance (gradient_boosting, held-out year)

Drop in day-level PR-AUC when one feature is shuffled.

| feature | mean_drop | std_drop |
|---|---|---|
| competitor_rate_spread_bps | 0.2604 | 0.0298 |
| rate_spread_chg10 | 0.0756 | 0.0100 |
| bank_cds_z | 0.0427 | 0.0092 |
| market_cds_chg5 | 0.0245 | 0.0125 |
| top10_share | 0.0193 | 0.0047 |
| bank_cds_bps | 0.0144 | 0.0099 |
| bank_cds_chg5 | 0.0098 | 0.0093 |
| cum_out_5d | 0.0076 | 0.0014 |
| cum5_z | 0.0071 | 0.0014 |
| cum_out_10d | 0.0065 | 0.0042 |
| z_retail_uninsured | 0.0053 | 0.0021 |
| z_total_lag2 | 0.0050 | 0.0023 |
| market_equity_drawdown_pct | 0.0045 | 0.0016 |
| total_outflow_pct_lag2 | 0.0022 | 0.0009 |
| outflow_velocity | 0.0013 | 0.0006 |
| outflow_acceleration | 0.0009 | 0.0002 |
| z_total_lag1 | 0.0009 | 0.0015 |
| bank_equity_drawdown_pct | 0.0008 | 0.0026 |
| cum_out_20d | 0.0006 | 0.0103 |
| max_segment_z | 0.0004 | 0.0005 |
