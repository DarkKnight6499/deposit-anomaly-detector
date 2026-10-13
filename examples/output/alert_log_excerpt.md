# Alert log excerpt (first 25 non-quiet days)

Synthetic data. Severity: WATCH (>= 0.5x threshold), ALERT, CRITICAL, DATA_QUALITY (routed).

| date | score | probability | severity | route | dq_reason | consecutive_alert_days | explanation |
|---|---|---|---|---|---|---|---|
| 2024-01-18 | 0.554 | 0.634 | WATCH | liquidity |  | 0 | rate_spread_chg10=16.33 (+2.60); bank_cds_bps=78.05 (-1.09); competitor_rate_spread_bps=49.14 (-0.71) |
| 2024-01-19 | 0.601 | 0.685 | ALERT | liquidity |  | 1 | rate_spread_chg10=23.10 (+2.69); bank_cds_bps=77.80 (-1.24); competitor_rate_spread_bps=56.62 (-0.53) |
| 2024-01-20 | 0.652 | 0.736 | ALERT | liquidity |  | 2 | rate_spread_chg10=30.21 (+2.72); bank_cds_bps=82.02 (-0.66); competitor_rate_spread_bps=63.08 (-0.62) |
| 2024-01-21 | 0.810 | 0.858 | CRITICAL | liquidity |  | 3 | rate_spread_chg10=35.82 (+2.79); competitor_rate_spread_bps=69.08 (-0.54); bank_cds_bps=82.81 (-0.30) |
| 2024-01-22 | 0.758 | 0.824 | CRITICAL | liquidity |  | 4 | rate_spread_chg10=43.04 (+2.87); bank_cds_bps=82.63 (-0.59); z_retail_uninsured=-0.16 (-0.41) |
| 2024-01-23 | 0.741 | 0.811 | CRITICAL | liquidity |  | 5 | rate_spread_chg10=51.48 (+2.78); bank_cds_bps=79.53 (-0.83); market_cds_chg5=5.27 (+0.29) |
| 2024-01-24 | 0.849 | 0.879 | CRITICAL | liquidity |  | 6 | rate_spread_chg10=57.70 (+2.72); bank_cds_bps=80.61 (-0.74); z_retail_uninsured=0.95 (+0.40) |
| 2024-01-25 | 0.889 | 0.898 | CRITICAL | liquidity |  | 7 | rate_spread_chg10=62.89 (+2.61); bank_cds_bps=81.79 (-0.63); z_retail_uninsured=2.96 (+0.44) |
| 2024-01-26 | 0.931 | 0.915 | CRITICAL | liquidity |  | 8 | rate_spread_chg10=64.13 (+2.66); market_cds_chg5=7.84 (+0.45); z_retail_uninsured=4.76 (+0.44) |
| 2024-01-27 | 0.945 | 0.921 | CRITICAL | liquidity |  | 9 | rate_spread_chg10=65.73 (+2.31); competitor_rate_spread_bps=107.94 (+1.15); bank_cds_bps=81.65 (-0.51) |
| 2024-01-28 | 0.965 | 0.927 | CRITICAL | liquidity |  | 10 | rate_spread_chg10=64.69 (+2.27); competitor_rate_spread_bps=113.82 (+1.38); z_retail_uninsured=1.66 (+0.44) |
| 2024-01-29 | 0.990 | 0.935 | CRITICAL | liquidity |  | 11 | competitor_rate_spread_bps=119.85 (+3.40); rate_spread_chg10=63.23 (+1.69); bank_cds_bps=81.52 (-0.46) |
| 2024-01-30 | 0.986 | 0.934 | CRITICAL | liquidity |  | 12 | competitor_rate_spread_bps=123.81 (+3.39); rate_spread_chg10=60.73 (+1.81); bank_cds_bps=79.35 (-0.89) |
| 2024-01-31 | 0.930 | 0.915 | CRITICAL | liquidity |  | 13 | competitor_rate_spread_bps=132.65 (+3.14); rate_spread_chg10=63.58 (+1.76); bank_cds_bps=76.30 (-1.01) |
| 2024-02-01 | 0.986 | 0.934 | CRITICAL | liquidity |  | 14 | competitor_rate_spread_bps=143.65 (+3.59); rate_spread_chg10=68.69 (+1.76); bank_cds_bps=79.19 (-0.83) |
| 2024-02-02 | 0.993 | 0.936 | CRITICAL | liquidity |  | 15 | competitor_rate_spread_bps=154.01 (+3.57); rate_spread_chg10=70.43 (+1.64); bank_cds_bps=81.51 (-0.39) |
| 2024-02-03 | 0.979 | 0.932 | CRITICAL | liquidity |  | 16 | competitor_rate_spread_bps=160.34 (+3.67); rate_spread_chg10=72.35 (+1.82); bank_cds_bps=75.47 (-1.01) |
| 2024-02-04 | 0.933 | 0.916 | CRITICAL | liquidity |  | 17 | competitor_rate_spread_bps=166.41 (+3.40); rate_spread_chg10=72.22 (+1.62); bank_cds_bps=72.78 (-1.01) |
| 2024-02-05 | 0.976 | 0.931 | CRITICAL | liquidity |  | 18 | competitor_rate_spread_bps=167.12 (+3.72); rate_spread_chg10=66.76 (+1.70); bank_cds_bps=74.92 (-0.93) |
| 2024-02-06 | 0.983 | 0.933 | CRITICAL | liquidity |  | 19 | competitor_rate_spread_bps=164.94 (+3.49); rate_spread_chg10=57.00 (+1.92); bank_cds_bps=77.22 (-0.91) |
| 2024-02-07 | 0.940 | 0.919 | CRITICAL | liquidity |  | 20 | competitor_rate_spread_bps=164.66 (+3.43); rate_spread_chg10=50.84 (+1.75); bank_cds_bps=74.72 (-1.01) |
| 2024-02-08 | 0.992 | 0.936 | CRITICAL | liquidity |  | 21 | competitor_rate_spread_bps=166.25 (+4.03); rate_spread_chg10=46.40 (+1.82); bank_cds_bps=78.06 (-0.61) |
| 2024-02-09 | 0.989 | 0.935 | CRITICAL | liquidity |  | 22 | competitor_rate_spread_bps=168.54 (+3.86); rate_spread_chg10=44.74 (+1.81); bank_cds_bps=79.78 (-0.67) |
| 2024-02-10 | 0.982 | 0.933 | CRITICAL | liquidity |  | 23 | competitor_rate_spread_bps=167.92 (+3.43); rate_spread_chg10=35.27 (+1.91); bank_cds_bps=78.25 (-0.91) |
| 2024-02-11 | 0.979 | 0.932 | CRITICAL | liquidity |  | 24 | competitor_rate_spread_bps=169.44 (+3.43); rate_spread_chg10=25.79 (+1.84); bank_cds_bps=77.05 (-0.96) |
