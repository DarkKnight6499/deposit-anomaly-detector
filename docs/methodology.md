# Methodology

All data is synthetic and every parameter below is invented (stylized). Nothing here is calibrated to a real bank.

## 1. Synthetic data and how labels are made

Code: `src/deposit_anomaly/generator.py` (independent of the detector). Seeded with `numpy.random.default_rng`.

- Period: 2019-06-01 to 2024-12-31, daily (2,041 days). The first 214 days are warm-up for features; the modelling window is 2020-01-01 to 2024-12-31 (1,827 days).
- Six segments: retail insured, retail uninsured, small business, corporate operational, corporate non-operational, brokered. Opening balances (USD m): 18,000 / 6,000 / 5,000 / 8,000 / 7,000 / 4,000.
- Base net flow (percent of prior balance): segment growth drift + Student-t(5) noise (unit variance) with one common factor (loading 0.4) + calendar effects. Weekend noise is scaled by 0.15, holiday noise by 0.25. Balances follow flows, with a weak pull back to the growth trend (0.4% of the gap per day, capped at 1%) so segments rebuild slowly after stress.
- Calendar effects (percent of prior balance on the day): payroll days (1st and 15th, rolled back to a business day), the day after payroll, month-end and first business day, tax dates (15 Jan, Apr, Jun, Sep, rolled forward), biweekly small-business paydays, and a December retail spending drift. Holidays come from the US federal calendar.
- Market and operational series: bank and market CDS-like spreads (AR(1) in logs), bank and market equity (random walks), a 2-year rate path, a competitor rate spread, the share of daily outflow from the 10 largest accounts (`top10_outflow_share`), a ledger reconciliation gap, and a payment-queue backlog.

### Injected events (33 in total) and their label windows

A label window starts on the first day any injected effect exists (including market-only precursors) and ends on the last day of the injected outflow. Start dates are the schedule dates in `EVENT_SCHEDULE` jittered by up to 2 days.

| Type | Count | Window | Effect |
|---|---|---|---|
| `idiosyncratic_run` | 7 | 3 precursor days (market only) then 10 to 14 outflow days, ramp 3 days, 3-day tail | Daily outflow up to U(1.6, 2.8)% of non-operational corporate, U(0.7, 1.4)% of uninsured retail, small amounts elsewhere; bank CDS +U(150, 300) bps; bank equity -U(18, 32)% (log); top-10 share up to 0.70 |
| `flight_to_quality` | 4 | 2 precursor days then 12 to 20 outflow days | Smaller outflows from non-operational corporate U(0.4, 0.8)%, brokered U(0.4, 0.7)%, uninsured retail U(0.15, 0.3)%, inflow to insured retail; market CDS +U(60, 110) bps, bank CDS +40% of that, equity falls, 2-year rate falls |
| `slow_bleed` | 4 | 55 to 70 days; outflows start on day 6 and ramp over 10 days | About 0.05 to 0.11% a day from uninsured retail, brokered, non-operational corporate; competitor rate spread +U(80, 140) bps ramping over 20 days |
| `data_glitch` | 11 | 2 days | One segment's reported flow carries a spurious record of U(2.5, 5)% of its balance on day 1, reversed exactly on day 2. The ledger balance never moves, so the reconciliation gap is large and top-10 share is about 0.95 |
| `payment_outage` | 7 | 1 to 3 delay days plus a release day | 90% of net flows of small business and corporate segments are held, then posted on the release day. Payment queue backlog rises |

Unlabelled nuisance (8 windows, `NUISANCE_SCHEDULE`): four 35 to 40 day competitor-spread bumps (+U(40, 80) bps) and four 12 day bank-CDS bumps (+U(40, 90) bps) with no outflow effect. They exist so market indicators alone are not a perfect tell. Windows may not overlap labelled events; the generator raises if they do.

Labels (separate frame): `event_type`, `event_id`, `y_liquidity` (run, flight to quality, slow bleed) and `y_data_quality` (glitch, outage). `critical_day` for liquidity events is the day cumulative injected outflow reaches 50% of its total; for data events it is the first day. Lead time is measured against `critical_day`.

## 2. Features (`features.py`)

Every feature at day t uses data observed up to and including day t (end-of-day scoring). Nothing from t+1 is used. This is tested by perturbing and truncating the future (`tests/test_features.py`).

- Outflow: total and per-segment outflow as percent of prior-day balance (positive means outflow).
- Seasonality-adjusted residual z-scores (total and per segment): expected outflow from a robust least-squares calendar model (weekday, weekend, holiday, payroll, after-payroll, month-end, first business day, tax day, small-business payday), refit every 21 days on past rows only, with residuals beyond 5 robust sds trimmed before the final fit. Scale is the MAD-based sd of the previous 90 residuals. The model needs 120 rows before its first fit.
- Lagged flows (lags 1, 2, 5 of total outflow and total z), cumulative outflow over 5, 10, 20 days, and a robust z of the 5-day cumulative outflow.
- Breadth: share of segments with z above 2, share above 1, and the largest segment z.
- Velocity and acceleration: change in the 5-day mean outflow over 3 days, and the change of that change.
- Concentration proxy: top-10 share and its robust z.
- Operational: reconciliation gap, payment queue backlog.
- Market: bank CDS level, 5-day change, robust z of its log; market CDS 5-day change; bank and market equity drawdown from the 60-day high; 2-year rate 5-day change; competitor spread level and 10-day change.

42 features in total (`FEATURE_COLUMNS`). The longest rolling window is 90 days (`LOOKBACK_DAYS`). The calendar model uses an expanding window of past rows; that is the one memory longer than the purge gap, and it is backward-looking only.

## 3. Models (`models.py`)

| Model | Description |
|---|---|
| `rules` | Static thresholds: the 99.5th percentile of each daily outflow series (total and six segments) on the training window, frozen. Score is the largest ratio to a threshold |
| `robust_z` | Score is the largest seasonality-adjusted outflow z across total, 5-day cumulative and the six segments |
| `isolation_forest` | 300 trees on all features; unsupervised |
| `autoencoder` | `MLPRegressor` (24-8-24), trained to reconstruct clipped standardised features on days with no labelled event of any type (semi-supervised); score is reconstruction error |
| `gradient_boosting` | `GradientBoostingClassifier` (150 trees, depth 3, learning rate 0.05, subsample 0.8) with inverse-frequency sample weights for imbalance |
| `random_forest` | `RandomForestClassifier` (300 trees, min leaf 3, `balanced_subsample` class weights) |

The supervised target is `y_liquidity`. Glitch and outage days are negatives, so a supervised model is trained to treat them as non-liquidity. `rules_fixed_policy` is the rules baseline alerting at ratio 1.0 with no tuning.

## 4. Validation design (`splits.py`, `pipeline.py`)

- Expanding-window CV, five folds. Validation blocks: 2021-07 to 2021-12, 2022-H1, 2022-H2, 2023-H1, 2023-H2. Training is every day before `val_start - 90 days` (purge gap = longest rolling window).
- Final held-out year: 2024-01-01 to 2024-12-31, never used for fitting, thresholds or tuning. The final model trains on days up to 2023-10-02 (same 90-day purge).
- Cost-sensitive thresholds are chosen on the concatenated out-of-fold scores (2021-07-01 to 2023-12-31) and then frozen for the held-out year.

## 5. Evaluation (`evaluation.py`)

- Detection: a liquidity event is detected if at least one alert day falls inside its label window. Lead time = `critical_day` minus the first alert day (positive means before the critical day; negative means late). Means are over detected events.
- False alarms: an alert day is justified if it is inside a liquidity window or up to 5 days after it. Other alert days are false. False alarms are counted as 7-day blocks with at least one false alert day, reported per year. Alerts inside glitch or outage windows are false liquidity alarms and are also tallied as data-quality misroutes.
- Day level: precision and recall at the chosen threshold, and PR-AUC (average precision) on `y_liquidity`.
- Cost: `100 x missed liquidity events + 1 x false-alarm weeks` (assumed, stylized). The threshold search runs over a 200-point quantile grid plus a no-alert option. Many thresholds tie at the minimum; the midpoint of the lowest and highest minimisers is used. Both ends are non-increasing in the miss cost, so the threshold falls as misses become costlier (tested).
- Data-quality routing (`dq.py`): days where the reconciliation gap or payment queue exceeds median + 10 robust sd (fitted on training features, no labels) are labelled data-quality and their liquidity alerts are suppressed. Tables report results both raw and routed.
- Calibration: Platt scaling fitted on out-of-fold scores; Brier score and 10-bin expected calibration error on the held-out year.
- Intervals: PR-AUC by moving-block bootstrap (10-day blocks, 500 resamples); detection rate by exact binomial interval; false alarms per year by exact Poisson interval. With 4 held-out liquidity events the detection intervals are very wide.

## 6. Interpretability and model risk (`interpret.py`, `risk.py`)

- Permutation importance on the held-out year (drop in PR-AUC, 10 shuffles), SHAP `TreeExplainer` values for gradient boosting (log-odds, additivity tested), top-3 contributors per alert day, and one-way partial dependence.
- PSI per feature (training window vs held-out year, with and without event days), PSI by quarter, and PSI of the model score (equal-width bins on [0, 1], out-of-fold scores as the reference). Status bands: below 0.10 OK, below 0.25 WATCH, otherwise ACTION.

## 7. Monitoring (`monitoring.py`)

`build_alert_log` scores each day and writes severity: WATCH (score at least half the threshold), ALERT (at or above the threshold), CRITICAL (calibrated probability at least 0.90, or 3 consecutive alert days), DATA_QUALITY (router fired; no liquidity severity is raised). Each flagged day carries the top SHAP contributors. `drift_report` gives feature PSI plus a score PSI row.

## 8. Known weaknesses of the design

See the README limitations and `docs/model_card.md`.
