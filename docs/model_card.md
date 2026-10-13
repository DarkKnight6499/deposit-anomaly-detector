# Model card: deposit-anomaly-detector

## Intended use

A research and teaching example of how to build, validate, explain and monitor a detector for abnormal deposit outflows. It is not a production liquidity control, not a regulatory model and not a substitute for a bank's contingency funding process. Do not use it for production decisions without independent validation on the user's own data.

## Data

Fully synthetic, produced by seeded code in `src/deposit_anomaly/generator.py`. 2,041 daily rows (2019-06-01 to 2024-12-31), six deposit segments, market and operational indicators, and 33 labelled injected events (7 runs, 4 flights to quality, 4 slow bleeds, 11 data glitches, 7 payment outages) plus 8 unlabelled benign market bumps. No real customer, bank or market data is used. Label rules are in `docs/methodology.md`.

## Models

Rules baseline, robust-z baseline, Isolation Forest, MLP autoencoder, gradient boosting and random forest. The primary supervised model is gradient boosting; its output is a score in [0, 1] that is roughly, not exactly, calibrated.

## Metrics (held-out year 2024, thresholds frozen from out-of-fold scores)

See the README tables. For gradient boosting: PR-AUC 0.937 (block-bootstrap 95% interval 0.826 to 0.995), 4 of 4 liquidity events detected (exact 95% interval 0.398 to 1.000), mean lead 13.25 days, 1.996 false-alarm weeks per year, 0 of 5 data-quality events raised as liquidity alerts. Across 8 other generator seeds the mean held-out PR-AUC is 0.921 (sd 0.043) and detection 0.906 (sd 0.129).

## Limitations

- The held-out year has 4 liquidity events and 5 data-quality events. Event-level numbers are anecdotal; the intervals are wide.
- The generator and the detector were written together. The injected signals (competitor spread, bank CDS, outflow concentration, reconciliation gap) are stronger and cleaner than real ones, and supervised models are rewarded for learning them. Treat the supervised advantage over baselines as an upper bound.
- Labels mark whole windows, including precursor days where only market indicators move, so a "false" alarm just before a window is possible but counted as false.
- Thresholds depend on assumed costs (a missed event is 100 times a false-alarm week); different costs give different thresholds.
- Data-quality routing succeeded on every glitch and outage day because the synthetic reconciliation gap and payment queue are clean tells. Real operational signals are noisier and often delayed.
- Real runs are rarer, faster and more heterogeneous than the injected ones, and real stress can change what normal looks like (regime change), which this design does not model.
- PSI flags many features as drifting during stress periods; it is a monitoring prompt, not proof of model failure.
- The calendar adjustment model learns from all past rows, including stress and rebuilding days, so z-scores on event-free days are not exactly centred (mean total z -0.33, sd 1.44).

## Ethical notes

Synthetic data only; no personal data. Alerts could prompt costly liquidity actions or customer-facing steps. Any real deployment needs human review of alerts, documented overrides, change control, ongoing back-testing, and fairness and conduct review if segment-level outputs reach customers.
