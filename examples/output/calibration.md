# Held-out year calibration (10 equal-width bins)

Raw columns apply only to the probabilistic classifiers; others have arbitrary score scales.

| model | brier_platt | ece_platt | brier_raw | ece_raw |
|---|---|---|---|---|
| rules | 0.2026 | 0.1312 |  |  |
| robust_z | 0.1773 | 0.1185 |  |  |
| isolation_forest | 0.1682 | 0.0629 |  |  |
| autoencoder | 0.1666 | 0.1227 |  |  |
| gradient_boosting | 0.0559 | 0.0905 | 0.0496 | 0.0338 |
| random_forest | 0.0752 | 0.0902 | 0.0689 | 0.0769 |
