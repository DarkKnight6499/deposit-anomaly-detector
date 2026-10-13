# Expanding-window CV: day-level PR-AUC by fold

Purge gap = 90 days. Fold validation blocks: 2021H2, 2022H1, 2022H2, 2023H1, 2023H2.

| model | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| autoencoder | 0.853 | 0.915 | 0.684 | 0.590 | 0.576 |
| gradient_boosting | 0.844 | 1.000 | 0.924 | 0.846 | 0.926 |
| isolation_forest | 0.831 | 0.839 | 0.672 | 0.547 | 0.457 |
| random_forest | 0.908 | 0.997 | 0.956 | 0.935 | 0.940 |
| robust_z | 0.754 | 0.606 | 0.573 | 0.543 | 0.360 |
| rules | 0.369 | 0.352 | 0.530 | 0.455 | 0.294 |
