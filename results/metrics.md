# Test-split results

Written by `src/run_eval.py`; do not edit by hand.

- category: `bottle`, seeds: 0, 1, 2; cells are mean ± sample std (ddof=1)
- calibrated threshold: quantile 1 - 0.01 of `val_normal` scores, fixed before this run
- `f1_max (oracle)` picks its threshold with test labels: an upper bound, not achieved performance
- `mahalanobis` is image-level only; its anomaly map is its image score repeated, so its pixel metrics carry no localization
- latency is measured separately in `results/latency.md`

| model | image_auroc | pixel_auroc | au_pro | recall @ calibrated thr. | FPR @ calibrated thr. | f1_max (oracle) | memory_bank_mb |
|---|---|---|---|---|---|---|---|
| mahalanobis | 0.998 ± 0.000 | 0.763 ± 0.000 | 0.210 ± 0.000 | 0.921 ± 0.027 | 0.000 ± 0.000 | 0.992 ± 0.000 | n/a |
| autoencoder | 0.909 ± 0.007 | 0.733 ± 0.008 | 0.479 ± 0.004 | 0.582 ± 0.177 | 0.000 ± 0.000 | 0.907 ± 0.006 | n/a |
| patchcore | 1.000 ± 0.000 | 0.980 ± 0.000 | 0.920 ± 0.003 | 1.000 ± 0.000 | 0.000 ± 0.000 | 1.000 ± 0.000 | 7.67 |
