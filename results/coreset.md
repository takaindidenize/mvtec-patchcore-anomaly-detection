# Coreset ratio sweep

Written by `src/coreset_sweep.py`; do not edit by hand.

- category: `bottle`, seed: `0`, backbone: `wide_resnet50_2`
- query latency: single-image `predict` over 42 `val_normal` images, 12 CPU threads, Intel64 Family 6 Model 154 Stepping 3, GenuineIntel
- no test image was read

| coreset_ratio | bank patches | memory_bank_mb | fit s | query p50 ms | query p95 ms | val_normal median score |
|---|---|---|---|---|---|---|
| 0.1 | 13092 | 76.71 | 285.6 | 333.5 | 348.9 | 27.179 |
| 0.01 (config) | 1309 | 7.67 | 40.5 | 167.8 | 176.0 | 30.821 |
| 0.001 | 130 | 0.76 | 17.4 | 116.1 | 126.9 | 41.794 |
