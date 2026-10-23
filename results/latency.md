# Latency benchmark

Written by `src/benchmark.py`; do not edit by hand.

- category: `bottle`, seed: `0`
- 200 single-image `POST /predict` requests per model after 5 warm-up requests, sent one at a time
- client = round trip seen by the caller (in-process `TestClient`, no network); server = the `latency_ms` the service reports for decode, preprocessing, model and heatmap
- 12 CPU threads, Intel64 Family 6 Model 154 Stepping 3, GenuineIntel
- `first request` is the first request after startup warm-up

| model | client p50 ms | client p95 ms | server p50 ms | server p95 ms | first request ms |
|---|---|---|---|---|---|
| mahalanobis | 158.9 | 175.8 | 154.5 | 171.5 | 188.3 |
| autoencoder | 45.1 | 52.1 | 40.6 | 47.5 | 48.0 |
| patchcore | 207.4 | 219.6 | 203.8 | 215.9 | 192.6 |
