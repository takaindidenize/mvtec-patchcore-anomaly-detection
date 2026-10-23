# MVTec AD — One-Class Anomaly Detection

Detects manufacturing defects in product images and localizes the defective region.
The model is trained on defect-free images only, so this is one-class (unsupervised)
anomaly detection, not binary classification. A trained model is served over a REST
API and packaged in a CPU-only Docker image.

The implementation contract is [AGENTS.md](AGENTS.md); per-task decisions are recorded in
[docs/DECISIONS.md](docs/DECISIONS.md).

**Scope.** Only the `bottle` category has been trained and evaluated. Nothing below
says how the models behave on the other MVTec AD categories.

## Results

Every number in this README is copied from `results/` by `python -m src.readme`; the
files in `results/` are written by the scripts that measured them. Nothing is typed by
hand, and a value that has not been measured reads `TBD`.

### Test-split metrics

The test split was opened exactly once, after the design was frozen. Thresholds were
calibrated on held-out normal images only.

<!-- BEGIN metrics -->
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
<!-- END metrics -->

How to read it:

- `image_auroc`, `pixel_auroc` and `au_pro` are threshold-free and are the primary
  comparison metrics.
- The two columns at the calibrated threshold show what the deployed operating point
  does. The threshold is a quantile of the held-out normal scores, so a small
  validation set puts it close to the highest normal score. That makes it
  conservative: the false positive rate stays low, and recall can fall well below what
  the threshold-free metrics suggest. The weaker the model, the larger this gap.
- `f1_max (oracle)` chooses its threshold using the test labels. It is an upper bound,
  not achieved performance.
- The Mahalanobis baseline scores whole images. Its anomaly map is the image score
  repeated at every pixel, so its pixel-level numbers do not measure localization.
- Accuracy is not reported: it depends on the threshold and misleads under class
  imbalance.

### Coreset ratio

The ratio was chosen from this table and the held-out normal images alone, without
reference to test data.

<!-- BEGIN coreset -->
Written by `src/coreset_sweep.py`; do not edit by hand.

- category: `bottle`, seed: `0`, backbone: `wide_resnet50_2`
- query latency: single-image `predict` over 42 `val_normal` images, 12 CPU threads, Intel64 Family 6 Model 154 Stepping 3, GenuineIntel
- no test image was read

| coreset_ratio | bank patches | memory_bank_mb | fit s | query p50 ms | query p95 ms | val_normal median score |
|---|---|---|---|---|---|---|
| 0.1 | 13092 | 76.71 | 285.6 | 333.5 | 348.9 | 27.179 |
| 0.01 (config) | 1309 | 7.67 | 40.5 | 167.8 | 176.0 | 30.821 |
| 0.001 | 130 | 0.76 | 17.4 | 116.1 | 126.9 | 41.794 |
<!-- END coreset -->

### Latency

<!-- BEGIN latency -->
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
<!-- END latency -->

## Method

| Model | Idea |
|---|---|
| `mahalanobis` | Global average-pooled features of the deepest backbone stage; Mahalanobis distance to the training distribution, with Ledoit-Wolf shrinkage for the covariance. Image-level only. |
| `autoencoder` | Convolutional autoencoder trained on normal images; the anomaly map is the per-pixel reconstruction error. |
| `patchcore` | Locally aggregated features from two backbone stages, a greedy k-center coreset as memory bank, nearest-neighbour distance per patch, upsampled and Gaussian-smoothed. |

Protocol:

- `train/good` is split deterministically into `train_fit`, used to build the model, and
  `val_normal`, used only to calibrate the threshold. The MVTec test split is untouched
  until the single evaluation.
- The threshold is `numpy.quantile(val_scores, 1 - target_fpr)`: on normal parts the
  configuration is meant to raise a false alarm roughly `target_fpr` of the time.
- Preprocessing is defined once, stored in the artifact bundle and read from there by
  the API. There is no data augmentation.
- PatchCore uses an ImageNet-pretrained `wide_resnet50_2` in inference mode only. That
  backbone is not trained here.
- Every random operation takes an explicit seed from the config, and every model is run
  with the seeds listed above.

## Requirements

- Python 3.12
- GNU Make (Windows: available through MSYS2 or `winget install ezwinports.make`);
  without it, run the `python -m ...` commands from the `Makefile` directly
- Docker, for the container targets

## Quick start

```bash
make setup
make lint
make test
```

The dataset is not committed. Download MVTec AD and extract it so that the layout is
`data/mvtec/<category>/{train,test,ground_truth}/...`.

## Reproduce the results

```bash
python -m src.data.sanity --output results/data_sanity.md
python -m src.train --config configs/bottle.yaml --model mahalanobis --seed 0
python -m src.train --config configs/bottle.yaml --model autoencoder --seed 0
python -m src.train --config configs/bottle.yaml --model patchcore --seed 0
python -m src.coreset_sweep --config configs/bottle.yaml --seed 0
python -m src.run_eval --config configs/bottle.yaml
python -m src.benchmark --config configs/bottle.yaml --seed 0
python -m src.readme
```

Repeat the three `train` commands for each seed listed in the results table before
running `src.run_eval`. The evaluation refuses to overwrite `results/metrics.md`,
because the test split is meant to be opened once.

## API

```bash
make serve    # or: python -m uvicorn src.api.main:app --port 8000
```

The model is loaded once at startup from the bundle named by `MODEL_ARTIFACT`.

```bash
curl -F "file=@image.png;type=image/png" http://localhost:8000/predict
curl http://localhost:8000/health
curl http://localhost:8000/info
```

`POST /predict` returns:

```json
{
  "score": "<float>",
  "threshold": "<float>",
  "is_anomalous": "<bool>",
  "heatmap_png_base64": "<base64 PNG: anomaly map overlaid on the preprocessed input>",
  "latency_ms": "<float>",
  "model_version": "<category>_<model>_seed<seed>"
}
```

Errors: `413` for a file over the size limit, `422` for an unsupported type or an
undecodable image, `503` when the model artifact is not loaded. Malformed input never
produces a `500`.

## Docker

```bash
docker build -t mvtec-ad:latest .
docker run --rm -p 8000:8000 mvtec-ad:latest
```

The image is multi-stage and CPU-only. The backbone weights and the single deployed
bundle (`artifacts/bottle_patchcore_seed0.joblib`) are baked in, so the container needs
no network at run time.

## Limitations

- One category. Behaviour on textures and on objects with large natural variation is
  unmeasured.
- Latency was measured in-process on a development CPU. It excludes network time and
  will differ on other hardware.
- The calibrated false positive rate describes the validation set. The rate observed on
  the test split is in the metrics table, and it need not match the target.
- Any change to the models, preprocessing or threshold after the evaluation would
  invalidate the reported test numbers.
