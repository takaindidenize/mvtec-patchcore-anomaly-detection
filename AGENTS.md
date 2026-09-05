# AGENTS.md — Implementation Contract

> **Read this file completely before writing any code.**
> This is a specification, not a suggestion. Where this file states a value, use that value.
> Where it does not, **ask the user** — do not choose for them.
> Companion document: `docs/PLAN.md` explains *why* these rules exist. This file states *what* to build.

---

## 1. Project context (assume no prior knowledge)

Build a system that detects manufacturing defects in product images and localizes the defective region.

**Key constraint that shapes everything:** the model is trained on **defect-free images only**. There are no labeled defect examples at training time. This is one-class (unsupervised) anomaly detection, **not** binary classification.

**Dataset:** MVTec AD. 15 categories of industrial objects and textures. Training split contains only `good` images. Test split contains `good` images plus several defect types, with pixel-level ground truth masks for defects.

**Deliverable:** a reproducible training pipeline, a comparative results table, and a REST API serving the trained model, all packaged in Docker.

**Audience:** engineering recruiters at manufacturing companies. Code quality, methodological rigor, and reproducibility matter as much as the score.

---

## 2. Non-negotiable rules

Violating any of these invalidates the project. If a task appears to require breaking one, stop and ask.

1. **The test split is opened exactly once**, after every design decision is frozen. Never tune hyperparameters, thresholds, backbones, or preprocessing against test metrics.
2. **Normalization and threshold statistics are computed from training data only.** Never fit a scaler, compute a mean, or calibrate a threshold using test images.
3. **No number in any markdown file is typed by hand.** Every metric in `results/` is written by a script. If a table needs updating, re-run the script.
4. **Never invent or estimate a metric value.** If a number is not yet measured, write `TBD`.
5. **The dataset is never committed.** `data/` is gitignored. Model artifacts in `artifacts/` are gitignored except the single deployed bundle.
6. **Do not add a dependency** that is not in Section 4 without asking first.
7. **Notebooks never contain production logic.** `notebooks/` may import from `src/`. `src/` never imports from `notebooks/`.
8. **All code, comments, docstrings, commit messages, and README content are in English.**
9. **Preprocessing is defined once**, stored in the artifact bundle, and read from there by the serving code. Serving code must not hardcode resize dimensions or normalization constants.
10. **Every random operation takes an explicit seed** passed from config. No implicit global randomness.

---

## 3. Environment

| Item | Value |
|---|---|
| Python | 3.12 (amended from 3.11 — see note) |
| Compute | CPU-only. No CUDA-specific code paths. |
| OS target | Linux container; development may be on Windows |
| Package manager | `pip` with `pyproject.toml` |
| Formatter / linter | `ruff` (format + check) |
| Test runner | `pytest` |

The model must run on CPU within reasonable memory. Do not assume a GPU exists at any point.

> **Amendment (approved by the user):** Python is 3.12, not the 3.11 originally fixed here. 3.11 was not installed on the development machine and every dependency in Section 4 supports 3.12. The Dockerfile uses `python:3.12-slim` so development and production stay on the same interpreter. Recorded in `docs/DECISIONS.md`.

---

## 4. Allowed dependencies

**Runtime:** `torch` (CPU build), `torchvision`, `timm`, `numpy`, `scipy`, `scikit-learn`, `Pillow`, `pyyaml`, `joblib`, `fastapi`, `uvicorn[standard]`, `python-multipart`, `pydantic` (v2)

**Development:** `pytest`, `pytest-cov`, `ruff`, `tqdm`, `matplotlib` (figures only, never in the serving path)

**Optional, Phase 3 reference run only:** `anomalib`. This is used once to obtain a reference score. It must not be a runtime dependency of the final system.

Anything else requires the user's approval.

---

## 5. Data contract

Expected layout after the user downloads and extracts MVTec AD:

```
data/mvtec/
└── <category>/                  # e.g. bottle
    ├── train/
    │   └── good/                # defect-free images only
    ├── test/
    │   ├── good/
    │   └── <defect_type>/       # e.g. broken_large, contamination
    └── ground_truth/
        └── <defect_type>/       # PNG masks, one per test defect image
```

Rules:

- `test/good/` images have **no** mask file. Represent their mask as an all-zero array of the image's spatial size. Do not crash, do not skip them.
- A mask file matches its image by stem: `000.png` in `test/broken_large/` corresponds to `000_mask.png` in `ground_truth/broken_large/`. **Verify this pairing rule against the actual extracted data before relying on it** and report what you find.
- Label convention: `0` = normal (`good`), `1` = anomalous.
- If any image and its mask differ in spatial size, resize the **mask** with nearest-neighbor interpolation. Never resize the mask with bilinear interpolation — that produces non-binary values.

---

## 6. Splits

MVTec ships a fixed train/test split. Do not re-split it. One additional split is required:

- **`train_fit`** — 80% of `train/good`. Used to build the model.
- **`val_normal`** — 20% of `train/good`. Used **only** for threshold calibration (Section 9). Contains no anomalies.
- **`test`** — the untouched MVTec test split. Used once, for final metrics.

The split of `train/good` is deterministic given `seed`. A unit test must assert that `train_fit` and `val_normal` file path sets are disjoint.

---

## 7. Preprocessing

Fixed values. Do not change without asking.

| Parameter | Value |
|---|---|
| Resize (shorter side) | 256 |
| Center crop | 224 × 224 |
| Interpolation | bilinear for images, nearest for masks |
| Channel order | RGB |
| Scale | `[0, 1]` then normalize |
| Normalization mean | `[0.485, 0.456, 0.406]` |
| Normalization std | `[0.229, 0.224, 0.225]` |

No data augmentation. The model must learn the exact distribution of normal images; augmentation widens it and hurts one-class detection.

Masks receive resize and center crop only — no normalization.

---

## 8. Model interface

Every model implements this protocol. Define it in `src/models/base.py`.

```python
from typing import Protocol
import numpy as np
import torch


class AnomalyModel(Protocol):
    name: str

    def fit(self, loader: "torch.utils.data.DataLoader") -> None:
        """Build the model from defect-free images only."""

    def predict(self, images: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """
        Args:
            images: (B, 3, 224, 224) float32, already preprocessed.
        Returns:
            image_scores: (B,) float32. Higher = more anomalous. Unbounded.
            anomaly_maps: (B, 224, 224) float32, same scale as image_scores.
        """

    def save(self, path: str) -> None: ...

    @classmethod
    def load(cls, path: str) -> "AnomalyModel": ...
```

Three implementations are required, in this order:

| File | Name | Notes |
|---|---|---|
| `src/models/baseline_mahalanobis.py` | `mahalanobis` | Global average-pooled backbone features; Mahalanobis distance to the training distribution. Image-level score only — return a uniform anomaly map and document this limitation. |
| `src/models/autoencoder.py` | `autoencoder` | Convolutional autoencoder trained on `train_fit`. Anomaly map = per-pixel reconstruction error. |
| `src/models/patchcore.py` | `patchcore` | See Section 8.1. |

### 8.1 PatchCore specification

| Parameter | Value |
|---|---|
| Backbone | `wide_resnet50_2`, pretrained, `features_only=True` |
| Feature layers | The two layers at stride 8 and stride 16 (ResNet `layer2` and `layer3`) |
| Layer index resolution | **Do not hardcode `out_indices` from memory.** Instantiate the model, run one dummy forward pass, print the output shapes, and select the indices matching stride 8 and 16. Log the resolved indices. |
| Feature fusion | Bilinearly upsample the deeper map to the shallower map's spatial size, concatenate along channels |
| Local aggregation | `avg_pool2d(kernel_size=3, stride=1, padding=1)` |
| Coreset selection | Greedy k-center, `coreset_ratio` from config (default `0.01`) |
| Nearest neighbour | `sklearn.neighbors.NearestNeighbors`, `n_neighbors=1`, Euclidean |
| Map upsampling | Bilinear to 224×224 |
| Map smoothing | Gaussian, `sigma=4` |
| Image score | Maximum value of the smoothed anomaly map |

The backbone is used in inference mode only. No gradients, no fine-tuning.

Efficiency requirement: each batch is embedded **once** during `fit`. Do not call the embedding function twice on the same batch.

---

## 9. Threshold calibration

This is the part most implementations get wrong. Read carefully.

The training data contains **no anomalies**, so a threshold cannot be chosen by maximizing F1 on labeled data without touching the test set. The correct procedure:

1. Score every image in `val_normal` (held-out normal images the model never saw during `fit`).
2. Set `threshold = numpy.quantile(val_scores, 1 - target_fpr)`, with `target_fpr` from config (default `0.01`).
3. Interpretation: on normal parts, this configuration raises a false alarm roughly 1% of the time.
4. Store the threshold and the `target_fpr` used in the artifact bundle.

Reporting rules:

- `image_auroc`, `pixel_auroc`, `au_pro` are threshold-free and are the primary comparison metrics.
- Operating-point metrics (recall and false positive rate at the calibrated threshold) are reported **using the calibrated threshold only**.
- `f1_max` over the test set may be reported, but must be labeled explicitly as an **oracle upper bound**, because selecting it uses test labels. Never present it as achieved performance.

---

## 10. Metrics

Implemented in `src/evaluate.py`.

| Key | Definition | Level |
|---|---|---|
| `image_auroc` | ROC AUC of image scores vs. binary image labels | image |
| `pixel_auroc` | ROC AUC of flattened anomaly maps vs. flattened binary masks | pixel |
| `au_pro` | Area under the per-region-overlap curve, integrated up to FPR 0.3 | pixel |
| `f1_max` | Best F1 over all thresholds — **oracle, label as such** | image |
| `recall_at_threshold` | Recall at the calibrated threshold | image |
| `fpr_at_threshold` | False positive rate at the calibrated threshold | image |
| `latency_p50_ms`, `latency_p95_ms` | Single-image end-to-end inference latency | system |
| `memory_bank_mb` | Size of the stored memory bank | system |

`accuracy` is **not** reported. It is threshold-dependent and misleading under class imbalance.

Every model is run with `seeds: [0, 1, 2]`. Report mean ± standard deviation. A single-run number is not acceptable.

---

## 11. Configuration schema

`configs/<category>.yaml`. This is the complete set of keys; add none silently.

```yaml
category: bottle
data_root: data/mvtec
seed: 0
validation_ratio: 0.2
target_fpr: 0.01

preprocessing:
  resize: 256
  center_crop: 224
  mean: [0.485, 0.456, 0.406]
  std: [0.229, 0.224, 0.225]

model:
  name: patchcore          # mahalanobis | autoencoder | patchcore
  backbone: wide_resnet50_2
  coreset_ratio: 0.01
  gaussian_sigma: 4

training:
  batch_size: 8
  num_workers: 0           # 0 for Windows compatibility

output:
  artifacts_dir: artifacts
  results_dir: results
```

---

## 12. Artifact bundle

One file: `artifacts/<category>_<model_name>_seed<seed>.joblib`. Serving loads this and nothing else.

```python
{
    "schema_version": 1,
    "category": str,
    "model_name": str,
    "backbone": str,
    "resolved_out_indices": list[int],
    "preprocessing": dict,          # verbatim copy of the config block
    "feature_map_size": tuple[int, int],
    "memory_bank": np.ndarray,      # float32, PatchCore only
    "coreset_ratio": float,
    "gaussian_sigma": int,
    "threshold": float,
    "target_fpr": float,
    "seed": int,
    "created_at": str,              # ISO 8601
    "git_commit": str,
}
```

The serving code reads preprocessing parameters from this dict. If it hardcodes them anywhere, the implementation is wrong.

---

## 13. API contract

FastAPI application in `src/api/main.py`. The model is loaded **once at application startup**, never per request.

### `POST /predict`

Request: `multipart/form-data`, field name `file`. Accepted types: `image/png`, `image/jpeg`. Maximum size: 10 MB.

Response `200`:

```json
{
  "score": 3.417,
  "threshold": 2.884,
  "is_anomalous": true,
  "heatmap_png_base64": "iVBORw0KGgo...",
  "latency_ms": 62.4,
  "model_version": "bottle_patchcore_seed0"
}
```

`heatmap_png_base64` is a 224×224 PNG: the anomaly map rendered as a colormap overlaid on the preprocessed input.

Error responses:

| Status | Condition | Body |
|---|---|---|
| `413` | File exceeds 10 MB | `{"detail": "File too large"}` |
| `422` | Unsupported type or undecodable image | `{"detail": "<reason>"}` |
| `503` | Model artifact not loaded | `{"detail": "Model not loaded"}` |

Never return `500` for malformed user input.

### `GET /health`

```json
{"status": "ok", "model_loaded": true, "category": "bottle", "model_version": "bottle_patchcore_seed0"}
```

Returns `503` with `"model_loaded": false` when the artifact is missing.

### `GET /info`

Returns the artifact bundle metadata excluding `memory_bank`.

---

## 14. Commands

Every command in the `Makefile` must work from a clean checkout after `make setup`.

```
make setup          # install dependencies
make lint           # ruff format --check && ruff check
make test           # pytest
make train CONFIG=configs/bottle.yaml
make eval CONFIG=configs/bottle.yaml     # writes results/metrics.md
make bench CONFIG=configs/bottle.yaml    # writes results/latency.md
make serve          # uvicorn, local
make docker-build
make docker-run
```

---

## 15. Task sequence

Complete tasks in order. Each has a definition of done that must be verifiable by running a command. Do not begin a task before its predecessor's DoD passes.

| # | Task | Definition of done |
|---|---|---|
| T0 | Repo scaffold, `pyproject.toml`, `Makefile`, `ruff` config, CI workflow | `make lint` and `make test` both pass on an empty test suite |
| T1 | `src/data/mvtec.py`: dataset, splits, dataloaders | Test asserts correct image counts, tensor shapes `(3,224,224)`, mask shapes `(224,224)`, binary mask values, and disjoint `train_fit` / `val_normal` path sets |
| T2 | Data sanity report | A script prints per-category image counts, defect types, image/mask size mismatches, and any broken image–mask pairing. Findings reported to the user before proceeding. |
| T3 | `src/evaluate.py`: all metrics from Section 10, and `src/calibrate.py`: threshold calibration from `val_normal` (Section 9) | Test feeds random scores and asserts `image_auroc` ≈ 0.5 ± 0.05; feeds perfect scores and asserts `image_auroc` == 1.0; a test asserts the calibrated threshold equals `numpy.quantile(val_scores, 1 - target_fpr)` |
| T4 | Mahalanobis baseline | `make train` writes `artifacts/<category>_mahalanobis_seed<seed>.joblib` containing every key in Section 12. **No test image is read.** |
| T5 | Autoencoder baseline | Same, for `autoencoder` |
| T6 | PatchCore | Same, for `patchcore`; a test asserts identical output for two runs with the same seed |
| T7 | Coreset ratio sweep: `0.1`, `0.01`, `0.001` | `results/coreset.md` tabulates `memory_bank_mb` and fit/query latency per ratio; the chosen ratio is justified **without reference to test data** |
| T8 | Design freeze, then the single unlock evaluation | `make eval` is run exactly once and writes every row (3 models × 3 seeds, mean ± std) to `results/metrics.md` |
| T9 | FastAPI service | `make test` covers all endpoints and all error statuses via `TestClient` |
| T10 | Latency benchmark | `results/latency.md` contains p50 and p95 over at least 100 single-image requests |
| T11 | Dockerfile, multi-stage, CPU-only torch | `make docker-run` serves a working `/health`; final image under 2 GB |
| T12 | English `README.md` with the real results table | Every number in the README is copied from `results/`, none typed by hand |

> **Amendment (approved by the user):** T3–T8 were rewritten. As originally written, T4–T7 required `make eval` to write test metrics before T8, which Section 16 forbids and which breaks Rule 1 ("the test split is opened exactly once, after every design decision is frozen"). Threshold calibration touches only `val_normal`, so it moved into T3; T4–T7 now build models and artifact bundles without reading a single test image; and T8 is the one and only evaluation run, which emits every row of `results/metrics.md` at once. Recorded in `docs/DECISIONS.md`.

---

## 16. Prohibited actions

Stop and ask the user rather than doing any of these:

- Computing test metrics before T8 is complete
- Changing any value fixed in Sections 3, 4, 7, 8.1, or 11
- Adding a dependency outside Section 4
- Writing a number into markdown that no script produced
- Committing anything under `data/`
- Introducing multiprocessing, distributed training, or GPU-specific code
- Refactoring across more than one module in a single commit
- Marking a task done when its DoD command fails
- Replacing a failing test with a weaker assertion to make it pass
- Deleting or rewriting `docs/DECISIONS.md` entries

---

## 17. Working protocol

- **When the spec is silent, ask.** Do not choose a default and proceed. State the options and their trade-offs, then wait.
- **When reality contradicts the spec** — e.g. the mask naming rule in Section 5 does not match the extracted data — report the discrepancy and propose a fix. Do not silently work around it.
- **One task per commit.** Commit message format: `T<number>: <imperative summary>`, e.g. `T6: implement PatchCore with greedy coreset selection`.
- **After each task**, append one line to `docs/DECISIONS.md`: what was decided, and why. Keep it to a single sentence.
- **Report measured numbers, never expected ones.** If a metric is lower than hoped, report it as measured and say so.

---

## 18. Glossary

| Term | Meaning |
|---|---|
| One-class anomaly detection | Training on normal examples only; scoring by deviation from the learned normal distribution |
| Data leakage | Information from the evaluation set influencing training or model selection, producing inflated metrics |
| Memory bank | The stored collection of patch embeddings from normal training images, queried at inference |
| Coreset | A small subset chosen to preserve the coverage of a larger set, reducing memory and query cost |
| Anomaly map | Per-pixel anomaly score array, same spatial size as the input |
| Operating point | A specific threshold, and the recall / false-positive rate it produces |
| Train/serve skew | A mismatch between preprocessing at training time and at serving time, causing silent degradation |
| AU-PRO | Area under the per-region-overlap curve; a segmentation metric that prevents large defects from dominating small ones |
