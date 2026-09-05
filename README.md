# MVTec AD — One-Class Anomaly Detection

Detects manufacturing defects in product images and localizes the defective region.
The model is trained on defect-free images only, so this is one-class (unsupervised)
anomaly detection, not binary classification.

The implementation contract is [AGENTS.md](AGENTS.md); per-task decisions are recorded in
[docs/DECISIONS.md](docs/DECISIONS.md).

## Status

Scaffold only. No model has been trained yet.

## Results

TBD — every number in this section is written by a script into `results/` and copied here.
Nothing is typed by hand.

## Requirements

- Python 3.12
- GNU Make (Windows: available through MSYS2 or `winget install ezwinports.make`)
- Docker, for the container targets

## Quick start

```bash
make setup
make lint
make test
```

The dataset is not committed. Download MVTec AD and extract it so that the layout is
`data/mvtec/<category>/{train,test,ground_truth}/...`.
