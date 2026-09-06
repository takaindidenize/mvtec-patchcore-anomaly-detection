# Decisions

One line per task: what was decided, and why.

- **Setup:** The project lives outside OneDrive at `C:\dev\mvtec-ad`, because syncing a 4.9 GB dataset of thousands of small image files slows I/O and can lock files during training.
- **Setup:** Python is pinned to 3.12 rather than the 3.11 fixed in AGENTS.md section 3, because 3.11 was not installed and every dependency in section 4 supports 3.12; the Dockerfile will use `python:3.12-slim` so development and production stay on the same interpreter (approved by the user).
- **Setup:** Threshold calibration moves ahead of T4 and every test metric is produced by a single unlock run after T7, because the T4-T7 definitions of done required test metrics that section 16 and rule 1 forbid before the design is frozen (approved by the user).
- **Setup:** `bottle` is the only category until the full pipeline works end to end, because each additional category costs nine CPU runs and adds no new code path (approved by the user).
- **T0:** torch and torchvision are installed from the PyTorch CPU wheel index in both the Makefile and CI, because the default PyPI wheels bundle CUDA on Linux and would alone exceed the 2 GB image budget of T11.
- **T0:** Ruff is restricted to Python files via `include`, because it otherwise reformats the Python snippets embedded in `AGENTS.md`, and the contract document must not be rewritten by a formatter.
- **T1:** The split is a pure function over a list of paths, tested without the dataset, while the dataset-backed tests carry a skip marker, because CI never has the data yet determinism and disjointness still have to be proven there; the seed sweep of section 10 is driven by a `load_config(..., seed=)` override rather than a new config key, since the section 11 schema is declared complete.
- **T2:** The audit checks mask binarity and image/mask size agreement on top of the counts and pairing that section 5 asks for, because a non-binary or misaligned mask would corrupt every pixel metric without raising an error; all 15 categories came back clean, so the pairing rule the dataset code relies on is now verified rather than assumed.
- **T3:** AU-PRO draws its thresholds from the clean-pixel quantiles *and* a linear span of the score range, then keeps the best overlap at each false positive rate, because a perfect predictor collapses the clean scores onto a single value and a quantile-only grid would score that case zero instead of one; threshold calibration ships alongside the metrics since it reads only `val_normal` and must exist before any model is fitted.
