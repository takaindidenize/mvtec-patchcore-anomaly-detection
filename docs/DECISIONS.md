# Decisions

One line per task: what was decided, and why.

- **Setup:** The project lives outside OneDrive at `C:\dev\mvtec-ad`, because syncing a 4.9 GB dataset of thousands of small image files slows I/O and can lock files during training.
- **Setup:** Python is pinned to 3.12 rather than the 3.11 fixed in AGENTS.md section 3, because 3.11 was not installed and every dependency in section 4 supports 3.12; the Dockerfile will use `python:3.12-slim` so development and production stay on the same interpreter (approved by the user).
- **Setup:** Threshold calibration moves ahead of T4 and every test metric is produced by a single unlock run after T7, because the T4-T7 definitions of done required test metrics that section 16 and rule 1 forbid before the design is frozen (approved by the user).
- **Setup:** `bottle` is the only category until the full pipeline works end to end, because each additional category costs nine CPU runs and adds no new code path (approved by the user).
- **T0:** torch and torchvision are installed from the PyTorch CPU wheel index in both the Makefile and CI, because the default PyPI wheels bundle CUDA on Linux and would alone exceed the 2 GB image budget of T11.
- **T0:** Ruff is restricted to Python files via `include`, because it otherwise reformats the Python snippets embedded in `AGENTS.md`, and the contract document must not be rewritten by a formatter.
