# Every target must work from a clean checkout after `make setup`.
# Development happens on Windows; the deployment target is a Linux container.

PYTHON ?= python
PIP := $(PYTHON) -m pip

CONFIG ?= configs/bottle.yaml
SEED ?= 0

# torch is always installed from the CPU wheel index. The default PyPI wheels
# bundle CUDA on Linux, which alone would exceed the 2 GB image budget.
TORCH_CPU_INDEX := https://download.pytorch.org/whl/cpu

IMAGE := mvtec-ad:latest

.PHONY: setup lint format test train eval bench serve docker-build docker-run

setup:
	$(PIP) install --upgrade pip
	$(PIP) install --index-url $(TORCH_CPU_INDEX) torch torchvision
	$(PIP) install -e ".[dev]"

lint:
	$(PYTHON) -m ruff format --check .
	$(PYTHON) -m ruff check .

format:
	$(PYTHON) -m ruff format .
	$(PYTHON) -m ruff check --fix .

test:
	$(PYTHON) -m pytest

train:
	$(PYTHON) -m src.train --config $(CONFIG) --seed $(SEED)

eval:
	$(PYTHON) -m src.evaluate --config $(CONFIG) --seed $(SEED)

bench:
	$(PYTHON) -m src.benchmark --config $(CONFIG) --seed $(SEED)

serve:
	$(PYTHON) -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000

docker-build:
	docker build -t $(IMAGE) .

docker-run:
	docker run --rm -p 8000:8000 $(IMAGE)
