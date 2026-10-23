# Multi-stage, CPU-only image serving the deployed bundle.
#
# The builder installs dependencies and downloads the backbone weights; the final
# stage copies only the finished virtualenv, the weights cache and the one bundle, so
# no compiler, pip cache or build tooling ships.

FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# torch comes from the CPU wheel index: the default Linux wheels bundle CUDA.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision

WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

# Inference needs neither C++ headers nor the packages' own test suites.
RUN cd /opt/venv/lib/python3.12/site-packages \
    && rm -rf torch/include torch/test \
    && find . -type d -name tests -prune -exec rm -rf {} +

# Bake the pretrained backbone into the image so the container never needs the network.
RUN python -c "import timm; timm.create_model('wide_resnet50_2', pretrained=True, features_only=True)"


FROM python:3.12-slim

ENV PATH="/opt/venv/bin:$PATH" \
    HF_HOME=/opt/hf \
    HF_HUB_OFFLINE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_ARTIFACT=/app/artifacts/bottle_patchcore_seed0.joblib

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/hf /opt/hf

RUN useradd --create-home --uid 1000 app
WORKDIR /app
COPY --chown=app:app artifacts/bottle_patchcore_seed0.joblib /app/artifacts/bottle_patchcore_seed0.joblib
USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

# One worker: the model lives in this process, and the spec rules out multiprocessing.
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
