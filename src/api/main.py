"""REST service for the trained anomaly model (AGENTS.md section 13).

The model is loaded once at application startup. Preprocessing parameters come from
the artifact bundle; nothing about resizing or normalization is hardcoded here.
"""

from __future__ import annotations

import io
import logging
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from src.api.heatmap import render_heatmap_png_base64
from src.bundle import load_bundle
from src.config import PreprocessingConfig
from src.data.mvtec import preprocess_image
from src.models.loading import model_from_bundle

LOGGER = logging.getLogger(__name__)

#: Environment variable naming the bundle to serve.
ARTIFACT_ENV = "MODEL_ARTIFACT"
DEFAULT_ARTIFACT = "artifacts/bottle_patchcore_seed0.joblib"

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ACCEPTED_CONTENT_TYPES = {"image/png": "PNG", "image/jpeg": "JPEG"}

#: Bundle entries that are not metadata: bulky, and not JSON-serializable.
NON_METADATA_KEYS = ("memory_bank", "model_state")


class PredictResponse(BaseModel):
    score: float
    threshold: float
    is_anomalous: bool
    heatmap_png_base64: str
    latency_ms: float
    model_version: str


@dataclass
class LoadedModel:
    bundle: dict[str, Any]
    model: Any
    preprocessing: PreprocessingConfig

    @property
    def version(self) -> str:
        bundle = self.bundle
        return f"{bundle['category']}_{bundle['model_name']}_seed{bundle['seed']}"


def _load(path: Path) -> LoadedModel | None:
    if not path.is_file():
        LOGGER.error("model artifact not found at %s", path)
        return None
    try:
        bundle = load_bundle(path)
        preprocessing = bundle["preprocessing"]
        return LoadedModel(
            bundle=bundle,
            model=model_from_bundle(bundle),
            preprocessing=PreprocessingConfig(
                resize=preprocessing["resize"],
                center_crop=preprocessing["center_crop"],
                mean=tuple(preprocessing["mean"]),
                std=tuple(preprocessing["std"]),
            ),
        )
    except Exception:
        LOGGER.exception("failed to load model artifact %s", path)
        return None


def create_app(artifact_path: str | Path | None = None) -> FastAPI:
    """Build the app. The artifact defaults to `$MODEL_ARTIFACT`, then the bottle PatchCore."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        path = Path(artifact_path or os.environ.get(ARTIFACT_ENV, DEFAULT_ARTIFACT))
        app.state.loaded = _load(path)
        yield

    app = FastAPI(title="MVTec AD anomaly detection", lifespan=lifespan)
    app.state.loaded = None

    def require_model() -> LoadedModel:
        loaded: LoadedModel | None = app.state.loaded
        if loaded is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        return loaded

    @app.get("/health")
    def health() -> JSONResponse:
        loaded: LoadedModel | None = app.state.loaded
        if loaded is None:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "unavailable",
                    "model_loaded": False,
                    "category": None,
                    "model_version": None,
                },
            )
        return JSONResponse(
            content={
                "status": "ok",
                "model_loaded": True,
                "category": loaded.bundle["category"],
                "model_version": loaded.version,
            }
        )

    @app.get("/info")
    def info() -> dict[str, Any]:
        bundle = require_model().bundle
        metadata = {key: value for key, value in bundle.items() if key not in NON_METADATA_KEYS}
        return {key: _jsonable(value) for key, value in metadata.items()}

    @app.post("/predict", response_model=PredictResponse)
    def predict(file: UploadFile) -> PredictResponse:
        loaded = require_model()

        expected_format = ACCEPTED_CONTENT_TYPES.get(file.content_type or "")
        if expected_format is None:
            raise HTTPException(
                status_code=422,
                detail=f"Unsupported content type '{file.content_type}'; "
                "send image/png or image/jpeg",
            )

        data = file.file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="File too large")

        start = time.perf_counter()
        image = _decode(data, expected_format)
        tensor = preprocess_image(image, loaded.preprocessing)

        scores, maps = loaded.model.predict(tensor[None])
        score = float(scores[0])
        threshold = float(loaded.bundle["threshold"])
        heatmap = render_heatmap_png_base64(
            tensor,
            maps[0],
            threshold,
            list(loaded.preprocessing.mean),
            list(loaded.preprocessing.std),
        )
        latency_ms = (time.perf_counter() - start) * 1000.0

        return PredictResponse(
            score=score,
            threshold=threshold,
            is_anomalous=score >= threshold,
            heatmap_png_base64=heatmap,
            latency_ms=latency_ms,
            model_version=loaded.version,
        )

    return app


def _decode(data: bytes, expected_format: str) -> Image.Image:
    """Decode an upload, turning every way it can be malformed into a 422."""
    try:
        image = Image.open(io.BytesIO(data))
        if image.format != expected_format:
            raise HTTPException(
                status_code=422,
                detail=f"Content does not match the declared type: found {image.format}",
            )
        image.load()
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as error:
        raise HTTPException(status_code=422, detail=f"Cannot decode image: {error}") from error
    return image


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item") and getattr(value, "ndim", None) == 0:
        return value.item()
    return value


app = create_app()
