"""Localhost HTTP interface. Start using scripts/start-api.ps1 or the CLI."""

from functools import lru_cache
import os
from typing import Annotated

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from . import __version__
from .catalog import CATALOG
from .engine import Detector, MAX_BATCH_CHARS, MAX_TEXT_CHARS, ModelUnavailable

app = FastAPI(title="Local Chinese AIGC detectors", version=__version__,
              description="Offline raw model scores and configurable passage annotations. Screening labels are uncalibrated.")


@lru_cache(maxsize=1)
def get_detector():
    return Detector()


Text = Annotated[str, Field(min_length=1, max_length=MAX_TEXT_CHARS, strict=True)]


class Options(BaseModel):
    model_config = ConfigDict(extra="forbid")
    models: list[str] | None = Field(default=None, min_length=1, max_length=len(CATALOG))
    overlap_tokens: int = Field(default=64, ge=0, le=256, strict=True)
    include_chunks: bool = True


class DetectRequest(Options):
    text: Text

    @field_validator("text")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("Text must not be blank")
        return value


class BatchRequest(Options):
    texts: list[Text] = Field(min_length=1, max_length=32)

    @field_validator("texts")
    @classmethod
    def validate_batch(cls, values):
        if any(not value.strip() for value in values):
            raise ValueError("Texts must not be blank")
        if sum(map(len, values)) > MAX_BATCH_CHARS:
            raise ValueError(f"Batch exceeds {MAX_BATCH_CHARS} characters")
        return values


class AnnotationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: Text
    models: list[str] | None = Field(default=None, min_length=1, max_length=len(CATALOG))
    segment_chars: int = Field(default=400, ge=128, le=2000, strict=True)
    min_segment_chars: int = Field(default=120, ge=1, le=2000, strict=True)
    high_threshold: float | None = Field(default=None, ge=0, le=1, strict=True)
    possible_threshold: float | None = Field(default=None, ge=0, le=1, strict=True)
    policy: str = "thesis-cnki-20260929-v1"
    include_text: bool = True


@app.get("/health")
def health():
    detector = get_detector()
    models = detector.list_models()
    return {"status": "ok" if all(model["installed"] for model in models) else "models_missing",
            "service": "local-aigc-detectors", "version": __version__, "offline": True,
            "device": detector.device, "process_id": os.getpid(),
            "installed_models": [model["id"] for model in models if model["installed"]]}


@app.get("/v1/models")
def list_models():
    return {"models": get_detector().list_models()}


@app.post("/v1/detect")
def detect(request: DetectRequest):
    try:
        return get_detector().detect(**request.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ModelUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/v1/detect-batch")
def detect_batch(request: BatchRequest):
    try:
        return {"items": get_detector().detect_batch(**request.model_dump())}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ModelUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/v1/annotate")
def annotate(request: AnnotationRequest):
    try:
        return get_detector().annotate(**request.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ModelUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
