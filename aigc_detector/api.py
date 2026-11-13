"""Localhost HTTP interface. Start using scripts/start-api.ps1 or the CLI."""

from functools import lru_cache
import os
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from . import __version__
from .catalog import CATALOG, CHINESE_MODELS, EXTRA_ENGLISH_MODELS, MODEL_GROUPS, SPECIALIZED_ENGLISH_MODELS
from .engine import Detector, MAX_BATCH_CHARS, MAX_TEXT_CHARS, ModelUnavailable

app = FastAPI(title="Local Chinese and English AIGC detectors", version=__version__,
              description="Offline raw model scores and configurable passage annotations. Screening labels are uncalibrated.")


@lru_cache(maxsize=1)
def get_detector():
    return Detector()


Text = Annotated[str, Field(min_length=1, max_length=MAX_TEXT_CHARS, strict=True)]


class Options(BaseModel):
    model_config = ConfigDict(extra="forbid")
    models: list[str] | None = Field(default=None, min_length=1, max_length=len(CATALOG))
    language: Literal["zh", "en"] = "zh"
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
    language: Literal["zh", "en"] = "zh"
    segment_chars: int | None = Field(default=None, ge=128, le=2000, strict=True)
    min_segment_chars: int | None = Field(default=None, ge=1, le=2000, strict=True)
    high_threshold: float | None = Field(default=None, ge=0, le=1, strict=True)
    possible_threshold: float | None = Field(default=None, ge=0, le=1, strict=True)
    policy: str | None = None
    include_text: bool = True


def annotation_profiles(detector):
    """Report annotation readiness separately from the raw detection groups."""
    from .fusion import load_profile, resolve_policy
    manifest = detector._manifest()
    result = {}
    for language in ("zh", "en"):
        policy = resolve_policy(None, language)
        try:
            profile, digest = load_profile(policy)
            required = dict(profile["revisions"])
            for assets in profile.get("statistical_assets", {}).values():
                required.update(assets)
            missing = [mid for mid, revision in required.items()
                       if manifest.get(mid, {}).get("revision") != revision or not detector._installed(mid, manifest)]
            result[language] = {"policy": policy, "profile_sha256": digest, "models": profile["models"],
                                "required_assets": list(required), "ready": not missing,
                                "missing_or_mismatched_assets": missing, "segment_chars": profile["segment_chars"],
                                "min_segment_chars": profile["min_segment_chars"]}
        except (OSError, ValueError) as error:
            result[language] = {"policy": policy, "ready": False, "error": str(error)}
    return result


@app.get("/health")
def health():
    detector = get_detector()
    models = detector.list_models()
    installed = {model["id"] for model in models if model["installed"]}
    return {"status": "ok" if set(CHINESE_MODELS) <= installed else "models_missing",
            "service": "local-aigc-detectors", "version": __version__, "offline": True,
            "device": detector.device, "process_id": os.getpid(),
            "installed_models": [model["id"] for model in models if model["installed"]],
            "annotation_profiles": annotation_profiles(detector),
            "model_groups": {language: {"models": list(ids), "ready": set(ids) <= installed}
                             for language, ids in MODEL_GROUPS.items()}}


@app.get("/v1/models")
def list_models():
    from .statistical import StatisticalDetector
    return {"models": get_detector().list_models(), "default_language": "zh", "model_groups": MODEL_GROUPS,
            "annotation_profiles": annotation_profiles(get_detector()),
            "extra_english_models": EXTRA_ENGLISH_MODELS,
            "specialized_english_models": SPECIALIZED_ENGLISH_MODELS,
            "statistical_backends": StatisticalDetector(device="cpu").list_backends()}


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
