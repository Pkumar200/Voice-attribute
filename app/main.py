from __future__ import annotations

import logging
import os
import time
import uuid

from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from app import audio_utils, inference
from app.logging_conf import configure_logging
from app.schemas import AgeBracketScore, AnalyzeResponse, ErrorResponse, GenderScore, LanguageScore

configure_logging(os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger("voice_attr.api")

MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", 15 * 1024 * 1024))  # 15MB safety cap


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the model at startup rather than on the first request so the first
    real call doesn't eat a multi-second cold-start penalty. Failures here
    are logged but non-fatal - the engine will retry / fall back lazily on
    first use (see inference.InferenceEngine).
    """
    t0 = time.perf_counter()
    try:
        inference.engine.ensure_loaded()
        logger.info(
            "model warm-up complete",
            extra={"backend": inference.engine.backend_name,
                   "warmup_ms": round((time.perf_counter() - t0) * 1000, 1)},
        )
    except Exception:
        logger.exception("model warm-up failed; will retry lazily on first request")

    try:
        inference.engine.ensure_language_loaded()
        logger.info(
            "language-id warm-up complete",
            extra={"backend": inference.engine.language_backend_name or "unavailable"},
        )
    except Exception:
        logger.exception("language-id warm-up failed; 'language' will be omitted")
    yield


app = FastAPI(
    title="Voice Attribute Inference Service",
    version="1.0.0",
    description=(
        "Infers caller gender and age bracket from a short audio clip. "
        "No audio is persisted; everything is processed in memory for the "
        "life of a single request."
    ),
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "backend": inference.engine.backend_name if inference.engine.is_loaded else "loading",
        "language_backend": inference.engine.language_backend_name or "unavailable",
    }


def _run_pipeline(samples, sr, contact_id: str) -> AnalyzeResponse:
    """Shared quality-check + inference path used by both REST and WS."""
    warnings: list[str] = []
    quality = audio_utils.assess_quality(samples, sr)
    warnings.extend(quality.reasons)

    if quality.quality == "insufficient":
        return AnalyzeResponse(
            contact_id=contact_id,
            gender=GenderScore(prediction="unknown", confidence=0.0),
            age_bracket=AgeBracketScore(prediction="unknown", confidence=0.0),
            processing_ms=0,  # filled in by caller
            audio_quality="insufficient",
            warnings=warnings or ["audio quality too low for a reliable estimate"],
        )

    result = inference.engine.predict(samples, sr)
    lang_result = inference.engine.predict_language(samples, sr)

    # if the signal is merely "degraded", damp confidence rather than
    # silently reporting model-native confidence as if nothing were wrong
    gender_conf = result.gender.confidence
    age_conf = result.age.confidence
    lang_conf = lang_result.confidence if lang_result else 0.0
    if quality.quality == "degraded":
        gender_conf = round(gender_conf * 0.75, 2)
        age_conf = round(age_conf * 0.75, 2)
        lang_conf = round(lang_conf * 0.75, 2)

    return AnalyzeResponse(
        contact_id=contact_id,
        gender=GenderScore(prediction=result.gender.prediction, confidence=gender_conf),
        age_bracket=AgeBracketScore(prediction=result.age.prediction, confidence=age_conf),
        language=LanguageScore(prediction=lang_result.prediction, confidence=lang_conf)
        if lang_result else None,
        processing_ms=0,
        audio_quality=quality.quality,
        warnings=warnings,
    )


@app.post("/analyze", response_model=AnalyzeResponse, responses={422: {"model": ErrorResponse}})
async def analyze(file: UploadFile = File(...), contact_id: str | None = None):
    """
    Accepts a multipart audio upload (wav/mp3/ogg/webm/opus/etc - anything
    ffmpeg understands) and returns gender + age-bracket estimates.
    """
    req_id = contact_id or str(uuid.uuid4())
    t0 = time.perf_counter()

    raw = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="audio payload too large")

    try:
        decoded = await audio_utils.decode_to_pcm(raw)
    except audio_utils.AudioDecodeError as e:
        logger.warning("audio decode failed", extra={"contact_id": req_id, "error": str(e)})
        return JSONResponse(
            status_code=422,
            content=ErrorResponse(
                contact_id=req_id, error="audio_decode_failed", detail=str(e)
            ).model_dump(),
        )
    finally:
        # raw bytes are not referenced anywhere else and go out of scope
        # here; nothing was written to disk at any point in this request.
        del raw

    try:
        response = _run_pipeline(decoded.samples, decoded.sample_rate, req_id)
    except Exception:
        logger.exception("inference failed", extra={"contact_id": req_id})
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                contact_id=req_id, error="inference_failed",
                detail="internal error during attribute inference",
            ).model_dump(),
        )

    response.processing_ms = int((time.perf_counter() - t0) * 1000)

    logger.info(
        "analyze complete",
        extra={
            "contact_id": req_id,
            "processing_ms": response.processing_ms,
            "audio_quality": response.audio_quality,
            "gender": response.gender.prediction,
            "age_bracket": response.age_bracket.prediction,
            "duration_s": round(decoded.duration_s, 2),
            "backend": inference.engine.backend_name,
        },
    )
    return response


@app.websocket("/ws/analyze")
async def analyze_stream(ws: WebSocket):
    """
    Real-time streaming variant (bonus task).

    Protocol:
      * Client connects, optionally sends a JSON text frame first:
            {"contact_id": "..."}
      * Client then sends binary frames of raw audio bytes (any codec
        ffmpeg can decode per-chunk is NOT assumed here - browsers should
        send small WAV/PCM or Opus frames; each frame is decoded
        independently, then samples are accumulated in memory).
      * Every time ~1.5s of new audio has accumulated, the server emits a
        progressive AnalyzeResponse JSON frame computed over *all* audio
        seen so far in this connection (more context -> better estimate).
      * Client sends a text frame "end" (or disconnects) to stop.

    Nothing received over this socket is ever written to disk; the sample
    buffer lives only for the lifetime of the connection and is discarded
    on close.
    """
    await ws.accept()
    contact_id = str(uuid.uuid4())
    buffer_samples: list = []
    total_new_samples = 0
    EMIT_EVERY_S = 1.5
    sr = audio_utils.TARGET_SR
    emit_threshold = int(EMIT_EVERY_S * sr)

    import numpy as np

    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break

            if "text" in msg and msg["text"] is not None:
                text = msg["text"]
                if text.strip().lower() == "end":
                    break
                try:
                    import json
                    payload = json.loads(text)
                    contact_id = payload.get("contact_id", contact_id)
                except Exception:
                    pass
                continue

            data = msg.get("bytes")
            if not data:
                continue

            t0 = time.perf_counter()
            try:
                decoded = await audio_utils.decode_to_pcm(data)
            except audio_utils.AudioDecodeError as e:
                await ws.send_json(
                    ErrorResponse(contact_id=contact_id, error="audio_decode_failed",
                                  detail=str(e)).model_dump()
                )
                continue
            finally:
                del data

            buffer_samples.append(decoded.samples)
            total_new_samples += len(decoded.samples)

            if total_new_samples >= emit_threshold:
                all_samples = np.concatenate(buffer_samples)
                try:
                    response = _run_pipeline(all_samples, sr, contact_id)
                except Exception:
                    logger.exception("stream inference failed", extra={"contact_id": contact_id})
                    response = None

                if response is not None:
                    response.processing_ms = int((time.perf_counter() - t0) * 1000)
                    await ws.send_json(response.model_dump())
                    logger.info(
                        "stream partial result",
                        extra={
                            "contact_id": contact_id,
                            "processing_ms": response.processing_ms,
                            "buffered_s": round(len(all_samples) / sr, 2),
                        },
                    )
                total_new_samples = 0

    except WebSocketDisconnect:
        pass
    finally:
        buffer_samples.clear()
        logger.info("stream connection closed", extra={"contact_id": contact_id})
