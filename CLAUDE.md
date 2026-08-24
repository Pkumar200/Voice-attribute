# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A FastAPI service that infers a caller's gender and age bracket from a short audio clip (`POST /analyze`, `WS /ws/analyze`, `GET /health`). No caller data is persisted anywhere — audio is piped through ffmpeg in-memory and never written to disk. See README.md for the full API contract and DESIGN.md for the design rationale.

## Commands

```bash
# local dev, no Docker (heuristic backend = fast, no model download)
MODEL_BACKEND=heuristic uvicorn app.main:app --reload

# exercise the real pretrained model
uvicorn app.main:app --reload

# docker
docker compose up --build

# tests (conftest.py forces heuristic backend so CI never downloads the model)
pytest tests/ -v
pytest tests/test_analyze.py::test_name -v   # single test

# accuracy/calibration harness (bonus, not part of CI)
python eval/eval_common_voice.py
```

Requires `ffmpeg` on PATH for local (non-Docker) runs. `requirements-dev.txt` adds `pytest`/`httpx` on top of `requirements.txt`.

## Architecture

Pipeline (see `README.md#architecture` for the diagram):

```
audio bytes -> ffmpeg decode (stdin/stdout pipes, no temp files) -> mono f32 PCM @16kHz
            -> quality gate (RMS/clipping/VAD heuristics, NumPy only)
            -> good/degraded -> wav2vec2 age/gender model -> response
            -> insufficient  -> short-circuit "unknown", skip inference entirely
```

- `app/audio_utils.py` — ffmpeg-based decode (bytes piped through subprocess stdin/stdout, 10s timeout) and the pre-inference quality gate (`assess_quality`) that classifies `good`/`degraded`/`insufficient` before paying for a model forward pass.
- `app/inference.py` — the `Predictor` protocol plus two implementations: the primary wav2vec2 model wrapper and `HeuristicPredictor` (pitch/energy only, zero downloads/deps beyond librosa). A module-level `engine` singleton (`InferenceEngine`) picks the backend per `MODEL_BACKEND` and loads it once at startup via the `lifespan` handler in `app/main.py`, not per-request. Swapping the model backend (e.g. for licensing reasons — see below) means adding one new `Predictor` class here, not touching the API layer.
- `app/main.py` — FastAPI app, async throughout (ffmpeg subprocess is awaited, never blocking the event loop). `_run_pipeline()` is the shared quality-check + inference logic used by both the REST endpoint and the WebSocket streaming endpoint — keep them sharing this function rather than duplicating pipeline logic. Structured JSON logs record `contact_id`, `processing_ms`, `audio_quality`, backend, and predictions — never raw audio or decoded samples.
- `app/schemas.py` — Pydantic request/response models (`AnalyzeResponse`, `GenderScore`, `AgeBracketScore`, `ErrorResponse`).
- `app/logging_conf.py` — structured JSON logging setup.

### Model backends (`MODEL_BACKEND` env var)

| value | behavior |
|---|---|
| `auto` (default) | try pretrained model; on load failure, log and fall back to heuristic |
| `model` | require the pretrained model; raise on load failure instead of degrading silently |
| `heuristic` | always use the zero-download pitch heuristic (what the test suite/CI uses) |

Primary model is `audeering/wav2vec2-large-robust-6-ft-age-gender` — **CC-BY-NC-SA-4.0, non-commercial only**; keep this in mind before suggesting it for anything production/commercial (README's "Model choice & licensing" section has the swap-out plan).

### Quality gate semantics (don't regress these)

- `insufficient` audio always returns `prediction: "unknown"` / `confidence: 0.0` — never dress up a guess as a real estimate.
- `degraded` audio still gets a best-effort prediction but confidence is damped by 25% (`_run_pipeline` in `app/main.py`), so downstream consumers can apply their own threshold.
- The model's "child" gender class is deliberately mapped to `"unknown"`, not forced into male/female.

### Privacy invariant

No caller audio or decoded PCM is ever written to disk or persisted beyond the lifetime of a single request/WS connection. Only the pretrained model's public weights are cached to disk (`hf-model-cache` Docker volume). Preserve this when touching `audio_utils.py` or the WS buffer handling in `main.py`.

### Tests

`tests/conftest.py` forces `MODEL_BACKEND=heuristic` and provides synthetic-audio fixtures so the suite never needs network access or a multi-hundred-MB checkpoint download. `tests/test_analyze.py` runs integration tests against a live FastAPI app instance (via httpx).
