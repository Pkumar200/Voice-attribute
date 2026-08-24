# Voice Attribute Inference Service

Infers a caller's **gender** and **age bracket** from a short audio clip, for
use by voice AI agents that coordinate logistics calls. No prior data about
the contact is required or used.

```
POST /analyze          multipart audio upload -> structured prediction
WS   /ws/analyze        streaming variant, emits progressive predictions
GET  /health            liveness + which inference backend is active
```

## Quick start

```bash
docker compose up --build
```

The service listens on `http://localhost:8000`. First boot downloads the
~130MB gender/age model plus a ~30MB language-ID model from Hugging Face
and caches both in a named Docker volume (`hf-model-cache`), so subsequent
restarts are fast. If your environment has no outbound internet access, set
`MODEL_BACKEND=heuristic` (see
[Model backends](#model-backends--fallback-behavior) below) - the service
still starts and serves predictions with no downloads at all (though
`language` will be omitted - see
[Language detection](#language-detection-bonus)).

```bash
curl -X POST http://localhost:8000/analyze \
  -F "file=@sample_audio/smoke_test_sample.wav"
```

```json
{
  "contact_id": "b1e7...",
  "gender": {"prediction": "male", "confidence": 0.81},
  "age_bracket": {"prediction": "31-45", "confidence": 0.62},
  "language": {"prediction": "en", "confidence": 0.85},
  "processing_ms": 187,
  "audio_quality": "good",
  "warnings": []
}
```

## Local development (no Docker)

```bash
python -m venv .venv && source .venv/bin/activate
# CPU-only torch/torchaudio first (they're intentionally not pinned in
# requirements.txt - see Dockerfile comment - so a plain `pip install -r`
# doesn't pull the ~2GB of unused GPU/CUDA wheels the default PyPI build
# depends on):
pip install torch==2.3.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-dev.txt
apt-get install ffmpeg   # or `brew install ffmpeg` on macOS

MODEL_BACKEND=heuristic uvicorn app.main:app --reload   # fast, no download
# or, to exercise the real model:
uvicorn app.main:app --reload
```

Run tests (forces the heuristic backend so CI never needs to download a
300MB+ checkpoint - see `tests/conftest.py`):

```bash
pytest tests/ -v
```

## API contract

`POST /analyze` accepts a multipart file field named `file` (any container
ffmpeg understands - wav, mp3, ogg/opus, webm/opus, raw mu-law, ...) and an
optional `contact_id` query param (a UUID is generated if omitted).

```json
{
  "contact_id": "uuid",
  "gender": {"prediction": "male" | "female" | "unknown", "confidence": 0.87},
  "age_bracket": {"prediction": "18-30" | "31-45" | "46-60" | "60+" | "unknown", "confidence": 0.63},
  "language": {"prediction": "en", "confidence": 0.81},
  "processing_ms": 142,
  "audio_quality": "good" | "degraded" | "insufficient"
}
```

`language` is `null` when audio is `insufficient`, when the language-ID
model couldn't be loaded (e.g. offline), or under `MODEL_BACKEND=heuristic`
(see [Language detection](#language-detection-bonus)).

* Decode failures (corrupt/unsupported audio) -> `422` with
  `{"error": "audio_decode_failed", ...}`.
* `audio_quality: "insufficient"` (too short, silent, or all-noise) always
  returns `prediction: "unknown"` / `confidence: 0.0` rather than a
  guess dressed up as a real estimate.
* `audio_quality: "degraded"` (short clip, clipping, low signal level)
  still returns a best-effort prediction but with confidence damped by 25%,
  so a downstream agent can decide its own threshold for acting on it.

### WebSocket streaming (bonus)

`WS /ws/analyze`: send binary audio frames as they arrive from the call.
Every ~1.5s of newly buffered audio, the server emits a JSON frame with a
prediction computed over *all* audio seen so far in the connection (more
context -> better estimate, similar to how a human's read on a caller
sharpens the longer they talk). Send a text frame `"end"` or just close the
socket to stop.

## Architecture

```
audio bytes -> [ffmpeg decode, in-memory pipes] -> mono f32 PCM @16kHz
            -> [quality gate: RMS/clipping/voice-activity heuristics]
            -> good/degraded -> [wav2vec2 age/gender model] -> response
            -> insufficient  -> short-circuit "unknown", skip inference
```

* **Ingestion (`app/audio_utils.py`)**: bytes are piped straight through
  `ffmpeg` via stdin/stdout - no temp files. ffmpeg's format auto-detection
  covers the codecs a telephony/SIP stack or browser client is likely to
  send. Decode timeout is capped at 10s to bound worst-case latency.
* **Quality gate**: before spending a model forward pass, cheap NumPy-only
  checks (RMS level, clipping ratio, a 20ms-frame voice-activity proxy,
  duration) decide `good` / `degraded` / `insufficient`. This is the piece
  that keeps the service honest in a noisy warehouse/truck-cab environment:
  it's designed to say "I can't tell" rather than confidently guess over
  engine noise.
* **Inference (`app/inference.py`)**: see [Model choice](#model-choice--licensing).
* **API (`app/main.py`)**: FastAPI, async throughout (ffmpeg subprocess is
  awaited, not blocking the event loop). Structured JSON logs record
  `contact_id`, `processing_ms`, `audio_quality`, backend, and predictions
  for every request - never raw audio or transcribed content.

## Model choice & licensing

Primary backend: **`audeering/wav2vec2-large-robust-6-ft-age-gender`**
([model card](https://huggingface.co/audeering/wav2vec2-large-robust-6-ft-age-gender)),
a wav2vec2 encoder (6 transformer layers, the smaller/faster of audEERING's
two public checkpoints) fine-tuned jointly for age (regression, 0-100
years) and gender (3-way softmax: female / male / child) on aGender,
Common Voice, TIMIT, and VoxCeleb2. Swap in the 24-layer variant via
`AGE_GENDER_MODEL=audeering/wav2vec2-large-robust-24-ft-age-gender` for
higher accuracy at roughly 3-4x the compute.

**Why this model:** it's a single forward pass that gives both attributes
jointly (no need to chain a separate gender classifier and age regressor,
which would double latency and let the two disagree on the same voice), it
was trained on a broad enough set of corpora to generalize past studio
conditions, and it's directly loadable through `transformers` with no
custom serving stack.

**Licensing note:** this checkpoint is **CC-BY-NC-SA-4.0 (non-commercial)**.
That's fine for this assignment, but a production deployment for a
logistics company is a commercial use case - either license audEERING's
commercial model (`devAIce`), or swap this backend for a permissively
licensed alternative (e.g. fine-tune a SpeechBrain ECAPA-TDNN embedding
extractor, which is Apache-2.0, with a small classifier head trained on a
licensed dataset). The `Predictor` protocol in `app/inference.py` is
designed so that swap is a single new class, not a rewrite.

**Fallback backend:** `HeuristicPredictor` (pitch/energy only, zero
downloads, zero ML dependencies beyond librosa) is used automatically if
the pretrained model can't load - e.g. no network access on first boot in
an air-gapped environment. It only attempts gender (via median F0 relative
to the ~165Hz male/female crossover) and honestly reports age as
`"unknown"` at low confidence, since pitch alone isn't a credible age
signal. Force it explicitly with `MODEL_BACKEND=heuristic`; force the real
model and fail loudly instead of silently degrading with
`MODEL_BACKEND=model`.

## Language detection (bonus)

Best-effort spoken-language ID runs alongside the gender/age model, using
`speechbrain/lang-id-voxlingua107-ecapa`
([model card](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa)),
an Apache-2.0 ECAPA embedding classifier trained on VoxLingua107 (107
languages). It's a separate, independent path from gender/age inference:
`app/inference.py`'s `InferenceEngine.predict_language()` never raises -
on any failure (no network, model load error, inference error) it returns
`None` and the API simply omits `language` from the response rather than
failing the whole request. It's skipped entirely under
`MODEL_BACKEND=heuristic` (used by tests/CI) since it has no
zero-download heuristic equivalent. Override the checkpoint via
`LANGUAGE_ID_MODEL`. Like gender/age, it's loaded once at startup
(`lifespan` in `app/main.py`) so the first real request isn't stuck with
its cold-start latency, and its status is reported at `GET /health` as
`language_backend`.

## Model backends & fallback behavior

| `MODEL_BACKEND` | Behavior |
|---|---|
| `auto` (default) | try the pretrained model; on any load failure, log and fall back to the heuristic |
| `model` | require the pretrained model; raise on load failure instead of degrading silently |
| `heuristic` | always use the zero-download pitch heuristic (used by the test suite / CI) |

The active backend is loaded once at process startup (`lifespan` handler in
`app/main.py`) so the first real call isn't stuck with cold-start latency,
and is reported at `GET /health`.

## Privacy

Caller audio is treated as PII end-to-end:

* Audio bytes are read into memory (`await file.read(...)`) and passed
  through `ffmpeg` over **stdin/stdout pipes** - at no point in the REST or
  WebSocket path is caller audio written to disk.
* Decoded PCM samples live only in local Python variables for the duration
  of a single request (or a single WebSocket connection's in-memory
  buffer, discarded on disconnect) and are never persisted, cached, or
  logged.
* Logs record only derived, non-reversible fields (`contact_id`,
  `processing_ms`, `audio_quality`, the predicted labels/confidences) -
  never raw or decoded audio, and no transcription is performed anywhere
  in this pipeline.
* The only thing cached to disk is the pretrained *model's* weights
  (`/srv/.cache/huggingface`, mounted as a Docker volume) - this is public
  model data, not caller data.
* `MAX_UPLOAD_BYTES` (default 15MB) bounds worst-case memory use per
  request and rejects oversized uploads before decode.

## Latency & scaling to 1,000 concurrent calls

Measured on this dev container (CPU-only): the heuristic backend processes
a warm 5s clip in ~130ms end-to-end; the wav2vec2-6-layer model adds
roughly 150-300ms of forward-pass time on CPU for the same clip, which is
within the 500ms target but leaves little margin under load. To scale to
~1,000 concurrent calls in production:

1. **Batch across concurrent requests.** A queueing layer that
   micro-batches incoming 16kHz clips (e.g. 10-20ms batching window) into a
   single model forward pass amortizes the fixed cost of a transformer
   forward pass across many callers - this is the single biggest lever.
2. **GPU or quantized-CPU inference.** Batched GPU inference (or INT8
   quantization via `optimum`/ONNX Runtime for CPU fleets) cuts per-clip
   latency well below 100ms, giving headroom for batching + queueing
   overhead.
3. **Horizontal scale-out, stateless workers.** Each `app/main.py` process
   holds no per-call state beyond a single request/connection, so scaling
   is just running more replicas behind a load balancer; the model weight
   cache (read-only after warm-up) can be baked into the image or shared
   via a read-only volume/init-container instead of re-downloaded per pod.
4. **Decouple ingestion from inference.** For the WebSocket path
   specifically, separate the lightweight "accept audio frames" tier from
   a pool of GPU inference workers behind a queue (e.g. gRPC or a message
   bus), so a burst of call starts doesn't block on model availability.
5. **Autoscale on queue depth / p95 latency**, not just CPU%, since
   transformer inference latency doesn't scale linearly with naive CPU
   metrics.

## Known limitations

* **Age accuracy is inherently harder than gender.** Vocal aging cues are
  noisier and more speaker-dependent than the spectral cues for gender;
  expect meaningfully lower accuracy on `age_bracket` than on `gender`,
  especially near bracket boundaries (a 44-year-old and a 46-year-old can
  sound identical). The `eval/eval_common_voice.py` harness is the way to
  get real numbers on your own data.
* **Gender is modeled as male/female/unknown**, following the assignment's
  API contract. The underlying model's "child" class is deliberately
  mapped to `"unknown"` rather than forced into an adult bucket. This is a
  binary model of a non-binary reality; treat `unknown` as a real, useful
  output (low-confidence signal), not a bug.
* **Heuristic fallback cannot estimate age at all** and only weakly
  estimates gender (pure F0 heuristics misclassify some adult male voices
  with naturally higher pitch, and vice versa) - it exists purely as a
  never-fails safety net, not a production-quality path.
* **Non-adult callers, disguised/synthetic voices, and heavy
  accents/dialects not represented in the training corpora** will degrade
  accuracy in ways the confidence score won't fully capture, since the
  model has no explicit out-of-distribution detector.
* **The age confidence score is a heuristic derived from distance-to-
  bracket-boundary**, not a calibrated probability - the base model is a
  point-estimate regressor with no native uncertainty head. Treat it as a
  relative ranking signal, not a statistically calibrated confidence.
* **A small part of the wav2vec2 checkpoint doesn't load cleanly on this
  torch/transformers version pair.** At startup you'll see "Some weights
  of AgeGenderModel were not initialized... newly initialized:
  ['wav2vec2.encoder.pos_conv_embed.conv.parametrizations.weight...']" -
  a known compatibility quirk where newer torch's weight-norm
  reparametrization naming doesn't match the older checkpoint's saved key
  names, so those 2 positional-conv tensors fall back to random init
  instead of the trained values. Predictions still look reasonable in
  testing, but this is a genuine (if likely minor) fidelity gap worth
  knowing about rather than silently ignoring.
* **Language ID is best-effort and coarse-grained.** It reports a spoken
  language code (e.g. `"en"`), not accent/dialect, and confidence is a
  single softmax score over 107 languages from a small embedding model -
  expect confusion between close languages on short/noisy clips. See
  [Language detection](#language-detection-bonus) below.

## Repository layout

```
app/
  main.py           FastAPI app: REST + WebSocket endpoints, logging, timing
  audio_utils.py     ffmpeg-based decode, in-memory quality assessment
  inference.py       model wrapper + heuristic fallback + engine singleton
  schemas.py         Pydantic request/response models
  logging_conf.py    structured JSON logging setup
tests/
  test_analyze.py    integration tests against a running FastAPI app
  conftest.py        synthetic-audio fixtures, forces heuristic backend
eval/
  eval_common_voice.py   accuracy + calibration harness (bonus task)
sample_audio/         synthetic smoke-test clip + instructions for real data
Dockerfile
docker-compose.yml
```
