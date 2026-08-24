# Design write-up (~200 words)

I built a FastAPI service that pipes uploaded audio through ffmpeg
in-memory (no disk writes, ever - important given caller audio is PII) into
mono 16kHz PCM, runs cheap NumPy quality checks (RMS, clipping, a
frame-level voice-activity proxy) to decide `good`/`degraded`/`insufficient`
*before* paying for inference, then predicts gender and age jointly with
`audeering/wav2vec2-large-robust-6-ft-age-gender`. I chose it because it's
a single forward pass for both attributes (avoids two models disagreeing on
the same voice), generalizes past studio audio (trained on Common
Voice/TIMIT/VoxCeleb2/aGender), and drops straight into `transformers`. It's
CC-BY-NC-SA (non-commercial) - a real deployment needs a licensed or
self-trained replacement, which the `Predictor` protocol makes a one-class
swap. A pitch-only heuristic fallback keeps the service alive with zero
downloads if the model can't load. Spoken-language ID runs alongside it via
SpeechBrain's VoxLingua107 ECAPA model - a separate, best-effort path that
never blocks the core gender/age result if it fails to load.

With more time: fine-tune a permissively-licensed encoder (SpeechBrain
ECAPA, Apache-2.0) on labeled logistics-call-like data for a commercial-safe
model, add a real VAD instead of the frame-energy proxy, and calibrate
confidence using the eval harness's ECE metric as the loss to minimize.

To reach 1,000 concurrent calls: micro-batch concurrent requests into
single forward passes, move inference to GPU (or INT8/ONNX on CPU),
run stateless horizontally-scaled workers behind a queue decoupled from
ingestion, and autoscale on queue depth/p95 latency rather than raw CPU%.
