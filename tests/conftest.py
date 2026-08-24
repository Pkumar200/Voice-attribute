"""
Tests force MODEL_BACKEND=heuristic so CI/smoke-testing never depends on
downloading the ~300MB pretrained checkpoint from Hugging Face. This must
be set BEFORE `app.main` is imported, since the model is warmed up on
FastAPI's startup event.
"""
import io
import os

os.environ.setdefault("MODEL_BACKEND", "heuristic")

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c


def _sine_wav_bytes(freq_hz: float, seconds: float = 3.0, sr: int = 16000) -> bytes:
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    # a few harmonics so it looks a bit more voice-like than a pure tone,
    # plus a touch of noise so it isn't a degenerate zero-crossing signal
    signal = (
        0.6 * np.sin(2 * np.pi * freq_hz * t)
        + 0.2 * np.sin(2 * np.pi * freq_hz * 2 * t)
        + 0.05 * np.random.default_rng(0).normal(size=t.shape)
    ).astype(np.float32)
    signal = signal / np.max(np.abs(signal))
    buf = io.BytesIO()
    sf.write(buf, signal, sr, format="WAV", subtype="FLOAT")
    buf.seek(0)
    return buf.read()


@pytest.fixture
def low_pitch_wav_bytes() -> bytes:
    """~110Hz fundamental - in typical adult-male F0 range."""
    return _sine_wav_bytes(110.0)


@pytest.fixture
def high_pitch_wav_bytes() -> bytes:
    """~220Hz fundamental - in typical adult-female F0 range."""
    return _sine_wav_bytes(220.0)


@pytest.fixture
def silence_wav_bytes() -> bytes:
    sr = 16000
    seconds = 3.0
    signal = np.zeros(int(sr * seconds), dtype=np.float32)
    buf = io.BytesIO()
    sf.write(buf, signal, sr, format="WAV", subtype="FLOAT")
    buf.seek(0)
    return buf.read()


@pytest.fixture
def too_short_wav_bytes() -> bytes:
    sr = 16000
    seconds = 0.2
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    signal = (0.5 * np.sin(2 * np.pi * 150 * t)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, signal, sr, format="WAV", subtype="FLOAT")
    buf.seek(0)
    return buf.read()
