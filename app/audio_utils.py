"""
Audio ingestion helpers.

Design goals:
  * Accept whatever codec/container a telephony/SIP stack throws at us
    (a-law/mu-law WAV, mp3, ogg/opus, webm/opus from browsers, raw PCM, ...).
  * Never touch disk. Everything is piped through ffmpeg in-memory and the
    decoded samples live only in process memory for the lifetime of the
    request (see PRIVACY.md / README "Privacy" section).
  * Produce a single, well-understood representation for the rest of the
    pipeline: float32 mono PCM at 16 kHz.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger("voice_attr.audio")

TARGET_SR = 16_000
MIN_USABLE_SECONDS = 0.75       # below this we can't say anything meaningful
MIN_GOOD_SECONDS = 2.0          # below this, downgrade "good" -> "degraded"
CLIP_RATIO_BAD = 0.02           # >2% of samples clipped -> quality hit
SILENCE_RMS_DBFS = -50.0        # below this we treat the chunk as silence


class AudioDecodeError(Exception):
    """Raised when ffmpeg cannot make sense of the uploaded bytes."""


@dataclass
class DecodedAudio:
    samples: np.ndarray   # float32, mono, range [-1, 1]
    sample_rate: int
    duration_s: float


async def decode_to_pcm(raw_bytes: bytes) -> DecodedAudio:
    """
    Decode arbitrary audio bytes to mono float32 PCM @ TARGET_SR using ffmpeg.

    We shell out to ffmpeg via stdin/stdout pipes so no intermediate file is
    ever written to disk. ffmpeg auto-detects the input container/codec
    (wav/mp3/ogg-opus/webm-opus/flac/raw mu-law, etc.), which covers the
    range of codecs telephony and browser clients commonly send.
    """
    if not raw_bytes:
        raise AudioDecodeError("empty audio payload")

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-i", "pipe:0",
        "-ar", str(TARGET_SR),
        "-ac", "1",
        "-f", "f32le",       # raw little-endian float32 -> trivial to np.frombuffer
        "pipe:1",
    ]

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(input=raw_bytes), timeout=10.0
        )
    except asyncio.TimeoutError:
        proc.kill()
        raise AudioDecodeError("ffmpeg decode timed out")

    if proc.returncode != 0 or not stdout:
        raise AudioDecodeError(
            f"ffmpeg failed (code={proc.returncode}): {stderr.decode(errors='ignore')[:500]}"
        )

    samples = np.frombuffer(stdout, dtype=np.float32)
    duration_s = len(samples) / TARGET_SR

    # Explicitly drop the raw bytes reference ASAP; nothing else in this
    # function retains the original compressed payload.
    del raw_bytes

    return DecodedAudio(samples=samples, sample_rate=TARGET_SR, duration_s=duration_s)


@dataclass
class QualityReport:
    quality: str          # "good" | "degraded" | "insufficient"
    rms_dbfs: float
    clip_ratio: float
    voiced_ratio: float
    duration_s: float
    reasons: list[str]


def assess_quality(samples: np.ndarray, sr: int) -> QualityReport:
    """
    Cheap, dependency-light signal-quality heuristics that run in <1ms so we
    can decide *before* the (relatively) expensive model forward pass
    whether a prediction is even worth trusting.
    """
    reasons: list[str] = []
    duration_s = len(samples) / sr if sr else 0.0

    if duration_s < MIN_USABLE_SECONDS:
        return QualityReport(
            quality="insufficient",
            rms_dbfs=-120.0,
            clip_ratio=0.0,
            voiced_ratio=0.0,
            duration_s=duration_s,
            reasons=[f"clip too short ({duration_s:.2f}s < {MIN_USABLE_SECONDS}s)"],
        )

    rms = float(np.sqrt(np.mean(np.square(samples)) + 1e-12))
    rms_dbfs = 20.0 * np.log10(rms + 1e-12)

    clip_ratio = float(np.mean(np.abs(samples) >= 0.999))

    # crude voice-activity proxy: fraction of 20ms frames whose energy is
    # above the silence floor. Good enough to catch "mostly dead air /
    # engine noise, barely any speech" without pulling in a VAD model.
    frame_len = int(0.02 * sr)
    if frame_len > 0 and len(samples) >= frame_len:
        n_frames = len(samples) // frame_len
        framed = samples[: n_frames * frame_len].reshape(n_frames, frame_len)
        frame_rms_dbfs = 20.0 * np.log10(np.sqrt(np.mean(framed ** 2, axis=1)) + 1e-12)
        voiced_ratio = float(np.mean(frame_rms_dbfs > SILENCE_RMS_DBFS))
    else:
        voiced_ratio = 0.0

    quality = "good"

    if rms_dbfs < SILENCE_RMS_DBFS or voiced_ratio < 0.15:
        quality = "insufficient"
        reasons.append(f"mostly silence/noise (voiced_ratio={voiced_ratio:.2f})")

    if clip_ratio > CLIP_RATIO_BAD:
        quality = "insufficient" if quality == "insufficient" else "degraded"
        reasons.append(f"clipping detected ({clip_ratio:.2%} of samples)")

    if duration_s < MIN_GOOD_SECONDS and quality == "good":
        quality = "degraded"
        reasons.append(f"short clip ({duration_s:.2f}s) reduces confidence")

    if -50.0 <= rms_dbfs < -35.0 and quality == "good":
        quality = "degraded"
        reasons.append(f"low signal level ({rms_dbfs:.1f} dBFS)")

    return QualityReport(
        quality=quality,
        rms_dbfs=rms_dbfs,
        clip_ratio=clip_ratio,
        voiced_ratio=voiced_ratio,
        duration_s=duration_s,
        reasons=reasons,
    )
