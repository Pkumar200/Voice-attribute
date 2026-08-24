def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_analyze_low_pitch_predicts_male(client, low_pitch_wav_bytes):
    r = client.post(
        "/analyze",
        files={"file": ("clip.wav", low_pitch_wav_bytes, "audio/wav")},
    )
    assert r.status_code == 200
    body = r.json()

    assert body["gender"]["prediction"] in ("male", "female", "unknown")
    # with the heuristic backend, a clean ~110Hz tone should read as "male"
    assert body["gender"]["prediction"] == "male"
    assert 0.0 <= body["gender"]["confidence"] <= 1.0
    assert body["age_bracket"]["prediction"] in (
        "18-30", "31-45", "46-60", "60+", "unknown"
    )
    assert body["audio_quality"] in ("good", "degraded", "insufficient")
    assert isinstance(body["processing_ms"], int)
    assert body["processing_ms"] >= 0


def test_analyze_high_pitch_predicts_female(client, high_pitch_wav_bytes):
    r = client.post(
        "/analyze",
        files={"file": ("clip.wav", high_pitch_wav_bytes, "audio/wav")},
    )
    assert r.status_code == 200
    assert r.json()["gender"]["prediction"] == "female"


def test_analyze_silence_reports_insufficient_quality(client, silence_wav_bytes):
    r = client.post(
        "/analyze",
        files={"file": ("clip.wav", silence_wav_bytes, "audio/wav")},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["audio_quality"] == "insufficient"
    assert body["gender"]["prediction"] == "unknown"
    assert body["gender"]["confidence"] == 0.0


def test_analyze_too_short_clip_is_insufficient(client, too_short_wav_bytes):
    r = client.post(
        "/analyze",
        files={"file": ("clip.wav", too_short_wav_bytes, "audio/wav")},
    )
    assert r.status_code == 200
    assert r.json()["audio_quality"] == "insufficient"


def test_analyze_garbage_bytes_returns_422():
    from app.main import app
    from fastapi.testclient import TestClient

    with TestClient(app) as c:
        r = c.post(
            "/analyze",
            files={"file": ("clip.wav", b"not actually audio data", "audio/wav")},
        )
    assert r.status_code == 422
    assert r.json()["error"] == "audio_decode_failed"


def test_analyze_language_omitted_under_heuristic_backend(client, low_pitch_wav_bytes):
    # MODEL_BACKEND=heuristic (forced in conftest.py) also disables the
    # language-id path, since it has no zero-download heuristic equivalent -
    # the field should be gracefully omitted (null), not an error.
    r = client.post(
        "/analyze",
        files={"file": ("clip.wav", low_pitch_wav_bytes, "audio/wav")},
    )
    assert r.status_code == 200
    assert r.json()["language"] is None


def test_analyze_contact_id_echoed(client, low_pitch_wav_bytes):
    r = client.post(
        "/analyze?contact_id=test-contact-123",
        files={"file": ("clip.wav", low_pitch_wav_bytes, "audio/wav")},
    )
    assert r.status_code == 200
    assert r.json()["contact_id"] == "test-contact-123"
