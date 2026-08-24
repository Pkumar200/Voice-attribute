"""
Attribute inference engine.

Primary path: audEERING's wav2vec2-large-robust-*-ft-age-gender model
(https://huggingface.co/audeering/wav2vec2-large-robust-6-ft-age-gender),
a wav2vec2 encoder fine-tuned on aGender + Common Voice + TIMIT + VoxCeleb2
for joint age (regression, years/100) and gender (3-way softmax: child /
female / male) prediction. It is CC-BY-NC-SA-4.0 (non-commercial) - see
README "Model choice & licensing" for the commercial alternative.

Fallback path: a zero-dependency, zero-download pitch/energy heuristic
(librosa YIN F0 tracking). This exists so the service still boots and
returns *something* with an honestly low confidence when:
  * the container has no network access to Hugging Face at first boot, or
  * torch/transformers fail to load (e.g. constrained CPU-only image), or
  * MODEL_BACKEND=heuristic is set explicitly (useful for tests/CI, which
    should not depend on downloading a 300MB+ checkpoint).

Both paths implement the same `Predictor` interface so main.py doesn't care
which one is active; the active backend is reported in logs and can be
surfaced via /health.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np

logger = logging.getLogger("voice_attr.inference")

AGE_BRACKETS = ["18-30", "31-45", "46-60", "60+"]


def bucket_age(years: float) -> str:
    if years < 18:
        # Under-18 callers aren't a bracket we report on for this use case;
        # collapse into the youngest adult bucket rather than inventing a
        # new label the API contract doesn't define.
        return "18-30"
    if years <= 30:
        return "18-30"
    if years <= 45:
        return "31-45"
    if years <= 60:
        return "46-60"
    return "60+"


@dataclass
class GenderResult:
    prediction: str
    confidence: float


@dataclass
class AgeResult:
    prediction: str
    confidence: float


@dataclass
class InferenceResult:
    gender: GenderResult
    age: AgeResult
    backend: str
    model_latency_ms: float


@dataclass
class LanguageResult:
    prediction: Optional[str]
    confidence: float


class Predictor(Protocol):
    def predict(self, samples: np.ndarray, sr: int) -> InferenceResult: ...


class LanguagePredictor(Protocol):
    def predict(self, samples: np.ndarray, sr: int) -> LanguageResult: ...


# --------------------------------------------------------------------------
# Heuristic fallback (no downloads, no torch required at all)
# --------------------------------------------------------------------------
class HeuristicPredictor:
    """
    Acoustic-feature heuristic used when the pretrained model is unavailable.

    Gender: median fundamental frequency (F0) via librosa's YIN estimator.
    Typical adult male speaking F0 is ~85-165 Hz, adult female ~165-255 Hz
    (Titze, 1994). We map distance from the ~165 Hz crossover to a
    confidence score, and only vote if a large enough fraction of frames
    are voiced (otherwise F0 is noise-dominated garbage).

    Age: F0 alone is a weak age predictor (huge inter-speaker variance), so
    we deliberately report low confidence and lean on "unknown" rather than
    letting a shaky heuristic masquerade as a real estimate for something
    it can't actually see (e.g. this cannot use timbre/formant-drift cues a
    learned model would). This is intentional and documented as a
    known limitation in the README.
    """

    name = "heuristic-f0"

    def predict(self, samples: np.ndarray, sr: int) -> InferenceResult:
        import librosa

        t0 = time.perf_counter()

        f0, voiced_flag, _voiced_prob = librosa.pyin(
            samples,
            fmin=60,
            fmax=400,
            sr=sr,
            frame_length=2048,
        )
        voiced_ratio = float(np.nanmean(voiced_flag)) if len(voiced_flag) else 0.0
        f0_voiced = f0[voiced_flag] if voiced_flag is not None else np.array([])

        if len(f0_voiced) < 5 or voiced_ratio < 0.1:
            gender = GenderResult("unknown", 0.0)
        else:
            median_f0 = float(np.nanmedian(f0_voiced))
            crossover = 165.0
            # distance from crossover -> confidence, saturating at 60Hz away
            dist = min(abs(median_f0 - crossover), 60.0) / 60.0
            confidence = round(0.5 + 0.45 * dist, 2)  # 0.5..0.95
            if median_f0 < crossover:
                gender = GenderResult("male", confidence)
            else:
                gender = GenderResult("female", confidence)

        # Age: heuristics on F0/jitter are not reliable enough to assert a
        # bracket with any honesty -> always low-confidence "unknown".
        age = AgeResult("unknown", 0.1)

        model_latency_ms = (time.perf_counter() - t0) * 1000
        return InferenceResult(gender=gender, age=age, backend=self.name,
                                model_latency_ms=model_latency_ms)


# --------------------------------------------------------------------------
# Pretrained wav2vec2 age/gender model
# --------------------------------------------------------------------------
class Wav2Vec2AgeGenderPredictor:
    name = "wav2vec2-age-gender"

    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        import torch
        import torch.nn as nn
        from transformers import Wav2Vec2Processor
        from transformers.models.wav2vec2.modeling_wav2vec2 import (
            Wav2Vec2Model,
            Wav2Vec2PreTrainedModel,
        )

        self.torch = torch
        model_name = model_name or os.environ.get(
            "AGE_GENDER_MODEL", "audeering/wav2vec2-large-robust-6-ft-age-gender"
        )
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # --- model definition, mirrors audEERING's published model card ---
        # (nested classes close over `torch`/`nn` from this __init__ scope)
        class ModelHead(nn.Module):
            def __init__(self, config, num_labels):
                super().__init__()
                self.dense = nn.Linear(config.hidden_size, config.hidden_size)
                self.dropout = nn.Dropout(config.final_dropout)
                self.out_proj = nn.Linear(config.hidden_size, num_labels)

            def forward(self, features, **kwargs):
                x = self.dropout(features)
                x = self.dense(x)
                x = torch.tanh(x)
                x = self.dropout(x)
                x = self.out_proj(x)
                return x

        class AgeGenderModel(Wav2Vec2PreTrainedModel):
            def __init__(self, config):
                super().__init__(config)
                self.config = config
                self.wav2vec2 = Wav2Vec2Model(config)
                self.age = ModelHead(config, 1)
                self.gender = ModelHead(config, 3)
                self.init_weights()

            def forward(self, input_values):
                outputs = self.wav2vec2(input_values)
                hidden_states = outputs[0].mean(dim=1)
                logits_age = self.age(hidden_states)
                logits_gender = torch.softmax(self.gender(hidden_states), dim=-1)
                return hidden_states, logits_age, logits_gender

        logger.info("loading age/gender model %s on %s", model_name, self.device)
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = AgeGenderModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        # id2label from config tells us the true class order rather than
        # trusting a hardcoded guess (audEERING's public checkpoints use
        # {0: 'female', 1: 'male', 2: 'child'}, but we defer to config).
        id2label = getattr(self.model.config, "id2label", None)
        if id2label:
            self.gender_labels = [id2label[i] for i in range(len(id2label))]
        else:
            self.gender_labels = ["female", "male", "child"]

    def predict(self, samples: np.ndarray, sr: int) -> InferenceResult:
        t0 = time.perf_counter()
        inputs = self.processor(samples, sampling_rate=sr, return_tensors="np")
        input_values = self.torch.from_numpy(inputs["input_values"]).to(self.device)

        with self.torch.no_grad():
            _hidden, logits_age, logits_gender = self.model(input_values)

        age_years = float(logits_age.squeeze().cpu().numpy()) * 100.0
        age_years = max(0.0, min(100.0, age_years))

        gender_probs = logits_gender.squeeze().cpu().numpy()
        label_idx = int(np.argmax(gender_probs))
        raw_label = self.gender_labels[label_idx]
        confidence = float(gender_probs[label_idx])

        # collapse the model's 3-way {female, male, child} onto our API's
        # {male, female, unknown} contract - "child" callers aren't a
        # target persona for this system, and forcing them into an adult
        # bucket would be actively misleading, so we surface "unknown".
        if raw_label == "child":
            gender = GenderResult("unknown", round(confidence, 2))
        elif raw_label in ("male", "female"):
            gender = GenderResult(raw_label, round(confidence, 2))
        else:
            gender = GenderResult("unknown", 0.0)

        # age confidence: the model is a point-estimate regressor with no
        # native uncertainty head. We derive a rough confidence from how
        # far the estimate sits from a bracket boundary (more margin =
        # more confidence), clipped to a sane range. This is a heuristic
        # on top of a real model output, and is called out as such in the
        # README rather than presented as calibrated probability.
        bracket = bucket_age(age_years)
        boundaries = [18, 30, 45, 60, 100]
        dist_to_boundary = min(abs(age_years - b) for b in boundaries)
        age_confidence = round(min(0.95, 0.4 + dist_to_boundary / 30.0), 2)
        age = AgeResult(bracket, age_confidence)

        model_latency_ms = (time.perf_counter() - t0) * 1000
        return InferenceResult(gender=gender, age=age, backend=self.name,
                                model_latency_ms=model_latency_ms)


# --------------------------------------------------------------------------
# Language identification (bonus task) - best-effort, entirely independent
# of the gender/age path so a failure or absence here never blocks the
# core prediction.
# --------------------------------------------------------------------------
class SpeechBrainLanguageIdPredictor:
    """
    Best-effort spoken-language ID via SpeechBrain's VoxLingua107 ECAPA
    model (https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa),
    an Apache-2.0 embedding classifier trained on 107 languages. Runs on
    the same 16kHz mono PCM the rest of the pipeline already produces.
    """

    name = "speechbrain-lang-id"

    def __init__(self, model_name: Optional[str] = None):
        from speechbrain.inference.classifiers import EncoderClassifier

        model_name = model_name or os.environ.get(
            "LANGUAGE_ID_MODEL", "speechbrain/lang-id-voxlingua107-ecapa"
        )
        savedir = os.path.join(
            os.environ.get("HF_HOME", "/tmp"), "speechbrain-lang-id"
        )
        logger.info("loading language-id model %s", model_name)
        self.classifier = EncoderClassifier.from_hparams(
            source=model_name, savedir=savedir, run_opts={"device": "cpu"}
        )

    def predict(self, samples: np.ndarray, sr: int) -> LanguageResult:
        import torch

        signal = torch.from_numpy(samples).float().unsqueeze(0)
        _out_prob, score, _index, text_lab = self.classifier.classify_batch(signal)

        label = None
        if text_lab:
            # this checkpoint's labels look like "en: English"; keep just
            # the ISO code so the field matches the "en"/"es" style the
            # schema documents.
            label = str(text_lab[0]).split(":")[0].strip() or None

        confidence = 0.0
        if score is not None:
            # classify_batch's `score` is a log-posterior; exponentiate to
            # get a real 0..1 probability rather than surfacing a raw
            # log-likelihood as if it were a confidence.
            confidence = float(torch.exp(score).squeeze().item())
            confidence = max(0.0, min(1.0, confidence))

        return LanguageResult(prediction=label, confidence=round(confidence, 2))


# --------------------------------------------------------------------------
# Engine: lazy singleton with automatic fallback
# --------------------------------------------------------------------------
class InferenceEngine:
    def __init__(self):
        self._predictor: Optional[Predictor] = None
        self._backend_name: Optional[str] = None
        self._lang_predictor: Optional[LanguagePredictor] = None
        self._lang_backend_name: Optional[str] = None
        self._lang_load_attempted = False

    def _load(self) -> Predictor:
        backend_pref = os.environ.get("MODEL_BACKEND", "auto")  # auto|model|heuristic

        if backend_pref == "heuristic":
            return HeuristicPredictor()

        if backend_pref in ("auto", "model"):
            try:
                return Wav2Vec2AgeGenderPredictor()
            except Exception:
                if backend_pref == "model":
                    raise
                logger.exception(
                    "failed to load pretrained age/gender model; "
                    "falling back to acoustic heuristic"
                )
                return HeuristicPredictor()

        raise ValueError(f"unknown MODEL_BACKEND={backend_pref!r}")

    def ensure_loaded(self) -> None:
        if self._predictor is None:
            self._predictor = self._load()
            self._backend_name = getattr(self._predictor, "name", "unknown")

    @property
    def backend_name(self) -> str:
        return self._backend_name or "not-loaded"

    @property
    def is_loaded(self) -> bool:
        return self._predictor is not None

    def predict(self, samples: np.ndarray, sr: int) -> InferenceResult:
        self.ensure_loaded()
        assert self._predictor is not None
        return self._predictor.predict(samples, sr)

    def _load_language(self) -> Optional[LanguagePredictor]:
        backend_pref = os.environ.get("MODEL_BACKEND", "auto")
        if backend_pref == "heuristic":
            # zero-download mode (also what tests/CI force) - language ID
            # has no dependency-free heuristic equivalent, so it's simply
            # unavailable here rather than faking a guess.
            return None
        try:
            return SpeechBrainLanguageIdPredictor()
        except Exception:
            logger.exception(
                "failed to load language-id model; "
                "'language' will be omitted from responses"
            )
            return None

    def ensure_language_loaded(self) -> None:
        if not self._lang_load_attempted:
            self._lang_load_attempted = True
            self._lang_predictor = self._load_language()
            self._lang_backend_name = getattr(self._lang_predictor, "name", None)

    @property
    def language_backend_name(self) -> Optional[str]:
        return self._lang_backend_name

    def predict_language(self, samples: np.ndarray, sr: int) -> Optional[LanguageResult]:
        """Best-effort: returns None (never raises) if unavailable or on failure."""
        self.ensure_language_loaded()
        if self._lang_predictor is None:
            return None
        try:
            return self._lang_predictor.predict(samples, sr)
        except Exception:
            logger.exception("language-id inference failed")
            return None


# module-level singleton, imported by main.py / ws handlers
engine = InferenceEngine()
