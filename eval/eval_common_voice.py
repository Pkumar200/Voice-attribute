#!/usr/bin/env python3
"""
Eval harness (bonus task).

Runs the inference engine against a labeled dataset and reports accuracy
and confidence calibration for gender and age-bracket predictions.

Expected input: Mozilla Common Voice ships a `validated.tsv` (or similar)
with columns including `path`, `age`, `gender` for a subset of
demographically-labeled clips. Common Voice's raw `age` values are already
bucketed (teens, twenties, thirties, forties, fifties, sixties, ...), which
we remap to this service's brackets.

Usage:
    python eval/eval_common_voice.py \\
        --tsv /path/to/cv-corpus/en/validated.tsv \\
        --clips-dir /path/to/cv-corpus/en/clips \\
        --limit 500

Only clips with both `age` and `gender` populated are scored (Common Voice
leaves both blank for most contributors, so pass --limit generously or omit
it to scan the whole file until enough labeled rows are found).

Note: running this against the pretrained-model backend requires network
access to Hugging Face at process start (to download the checkpoint) -
set MODEL_BACKEND=heuristic to eval the fallback instead, or pre-populate
the HF cache as the Dockerfile does.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import audio_utils, inference  # noqa: E402

CV_AGE_TO_BRACKET = {
    "teens": "18-30",
    "twenties": "18-30",
    "thirties": "31-45",
    "fourties": "31-45",
    "forties": "31-45",
    "fifties": "46-60",
    "sixties": "46-60",
    "seventies": "60+",
    "eighties": "60+",
    "nineties": "60+",
}

CV_GENDER_MAP = {
    "male": "male",
    "male_masculine": "male",
    "female": "female",
    "female_feminine": "female",
}


def load_labeled_rows(tsv_path: Path, limit: int | None):
    rows = []
    with open(tsv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            gender = CV_GENDER_MAP.get((row.get("gender") or "").strip().lower())
            bracket = CV_AGE_TO_BRACKET.get((row.get("age") or "").strip().lower())
            if not gender or not bracket:
                continue
            rows.append({"path": row["path"], "gender": gender, "age_bracket": bracket})
            if limit and len(rows) >= limit:
                break
    return rows


def expected_calibration_error(confidences: list[float], corrects: list[bool], n_bins: int = 10) -> float:
    """Standard ECE: weighted average gap between confidence and accuracy per bin."""
    confidences = np.array(confidences)
    corrects = np.array(corrects, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(confidences)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if not np.any(mask):
            continue
        bin_conf = confidences[mask].mean()
        bin_acc = corrects[mask].mean()
        ece += (mask.sum() / n) * abs(bin_conf - bin_acc)
    return float(ece)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", required=True, type=Path, help="path to Common Voice validated.tsv")
    ap.add_argument("--clips-dir", required=True, type=Path, help="path to the clips/ directory")
    ap.add_argument("--limit", type=int, default=200)
    args = ap.parse_args()

    rows = load_labeled_rows(args.tsv, args.limit)
    if not rows:
        print("No labeled rows found (need both `age` and `gender` populated). Exiting.")
        return

    print(f"Evaluating on {len(rows)} labeled clips (backend={inference.engine.backend_name or 'lazy'})...")

    gender_correct, gender_conf, gender_hit = [], [], []
    age_correct, age_conf, age_hit = [], [], []
    latencies_ms = []
    skipped = 0

    for i, row in enumerate(rows):
        clip_path = args.clips_dir / row["path"]
        if not clip_path.exists():
            skipped += 1
            continue

        raw = clip_path.read_bytes()
        import asyncio
        try:
            decoded = asyncio.run(audio_utils.decode_to_pcm(raw))
        except audio_utils.AudioDecodeError:
            skipped += 1
            continue

        quality = audio_utils.assess_quality(decoded.samples, decoded.sample_rate)
        if quality.quality == "insufficient":
            skipped += 1
            continue

        t0 = time.perf_counter()
        result = inference.engine.predict(decoded.samples, decoded.sample_rate)
        latencies_ms.append((time.perf_counter() - t0) * 1000)

        g_hit = result.gender.prediction == row["gender"]
        gender_hit.append(g_hit)
        gender_conf.append(result.gender.confidence)

        a_hit = result.age.prediction == row["age_bracket"]
        age_hit.append(a_hit)
        age_conf.append(result.age.confidence)

        if (i + 1) % 50 == 0:
            print(f"  ...{i + 1}/{len(rows)} processed")

    n = len(gender_hit)
    if n == 0:
        print("No clips were successfully scored (all skipped). Check --clips-dir path.")
        return

    print()
    print(f"Scored {n} clips (skipped {skipped} missing/undecodable/low-quality)")
    print(f"Mean inference latency: {np.mean(latencies_ms):.1f}ms  (p95: {np.percentile(latencies_ms, 95):.1f}ms)")
    print()
    print("Gender:")
    print(f"  Accuracy: {np.mean(gender_hit):.3f}")
    print(f"  Mean confidence: {np.mean(gender_conf):.3f}")
    print(f"  Expected Calibration Error (ECE): {expected_calibration_error(gender_conf, gender_hit):.3f}")
    print()
    print("Age bracket:")
    print(f"  Accuracy: {np.mean(age_hit):.3f}")
    print(f"  Mean confidence: {np.mean(age_conf):.3f}")
    print(f"  Expected Calibration Error (ECE): {expected_calibration_error(age_conf, age_hit):.3f}")


if __name__ == "__main__":
    main()
