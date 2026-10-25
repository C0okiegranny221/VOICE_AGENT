"""Local speech-to-text.

- hinglish (default): Oriserve Whisper-Hindi2Hinglish-Apex via transformers on the M2 GPU (MPS).
  Hindi, Hinglish and Indian-accented English all come out in Roman script
  ("Meri flight tin ghante late hai"), which keeps English airline terms searchable.
- english: Whisper large-v3-turbo via mlx-whisper, English only.
"""
import time
from functools import lru_cache

import numpy as np

from config import CONFIG

MIN_SECONDS = 0.3


@lru_cache(maxsize=1)
def _hinglish_pipeline():
    import torch
    from transformers import pipeline

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    return pipeline("automatic-speech-recognition", model=CONFIG.stt_hinglish_model, device=device,
                    dtype=torch.float16 if device == "mps" else torch.float32)


def _transcribe_hinglish(audio: np.ndarray) -> str:
    # The model writes Hinglish under Whisper's English language token.
    out = _hinglish_pipeline()({"raw": audio, "sampling_rate": 16000},
                               generate_kwargs={"language": "en", "task": "transcribe"})
    return out["text"].strip()


def _transcribe_english(audio: np.ndarray) -> str:
    import mlx_whisper

    result = mlx_whisper.transcribe(
        audio, path_or_hf_repo=CONFIG.stt_model, language="en", initial_prompt=CONFIG.stt_vocab,
        condition_on_previous_text=False, temperature=0.0,
    )
    text = result.get("text", "").strip()
    # Whisper sometimes echoes the vocabulary prompt on near-silent input.
    return "" if text and text.rstrip(".") in CONFIG.stt_vocab else text


def transcribe(audio: np.ndarray, sample_rate: int = 16000) -> str:
    """Transcribe float32 mono 16 kHz audio. Returns '' for silence or very short audio."""
    if sample_rate != 16000:
        raise ValueError("expected 16 kHz audio")
    audio = np.asarray(audio, dtype=np.float32).flatten()
    if len(audio) < MIN_SECONDS * sample_rate or float(np.abs(audio).max(initial=0)) < 1e-3:
        return ""
    return _transcribe_hinglish(audio) if CONFIG.stt_engine == "hinglish" else _transcribe_english(audio)


def warmup() -> float:
    t0 = time.perf_counter()
    rng = np.random.default_rng(0)
    transcribe((rng.standard_normal(16000) * 0.01).astype(np.float32))
    return time.perf_counter() - t0
