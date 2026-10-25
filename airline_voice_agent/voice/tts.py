"""Local text-to-speech with Kokoro-82M (24 kHz).

Each piece of text is routed by script: Devanagari (Hindi / Hinglish replies) goes to Kokoro's Hindi
pipeline and voice, anything else to the English one. Both pipelines share one copy of the model.
"""
import re
import time
from functools import lru_cache

import numpy as np

from config import CONFIG

DEVANAGARI = re.compile(r"[ऀ-ॿ]")

ENGLISH_REPLACEMENTS = [
    (r"(?:₹|\bRs\.?|\bINR)\s?([\d,]+(?:\.\d+)?)", r"\1 rupees"),
    (r"\b(\d+(?:\.\d+)?)\s?kgs?\b", r"\1 kilograms"),
    (r"\b(\d+(?:\.\d+)?)\s?km\b", r"\1 kilometres"),
    (r"\b(\d+(?:\.\d+)?)\s?(?:hrs?|hours?)\b", r"\1 hours"),
    (r"\b(\d+(?:\.\d+)?)\s?mins?\b", r"\1 minutes"),
    (r"\b(\d+(?:\.\d+)?)\s?ml\b", r"\1 millilitres"),
    (r"\b(\d+(?:\.\d+)?)\s?cm\b", r"\1 centimetres"),
    (r"(\d)\s?%", r"\1 percent"),
    (r"\b6E\b", "six E"),
    (r"\bDGCA\b", "D G C A"),
    (r"\bPNR\b", "P N R"),
    (r"\bCAR\b", "C A R"),
    (r"&", " and "),
]
HINDI_REPLACEMENTS = [
    (r"(?:₹|\bRs\.?|\bINR)\s?([\d,]+(?:\.\d+)?)", r"\1 रुपये"),
    (r"\b(\d+(?:\.\d+)?)\s?kgs?\b", r"\1 किलो"),
    (r"\b(\d+(?:\.\d+)?)\s?km\b", r"\1 किलोमीटर"),
    (r"\b(\d+(?:\.\d+)?)\s?(?:hrs?|hours?)\b", r"\1 घंटे"),
    (r"\b(\d+(?:\.\d+)?)\s?mins?\b", r"\1 मिनट"),
    (r"\b(\d+(?:\.\d+)?)\s?ml\b", r"\1 मिलीलीटर"),
    (r"(\d)\s?%", r"\1 प्रतिशत"),
    (r"\b6E\b", "सिक्स ई"),
    (r"\bDGCA\b", "डी जी सी ए"),
    (r"\bPNR\b", "पी एन आर"),
    (r"\bIndiGo\b", "इंडिगो"),
    (r"&", " और "),
]
COMMON = [(r"[*#_`>|]", " "), (r"https?://\S+", "the IndiGo website")]


def is_hindi(text: str) -> bool:
    return bool(DEVANAGARI.search(text))


def normalize(text: str) -> str:
    """Make text speakable: currency, units and acronyms spelled out; markdown stripped."""
    for pattern, repl in (HINDI_REPLACEMENTS if is_hindi(text) else ENGLISH_REPLACEMENTS) + COMMON:
        text = re.sub(pattern, repl, text)
    text = re.sub(r"(\d),(\d{3})", r"\1\2", text)  # 10,000 -> 10000 (read as a number)
    return re.sub(r"\s+", " ", text).strip()


@lru_cache(maxsize=1)
def _english():
    from kokoro import KPipeline

    return KPipeline(lang_code=CONFIG.tts_lang, repo_id="hexgrad/Kokoro-82M")


@lru_cache(maxsize=1)
def _hindi():
    from kokoro import KPipeline

    return KPipeline(lang_code="h", repo_id="hexgrad/Kokoro-82M", model=_english().model)


def synthesize(text: str, voice: str | None = None) -> np.ndarray:
    """Float32 mono audio at CONFIG.tts_rate for one piece of text."""
    hindi = is_hindi(text) or (voice or "").startswith(("hf_", "hm_"))
    pipeline = _hindi() if hindi else _english()
    voice = voice or (CONFIG.tts_voice_hindi if hindi else CONFIG.tts_voice)
    chunks = [r.audio for r in pipeline(normalize(text), voice=voice) if r.audio is not None]
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate([np.asarray(c, dtype=np.float32) for c in chunks])


def warmup() -> float:
    t0 = time.perf_counter()
    synthesize("Hello.")
    synthesize("नमस्ते।")
    return time.perf_counter() - t0
