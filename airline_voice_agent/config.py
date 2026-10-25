"""Central configuration. Override any value with an environment variable of the same name."""
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")


def _env(name: str, default):
    raw = os.getenv(name)
    if raw is None:
        return default
    return type(default)(raw)


# OpenAI-compatible providers use the `openai` SDK with a base_url; "claude" uses the Anthropic SDK.
PROVIDERS = {
    "gemini": {  # free tier: rate-limited; free-tier prompts may be used by Google to improve its models
        "kind": "openai_compatible",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "key_env": "GEMINI_API_KEY",
        # gemini-3.8-flash is Google's newest Flash but was overloaded on the free tier on 2026-10-16
        # (503 "high demand" / 40 s timeouts); 3.5 Flash answered reliably. Fallbacks are tried in order
        # when a model is overloaded, times out or is rate-limited before it starts answering.
        "model": "gemini-3.5-flash",
        "fallback_models": ["gemini-3.5-flash-lite"],
        "free": True,
    },
    "groq": {  # free tier: rate-limited, low daily token cap
        "kind": "openai_compatible",
        "base_url": "https://api.groq.com/openai/v1",
        "key_env": "GROQ_API_KEY",
        "model": "openai/gpt-oss-120b",
        "fallback_models": ["openai/gpt-oss-20b"],
        "free": True,
    },
    "claude": {  # paid API
        "kind": "anthropic",
        "key_env": "ANTHROPIC_API_KEY",
        "model": "claude-opus-5-5",
        "free": False,
    },
}


def require_api_key() -> None:
    """Exit with a clear message if the selected provider's key is missing."""
    import sys

    if not CONFIG.api_key():
        sys.exit(f"{CONFIG.provider_info['key_env']} is not set (PROVIDER={CONFIG.provider}). "
                 f"Copy .env.example to .env and add it.")


@dataclass(frozen=True)
class Config:
    # Paths
    pdf_dir: Path = ROOT / "data" / "pdfs"
    sources_file: Path = ROOT / "ingest" / "sources.yaml"
    processed_dir: Path = ROOT / "data" / "processed"
    tables_dir: Path = ROOT / "data" / "tables"
    lancedb_dir: Path = ROOT / "data" / "lancedb"
    policy_db: Path = ROOT / "data" / "policy.db"
    mock_bookings: Path = ROOT / "data" / "mock" / "bookings.json"  # fictional demo bookings
    calls_dir: Path = ROOT / "calls"

    # LLM. PROVIDER picks the backend (see PROVIDERS below); MODEL overrides its default model.
    provider: str = _env("PROVIDER", "gemini")
    model: str = _env("MODEL", "")
    live_effort: str = _env("LIVE_EFFORT", "low")
    offline_effort: str = _env("OFFLINE_EFFORT", "high")
    live_max_tokens: int = 16000  # includes the model's thinking tokens

    def __post_init__(self):
        if self.provider not in PROVIDERS:
            raise ValueError(f"PROVIDER must be one of {sorted(PROVIDERS)}, got {self.provider!r}")
        if not self.model:
            object.__setattr__(self, "model", PROVIDERS[self.provider]["model"])

    @property
    def provider_info(self) -> dict:
        return PROVIDERS[self.provider]

    def api_key(self) -> str | None:
        return os.getenv(self.provider_info["key_env"])

    # Retrieval
    # Multilingual (Hindi / Hinglish / English questions against English documents). The English-only setup
    # measured earlier was bge-small-en-v1.5 + ms-marco-MiniLM-L-6-v2 (see report/04_indexing_retrieval).
    embed_model: str = _env("EMBED_MODEL", "BAAI/bge-m3")
    rerank_model: str = _env("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
    rerank_max_length: int = 512
    # Off: on Roman-script Hinglish (what the Hinglish STT writes) both rerankers made ranking worse
    # (bge-m3 alone 8/10 top-1 in 34 ms; + v2-m3 rerank 5/10 in 1.7 s; + MiniLM 4/10). See report.
    rerank: bool = _env("RERANK", "0") == "1"
    # English turns are reranked with the English cross-encoder: on the bge-m3 index it lifts English top-1 from
    # 14/20 to 16/20 (+~100 ms) but drops Hinglish top-1 from 8/12 to 4/12, so it is applied by turn language.
    rerank_english: bool = _env("RERANK_ENGLISH", "1") == "1"
    rerank_english_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    candidates: int = 20
    top_k: int = 4
    chunk_min_tokens: int = 300
    chunk_max_tokens: int = 800
    chunk_overlap_tokens: int = 80

    # Speech. STT_ENGINE: hinglish (default; Oriserve Whisper fine-tune, writes Hindi/Hinglish/Indian English in
    # Roman script, runs on MPS via transformers) | english (mlx-whisper large-v3-turbo, English only).
    stt_engine: str = _env("STT_ENGINE", "hinglish")
    stt_hinglish_model: str = "Oriserve/Whisper-Hindi2Hinglish-Apex"
    stt_model: str = "mlx-community/whisper-large-v3-turbo"
    stt_vocab: str = (
        "IndiGo, 6E, PNR, DGCA, Saver fare, Flexi fare, web check-in, excess baggage, "
        "cabin baggage, checked baggage, refund, denied boarding, cancellation, rupees."
    )
    mic_rate: int = 16000
    tts_rate: int = 24000
    tts_voice: str = _env("TTS_VOICE", "af_heart")  # English replies
    tts_voice_hindi: str = _env("TTS_VOICE_HINDI", "hf_alpha")  # replies in Devanagari (hf_beta, hm_omega, hm_psi)
    tts_lang: str = "a"  # Kokoro: 'a' = American English, 'h' = Hindi
    vad_silence_ms: int = 600
    vad_min_speech_ms: int = 300
    idle_timeout_s: int = 60


CONFIG = Config()
