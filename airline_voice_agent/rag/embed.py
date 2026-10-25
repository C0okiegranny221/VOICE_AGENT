"""Local embedding + reranking models (loaded once, on the M2 GPU when available)."""
from functools import lru_cache

import numpy as np
import torch

from config import CONFIG

# English bge v1.5 models want this instruction on queries (not passages); bge-m3 uses none.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def _query_prefix() -> str:
    return QUERY_INSTRUCTION if "-en-v1.5" in CONFIG.embed_model else ""


def _device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


@lru_cache(maxsize=1)
def embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(CONFIG.embed_model, device=_device())


@lru_cache(maxsize=4)
def reranker(name: str | None = None, max_length: int | None = None):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(name or CONFIG.rerank_model, device=_device(), max_length=max_length or CONFIG.rerank_max_length)


def embed_passages(texts: list[str]) -> np.ndarray:
    return embedder().encode(texts, batch_size=8, normalize_embeddings=True, show_progress_bar=False)


def embed_query(text: str) -> np.ndarray:
    return embedder().encode([_query_prefix() + text], normalize_embeddings=True)[0]


def rerank_scores(query: str, passages: list[str], model: str | None = None, max_length: int | None = None) -> list[float]:
    if not passages:
        return []
    return reranker(model, max_length).predict([(query, p) for p in passages], show_progress_bar=False).tolist()


def warmup() -> None:
    embed_query("warmup")
    if CONFIG.rerank:
        rerank_scores("warmup", ["warmup"])
    if CONFIG.rerank_english:
        rerank_scores("warmup", ["warmup"], CONFIG.rerank_english_model)
