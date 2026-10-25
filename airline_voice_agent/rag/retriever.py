"""Hybrid retrieval: BM25 + vector search (LanceDB), reciprocal-rank fusion, cross-encoder rerank.

CLI:  python -m rag.retriever "cabin baggage allowance international" [--route international]
"""
import argparse
import re
import time
from dataclasses import dataclass, field
from functools import lru_cache

import lancedb

from config import CONFIG
from rag.embed import embed_query, rerank_scores

TABLE = "chunks"
RRF_K = 60
COLUMNS = [
    "chunk_id", "doc_id", "title", "source_url", "retrieved_on",
    "route_scope", "jurisdiction", "page", "section", "topic", "text",
]
# Foreign passenger-rights notices are only retrieved when the caller mentions that region.
JURISDICTION_TERMS = {
    "europe": r"\b(europe|european|eu|eu261|amsterdam|paris|london|manchester|istanbul|athens|copenhagen)\b",
    "thailand": r"\b(thailand|thai|bangkok|phuket|krabi|chiang mai)\b",
    "vietnam": r"\b(vietnam|vietnamese|hanoi|ho chi minh|saigon|da nang)\b",
}


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    source_url: str
    retrieved_on: str
    route_scope: str
    jurisdiction: str
    page: int
    section: str
    topic: str
    text: str
    score: float = 0.0


@dataclass
class Retrieval:
    query: str
    chunks: list[Chunk]
    timings_ms: dict = field(default_factory=dict)


@lru_cache(maxsize=1)
def _table():
    return lancedb.connect(str(CONFIG.lancedb_dir)).open_table(TABLE)


def build_query(question: str, previous_question: str | None = None, caller_details: dict | None = None) -> str:
    """Make follow-ups self-contained: 'what about infants?' + the previous question + known details."""
    parts = [question]
    if previous_question:
        parts.append(previous_question)
    if caller_details:
        parts.extend(str(v) for k, v in caller_details.items() if v and k in ("route_type", "fare_type", "passenger_type"))
    return " ".join(parts)


def build_filter(query: str, route_type: str | None, jurisdiction_filter: bool = True) -> str:
    """SQL filter: matching route scope (if known) and Indian rules plus any region the caller named."""
    regions = ["india"] + [j for j, pattern in JURISDICTION_TERMS.items() if re.search(pattern, query, re.I)]
    if not jurisdiction_filter:
        regions = ["india", *JURISDICTION_TERMS]
    clauses = ["jurisdiction IN (" + ", ".join(f"'{r}'" for r in regions) + ")"]
    if route_type in ("domestic", "international"):
        clauses.append(f"route_scope IN ('{route_type}', 'both')")
    return " AND ".join(clauses)


def _rows(builder, where: str, limit: int, score_column: str) -> list[dict]:
    # Selecting the score column explicitly opts in to LanceDB's future projection behaviour.
    return builder.where(where, prefilter=True).select(COLUMNS + [score_column]).limit(limit).to_list()


def retrieve(
    question: str,
    previous_question: str | None = None,
    caller_details: dict | None = None,
    top_k: int | None = None,
    rerank_model: str | None = None,
    rerank_max_length: int | None = None,
    jurisdiction_filter: bool = True,
    rerank: bool | None = None,
    language: str | None = None,
) -> Retrieval:
    """rerank_model / rerank_max_length / jurisdiction_filter / rerank exist for experiments; defaults come from config."""
    top_k = top_k or CONFIG.top_k
    t0 = time.perf_counter()
    query = build_query(question, previous_question, caller_details)
    where = build_filter(query, (caller_details or {}).get("route_type"), jurisdiction_filter)
    table = _table()

    vector_hits = _rows(table.search(embed_query(query).tolist(), query_type="vector"), where, CONFIG.candidates, "_distance")
    t1 = time.perf_counter()
    try:
        keyword_hits = _rows(table.search(query, query_type="fts"), where, CONFIG.candidates, "_score")
    except Exception:
        keyword_hits = []  # e.g. a query of only stop words
    t2 = time.perf_counter()

    # Reciprocal-rank fusion of the two result lists.
    fused: dict[str, float] = {}
    rows: dict[str, dict] = {}
    for hits in (vector_hits, keyword_hits):
        for rank, row in enumerate(hits):
            fused[row["chunk_id"]] = fused.get(row["chunk_id"], 0.0) + 1.0 / (RRF_K + rank + 1)
            rows[row["chunk_id"]] = row
    candidates = sorted(fused, key=fused.get, reverse=True)[: CONFIG.candidates]

    # Rerank against the caller's own words (plus the previous question, for follow-ups).
    rerank_query = f"{question} ({previous_question})" if previous_question else question
    # Explicit rerank flag > a named reranker (experiments) > English-turn reranking > the global setting.
    if rerank is None and not rerank_model and language == "english" and CONFIG.rerank_english:
        rerank, rerank_model = True, CONFIG.rerank_english_model
    do_rerank = rerank if rerank is not None else (bool(rerank_model) or CONFIG.rerank)
    if do_rerank:
        passages = [f"{rows[c]['section']}\n{rows[c]['text']}" for c in candidates]
        scores = rerank_scores(rerank_query, passages, rerank_model, rerank_max_length)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)[:top_k]
    else:  # fused keyword + vector order only
        ranked = [(c, fused[c]) for c in candidates[:top_k]]
    t3 = time.perf_counter()

    chunks = [Chunk(**{k: rows[cid][k] for k in COLUMNS}, score=float(s)) for cid, s in ranked]
    timings = {"vector": (t1 - t0) * 1000, "keyword": (t2 - t1) * 1000, "rerank": (t3 - t2) * 1000, "total": (t3 - t0) * 1000}
    return Retrieval(query=query, chunks=chunks, timings_ms={k: round(v, 1) for k, v in timings.items()})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--previous")
    ap.add_argument("--route", choices=["domestic", "international"])
    ap.add_argument("-k", type=int, default=CONFIG.top_k)
    args = ap.parse_args()

    from rag.embed import warmup

    warmup()
    r = retrieve(args.question, args.previous, {"route_type": args.route} if args.route else None, args.k)
    print(f"query: {r.query}\ntimings (ms): {r.timings_ms}")
    for i, c in enumerate(r.chunks, 1):
        print(f"\n[{i}] {c.score:+.2f}  {c.title} › {c.section}  (p{c.page}, {c.route_scope}, {c.jurisdiction}, {c.topic})")
        print("    " + c.text[:300].replace("\n", " "))


if __name__ == "__main__":
    main()
