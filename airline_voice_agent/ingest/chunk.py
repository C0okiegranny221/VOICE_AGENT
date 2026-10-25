"""Sections -> retrieval chunks with metadata.

- Small consecutive sections under the same level-1 clause are merged up to chunk_min_tokens.
- Sections over chunk_max_tokens are split on paragraph (then sentence) boundaries with overlap.
- search_text = "<title> › <heading path>\n<text>" is what gets embedded and keyword-indexed.
Token counts are approximated as words * 1.3 (no tokenizer needed for sizing).
"""
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CONFIG  # noqa: E402
from reporting import log_entry, md_table, save_artifact  # noqa: E402

MIN_CHUNK_TOKENS = 25

TOPIC_RULES = [  # first match on heading, else most keyword hits in text
    ("baggage", r"baggage|luggage|cabin bag|hand bag|excess|lithium|firearm|liquid"),
    ("refund", r"refund|look[- ]in|credit shell"),
    ("change_cancellation", r"cancel|change|name correction|reschedul|no[- ]show"),
    ("delay_disruption", r"delay|denied boarding|downgrad|disrupt|compensat|right to care|schedule"),
    ("special_assistance", r"disab|wheelchair|medical|pregnan|minor|infant|special assistance|oxygen|autism"),
    ("check_in_boarding", r"check[- ]in|boarding|identity|travel document|visa"),
    ("fees_tariff", r"tariff|fare|tax|fee|charge|payment|invoice|currency"),
    ("liability_claims", r"liabilit|claim|damage|lost|grievance|dispute|redress|complaint"),
    ("definitions", r"definition|interpretation|means"),
]


def approx_tokens(text: str) -> int:
    return int(len(text.split()) * 1.3)


def classify_topic(heading: str, text: str) -> str:
    for topic, pattern in TOPIC_RULES:
        if re.search(pattern, heading, re.I):
            return topic
    hits = {t: len(re.findall(p, text, re.I)) for t, p in TOPIC_RULES}
    best = max(hits, key=hits.get)
    return best if hits[best] >= 2 else "general"


SENTENCE_RE = re.compile(r"(?<=[.;:])\s+(?=[A-Z(])")


def _split_units(text: str, max_tokens: int) -> list[str]:
    """Paragraphs, with any paragraph over max_tokens broken into sentences."""
    units = []
    for para in text.split("\n"):
        if approx_tokens(para) <= max_tokens:
            units.append(para)
        else:
            units.extend(s for s in SENTENCE_RE.split(para) if s)
    return units


def _tail(units: list[str], overlap_tokens: int) -> str:
    """The last sentences of `units`, up to overlap_tokens, to repeat at the start of the next piece."""
    sentences = [s for u in units for s in SENTENCE_RE.split(u) if s]
    tail: list[str] = []
    for s in reversed(sentences):
        if tail and approx_tokens(" ".join([s] + tail)) > overlap_tokens:
            break
        tail.insert(0, s)
    return " ".join(tail) if approx_tokens(" ".join(tail)) <= overlap_tokens else ""


def split_long(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    if approx_tokens(text) <= max_tokens:
        return [text]
    pieces, cur = [], []
    for unit in _split_units(text, max_tokens):
        if cur and approx_tokens("\n".join(cur + [unit])) > max_tokens:
            pieces.append("\n".join(cur))
            tail = _tail(cur, overlap_tokens)
            cur = [tail] if tail and approx_tokens(tail + unit) <= max_tokens else []
        cur.append(unit)
    if cur:
        pieces.append("\n".join(cur))
    return pieces


def merge_small(sections: list[dict], min_tokens: int, max_tokens: int) -> list[dict]:
    """Merge consecutive small sections that share a level-1 clause."""
    merged = []
    for sec in sections:
        prev = merged[-1] if merged else None
        same_clause = prev and prev["heading_path"][:2] == sec["heading_path"][:2]
        if (
            prev
            and same_clause
            and approx_tokens(prev["text"]) < min_tokens
            and approx_tokens(prev["text"] + sec["text"]) <= max_tokens
        ):
            label = sec["heading_path"][-1]
            prev["text"] += f"\n{label}\n{sec['text']}" if len(sec["heading_path"]) > 2 else f"\n{sec['text']}"
            prev["merged_headings"].append(label)
        else:
            merged.append({**sec, "merged_headings": []})
    return merged


def _jurisdiction(doc: dict, path: list[str]) -> str:
    """Whose rules a chunk states. Passenger Rights is split by region tab (Europe, Thailand, ...);
    everything else is Indian (IndiGo CoC under DGCA rules)."""
    if doc.get("jurisdiction") == "by_heading" and len(path) > 1:
        return path[1].split(" / ")[0].strip().lower()
    return doc.get("jurisdiction") or "india"


def chunk_document(processed: dict) -> list[dict]:
    doc = processed["doc"]
    sections = merge_small(processed["sections"], CONFIG.chunk_min_tokens, CONFIG.chunk_max_tokens)
    chunks = []
    for sec in sections:
        path = sec["heading_path"]
        # A merged section is labelled with all its headings: "11. Baggage / 11.1 ... / 11.2 ...".
        if sec["merged_headings"]:
            path = path[:-1] + [" / ".join([path[-1]] + sec["merged_headings"])[:160]]
        heading = " › ".join(path)
        for i, piece in enumerate(split_long(sec["text"], CONFIG.chunk_max_tokens, CONFIG.chunk_overlap_tokens)):
            if approx_tokens(piece) < MIN_CHUNK_TOKENS:
                continue  # stray fragments (page furniture, link labels)
            key = f"{doc['doc_id']}|{heading}|{i}"
            chunks.append(
                {
                    "chunk_id": hashlib.sha1(key.encode()).hexdigest()[:12],
                    "doc_id": doc["doc_id"],
                    "title": doc["title"],
                    "source_url": doc["source_url"] or "",
                    "retrieved_on": doc["retrieved_on"] or "",
                    "route_scope": doc["route_scope"],
                    "jurisdiction": _jurisdiction(doc, path),
                    "page": sec["page"],
                    "section": " › ".join(path[1:]) or doc["title"],
                    "topic": classify_topic(" ".join(path[1:]), piece),
                    "text": piece,
                    "search_text": f"{heading}\n{piece}",
                    "tokens": approx_tokens(piece),
                }
            )
    return chunks


def chunk_all() -> list[dict]:
    chunks = []
    for f in sorted(CONFIG.processed_dir.glob("*.json")):
        doc_chunks = chunk_document(json.loads(f.read_text()))
        print(f"chunked {f.stem}: {len(doc_chunks)} chunks")
        chunks.extend(doc_chunks)
    ids = [c["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids)), "duplicate chunk_id"
    (CONFIG.processed_dir / "chunks.jsonl").write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in chunks))
    _report(chunks)
    return chunks


def _report(chunks: list[dict]) -> None:
    import collections
    import random

    tokens = sorted(c["tokens"] for c in chunks)
    pct = lambda q: tokens[min(len(tokens) - 1, int(q * len(tokens)))]  # noqa: E731
    counts = lambda key: md_table([{key: k, "chunks": v} for k, v in collections.Counter(c[key] for c in chunks).most_common()])  # noqa: E731
    per_doc = md_table([{"doc_id": d, "chunks": n} for d, n in collections.Counter(c["doc_id"] for c in chunks).items()])
    stats = (
        f"Chunks: **{len(chunks)}**. Settings: min {CONFIG.chunk_min_tokens}, max {CONFIG.chunk_max_tokens}, "
        f"overlap {CONFIG.chunk_overlap_tokens} tokens (approx. words × 1.3), fragments under {MIN_CHUNK_TOKENS} dropped.\n\n"
        f"Tokens min / p10 / median / p90 / max: {tokens[0]} / {pct(.1)} / {pct(.5)} / {pct(.9)} / {tokens[-1]}\n\n"
        f"### Per document\n\n{per_doc}\n### Topics\n\n{counts('topic')}\n"
        f"### Route scope\n\n{counts('route_scope')}\n### Jurisdiction\n\n{counts('jurisdiction')}"
    )
    rng = random.Random(7)
    samples = "\n".join(
        f"### {c['chunk_id']} · {c['doc_id']} · p{c['page']} · {c['topic']} · {c['tokens']} tok\n\n```\n{c['search_text'][:1200]}\n```\n"
        for c in rng.sample(chunks, min(8, len(chunks)))
    )
    stats_file = save_artifact("chunking", "chunk_stats", "# Chunk statistics\n\n" + stats)
    sample_file = save_artifact("chunking", "chunk_samples", "# Random chunk samples (seed 7)\n\n" + samples)
    snapshot = save_artifact("chunking", "chunks_snapshot", "\n".join(json.dumps(c, ensure_ascii=False) for c in chunks), "jsonl")
    log_entry("chunking", f"Chunked into {len(chunks)} chunks",
              f"Tokens median {pct(.5)}, max {tokens[-1]}.", [stats_file, sample_file, snapshot])


if __name__ == "__main__":
    chunk_all()
