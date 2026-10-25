"""Retrieval spot checks and ablations, saved to report/04_indexing_retrieval/.

    python -m eval.retrieval_experiments                 # current config only
    python -m eval.retrieval_experiments --ablations     # also compare rerankers and the jurisdiction filter

Metrics per variant: hit@1 and hit@4 against retrieval_spot_checks.yaml, mean reciprocal rank,
and median / p90 retrieval latency (vector + keyword + rerank) on the M2.
"""
import argparse
import statistics
from pathlib import Path

import yaml

from agent.language import detect
from config import CONFIG
from rag.embed import rerank_scores, warmup
from rag.retriever import retrieve
from reporting import log_entry, md_table, save_artifact

CHECKS = Path(__file__).with_name("retrieval_spot_checks.yaml")

VARIANTS = {  # for the current index (bge-m3); the earlier bge-small index results are in the report history
    "current (MiniLM rerank on English turns only)": {"language": "auto"},
    "no rerank": {"rerank": False},
    "+ bge-reranker-v2-m3": {"rerank_model": "BAAI/bge-reranker-v2-m3"},
    "+ ms-marco-MiniLM-L-6-v2": {"rerank_model": "cross-encoder/ms-marco-MiniLM-L-6-v2"},
    "no-jurisdiction-filter": {"jurisdiction_filter": False},
}


def _matches(chunk, expect: list[str]) -> bool:
    for e in expect:
        doc_prefix, _, section_part = (p.strip() for p in e.partition("::"))
        if chunk.doc_id.startswith(doc_prefix) and section_part.lower() in chunk.section.lower():
            return True
    return False


def run_variant(name: str, kwargs: dict, checks: list[dict], k: int = 4) -> tuple[dict, list[dict]]:
    if kwargs.get("rerank_model"):
        rerank_scores("warmup", ["warmup"], kwargs["rerank_model"], kwargs.get("rerank_max_length"))
    rows, latencies, rr = [], [], []
    for c in checks:
        details = {"route_type": c["route"]} if c.get("route") else None
        kw = dict(kwargs)
        if kw.get("language") == "auto":
            kw["language"] = detect(c["q"])
        r = retrieve(c["q"], c.get("previous"), details, top_k=k, **kw)
        rank = next((i for i, ch in enumerate(r.chunks, 1) if _matches(ch, c["expect"])), None)
        rr.append(1 / rank if rank else 0.0)
        latencies.append(r.timings_ms["total"])
        rows.append({
            "question": c["q"],
            "rank": rank or "miss",
            "ms": round(r.timings_ms["total"]),
            "top1": f"{r.chunks[0].doc_id} › {r.chunks[0].section[:60]}" if r.chunks else "",
        })
    lat = sorted(latencies)
    summary = {
        "variant": name,
        "hit@1": f"{sum(row['rank'] == 1 for row in rows)}/{len(rows)}",
        f"hit@{k}": f"{sum(row['rank'] != 'miss' for row in rows)}/{len(rows)}",
        "MRR": round(statistics.mean(rr), 3),
        "median_ms": round(statistics.median(lat)),
        "p90_ms": round(lat[int(0.9 * (len(lat) - 1))]),
    }
    return summary, rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablations", action="store_true")
    ap.add_argument("--checks", default=CHECKS.name, help="spot-check file in eval/")
    args = ap.parse_args()

    checks = yaml.safe_load((CHECKS.parent / args.checks).read_text())
    warmup()
    variants = VARIANTS if args.ablations else {"current (MiniLM rerank on English turns only)": {"language": "auto"}}
    summaries, sections = [], []
    for name, kwargs in variants.items():
        summary, rows = run_variant(name, kwargs, checks)
        summaries.append(summary)
        sections.append(f"## {name}\n\n{md_table(rows)}")
        print(summary)

    config_note = (
        f"Config: embed `{CONFIG.embed_model}`, rerank {'on' if CONFIG.rerank else 'off'} by default, "
        f"{CONFIG.candidates} fused candidates → top {CONFIG.top_k}. {len(checks)} spot checks from "
        f"`eval/{args.checks}`. Latency is warm, on the M2 (MPS)."
    )
    body = f"# Retrieval experiments\n\n{config_note}\n\n## Summary\n\n{md_table(summaries)}\n" + "\n".join(sections)
    name = ("retrieval_ablations" if args.ablations else "retrieval_spot_checks") + (
        "_" + Path(args.checks).stem.replace("retrieval_spot_checks", "").strip("_") if args.checks != CHECKS.name else "")
    name = name.rstrip("_")
    path = save_artifact("retrieval", name, body)
    log_entry("retrieval", name.replace("_", " ").capitalize(), md_table(summaries), [path])


if __name__ == "__main__":
    main()
