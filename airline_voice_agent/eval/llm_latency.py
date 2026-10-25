"""First-text latency of the live LLM turn across models and reasoning effort (same question, same
passages, tools on). Saved to report/07_agent_text/.

    python -m eval.llm_latency [--models gemini-3.5-flash gemini-3.5-flash-lite] [--efforts minimal low] [--repeats 3]
"""
import argparse
import statistics

from agent.llm_openai import OpenAICompatAgent
from agent.session import CallSession
from config import CONFIG, require_api_key
from rag.embed import warmup
from rag.retriever import retrieve
from reporting import log_entry, md_table, save_artifact

QUESTIONS = [
    "My two hour domestic flight is delayed by three hours. What do I get?",
    "How much cabin baggage can I carry?",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=[CONFIG.model, *CONFIG.provider_info.get("fallback_models", [])])
    ap.add_argument("--efforts", nargs="+", default=["minimal", "low"])
    ap.add_argument("--repeats", type=int, default=2)
    args = ap.parse_args()
    require_api_key()
    warmup()
    retrieved = {q: retrieve(q).chunks for q in QUESTIONS}
    rows, detail = [], []
    for model in args.models:
        for effort in args.efforts:
            firsts, totals, tools, errors = [], [], [], 0
            for _ in range(args.repeats):
                for q in QUESTIONS:
                    agent = OpenAICompatAgent(model=model, effort=effort)
                    agent.models = [model]  # measure this model only, no fallback
                    s = CallSession(mode="latency")
                    s.add_user_turn(q, retrieved[q])
                    text = "".join(agent.respond(s))
                    t = agent.last_turn
                    if str(t.stop_reason).startswith("error") or t.fallbacks:
                        errors += 1
                        detail.append({"model": model, "effort": effort, "question": q, "first_text_ms": None,
                                       "tools": [], "answer": f"ERROR {t.stop_reason} {t.fallbacks}"})
                        continue
                    firsts.append(t.latency_ms.get("first_text") or t.latency_ms["total"])
                    totals.append(t.latency_ms["total"])
                    tools.append(len(t.tool_calls))
                    detail.append({"model": model, "effort": effort, "question": q, "first_text_ms": firsts[-1],
                                   "total_ms": totals[-1], "tools": [c["name"] for c in t.tool_calls], "answer": text})
            rows.append({"model": model, "effort": effort, "n": len(firsts), "errors": errors,
                         "median_first_text_ms": round(statistics.median(firsts)) if firsts else None,
                         "max_first_text_ms": max(firsts) if firsts else None,
                         "median_total_ms": round(statistics.median(totals)) if totals else None,
                         "mean_tool_calls": round(statistics.mean(tools), 2) if tools else None})
            print(rows[-1])
    body = (f"# LLM latency — {CONFIG.provider}\n\nFirst text = time from sending the turn until the first spoken "
            f"text arrives (includes thinking and any tool round). Questions: {QUESTIONS}. Retrieval excluded.\n\n"
            f"{md_table(rows)}\n## Answers\n\n{md_table(detail, ['model', 'effort', 'first_text_ms', 'tools', 'answer'])}")
    path = save_artifact("agent", "llm_latency", body)
    log_entry("agent", f"LLM latency comparison ({CONFIG.provider})", md_table(rows), [path])


if __name__ == "__main__":
    main()
