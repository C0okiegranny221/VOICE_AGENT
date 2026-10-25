"""End-to-end answer and memory eval through the text pipeline (retrieval -> LLM -> tools), graded
by rules and by an LLM judge (the configured provider). Results go to report/09_eval/.

    python -m eval.run_eval                 # prints the plan and estimated cost, then exits
    python -m eval.run_eval --yes           # runs everything
    python -m eval.run_eval --yes --only bag-01 dly-02 --skip-conversations

Metrics
- must_contain: required figures/terms in the answer (number words count, e.g. "fifteen" = 15)
- source_hit:   an expected source was retrieved or an expected policy table was used via a tool
- judge:        correct / partial / wrong against the expected answer, plus grounded and voice-style flags
"""
import argparse
import json
import re
import statistics
import time
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from agent import structured
from agent.llm import make_agent
from agent.pipeline import complete_turn, run_turn
from agent.session import CallSession
from voice.tts import is_hindi
from config import CONFIG, require_api_key
from rag.embed import warmup
from reporting import log_entry, md_table, save_artifact

HERE = Path(__file__).parent
TOOL_TABLE = {
    "get_baggage_allowance": "baggage_allowance", "get_baggage_charges": "baggage_charges",
    "get_change_cancellation_rules": "change_cancellation_rules", "get_refund_timeline": "refund_timelines",
    "get_delay_entitlements": "delay_entitlements", "get_denied_boarding_compensation": "denied_boarding_compensation",
    "get_cancellation_compensation": "cancellation_compensation", "get_downgrade_reimbursement": "downgrade_reimbursement",
}
# Rough per-call cost on claude-opus-5-5 ($4 / $20 per MTok), from expected prompt sizes; printed before
# running. Free-tier providers cost nothing but are rate-limited (the SDK retries 429s with backoff).
EST_COST_PER_ANSWER, EST_COST_PER_JUDGE = 0.05, 0.02
FREE_TIER_PACE_S = 8  # pause between items on free tiers to stay under per-minute request limits

UNITS = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen "
                                    "fourteen fifteen sixteen seventeen eighteen nineteen".split())}
TENS = {w: 10 * i for i, w in enumerate("_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()) if w != "_"}
SCALES = {"hundred": 100, "thousand": 1000, "lakh": 100000}


def words_to_digits(text: str) -> str:
    """'rupees eight thousand' -> 'rupees 8000'; 'twenty one' -> '21'. Also drops thousands separators."""
    out, num, cur, active = [], 0, 0, False
    text = re.sub(r"\ba (hundred|thousand|lakh)\b", r"one \1", text.lower().replace("-", " "))
    tokens = re.findall(r"[a-z]+|\d[\d,]*(?:\.\d+)?|\S", text)

    def flush():
        nonlocal num, cur, active
        if active:
            out.append(str(num + cur))
        num, cur, active = 0, 0, False

    for tok in tokens:
        if tok in UNITS or tok in TENS:
            cur += UNITS.get(tok, 0) + TENS.get(tok, 0)
            active = True
        elif tok in SCALES and active:
            if SCALES[tok] == 100:
                cur *= 100
            else:
                num += cur * SCALES[tok]
                cur = 0
        elif tok == "and" and active:
            continue
        else:
            flush()
            out.append(tok.replace(",", "") if tok[0].isdigit() else tok)
    flush()
    return " ".join(out)


def contains_all(answer: str, required: list[str]) -> tuple[bool, list[str]]:
    norm = words_to_digits(answer)
    missing = [r for r in required if words_to_digits(r) not in norm]
    return not missing, missing


def source_hit(expected: list[str], turn: dict) -> bool | None:
    if not expected:
        return None
    retrieved = [r["source"] for r in turn["retrieved"]]
    tables = {TOOL_TABLE.get(c["name"]) for c in turn["tool_calls"]}
    for e in expected:
        prefix, _, part = (p.strip() for p in e.partition("::"))
        if prefix == "table" and part in tables:
            return True
        if any(src.startswith(prefix) and part.lower() in src.lower() for src in retrieved):
            return True
    return False


class Verdict(BaseModel):
    verdict: str = Field(description="correct | partial | wrong")
    grounded: bool = Field(description="Every policy fact and figure in the answer is supported by the expected answer or is a reasonable, non-contradicting addition")
    behaviour_ok: bool = Field(description="For clarify items: asked for the missing detail. For unanswerable: said it doesn't know / offered an agent. Else: answered.")
    voice_style_ok: bool = Field(description="Short (1-3 sentences), no lists/markdown/URLs, suitable to be read aloud")
    reason: str


JUDGE_SYSTEM = """You grade answers from an airline policy voice assistant against a reference. Judge meaning, not \
wording: numbers may be written as words. 'partial' = right direction but missing or wrong in a detail that matters \
to the caller. 'wrong' = contradicts the reference, gives a wrong figure, or fails the expected behaviour. Extra \
correct caveats (e.g. extraordinary circumstances, confirm at booking) are fine."""


def judge(question: str, expected: str, answer: str, item_type: str, context: str = "", language: str | None = None) -> Verdict:
    lang = f"<expected_reply_language>{language} (Hindi/Hinglish in Devanagari script)</expected_reply_language>\n" if language == "hinglish" else ""
    prompt = (f"{context}{lang}<question>{question}</question>\n<item_type>{item_type}</item_type>\n"
              f"<reference_answer>{expected}</reference_answer>\n<assistant_answer>{answer}</assistant_answer>")
    r = structured.parse(JUDGE_SYSTEM, prompt, Verdict, effort="medium", max_tokens=4000)
    if r.output is None:
        return Verdict(verdict="wrong", grounded=False, behaviour_ok=False, voice_style_ok=False,
                       reason=f"judge produced no verdict (stop_reason={r.stop_reason})")
    return r.output


def ask(session: CallSession, agent, text: str) -> dict:
    for _ in run_turn(session, agent, text):
        pass
    turn = complete_turn(session, agent)
    if CONFIG.provider_info["free"]:
        time.sleep(FREE_TIER_PACE_S)
    return turn


def run_questions(items) -> list[dict]:
    rows = []
    for item in items:
        session, agent = CallSession(mode="eval"), make_agent()  # fresh agent: each item is a new call
        turn = ask(session, agent, item["q"])
        ok, missing = contains_all(turn["assistant_text"], item.get("must_contain", []))
        v = judge(item["q"], item["expected_answer"], turn["assistant_text"], item.get("type", "answerable"),
                  language=item.get("expect_language"))
        lang_ok = None if not item.get("expect_language") else (
            is_hindi(turn["assistant_text"]) if item["expect_language"] == "hinglish" else not is_hindi(turn["assistant_text"]))
        rows.append({
            "id": item["id"], "type": item.get("type", "answerable"), "question": item["q"],
            "answer": turn["assistant_text"], "expected": item["expected_answer"],
            "must_contain_ok": ok, "missing": missing, "source_hit": source_hit(item.get("expected_sources", []), turn),
            "language_ok": lang_ok, "detected_language": turn.get("language"),
            "tools": [c["name"] for c in turn["tool_calls"]], "verdict": v.verdict, "grounded": v.grounded,
            "behaviour_ok": v.behaviour_ok, "voice_style_ok": v.voice_style_ok, "judge_reason": v.reason,
            "first_text_ms": turn["latency_ms"].get("llm_first_text"), "retrieval_ms": turn["latency_ms"].get("retrieval"),
            "cache_read_tokens": turn["usage"].get("cache_read_input_tokens"), "call_id": session.call_id,
            "model": turn.get("model"), "fallbacks": len(turn.get("model_fallbacks") or []),
        })
        print(f"{item['id']:8s} {v.verdict:8s} must={ok} src={rows[-1]['source_hit']} tools={rows[-1]['tools']}")
    return rows


def run_conversations(convs) -> list[dict]:
    rows = []
    for conv in convs:
        session, agent = CallSession(mode="eval"), make_agent()
        history = []
        for n, t in enumerate(conv["turns"], 1):
            turn = ask(session, agent, t["say"])
            history.append(f"CALLER: {t['say']}\nASSISTANT: {turn['assistant_text']}")
            if "check" not in t:
                continue
            ok, missing = contains_all(turn["assistant_text"], t.get("must_contain", []))
            context = "<conversation_so_far>\n" + "\n".join(history[:-1]) + "\n</conversation_so_far>\n"
            v = judge(t["say"], t["check"], turn["assistant_text"], "memory", context)
            rows.append({"id": conv["id"], "turn": n, "said": t["say"], "answer": turn["assistant_text"], "check": t["check"],
                         "model": turn.get("model"),
                         "must_contain_ok": ok, "missing": missing, "verdict": v.verdict, "judge_reason": v.reason,
                         "call_id": session.call_id})
            print(f"{conv['id']:30s} turn {n}: {v.verdict} must={ok}")
    return rows


def _rate(rows, key, value=True):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return f"{sum(v == value for v in vals)}/{len(vals)}" if vals else "-"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true", help="actually run (spends API credit)")
    ap.add_argument("--only", nargs="*", help="question ids to run")
    ap.add_argument("--skip-conversations", action="store_true")
    ap.add_argument("--questions", default="questions.yaml", help="question file in eval/")
    ap.add_argument("--conversations", default="conversations.yaml", help="conversation file in eval/")
    ap.add_argument("--tag", default="answer_eval", help="report artifact name")
    args = ap.parse_args()

    items = yaml.safe_load((HERE / args.questions).read_text())
    if args.only:
        items = [i for i in items if i["id"] in args.only]
    convs = [] if args.skip_conversations else yaml.safe_load((HERE / args.conversations).read_text())
    n_turns = sum(len(c["turns"]) for c in convs)
    n_checks = sum(1 for c in convs for t in c["turns"] if "check" in t)
    est = len(items) * (EST_COST_PER_ANSWER + EST_COST_PER_JUDGE) + n_turns * EST_COST_PER_ANSWER + n_checks * EST_COST_PER_JUDGE
    n_requests = 2 * (len(items) + n_turns) + len(items) + n_checks  # answers incl. ~1 tool round each, plus judges
    cost = ("free tier - no charge, but rate-limited; expect a few minutes of retries if the per-minute limit is hit"
            if CONFIG.provider_info["free"] else f"about ${est:.2f} (rough)")
    print(f"Plan: {len(items)} questions + {len(convs)} conversations ({n_turns} turns, {n_checks} checks) on "
          f"{CONFIG.provider}:{CONFIG.model}, ~{n_requests} requests.\nCost: {cost}.")
    if not args.yes:
        print("Re-run with --yes to start.")
        return

    require_api_key()
    warmup()
    q_rows = run_questions(items)
    c_rows = run_conversations(convs) if convs else []

    answerable = [r for r in q_rows if r["type"] == "answerable"]
    lat = [r["first_text_ms"] for r in q_rows if r["first_text_ms"]]
    summary = [{
        "questions": len(q_rows),
        "correct": _rate(q_rows, "verdict", "correct"), "partial": _rate(q_rows, "verdict", "partial"),
        "must_contain": _rate(q_rows, "must_contain_ok"), "source_hit": _rate(q_rows, "source_hit"),
        "grounded": _rate(q_rows, "grounded"), "behaviour_ok": _rate(q_rows, "behaviour_ok"),
        "voice_style_ok": _rate(q_rows, "voice_style_ok"), "language_ok": _rate(q_rows, "language_ok"),
        "answerable_correct": _rate(answerable, "verdict", "correct"),
        "median_first_text_ms": round(statistics.median(lat)) if lat else None,
    }]
    mem_summary = [{"memory_checks": len(c_rows), "correct": _rate(c_rows, "verdict", "correct"),
                    "must_contain": _rate(c_rows, "must_contain_ok")}] if c_rows else []

    cols = ["id", "type", "verdict", "must_contain_ok", "source_hit", "grounded", "behaviour_ok", "voice_style_ok",
            "language_ok", "tools", "first_text_ms", "model"]
    models_used = sorted({r["model"] for r in q_rows + c_rows if r.get("model")})
    body = (f"# Answer eval ({args.tag}) — {CONFIG.provider}, configured {CONFIG.model}, effort {CONFIG.live_effort}\n\n"
            f"Models that actually answered: {models_used} (free-tier fallback when the primary is over quota). "
            f"Judge: same provider. STT engine {CONFIG.stt_engine}, embeddings {CONFIG.embed_model}, rerank "
            f"{'on' if CONFIG.rerank else 'off'}. Question set: `eval/{args.questions}`, conversations `eval/{args.conversations}` "
            f"(draft expected answers, pending human review).\n\n## Summary\n\n{md_table(summary)}\n"
            + (f"## Memory\n\n{md_table(mem_summary)}\n" if mem_summary else "")
            + f"## Per question\n\n{md_table(q_rows, cols)}\n## Answers and judge reasons\n\n"
            + "\n".join(f"### {r['id']} — {r['verdict']}\n\n**Q:** {r['question']}\n\n**A:** {r['answer']}\n\n"
                        f"**Expected:** {r['expected']}\n\n**Judge:** {r['judge_reason']}\n" for r in q_rows)
            + ("\n## Memory checks\n\n" + "\n".join(
                f"### {r['id']} (turn {r['turn']}) — {r['verdict']}\n\n**Caller:** {r['said']}\n\n**A:** {r['answer']}\n\n"
                f"**Check:** {r['check']}\n\n**Judge:** {r['judge_reason']}\n" for r in c_rows) if c_rows else ""))
    path = save_artifact("eval", args.tag, body)
    save_artifact("eval", args.tag, {"summary": summary, "memory_summary": mem_summary, "questions": q_rows, "memory": c_rows})
    log_entry("eval", f"{args.tag} on {CONFIG.provider} (answered by {', '.join(models_used)})", md_table(summary) + ("\n" + md_table(mem_summary) if mem_summary else ""), [path])
    print(md_table(summary))
    if mem_summary:
        print(md_table(mem_summary))


if __name__ == "__main__":
    main()
