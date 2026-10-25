"""Post-call summary: one structured-output LLM call over the saved transcript, then a check that
every figure quoted in the summary was actually said by the agent or returned by a tool."""
import json
import re

from pydantic import BaseModel, Field

from agent import structured
from agent.session import CallSession
from config import CONFIG

SUMMARY_SYSTEM = """You summarise a completed customer-service call between a caller and an IndiGo policy voice \
assistant (a demo that answers from public policy documents). Use only what is in the transcript: do not add \
policy facts, figures or outcomes that were not said or returned by a tool. If the agent could not answer \
something, list it as unresolved. Write plainly for a human supervisor, in English, even if the call was in Hindi \
or Hinglish (quote the caller's words in their original language where it helps)."""


class QA(BaseModel):
    question: str = Field(description="What the caller asked, in plain words")
    answer_given: str = Field(description="What the agent told them, briefly")
    sources: list[str] = Field(description="Policy sources used (document and section), from retrieved passages or tools")
    figures_quoted: list[str] = Field(description="Exact figures the agent stated, e.g. '15 kg', 'INR 900', '7 working days'")


class Detail(BaseModel):
    field: str = Field(description="e.g. route_type, fare, passenger_type, payment_mode, flight")
    value: str


class CallSummary(BaseModel):
    caller_intents: list[str]
    caller_details: list[Detail] = Field(description="Facts the caller gave")
    qa: list[QA]
    unresolved: list[str] = Field(description="Questions the agent could not answer or that need a human")
    escalation_recommended: bool
    caller_sentiment: str = Field(description="e.g. 'neutral', 'frustrated → satisfied'")
    follow_up_actions: list[str]
    narrative_summary: str = Field(description="3-5 sentences")


def _transcript_text(session: CallSession) -> str:
    lines = []
    for t in session.turns:
        lines.append(f"[Turn {t['turn']}] CALLER: {t['user_text']}")
        for c in t.get("tool_calls", []):
            lines.append(f"  TOOL {c['name']}({json.dumps(c['input'])}) -> {json.dumps(c['output'], ensure_ascii=False)[:1500]}")
        if t.get("retrieved"):
            lines.append("  PASSAGES USED: " + "; ".join(r["source"] for r in t["retrieved"]))
        lines.append(f"[Turn {t['turn']}] AGENT: {t['assistant_text']}")
    return "\n".join(lines)


_NUM_RE = re.compile(r"\d+(?:[.,]\d+)*")


def _numbers(text: str) -> set[str]:
    return {n.replace(",", "") for n in _NUM_RE.findall(text)}


def check_figures(summary: dict, session: CallSession) -> dict:
    """Every number in figures_quoted must appear in the agent's speech or a tool result."""
    evidence = " ".join(t["assistant_text"] for t in session.turns)
    evidence += " " + " ".join(json.dumps(c["output"]) for t in session.turns for c in t.get("tool_calls", []))
    known = _numbers(evidence)
    unsupported = []
    for qa in summary.get("qa", []):
        for fig in qa.get("figures_quoted", []):
            missing = _numbers(fig) - known
            if missing:
                unsupported.append({"figure": fig, "numbers_not_found": sorted(missing)})
    return {"figures_checked": sum(len(q.get("figures_quoted", [])) for q in summary.get("qa", [])),
            "unsupported": unsupported, "passed": not unsupported}


def to_markdown(summary: dict) -> str:
    out = [f"# Call summary — {summary['call_id']}", "",
           f"Started {summary['started_at']} · ended {summary['ended_at']} · {summary['turns']} caller turns · mode {summary['mode']}", "",
           summary["narrative_summary"], "", "## Caller intents", ""]
    out += [f"- {i}" for i in summary["caller_intents"]] or ["- (none)"]
    if summary["caller_details"]:
        out += ["", "## Caller details", ""] + [f"- **{d['field']}**: {d['value']}" for d in summary["caller_details"]]
    out += ["", "## Questions and answers", ""]
    for n, qa in enumerate(summary["qa"], 1):
        out += [f"{n}. **Q:** {qa['question']}", f"   **A:** {qa['answer_given']}"]
        if qa["figures_quoted"]:
            out.append(f"   *Figures:* {', '.join(qa['figures_quoted'])}")
        if qa["sources"]:
            out.append(f"   *Sources:* {'; '.join(qa['sources'])}")
    out += ["", "## Unresolved", ""] + ([f"- {u}" for u in summary["unresolved"]] or ["- none"])
    out += ["", f"**Escalation recommended:** {'yes' if summary['escalation_recommended'] else 'no'} · "
                f"**Sentiment:** {summary['caller_sentiment']}", "", "## Follow-up actions", ""]
    out += [f"- {a}" for a in summary["follow_up_actions"]] or ["- none"]
    v = summary["validation"]
    out += ["", "## Figure check", "",
            f"{v['figures_checked']} figures checked: " + ("all supported by the transcript." if v["passed"] else
                                                          f"**{len(v['unsupported'])} not found in the transcript:** {v['unsupported']}")]
    return "\n".join(out) + "\n"


def generate(session: CallSession) -> dict | None:
    """Summarise the call, save summary.json / summary.md next to the transcript, and return the dict."""
    if not session.turns:
        return None
    r = structured.parse(SUMMARY_SYSTEM, f"<transcript>\n{_transcript_text(session)}\n</transcript>",
                         CallSummary, effort=CONFIG.offline_effort)
    if r.output is None:
        raise RuntimeError(f"summary not produced (stop_reason={r.stop_reason})")
    summary = {
        "call_id": session.call_id,
        "started_at": session.started_at,
        "ended_at": session.ended_at(),
        "mode": session.mode,
        "turns": len(session.turns),
        "summarised_by": f"{CONFIG.provider}:{CONFIG.model}",
        **r.output.model_dump(),
    }
    summary["validation"] = check_figures(summary, session)
    summary["usage"] = {"input_tokens": r.input_tokens, "output_tokens": r.output_tokens}
    (session.dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    (session.dir / "summary.md").write_text(to_markdown(summary))
    return summary
