"""One caller turn, shared by the text chat and the voice loop:
caller details -> retrieval -> LLM (streamed, with tools) -> turn record."""
import time

from agent import summary as summary_mod
from agent.session import CallSession
from rag.retriever import build_query, retrieve
from reporting import log_entry, md_table, save_artifact


def run_turn(session: CallSession, agent, text: str):
    """Yield the agent's spoken text for one caller utterance. Call complete_turn() afterwards."""
    t0 = time.perf_counter()
    previous = session.previous_user_text
    session.note_details(text)
    r = retrieve(text, previous, session.caller_details, language=session.language)
    session.add_user_turn(text, r.chunks, build_query(text, previous, session.caller_details))
    session.pending_latency = {"retrieval": round((time.perf_counter() - t0) * 1000)}
    yield from agent.respond(session)


def complete_turn(session: CallSession, agent, extra_latency: dict | None = None) -> dict:
    """Record the finished turn (text, tools, usage, latency) in the session and transcript."""
    turn = agent.last_turn
    latency = {**(extra_latency or {}), **session.pending_latency,
               "llm_first_text": turn.latency_ms.get("first_text"), "llm_total": turn.latency_ms.get("total")}
    _remember_booking(session, turn.tool_calls)
    record = session.finalize_turn(turn.text, latency, turn.usage, turn.stop_reason)
    record["model"], record["model_fallbacks"] = turn.model, turn.fallbacks
    return record


def _remember_booking(session: CallSession, tool_calls: list[dict]) -> None:
    """After a successful booking lookup, its facts become caller details (route type filters retrieval)."""
    for call in tool_calls:
        out = call.get("output") or {}
        if call["name"] == "get_booking" and out.get("found"):
            booking, seg = out["booking"], out["segments"][0]
            session.caller_details.update({
                "pnr": booking["pnr"], "route_type": seg["route_type"], "product": booking["product"],
                "payment_mode": booking["payment_mode"], "flight": f"{seg['flight_no']} {seg['from']}-{seg['to']}",
                "flight_status": seg["status"],
            })


def end_call(session: CallSession, reason: str) -> dict | None:
    """Summarise the call and copy transcript + summary into the report."""
    if not session.turns:
        return None
    try:
        summary = summary_mod.generate(session)
    except Exception as e:  # keep the transcript even if the summary fails
        summary = None
        print(f"[summary failed: {type(e).__name__}: {e}]")
    files = [save_artifact("calls", f"{session.call_id}_transcript", session.transcript_path.read_text(), "jsonl")]
    if summary:
        files.append(save_artifact("calls", f"{session.call_id}_summary", (session.dir / "summary.md").read_text()))
        files.append(save_artifact("calls", f"{session.call_id}_summary", summary))
    rows = [{"turn": t["turn"], "caller": t["user_text"][:70], "tools": ", ".join(c["name"] for c in t["tool_calls"]) or "-",
             "retrieval_ms": t["latency_ms"].get("retrieval"), "first_text_ms": t["latency_ms"].get("llm_first_text"),
             "cache_read": t["usage"].get("cache_read_input_tokens")} for t in session.turns]
    check = "" if not summary else f"\n\nSummary figure check: {'passed' if summary['validation']['passed'] else 'FAILED'}."
    log_entry("calls", f"Call {session.call_id} ({session.mode}, {len(session.turns)} turns, ended: {reason})",
              md_table(rows) + check, files)
    return summary
