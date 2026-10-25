"""Live smoke test of the configured LLM provider - run this first after adding an API key.

    python -m agent.provider_check

Checks, in about four requests: a plain streamed reply; a tool round trip (call -> result -> answer,
which exercises thought-signature echoing on Gemini 3); a second tool turn in the same conversation;
and a structured (schema-validated) output. Results are saved to report/07_agent_text/.
"""
import time
import traceback

from pydantic import BaseModel

from agent import structured
from agent.llm import make_agent
from agent.session import CallSession
from config import CONFIG, require_api_key
from reporting import log_entry, md_table, save_artifact


class Probe(BaseModel):
    route_type: str
    kilograms: int


def _turn(agent, session, text):
    session.add_user_turn(text, [])
    t0 = time.perf_counter()
    spoken = "".join(agent.respond(session))
    session.finalize_turn(spoken)
    t = agent.last_turn
    return spoken, t, round((time.perf_counter() - t0) * 1000)


def main() -> None:
    require_api_key()
    agent = make_agent()
    rows = []

    def check(name, fn):
        try:
            ok, detail, ms = fn()
        except Exception as e:  # report every failure, keep going
            ok, detail, ms = False, f"{type(e).__name__}: {e}"[:300], None
            traceback.print_exc()
        rows.append({"check": name, "ok": ok, "ms": ms, "detail": detail})
        print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")

    session = CallSession(mode="provider-check")

    def plain():
        s = CallSession(mode="provider-check")
        spoken, t, ms = _turn(agent, s, "Say hello in one short sentence.")
        return bool(spoken) and not t.stop_reason.startswith("error"), f"first text {t.latency_ms.get('first_text')} ms: {spoken[:80]!r}", ms

    def tool_round_trip():
        spoken, t, ms = _turn(agent, session, "How much checked baggage can I take on a domestic economy flight? Use the tool.")
        names = [c["name"] for c in t.tool_calls]
        ok = "get_baggage_allowance" in names and not any(c["is_error"] for c in t.tool_calls) and "15" in spoken.replace("fifteen", "15")
        return ok, f"tools {names}, stop {t.stop_reason}: {spoken[:100]!r}", ms

    def second_tool_turn():
        spoken, t, ms = _turn(agent, session, "And how much does an extra bag cost if I pre-book it?")
        ok = bool(t.tool_calls) and not str(t.stop_reason).startswith("error") and "900" in spoken.replace("nine hundred", "900")
        return ok, f"tools {[c['name'] for c in t.tool_calls]}, stop {t.stop_reason}: {spoken[:100]!r}", ms

    def structured_output():
        t0 = time.perf_counter()
        r = structured.parse("Extract the fields.", "International flight, bag is 23 kg.", Probe, effort=CONFIG.live_effort, max_tokens=2000)
        ok = r.output is not None and r.output.kilograms == 23
        return ok, f"{r.output!r}", round((time.perf_counter() - t0) * 1000)

    check("plain streamed reply", plain)
    check("tool round trip", tool_round_trip)
    check("second tool turn (history + signatures)", second_tool_turn)
    check("structured output", structured_output)

    passed = sum(r["ok"] for r in rows)
    body = f"# Provider check — {CONFIG.provider}:{CONFIG.model}\n\n{passed}/{len(rows)} passed.\n\n{md_table(rows)}"
    path = save_artifact("agent", f"provider_check_{CONFIG.provider}", body)
    log_entry("agent", f"Provider check {CONFIG.provider}:{CONFIG.model}: {passed}/{len(rows)} passed", md_table(rows), [path])


if __name__ == "__main__":
    main()
