"""Text-only version of the call: same retrieval, tools, memory and summary, typed instead of spoken.

    python chat.py            # type questions; 'bye' / 'that's all' or Ctrl+D ends the call
"""

from agent.end_detection import is_goodbye
from agent.llm import make_agent
from agent.pipeline import complete_turn, end_call, run_turn
from agent.prompts import GREETING, farewell
from agent.session import CallSession
from config import CONFIG, require_api_key
from rag.embed import warmup


def main() -> None:
    require_api_key()
    warmup()
    session, agent = CallSession(mode="text"), make_agent()
    print(f"[call {session.call_id} · {CONFIG.provider}:{CONFIG.model}]\nAGENT: {GREETING}")
    reason = "eof"
    try:
        while True:
            try:
                text = input("\nYOU: ").strip()
            except EOFError:
                break
            if not text:
                continue
            if is_goodbye(text):
                print(f"AGENT: {farewell(session.language)}")
                reason = "goodbye"
                break
            print("AGENT: ", end="", flush=True)
            for piece in run_turn(session, agent, text):
                print(piece, end="", flush=True)
            t = complete_turn(session, agent)
            tools = ", ".join(c["name"] for c in t["tool_calls"])
            print(f"\n  [retrieval {t['latency_ms']['retrieval']} ms · first text {t['latency_ms']['llm_first_text']} ms"
                  f"{' · tools: ' + tools if tools else ''}]")
    except KeyboardInterrupt:
        reason = "interrupted"
    print("\n[ending call, writing summary...]")
    summary = end_call(session, reason)
    if summary:
        print((session.dir / "summary.md").read_text())


if __name__ == "__main__":
    main()
