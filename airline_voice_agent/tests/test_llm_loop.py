"""Claude agent loop tests with a scripted fake Anthropic client (no network, no API key)."""
from types import SimpleNamespace as NS

import pytest

from agent import summary as summary_mod
from agent.llm import ClaudeAgent
from agent.prompts import REFUSAL_LINE
from agent.session import CallSession
from rag.retriever import Chunk


def block(type_, **kw):
    return NS(type=type_, **kw)


def message(content, stop_reason="end_turn"):
    usage = NS(input_tokens=100, output_tokens=20, cache_read_input_tokens=80, cache_creation_input_tokens=0)
    return NS(content=content, stop_reason=stop_reason, usage=usage, _request_id="req_test")


class FakeStream:
    def __init__(self, msg):
        self.msg = msg

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __iter__(self):
        for b in self.msg.content:
            if b.type == "text":
                for word in b.text.split(" "):
                    yield NS(type="text", text=word + " ")

    def get_final_message(self):
        return self.msg


class FakeClient:
    def __init__(self, script):
        self.script, self.requests = list(script), []
        self.beta = NS(messages=NS(stream=self._stream))

    def _stream(self, **kwargs):
        # Snapshot the history as sent, to check it is append-only across requests.
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        return FakeStream(self.script.pop(0))


def chunk(cid):
    return Chunk(cid, "indigo_coc_domestic", "IndiGo CoC", "", "", "domestic", "india", 14, "11.2", "baggage", "text " + cid)


@pytest.fixture
def session(tmp_path, monkeypatch):
    monkeypatch.setattr("agent.session.CONFIG", NS(calls_dir=tmp_path))
    return CallSession()


def test_tool_round_trip_appends_history_without_edits(session):
    client = FakeClient([
        message([block("thinking", thinking="", signature="sig"),
                 block("tool_use", id="tu1", name="get_baggage_allowance", input={"route_type": "domestic", "product": "economy"})],
                stop_reason="tool_use"),
        message([block("text", text="You can check in fifteen kilograms.")]),
    ])
    agent = ClaudeAgent(client=client)
    session.add_user_turn("how much baggage", [chunk("a"), chunk("b")])
    spoken = "".join(agent.respond(session))
    assert "fifteen kilograms" in spoken
    first, second = client.requests
    assert second["messages"][: len(first["messages"])] == first["messages"]  # append-only
    roles = [m["role"] for m in session.api_messages]
    assert roles == ["user", "assistant", "user", "assistant"]
    tool_result = session.api_messages[2]["content"][0]
    assert tool_result["tool_use_id"] == "tu1" and not tool_result["is_error"] and '"max_kg": 15' in tool_result["content"]
    assert session.api_messages[1]["content"][0].type == "thinking"  # thinking block echoed unchanged
    assert agent.last_turn.tool_calls[0]["name"] == "get_baggage_allowance"
    assert first["cache_control"] == {"type": "ephemeral"} and first["fallbacks"] == "default"
    assert all(t["strict"] and t["eager_input_streaming"] for t in first["tools"])


def test_invalid_tool_input_returns_error_result(session):
    client = FakeClient([
        message([block("tool_use", id="tu1", name="get_baggage_allowance", input={"route_type": "moon"})], stop_reason="tool_use"),
        message([block("text", text="Is that a domestic or international flight?")]),
    ])
    session.add_user_turn("baggage?", [])
    "".join(ClaudeAgent(client=client).respond(session))
    assert session.api_messages[2]["content"][0]["is_error"] is True


def test_refusal_speaks_fallback_line(session):
    client = FakeClient([message([], stop_reason="refusal")])
    session.add_user_turn("something", [])
    assert REFUSAL_LINE in "".join(ClaudeAgent(client=client).respond(session))


def test_passages_sent_once_per_call(session):
    session.add_user_turn("q1", [chunk("a"), chunk("b")])
    session.add_user_turn("q2", [chunk("b"), chunk("c")])
    second = session.api_messages[-1]["content"]
    assert 'id="c"' in second and 'id="b"' not in second


def test_transcript_written_per_turn(session):
    session.add_user_turn("q1", [chunk("a")])
    session.finalize_turn("answer one", {"retrieval": 100})
    session.add_user_turn("q2", [])
    session.finalize_turn("answer two")
    lines = session.transcript_path.read_text().strip().splitlines()
    assert len(lines) == 2 and '"answer two"' in lines[1]


def test_summary_figure_check(session):
    session.add_user_turn("baggage?", [])
    session.finalize_turn("You get fifteen kilograms, that is 15 kg, and an extra bag is rupees 900.")
    good = {"qa": [{"figures_quoted": ["15 kg", "INR 900"]}]}
    bad = {"qa": [{"figures_quoted": ["20 kg"]}]}
    assert summary_mod.check_figures(good, session)["passed"]
    assert not summary_mod.check_figures(bad, session)["passed"]
