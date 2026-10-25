"""OpenAI-compatible (Gemini/Groq) agent loop tests with a fake streaming client built from real SDK types."""
from types import SimpleNamespace as NS

import pytest
from openai.types.chat import ChatCompletionChunk

from agent.llm_openai import OPENAI_TOOLS, OpenAICompatAgent
from agent.session import CallSession

SIGNATURE = {"extra_content": {"google": {"thought_signature": "c2lnbmF0dXJl"}}}


def chunk(delta: dict, finish=None, usage=None) -> ChatCompletionChunk:
    return ChatCompletionChunk.model_validate({
        "id": "c1", "object": "chat.completion.chunk", "created": 0, "model": "gemini-3.8-flash",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}] if delta is not None else [],
        "usage": usage,
    })


def tool_call_stream(name, args_json, extra=None):
    call = {"index": 0, "id": "call_1", "type": "function", "function": {"name": name, "arguments": args_json}, **(extra or {})}
    return [chunk({"role": "assistant", "tool_calls": [call]}), chunk({}, finish="tool_calls"),
            chunk(None, usage={"prompt_tokens": 900, "completion_tokens": 40, "total_tokens": 940})]


def text_stream(text):
    words = text.split(" ")
    return [chunk({"content": w + (" " if i < len(words) - 1 else "")}) for i, w in enumerate(words)] + [
        chunk({}, finish="stop"), chunk(None, usage={"prompt_tokens": 1200, "completion_tokens": 30, "total_tokens": 1230})]


class FakeClient:
    def __init__(self, streams):
        self.streams, self.requests = list(streams), []
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **kwargs):
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        return iter(self.streams.pop(0))


@pytest.fixture
def session(tmp_path, monkeypatch):
    monkeypatch.setattr("agent.session.CONFIG", NS(calls_dir=tmp_path))
    s = CallSession()
    s.add_user_turn("how much baggage on a domestic flight?", [])
    return s


def test_tool_round_trip_echoes_thought_signature(session):
    client = FakeClient([
        tool_call_stream("get_baggage_allowance", '{"route_type": "domestic", "product": "economy"}', SIGNATURE),
        text_stream("You can check in fifteen kilograms."),
    ])
    agent = OpenAICompatAgent(provider="gemini", client=client, model="gemini-3.8-flash", effort="low")
    spoken = "".join(agent.respond(session))
    assert "fifteen kilograms" in spoken
    assistant, tool_msg = session.api_messages[1], session.api_messages[2]
    assert assistant["tool_calls"][0]["extra_content"] == SIGNATURE["extra_content"]  # echoed unchanged
    assert tool_msg["role"] == "tool" and tool_msg["tool_call_id"] == "call_1" and '"max_kg": 15' in tool_msg["content"]
    first, second = client.requests
    assert first["messages"][0]["role"] == "system"
    assert second["messages"][: len(first["messages"])] == first["messages"]  # append-only
    assert first["reasoning_effort"] == "low" and first["tool_choice"] == "auto"
    assert agent.last_turn.usage["input_tokens"] == 2100


def test_omitted_nullable_parameters_become_none(session):
    client = FakeClient([
        tool_call_stream("get_delay_entitlements", '{"delay_hours": 3, "route_type": "domestic"}'),
        text_stream("How long is the flight?"),
    ])
    agent = OpenAICompatAgent(provider="gemini", client=client, model="m")
    "".join(agent.respond(session))
    call = agent.last_turn.tool_calls[0]
    assert not call["is_error"]
    assert "need_more_information" in call["output"]["computed"]  # block_time_hours was None


def test_bad_json_arguments_return_tool_error(session):
    client = FakeClient([tool_call_stream("get_baggage_allowance", '{"route_type": '), text_stream("Sorry?")])
    agent = OpenAICompatAgent(provider="gemini", client=client, model="m")
    "".join(agent.respond(session))
    assert agent.last_turn.tool_calls[0]["is_error"]


def test_tool_schema_conversion():
    delay = next(t for t in OPENAI_TOOLS if t["function"]["name"] == "get_delay_entitlements")["function"]["parameters"]
    assert delay["properties"]["block_time_hours"]["type"] == "number"
    assert set(delay["required"]) == {"delay_hours", "route_type"}
    assert all(not isinstance(p.get("type"), list) for t in OPENAI_TOOLS for p in t["function"]["parameters"]["properties"].values())


class OverloadedThenOk(FakeClient):
    """First model raises 503; the fallback model streams normally."""

    def _create(self, **kwargs):
        if kwargs["model"] == "primary":
            import httpx
            from openai import InternalServerError

            req = httpx.Request("POST", "https://example.test")
            raise InternalServerError("high demand", response=httpx.Response(503, request=req), body=None)
        return super()._create(**kwargs)


def test_overloaded_model_falls_back_and_stays(session, monkeypatch):
    monkeypatch.setattr("agent.llm_openai.model_chain", lambda provider, model: ["primary", "backup"])
    client = OverloadedThenOk([text_stream("Hello there."), text_stream("Second answer.")])
    agent = OpenAICompatAgent(provider="gemini", client=client, model="primary")
    assert "Hello there." in "".join(agent.respond(session))
    assert agent.last_turn.model == "backup"
    assert agent.last_turn.fallbacks[0]["model"] == "primary"
    session.add_user_turn("again", [])
    "".join(agent.respond(session))
    assert [r["model"] for r in client.requests] == ["backup", "backup"]  # sticks to the model that answered
