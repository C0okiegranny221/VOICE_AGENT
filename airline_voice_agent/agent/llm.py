"""Streamed LLM turn with a manual tool loop.

    agent = make_agent()                  # provider from config (gemini by default)
    for text in agent.respond(session):   # text deltas, ready for the sentence splitter / TTS
        ...
    agent.last_turn                       # assistant text, tool calls, usage, latency, stop reason

Request shape: cached system prompt + strict tools, the session's full append-only message history,
top-level automatic prompt caching, low effort for live turns, and server-side refusal fallback.
"""
import json
import time
from dataclasses import dataclass, field

import anthropic

from agent.prompts import ERROR_LINE, REFUSAL_LINE, system_prompt
from agent.session import CallSession
from config import CONFIG, PROVIDERS
from rag.tools import TOOL_SCHEMAS, run_tool

FALLBACK_BETA = "server-side-fallback-2026-07-01"
MAX_TOOL_ROUNDS = 4
# Streaming + client tools: stream tool inputs as generated. The API then skips input validation,
# so run_tool() validates every input against its schema before executing.
STREAM_TOOLS = [{**t, "eager_input_streaming": True} for t in TOOL_SCHEMAS]


@dataclass
class TurnResult:
    text: str = ""
    tool_calls: list = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    latency_ms: dict = field(default_factory=dict)
    stop_reason: str | None = None
    request_ids: list = field(default_factory=list)
    model: str | None = None
    fallbacks: list = field(default_factory=list)  # models skipped this turn (overloaded / timed out)


def _echo_safe(content: list) -> list:
    """After a mid-output fallback, blocks before the last `fallback` marker that are model-internal
    (thinking, tool_use) must not be echoed back; text and later blocks are kept."""
    types = [b.type for b in content]
    if "fallback" not in types:
        return content
    last = len(types) - 1 - types[::-1].index("fallback")
    drop = {"thinking", "redacted_thinking", "tool_use"}
    return [b for i, b in enumerate(content) if i >= last or b.type not in drop]


def _usage_dict(usage) -> dict:
    keys = ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")
    return {k: getattr(usage, k, None) or 0 for k in keys}


class ClaudeAgent:
    def __init__(self, client: anthropic.Anthropic | None = None, model: str | None = None, effort: str | None = None):
        self.client = client or anthropic.Anthropic()
        self.model = model or (CONFIG.model if CONFIG.provider == "claude" else PROVIDERS["claude"]["model"])
        self.effort = effort or CONFIG.live_effort
        self.last_turn = TurnResult()

    def _stream(self, session: CallSession):
        return self.client.beta.messages.stream(
            model=self.model,
            max_tokens=CONFIG.live_max_tokens,
            system=[{"type": "text", "text": system_prompt()}],
            tools=STREAM_TOOLS,
            messages=session.api_messages,
            cache_control={"type": "ephemeral"},
            output_config={"effort": self.effort},
            betas=[FALLBACK_BETA],
            fallbacks="default",
        )

    def respond(self, session: CallSession):
        """Yield spoken text for one caller turn, running tools as needed."""
        result = TurnResult()
        self.last_turn = result
        t0 = time.perf_counter()
        usage_total: dict = {}
        for round_no in range(MAX_TOOL_ROUNDS + 1):
            try:
                with self._stream(session) as stream:
                    for event in stream:
                        if event.type == "text":
                            if "first_text" not in result.latency_ms:
                                result.latency_ms["first_text"] = round((time.perf_counter() - t0) * 1000)
                            result.text += event.text
                            yield event.text
                    response = stream.get_final_message()
            except ValueError:
                # Tool input JSON the SDK could not parse at all; re-issue the turn once.
                if round_no == 0:
                    continue
                raise
            except anthropic.APIError as e:
                result.stop_reason = f"error:{type(e).__name__}"
                result.text += ERROR_LINE
                yield ERROR_LINE
                break

            result.request_ids.append(getattr(response, "_request_id", None))
            for k, v in _usage_dict(response.usage).items():
                usage_total[k] = usage_total.get(k, 0) + v
            result.stop_reason = response.stop_reason
            session.add_message({"role": "assistant", "content": _echo_safe(response.content)})

            if response.stop_reason == "refusal":
                result.text += REFUSAL_LINE
                yield REFUSAL_LINE
                break
            tool_uses = [b for b in response.content if b.type == "tool_use"]
            if not tool_uses:
                break
            if response.stop_reason == "max_tokens":
                break  # a truncated tool input parses as a partial object; don't run it

            results, calls = [], []
            for block in tool_uses:
                t_tool = time.perf_counter()
                output, is_error = run_tool(block.name, block.input if isinstance(block.input, dict) else {})
                calls.append({"name": block.name, "input": block.input, "is_error": is_error,
                              "output": json.loads(output), "ms": round((time.perf_counter() - t_tool) * 1000, 1)})
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": output, "is_error": is_error})
            result.tool_calls.extend(calls)
            session.add_tool_results([{"role": "user", "content": results}], calls)
            if result.text and not result.text.endswith((" ", "\n")):
                result.text += " "
                yield " "

        result.usage = usage_total
        result.latency_ms["total"] = round((time.perf_counter() - t0) * 1000)
        result.text = result.text.strip()


def make_agent(provider: str | None = None, model: str | None = None, effort: str | None = None):
    """The live-turn agent for the configured (or given) provider."""
    provider = provider or CONFIG.provider
    if PROVIDERS[provider]["kind"] == "anthropic":
        return ClaudeAgent(model=model, effort=effort)
    from agent.llm_openai import OpenAICompatAgent

    return OpenAICompatAgent(provider=provider, model=model, effort=effort)

