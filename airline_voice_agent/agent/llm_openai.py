"""Streamed turn with a manual tool loop for OpenAI-compatible providers (Gemini, Groq).

Same interface as ClaudeAgent: respond(session) yields spoken text; last_turn holds the result.

Provider notes
- Tool schemas are converted to OpenAI function format. Nullable parameters ({"type": ["number","null"]})
  become optional parameters, since providers differ on type arrays; an omitted one is passed to the
  tool as None.
- Each assistant tool call is echoed back with every extra field the provider returned (Gemini 3 puts
  its thought signature there), so multi-step tool use keeps the model's reasoning state.
- Free tiers get overloaded. If a model is overloaded (5xx), times out or is rate-limited before any
  output, the request moves to the next model in the provider's fallback list instead of retrying for
  minutes. Once a model has answered in a call, the call stays on it (thought signatures are model-specific).
"""
import json
import time

import openai

from agent.llm import MAX_TOOL_ROUNDS, TurnResult
from agent.prompts import ERROR_LINE, system_prompt
from agent.session import CallSession
from config import CONFIG, PROVIDERS
from rag.tools import TOOL_SCHEMAS, run_tool


def _openai_tool(tool: dict) -> dict:
    schema = tool["input_schema"]
    props, required = {}, []
    for name, spec in schema["properties"].items():
        spec = dict(spec)
        types = spec.get("type")
        if isinstance(types, list):
            spec["type"] = next(t for t in types if t != "null")
        else:
            required.append(name)
        props[name] = spec
    return {"type": "function", "function": {
        "name": tool["name"], "description": tool["description"],
        "parameters": {"type": "object", "properties": props, "required": required},
    }}


OPENAI_TOOLS = [_openai_tool(t) for t in TOOL_SCHEMAS]
_ALL_PARAMS = {t["name"]: list(t["input_schema"]["properties"]) for t in TOOL_SCHEMAS}


TRANSIENT = (openai.InternalServerError, openai.APITimeoutError, openai.RateLimitError, openai.APIConnectionError)
REQUEST_TIMEOUT_S = 20.0


def make_client(provider: str) -> openai.OpenAI:
    import os

    info = PROVIDERS[provider]
    return openai.OpenAI(api_key=os.getenv(info["key_env"]), base_url=info["base_url"], max_retries=1,
                         timeout=REQUEST_TIMEOUT_S)


def model_chain(provider: str, model: str) -> list[str]:
    return [model] + [m for m in PROVIDERS[provider].get("fallback_models", []) if m != model]


def _prepend(first, rest):
    yield first
    yield from rest


def _fill_optional(name: str, args: dict) -> dict:
    """Optional (nullable) parameters the model left out are passed to the tool as None."""
    return {**{p: None for p in _ALL_PARAMS.get(name, [])}, **args} if isinstance(args, dict) else args


class OpenAICompatAgent:
    def __init__(self, provider: str | None = None, client: openai.OpenAI | None = None,
                 model: str | None = None, effort: str | None = None):
        self.provider = provider or CONFIG.provider
        self.client = client or make_client(self.provider)
        self.model = model or (CONFIG.model if self.provider == CONFIG.provider else PROVIDERS[self.provider]["model"])
        self.models = model_chain(self.provider, self.model)
        self.effort = effort or CONFIG.live_effort
        self.last_turn = TurnResult()
        self.fallback_log: list[dict] = []

    def _stream(self, session: CallSession):
        """Open a stream on the first model in the chain that accepts the request."""
        last_error = None
        for i, model in enumerate(self.models):
            try:
                stream = self._create(session, model)
                first = next(stream, None)  # surfaces 5xx / timeouts before any output is spoken
            except TRANSIENT as e:
                last_error = e
                self.fallback_log.append({"model": model, "error": f"{type(e).__name__} {getattr(e, 'status_code', '')}".strip()})
                continue
            if i:
                self.models = self.models[i:]  # stay on the model that answered
                self.model = model
            return _prepend(first, stream) if first is not None else iter(())
        raise last_error

    def _create(self, session: CallSession, model: str):
        return self.client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system_prompt()}, *session.api_messages],
            tools=OPENAI_TOOLS,
            tool_choice="auto",
            reasoning_effort=self.effort,
            max_completion_tokens=CONFIG.live_max_tokens,
            stream=True,
            stream_options={"include_usage": True},
        )

    def respond(self, session: CallSession):
        result = TurnResult()
        self.last_turn = result
        t0 = time.perf_counter()
        usage_total = {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}
        for _round in range(MAX_TOOL_ROUNDS + 1):
            text, calls, finish = "", {}, None
            try:
                for chunk in self._stream(session):
                    if chunk.usage:
                        usage_total["input_tokens"] += chunk.usage.prompt_tokens or 0
                        usage_total["output_tokens"] += chunk.usage.completion_tokens or 0
                        details = getattr(chunk.usage, "prompt_tokens_details", None)
                        usage_total["cache_read_input_tokens"] += getattr(details, "cached_tokens", 0) or 0
                    if not chunk.choices:
                        continue
                    choice = chunk.choices[0]
                    finish = choice.finish_reason or finish
                    delta = choice.delta
                    if delta.content:
                        if "first_text" not in result.latency_ms:
                            result.latency_ms["first_text"] = round((time.perf_counter() - t0) * 1000)
                        text += delta.content
                        result.text += delta.content
                        yield delta.content
                    for tc in delta.tool_calls or []:
                        entry = calls.setdefault(tc.index if tc.index is not None else len(calls),
                                                 {"id": None, "name": "", "arguments": "", "extra": {}})
                        if tc.id:
                            entry["id"] = tc.id
                        if tc.function and tc.function.name:
                            entry["name"] += tc.function.name
                        if tc.function and tc.function.arguments:
                            entry["arguments"] += tc.function.arguments
                        entry["extra"].update(tc.model_extra or {})
            except openai.APIError as e:
                result.stop_reason = f"error:{type(e).__name__}"
                result.text += ERROR_LINE
                yield ERROR_LINE
                break

            result.stop_reason = finish
            tool_calls = []
            for i, c in sorted(calls.items()):
                tool_calls.append({"id": c["id"] or f"call_{i}", "type": "function",
                                   "function": {"name": c["name"], "arguments": c["arguments"] or "{}"}, **c["extra"]})
            message = {"role": "assistant", "content": text or None}
            if tool_calls:
                message["tool_calls"] = tool_calls
            session.add_message(message)
            if not tool_calls:
                break
            if finish == "length":
                break  # truncated tool arguments; don't run them

            tool_messages, records = [], []
            for tc in tool_calls:
                name = tc["function"]["name"]
                t_tool = time.perf_counter()
                try:
                    args = json.loads(tc["function"]["arguments"])
                except json.JSONDecodeError:
                    args = None
                if isinstance(args, dict):
                    output, is_error = run_tool(name, _fill_optional(name, args))
                else:
                    output, is_error = json.dumps({"error": "INVALID_JSON arguments"}), True
                records.append({"name": name, "input": args, "is_error": is_error, "output": json.loads(output),
                                "ms": round((time.perf_counter() - t_tool) * 1000, 1)})
                tool_messages.append({"role": "tool", "tool_call_id": tc["id"], "content": output})
            result.tool_calls.extend(records)
            session.add_tool_results(tool_messages, records)
            if result.text and not result.text.endswith((" ", "\n")):
                result.text += " "
                yield " "

        result.usage = usage_total
        result.latency_ms["total"] = round((time.perf_counter() - t0) * 1000)
        result.text = result.text.strip()
        result.model = self.model
        result.fallbacks = list(self.fallback_log)
        self.fallback_log.clear()
