"""Structured (schema-validated) completions for the configured provider - used by the post-call
summary and the eval judge."""
from dataclasses import dataclass

from pydantic import BaseModel

from config import CONFIG, PROVIDERS


@dataclass
class Parsed:
    output: BaseModel | None
    stop_reason: str | None
    input_tokens: int = 0
    output_tokens: int = 0


def parse(system: str, user: str, schema: type[BaseModel], effort: str, max_tokens: int = 16000,
          provider: str | None = None, model: str | None = None) -> Parsed:
    provider = provider or CONFIG.provider
    model = model or (CONFIG.model if provider == CONFIG.provider else PROVIDERS[provider]["model"])
    if PROVIDERS[provider]["kind"] == "anthropic":
        import anthropic

        r = anthropic.Anthropic().messages.parse(
            model=model, max_tokens=max_tokens, system=system, output_config={"effort": effort},
            output_format=schema, messages=[{"role": "user", "content": user}],
        )
        return Parsed(r.parsed_output, r.stop_reason, r.usage.input_tokens, r.usage.output_tokens)

    from agent.llm_openai import TRANSIENT, make_client, model_chain

    client, last_error = make_client(provider), None
    for m in model_chain(provider, model):
        try:
            r = client.chat.completions.parse(
                model=m, max_completion_tokens=max_tokens, reasoning_effort=effort, response_format=schema,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            )
            break
        except TRANSIENT as e:  # overloaded / timed out: try the next model
            last_error = e
    else:
        raise last_error
    choice = r.choices[0]
    usage = r.usage
    return Parsed(choice.message.parsed, choice.finish_reason,
                  getattr(usage, "prompt_tokens", 0) or 0, getattr(usage, "completion_tokens", 0) or 0)
