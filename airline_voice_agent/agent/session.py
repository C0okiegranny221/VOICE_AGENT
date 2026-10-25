"""CallSession: the memory of one call.

- api_messages: exactly what is sent to the LLM, in the active provider's message format. Append-only:
  assistant turns are appended exactly as returned (Claude: full response.content with thinking and
  tool_use blocks; OpenAI-compatible: the message with its tool_calls and any provider extras such as
  Gemini thought signatures), never edited, so cached prefixes and reasoning state stay valid.
- turns: a readable record per caller turn, written to calls/<call_id>/transcript.jsonl as soon as
  the turn completes.
- Retrieved passages are sent once per call; later turns only carry passages not sent before.
"""
import datetime as dt
import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from agent import caller_details as details_mod
from config import CONFIG


@dataclass
class CallSession:
    call_id: str = field(default_factory=lambda: dt.datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6])
    started_at: str = field(default_factory=lambda: dt.datetime.now().isoformat(timespec="seconds"))
    mode: str = "text"  # text | voice
    api_messages: list = field(default_factory=list)
    turns: list = field(default_factory=list)
    caller_details: dict = field(default_factory=dict)
    sent_chunk_ids: set = field(default_factory=set)
    pending_latency: dict = field(default_factory=dict)
    language: str = "english"  # language of the latest caller turn: english | hinglish
    _pending: dict | None = None

    @property
    def dir(self) -> Path:
        d = CONFIG.calls_dir / self.call_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def transcript_path(self) -> Path:
        return self.dir / "transcript.jsonl"

    @property
    def previous_user_text(self) -> str | None:
        return self.turns[-1]["user_text"] if self.turns else None

    def note_details(self, text: str) -> None:
        """Update caller details and language from the new utterance (done before retrieval, which uses them)."""
        from agent import language

        self.caller_details = details_mod.update(self.caller_details, text)
        self.language = language.detect(text)

    def add_user_turn(self, text: str, chunks: list, retrieval_query: str = "") -> None:
        new_chunks = [c for c in chunks if c.chunk_id not in self.sent_chunk_ids]
        parts = []
        if new_chunks:
            passages = "\n".join(
                f'<passage id="{c.chunk_id}" source="{c.title} › {c.section}" page="{c.page}">\n{c.text}\n</passage>'
                for c in new_chunks
            )
            parts.append(f"<policy_passages>\n{passages}\n</policy_passages>")
        if self.caller_details:
            parts.append(f"<caller_details>{json.dumps(self.caller_details, ensure_ascii=False)}</caller_details>")
        parts.append(f"<caller_language>{self.language}</caller_language>")
        parts.append(f"<caller_said>{text}</caller_said>")
        self.api_messages.append({"role": "user", "content": "\n".join(parts)})
        self.sent_chunk_ids.update(c.chunk_id for c in new_chunks)
        self._pending = {
            "turn": len(self.turns) + 1,
            "at": dt.datetime.now().isoformat(timespec="seconds"),
            "user_text": text,
            "retrieval_query": retrieval_query,
            "retrieved": [{"chunk_id": c.chunk_id, "source": f"{c.doc_id} › {c.section}", "score": round(c.score, 3)} for c in chunks],
            "new_chunk_ids": [c.chunk_id for c in new_chunks],
            "caller_details": dict(self.caller_details),
            "language": self.language,
            "tool_calls": [],
        }

    def add_message(self, message: dict) -> None:
        """Append one provider-format message (assistant turn) to the history."""
        self.api_messages.append(message)

    def add_tool_results(self, messages: list[dict], calls: list[dict]) -> None:
        """Append the provider-format tool-result message(s) and record the calls for the transcript."""
        self.api_messages.extend(messages)
        if self._pending is not None:
            self._pending["tool_calls"].extend(calls)

    def finalize_turn(self, assistant_text: str, latency_ms: dict | None = None, usage: dict | None = None,
                      stop_reason: str | None = None) -> dict:
        record = {**(self._pending or {}), "assistant_text": assistant_text, "latency_ms": latency_ms or {},
                  "usage": usage or {}, "stop_reason": stop_reason}
        self.turns.append(record)
        self._pending = None
        with self.transcript_path.open("a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    def ended_at(self) -> str:
        return dt.datetime.now().isoformat(timespec="seconds")
