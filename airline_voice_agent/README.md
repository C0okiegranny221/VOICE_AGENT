# IndiGo policy voice agent (POC)

A phone-style voice assistant that answers IndiGo baggage, change/refund and disruption questions
from public IndiGo and DGCA documents, in English, Hindi or Hinglish, and looks up fictional demo bookings
by PNR. Runs on an M2 MacBook with the built-in mic and speakers.
The LLM is the only cloud component — Gemini 3.8 Flash on Google's free tier by default, with Groq
(free tier) or Claude (paid) as alternatives. Speech recognition, speech synthesis, turn detection,
embeddings and reranking all run locally.

```
mic → push-to-talk / Silero VAD → speech check → Hinglish Whisper (Oriserve Apex, Roman-script output)
  → language tag → hybrid retrieval (LanceDB BM25 + bge-m3 vectors, RRF) → LLM (streamed; policy-table and
  booking tools) → sentence splitter → Kokoro (Hindi voice for Devanagari replies, English voice otherwise) → speaker
```

Every turn is kept in the call's memory (full message history), and a structured summary is written
when the call ends. This is a demo built on public documents; it is not affiliated with IndiGo.

## Setup

```bash
uv sync                              # Python 3.11 env with all dependencies
brew install espeak-ng               # needed by Kokoro
cp .env.example .env                 # then add GEMINI_API_KEY (free, from aistudio.google.com)
.venv/bin/python -m agent.provider_check   # 4 live requests: streaming, tools, structured output
```

| `PROVIDER` | Default model | Key | Cost |
|---|---|---|---|
| `gemini` (default) | `gemini-3.8-flash` | `GEMINI_API_KEY` | Free tier, rate-limited; Google may use free-tier prompts to improve its models |
| `groq` | `openai/gpt-oss-120b` | `GROQ_API_KEY` | Free tier, rate-limited, low daily token cap |
| `claude` | `claude-opus-5-5` | `ANTHROPIC_API_KEY` | Paid (Claude API, billed separately from a Claude subscription) |

Set `MODEL=` in `.env` (or `--model` on `main.py`) to use another model from the same provider.

First use downloads the models (~5 GB: Hinglish Whisper Apex, bge-m3, Kokoro; English Whisper turbo if
`STT_ENGINE=english`).

| Setting (`.env`) | Default | Options |
|---|---|---|
| `STT_ENGINE` | `hinglish` | `english` (mlx Whisper turbo, English only) |
| `TTS_VOICE_HINDI` | `hf_alpha` | `hf_beta`, `hm_omega`, `hm_psi` (samples in `report/08_voice/audio_hinglish/`) |
| `TTS_VOICE` | `af_heart` | any Kokoro English voice |
| `EMBED_MODEL` | `BAAI/bge-m3` | `BAAI/bge-small-en-v1.5` (English only; rebuild the index after changing) |
| `RERANK` | `0` | `1` (off by default: reranking hurt Roman-script Hinglish) |

Demo bookings (fictional, `data/mock/bookings.json`): K7Q2PX / Mehta (on time), M4D8RT / Sharma (delayed),
Z9F3LW / Qureshi (cancelled), T2H6NB / Iyer (infant, booked yesterday), P5V1CJ / Das (denied boarding).

## Run

```bash
.venv/bin/python -m ingest.run_all           # PDFs → sections → chunks → index → policy tables
.venv/bin/python chat.py                     # text version of a call
.venv/bin/python main.py                     # voice, push-to-talk (Enter to start/stop)
.venv/bin/python main.py --vad               # voice, hands-free
```

Say "that's all" or "bye" to end a call (or Ctrl+C). Transcripts and summaries are in `calls/<call_id>/`.

## Tests and evaluation

```bash
.venv/bin/python -m pytest                               # unit tests (no network, no API key)
.venv/bin/python -m rag.retriever "can I carry a power bank" --route domestic
.venv/bin/python -m eval.retrieval_experiments --ablations   # retrieval spot checks and ablations
.venv/bin/python -m eval.voice_benchmark                 # STT/TTS/VAD benchmark on synthetic speech
.venv/bin/python -m eval.demo_smoke                      # end-to-end Hinglish + English call, no mic needed
.venv/bin/python -m eval.run_eval                        # answer + memory eval (shows cost; add --yes to run)
```

Every pipeline run, test run, benchmark, eval and call is also saved to `report/` (see `report/LOG.md`).

## Layout

| Path | What it does |
|---|---|
| `ingest/sources.yaml` | The source documents and how each is obtained and parsed |
| `ingest/fetch.py` | Downloads PDFs; prints IndiGo web pages to PDF with headless Chrome (no HTML parsing) |
| `ingest/parse.py` | PDF → sections with heading path and page (numbering or accordion-icon headings) |
| `ingest/chunk.py` | Sections → ~300–800 token chunks with topic, route and jurisdiction metadata |
| `ingest/build_index.py` | LanceDB table with vectors and a BM25 full-text index |
| `data/tables/*.csv`, `ingest/build_tables.py` | Policy figures with verbatim source quotes → `data/policy.db` (quotes are checked against the parsed text) |
| `rag/retriever.py` | Hybrid search, filters, rerank |
| `rag/tools.py` | LLM tools over the policy tables; disruption tools compute entitlements |
| `agent/` | Session memory, prompts, LLM loops (`llm.py` Claude, `llm_openai.py` Gemini/Groq), structured outputs, caller details, end-of-call detection, summary, provider check |
| `voice/` | Whisper STT, Kokoro TTS, Silero VAD, mic/speaker I/O, sentence splitter |
| `eval/` | Spot checks, ablations, voice benchmark, answer/memory eval |

## Known limitations

- Sources: DGCA CAR Sec 3 Series M Part IV (Jan 2023 rev.), IndiGo Conditions of Carriage (domestic and
  international), Passenger Rights notices, Baggage Policy — all as retrieved on 2026-10-16. IndiGo's
  fees and refunds pages and the Feb 2026 DGCA refund CAR (Series M Part II) were not collected.
- Change/cancellation fee amounts and per-kg excess baggage rates are not in the collected documents.
- Policy table rows are checked against the source text automatically but still need human review
  (`human_verified_on` column).
- Laptop speakers + mic: the mic is muted while the agent speaks, so the caller can't interrupt.
- Bookings are fictional demo data; nothing can be changed.
- Hinglish replies are spoken by Kokoro's Hindi voice from Devanagari text; English words inside them are
  transliterated. Code-mixed speech quality has only been checked by ear on a few samples.
- On the free tiers, answers are generally less precise than Claude's and requests are rate-limited;
  don't send real customer data through a free tier.
