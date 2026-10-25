# IndiGo Policy Voice Agent

A laptop-based, multilingual voice assistant proof of concept that answers passenger questions about IndiGo baggage policies, flight changes, refunds, and flight disruptions using public IndiGo and DGCA documents. It supports English, Hindi, and Hinglish, and can look up fictional demo bookings by PNR.

> **Disclaimer:** This is an independent proof of concept built from public documents. It is not affiliated with IndiGo. Demo bookings are fictional and cannot be changed.

## Features

- Voice interaction with push-to-talk or hands-free mode
- Speech recognition for Hindi-English/Hinglish
- Hybrid retrieval using keyword and semantic search
- LLM-generated responses with policy and booking tools
- Local speech synthesis in Hindi or English
- Session conversation history and end-of-call summaries
- Evaluation scripts for retrieval, voice components, and end-to-end calls

## Architecture

```text
Passenger
   |
Microphone → Silero VAD → Speech validation → Whisper STT
                                              |
                                       Language tagging
                                              |
                                  Hybrid policy retrieval
                                  /                   \
                         BM25 keyword search     BGE-M3 vectors
                                  \                   /
                                      RRF fusion
                                           |
                              Optional multilingual reranker
                                           |
                                  Conversation agent
                                  /                \
                              Cloud LLM       Policy/booking tools
                                                   |
                                         Policy tables / demo PNRs
                                           |
                                     Sentence splitting
                                           |
                                      Kokoro TTS
                                           |
                                        Speaker
```

The reranker is optional and disabled by default in the current configuration. Evaluate it on Hindi, English, and Roman-script Hinglish queries before enabling it.

## Technology Stack

| Area | Technology |
|---|---|
| Language / runtime | Python 3.11 |
| Environment and dependencies | `uv` |
| LLM providers | Gemini (default), Groq, or Claude |
| Speech-to-text | Oriserve Apex for Hinglish; optional MLX Whisper Turbo for English |
| Voice activity detection | Silero VAD |
| Text-to-speech | Kokoro TTS |
| TTS dependency | eSpeak NG |
| Vector and full-text retrieval | LanceDB |
| Embeddings | `BAAI/bge-m3` |
| Keyword retrieval | BM25 |
| Hybrid ranking | Reciprocal Rank Fusion (RRF) |
| Structured policy data | CSV and SQLite |
| Demo booking data | JSON |
| Testing | pytest |
| Web-page PDF capture | Headless Chrome |

## Requirements

- macOS on Apple Silicon is the primary tested environment
- Python 3.11
- `uv`
- Homebrew and `espeak-ng`
- An API key for one supported cloud LLM provider

Speech processing, embeddings, retrieval, and reranking run locally; the LLM is the cloud component. The first run downloads model files and may require several gigabytes of disk space.

## Setup

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <YOUR_REPOSITORY_DIRECTORY>
```

### 2. Install system dependency

```bash
brew install espeak-ng
```

### 3. Install Python dependencies

```bash
uv sync
```

### 4. Configure your API key safely

Create your local environment file from the example:

```bash
cp .env.example .env
```

Edit `.env` locally and add the API key for your chosen provider. **Never commit `.env` or paste API keys into source code.** Keep only placeholder values in `.env.example`, for example:

```dotenv
PROVIDER=gemini
MODEL=gemini-3.8-flash
GEMINI_API_KEY=
GROQ_API_KEY=
ANTHROPIC_API_KEY=
```

Use the variable names and supported model names that match your current application configuration. You only need to populate the key for the provider you use.

### 5. Check the configured provider

```bash
.venv/bin/python -m agent.provider_check
```

## Run the Application

Build the document index and policy tables:

```bash
.venv/bin/python -m ingest.run_all
```

Run the text-based assistant:

```bash
.venv/bin/python chat.py
```

Run voice mode with push-to-talk:

```bash
.venv/bin/python main.py
```

Run hands-free voice mode:

```bash
.venv/bin/python main.py --vad
```

Say **“that's all”** or **“bye”** to end a call. Transcripts and summaries are written under `calls/<call_id>/`.

## Configuration

Settings are configured through `.env`. Common options include:

| Variable | Purpose |
|---|---|
| `PROVIDER` | Selects `gemini`, `groq`, or `claude` |
| `MODEL` | Overrides the default model for the selected provider |
| `GEMINI_API_KEY` | Gemini API key |
| `GROQ_API_KEY` | Groq API key |
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `STT_ENGINE` | Selects Hinglish or English speech recognition |
| `TTS_VOICE_HINDI` | Selects a supported Hindi Kokoro voice |
| `TTS_VOICE` | Selects a Kokoro English voice |
| `EMBED_MODEL` | Selects the embedding model |
| `RERANK` | Enables or disables optional reranking |

Check `.env.example` and the application configuration for the exact supported values and defaults.

## Tests and Evaluation

Run unit tests:

```bash
.venv/bin/python -m pytest
```

Try a retrieval query:

```bash
.venv/bin/python -m rag.retriever "can I carry a power bank" --route domestic
```

Run retrieval experiments:

```bash
.venv/bin/python -m eval.retrieval_experiments --ablations
```

Run voice component benchmarks:

```bash
.venv/bin/python -m eval.voice_benchmark
```

Run an end-to-end demo without a microphone:

```bash
.venv/bin/python -m eval.demo_smoke
```

Run answer and memory evaluation:

```bash
.venv/bin/python -m eval.run_eval
```

Evaluation commands may make API calls and incur usage or cost depending on the selected provider. Check the command options before running them.

## Demo PNRs

The project includes fictional demo records for testing booking lookups. They do not represent real passengers or live airline systems. No actual booking modifications are supported.

## Data Sources and Limitations

The project uses public IndiGo and DGCA policy documents collected for the proof of concept. Coverage may be incomplete or out of date. Some fees and refund details are not present in the collected documents, and extracted policy-table entries may require human review.

Hinglish speech quality has only been checked on a limited set of samples. On the laptop setup, the microphone is muted while the agent speaks, so barge-in/interruption is not supported.

Do not send real customer or passenger data to third-party LLM providers, especially through free tiers, unless you have verified the provider's privacy, retention, and data-use terms and have appropriate authorization.

## Repository Hygiene and Secrets

Before pushing to GitHub:

- Keep `.env`, API keys, local virtual environments, call recordings, transcripts, generated indexes, and private data out of version control.
- Commit `.env.example` only when it contains placeholders, never real credentials.
- Review `git status` and staged files before every push.
- If a secret was committed previously, remove it from the repository and rotate/revoke it. Adding it to `.gitignore` does not erase it from Git history.

Example `.gitignore` entries:

```gitignore
# Secrets
.env
.env.*
!.env.example

# Python
.venv/
__pycache__/
*.py[cod]
.pytest_cache/

# Local outputs and potentially sensitive call data
calls/
*.wav
*.mp3

# Generated indexes / databases / reports (review before committing)
data/policy.db
data/index/
report/
```

Adjust the generated-data exclusions to match what you intentionally want to publish. Keep source policy data or evaluation artifacts only if you have permission to redistribute them.

## License

Add a license file before publishing if you intend others to reuse the code. Confirm the redistribution rights for all documents, datasets, model weights, and other third-party assets first.
