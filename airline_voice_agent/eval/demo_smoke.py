"""End-to-end demo smoke test without mic or speakers: synthetic caller speech -> STT -> retrieval ->
LLM + tools -> TTS. Saves the agent's spoken replies as WAVs and logs timings to report/08_voice/.

    python -m eval.demo_smoke
"""
import time

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from agent.llm import make_agent
from agent.pipeline import complete_turn, end_call, run_turn
from agent.session import CallSession
from config import CONFIG, require_api_key
from rag.embed import warmup as warm_retrieval
from reporting import log_entry, md_table, save_artifact, stage_dir
from voice import stt, tts
from voice.sentence_splitter import SentenceSplitter

CALLER = [  # (what the caller says, Kokoro voice used to fake their speech)
    ("मेरा पी एन आर एम फोर डी एट आर टी है, लास्ट नेम शर्मा। मेरी फ्लाइट लेट है, मुझे क्या मिलेगा?", "hm_omega"),
    ("And how much checked baggage can we take on this flight?", "am_michael"),
    ("ठीक है, और रिफंड चाहिए तो कितने दिन में आएगा?", "hm_omega"),
]


def main() -> None:
    require_api_key()
    t = time.perf_counter()
    warm_retrieval(); tts.warmup(); stt.warmup()
    ready = time.perf_counter() - t
    out_dir = stage_dir("voice") / "audio_hinglish"
    out_dir.mkdir(parents=True, exist_ok=True)
    session, agent, rows = CallSession(mode="demo-smoke"), make_agent(), []
    for n, (line, voice) in enumerate(CALLER, 1):
        audio = resample_poly(tts.synthesize(line, voice=voice), 2, 3).astype(np.float32)
        t0 = time.perf_counter()
        text = stt.transcribe(audio)
        t_stt = time.perf_counter()
        splitter, synth, first_audio = SentenceSplitter(), [], None
        for piece in run_turn(session, agent, text):
            for sentence in splitter.feed(piece):
                synth.append(tts.synthesize(sentence))
                first_audio = first_audio or time.perf_counter()
        for sentence in splitter.flush():
            synth.append(tts.synthesize(sentence))
            first_audio = first_audio or time.perf_counter()
        rec = complete_turn(session, agent, {"stt": round((t_stt - t0) * 1000)})
        wav = out_dir / f"demo_turn{n}_agent.wav"
        sf.write(wav, np.concatenate(synth) if synth else np.zeros(1, dtype=np.float32), CONFIG.tts_rate)
        rows.append({"turn": n, "language": rec["language"], "model": rec["model"], "stt_text": text,
                     "tools": ", ".join(c["name"] for c in rec["tool_calls"]) or "-",
                     "stt_ms": rec["latency_ms"]["stt"], "retrieval_ms": rec["latency_ms"]["retrieval"],
                     "llm_first_text_ms": rec["latency_ms"]["llm_first_text"],
                     "end_of_speech_to_first_audio_ms": round((first_audio - t0) * 1000) if first_audio else None,
                     "agent": rec["assistant_text"], "wav": wav.name})
        print({k: v for k, v in rows[-1].items() if k not in ("agent",)}, "\n  AGENT:", rec["assistant_text"])
    summary = end_call(session, "demo smoke test")
    body = (f"# Demo smoke test — {CONFIG.provider}:{agent.model}, STT {CONFIG.stt_engine}, embed {CONFIG.embed_model}, "
            f"rerank {'on' if CONFIG.rerank else 'off'}\n\nModels ready in {ready:.1f} s. Caller speech is synthetic "
            f"(Kokoro). First audio = time from end of caller speech until the first reply sentence is synthesized "
            f"(playback would start then).\n\n{md_table(rows)}\nSummary figure check: "
            f"{summary and summary['validation']['passed']}\n")
    path = save_artifact("voice", "demo_smoke", body)
    log_entry("voice", "Demo smoke test (Hinglish + English)", md_table(rows, ["turn", "language", "model", "tools",
              "stt_ms", "retrieval_ms", "llm_first_text_ms", "end_of_speech_to_first_audio_ms"]), [path])


if __name__ == "__main__":
    main()
