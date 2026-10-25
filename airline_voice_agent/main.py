"""Voice call on the laptop mic and speakers.

    python main.py            # push-to-talk: Enter to start speaking, Enter to stop
    python main.py --vad      # hands-free: speech detection decides when you've finished
    PROVIDER=claude python main.py   # provider comes from PROVIDER in .env (gemini by default)
    python main.py --model gemini-3.5-flash-lite --effort low

Flow per turn: mic -> Whisper -> retrieval -> LLM (streamed) -> sentence splitter -> Kokoro -> speaker.
The call ends on a goodbye, Ctrl+C, or (VAD mode) 60 s of silence; a summary is then written.
If it ever seems stuck, press Ctrl+\\ : it prints where every thread is, without quitting.
"""
import argparse
import faulthandler
import signal
import time

import numpy as np

from agent.end_detection import is_goodbye
from agent.llm import make_agent
from agent.pipeline import complete_turn, end_call, run_turn
from agent.prompts import GREETING, farewell
from agent.session import CallSession
from config import CONFIG, require_api_key
from rag.embed import warmup as warm_retrieval
from voice import stt, tts
from voice.audio_io import MicRecorder, Player, Speaker, device_names
from voice.sentence_splitter import SentenceSplitter
from voice.vad import speech_seconds

MIN_SPEECH_S = 0.3


def speak_and_wait(speaker: Speaker, text: str) -> None:
    """Speak a fixed message sentence by sentence, so Hindi and English sentences each get the right voice."""
    splitter = SentenceSplitter()
    for sentence in splitter.feed(text + " ") + splitter.flush():
        speaker.say(sentence)
    speaker.wait_done()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vad", action="store_true", help="hands-free turn detection instead of push-to-talk")
    ap.add_argument("--model", default=None)
    ap.add_argument("--effort", default=None)
    args = ap.parse_args()
    require_api_key()
    faulthandler.register(signal.SIGQUIT, all_threads=True, chain=False)  # Ctrl+\\ -> stack dump

    print("[loading models...]")
    t0 = time.perf_counter()
    warm_retrieval()
    tts.warmup()
    stt.warmup()
    print(f"[ready in {time.perf_counter() - t0:.1f} s]")

    player = Player()
    speaker = Speaker(player)
    mic = MicRecorder(player)
    session = CallSession(mode="voice-vad" if args.vad else "voice-ptt")
    agent = make_agent(model=args.model, effort=args.effort)
    print(f"[{device_names()}]")
    print(f"[call {session.call_id} · {CONFIG.provider}:{agent.model}]\nAGENT: {GREETING}")
    speak_and_wait(speaker, GREETING)

    reason = "interrupted"
    try:
        while True:
            audio = mic.listen_vad(CONFIG.idle_timeout_s) if args.vad else mic.push_to_talk()
            if audio is None:
                reason = "idle timeout"
                speak_and_wait(speaker, "I haven't heard anything for a while, so I'll end the call here. " + farewell("english"))
                break
            t_end_of_speech = time.perf_counter()
            seconds, level = len(audio) / CONFIG.mic_rate, float(np.abs(audio).max(initial=0))
            speech = speech_seconds(audio, CONFIG.mic_rate)
            print(f"  [heard {seconds:.1f} s, speech {speech:.1f} s, mic level {level:.3f}]")
            if speech < MIN_SPEECH_S:
                hint = " - the mic level is very low; check System Settings > Privacy > Microphone for your terminal" \
                    if level < 0.01 else ""
                print(f"  [no speech detected, please try again{hint}]")
                continue
            print("  [transcribing...]", flush=True)
            text = stt.transcribe(audio)
            t_stt = time.perf_counter()
            if not text:
                print("  [didn't catch that]")
                continue
            print(f"\nYOU: {text}")
            if is_goodbye(text):
                print(f"AGENT: {farewell(session.language)}")
                speak_and_wait(speaker, farewell(session.language))
                reason = "goodbye"
                break

            player.mark()
            splitter = SentenceSplitter()
            print("AGENT: ", end="", flush=True)
            for piece in run_turn(session, agent, text):
                print(piece, end="", flush=True)
                for sentence in splitter.feed(piece):
                    speaker.say(sentence)
            for sentence in splitter.flush():
                speaker.say(sentence)
            speaker.wait_done()

            first_audio = player.first_audio_at
            latency = {
                "audio_s": round(len(audio) / CONFIG.mic_rate, 2),
                "stt": round((t_stt - t_end_of_speech) * 1000),
                "end_of_speech_to_first_audio": round((first_audio - t_end_of_speech) * 1000) if first_audio else None,
            }
            t = complete_turn(session, agent, latency)
            print(f"\n  [stt {latency['stt']} ms · retrieval {t['latency_ms']['retrieval']} ms · "
                  f"first text {t['latency_ms']['llm_first_text']} ms · first audio "
                  f"{latency['end_of_speech_to_first_audio']} ms after you stopped]")
    except KeyboardInterrupt:
        reason = "interrupted"
    finally:
        mic.close()
        player.close()

    print("\n[ending call, writing summary...]")
    if end_call(session, reason):
        print((session.dir / "summary.md").read_text())


if __name__ == "__main__":
    main()
