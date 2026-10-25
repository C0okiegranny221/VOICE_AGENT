"""Microphone input and speaker output.

- Player: plays float32 chunks from a queue on one OutputStream; `speaking` is set while audio plays.
- Speaker: TTS worker thread - text pieces in, audio to the Player, so sentence N+1 is synthesized
  while sentence N plays.
- MicRecorder: one input stream for the whole call; push-to-talk (Enter to start, Enter to stop) or
  VAD-based turn detection. Mic input is ignored while the Player is speaking (laptop speakers would
  otherwise be heard by the mic).
"""
import queue
import threading
import time

import numpy as np
import sounddevice as sd

from config import CONFIG
from voice import tts


class Player:
    def __init__(self, sample_rate: int = CONFIG.tts_rate):
        self.sample_rate = sample_rate
        self.q: queue.Queue[np.ndarray] = queue.Queue()
        self._current = np.zeros(0, dtype=np.float32)
        self.speaking = threading.Event()
        self.first_audio_at: float | None = None
        self.last_audio_at = 0.0
        self._stream = sd.OutputStream(samplerate=sample_rate, channels=1, dtype="float32", callback=self._callback,
                                       blocksize=1024)
        self._stream.start()

    def _callback(self, outdata, frames, _time, _status):
        out = np.zeros(frames, dtype=np.float32)
        filled = 0
        while filled < frames:
            if not len(self._current):
                try:
                    self._current = self.q.get_nowait()
                except queue.Empty:
                    break
            take = min(frames - filled, len(self._current))
            out[filled : filled + take] = self._current[:take]
            self._current = self._current[take:]
            filled += take
        if filled:
            self.speaking.set()
            self.last_audio_at = time.perf_counter()
            if self.first_audio_at is None:
                self.first_audio_at = self.last_audio_at
        elif self.q.empty():
            self.speaking.clear()
        outdata[:, 0] = out

    def play(self, audio: np.ndarray) -> None:
        if len(audio):
            self.speaking.set()
            self.q.put(audio.astype(np.float32))

    def mark(self) -> None:
        """Start timing a new turn (first_audio_at is set by the next audio played)."""
        self.first_audio_at = None

    def wait_done(self, tail_s: float = 0.25) -> None:
        while not self.q.empty() or len(self._current):
            time.sleep(0.02)
        time.sleep(tail_s)
        self.speaking.clear()

    def close(self) -> None:
        self._stream.abort()  # see MicRecorder: stop() can block on macOS
        self._stream.close()


class Speaker:
    """Text pieces -> Kokoro -> Player, on a background thread."""

    def __init__(self, player: Player):
        self.player = player
        self.q: queue.Queue[str | None] = queue.Queue()
        self.synth_ms: list[float] = []
        self._idle = threading.Event()
        self._idle.set()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        while True:
            text = self.q.get()
            if text is None:
                return
            t0 = time.perf_counter()
            audio = tts.synthesize(text)
            self.synth_ms.append((time.perf_counter() - t0) * 1000)
            self.player.play(audio)
            if self.q.empty():
                self._idle.set()

    def say(self, text: str) -> None:
        self._idle.clear()
        self.q.put(text)

    def wait_done(self) -> None:
        self._idle.wait()
        self.player.wait_done()


class MicRecorder:
    """One input stream for the whole call. It is opened once and never stopped mid-call: on macOS,
    stopping an input stream while the output stream plays can block forever inside PortAudio/CoreAudio
    (seen in the first live call, hung in InputStream.stop()). Push-to-talk and VAD just read from the
    running stream's queue."""

    def __init__(self, player: Player | None = None, sample_rate: int = CONFIG.mic_rate):
        self.player = player
        self.sample_rate = sample_rate
        self.q: queue.Queue[np.ndarray] = queue.Queue()
        self._stream = sd.InputStream(samplerate=sample_rate, channels=1, dtype="float32", blocksize=512,
                                      callback=self._callback)
        self._stream.start()

    def _callback(self, indata, _frames, _time, _status):
        self.q.put(indata[:, 0].copy())

    def _drain(self) -> list[np.ndarray]:
        frames = []
        while True:
            try:
                frames.append(self.q.get_nowait())
            except queue.Empty:
                return frames

    def _muted(self) -> bool:
        if self.player is None:
            return False
        return self.player.speaking.is_set() or time.perf_counter() - self.player.last_audio_at < 0.3

    def push_to_talk(self) -> np.ndarray:
        """Enter starts recording, Enter stops. Returns 16 kHz float32 audio."""
        input("  [press Enter and speak]")
        self._drain()  # discard audio from before Enter
        input("  [recording... press Enter when done]")
        frames = self._drain()
        return np.concatenate(frames) if frames else np.zeros(0, dtype=np.float32)

    def listen_vad(self, idle_timeout_s: float | None = None) -> np.ndarray | None:
        """Wait for speech, return it once ~vad_silence_ms of silence follows. None on idle timeout."""
        from voice.vad import Utterances

        detector = Utterances(self.sample_rate)
        self._drain()
        deadline = time.perf_counter() + idle_timeout_s if idle_timeout_s else None
        while True:
            try:
                frame = self.q.get(timeout=0.1)
            except queue.Empty:
                frame = None
            if frame is not None:
                if self._muted():
                    detector.reset()
                    continue
                utterance = detector.feed(frame)
                if utterance is not None:
                    return utterance
            if deadline and not detector.in_speech and time.perf_counter() > deadline:
                return None

    def close(self) -> None:
        self._stream.abort()  # abort() doesn't wait on the driver the way stop() does
        self._stream.close()


def device_names() -> str:
    inp, out = sd.default.device
    return f"mic: {sd.query_devices(inp)['name']} · speaker: {sd.query_devices(out)['name']}"
