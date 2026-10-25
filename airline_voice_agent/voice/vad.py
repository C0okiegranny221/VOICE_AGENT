"""Turn detection with Silero VAD: feed 512-sample frames at 16 kHz, get back whole utterances."""
from functools import lru_cache

import numpy as np
import torch

from config import CONFIG

FRAME = 512  # samples per Silero frame at 16 kHz (32 ms)
PRE_ROLL_FRAMES = 10  # ~320 ms kept from before speech was detected, so first syllables aren't cut


@lru_cache(maxsize=1)
def _model():
    from silero_vad import load_silero_vad

    return load_silero_vad()


def speech_seconds(audio: np.ndarray, sample_rate: int = 16000) -> float:
    """Seconds of speech Silero finds in a recording. Used to skip Whisper on silence or room noise,
    where it invents text ("Thank you.", "I'm going to go to the next slide.")."""
    from silero_vad import get_speech_timestamps

    if not len(audio):
        return 0.0
    stamps = get_speech_timestamps(torch.from_numpy(audio.astype(np.float32)), _model(), sampling_rate=sample_rate)
    return sum(s["end"] - s["start"] for s in stamps) / sample_rate


class Utterances:
    def __init__(self, sample_rate: int = 16000, threshold: float = 0.5):
        from silero_vad import VADIterator

        self.sample_rate = sample_rate
        self.iterator = VADIterator(_model(), threshold=threshold, sampling_rate=sample_rate,
                                    min_silence_duration_ms=CONFIG.vad_silence_ms, speech_pad_ms=60)
        self.reset()

    def reset(self) -> None:
        self.iterator.reset_states()
        self.buffer = np.zeros(0, dtype=np.float32)
        self.pre_roll: list[np.ndarray] = []
        self.speech: list[np.ndarray] = []
        self.in_speech = False

    def feed(self, samples: np.ndarray) -> np.ndarray | None:
        """Add audio; return a complete utterance when speech ends (and was long enough)."""
        self.buffer = np.concatenate([self.buffer, samples.astype(np.float32)])
        while len(self.buffer) >= FRAME:
            frame, self.buffer = self.buffer[:FRAME], self.buffer[FRAME:]
            event = self.iterator(torch.from_numpy(frame), return_seconds=False)
            if self.in_speech:
                self.speech.append(frame)
            else:
                self.pre_roll = (self.pre_roll + [frame])[-PRE_ROLL_FRAMES:]
            if event and "start" in event and not self.in_speech:
                self.in_speech = True
                self.speech = list(self.pre_roll)
            elif event and "end" in event and self.in_speech:
                audio = np.concatenate(self.speech)
                self.in_speech, self.speech, self.pre_roll = False, [], []
                if len(audio) >= CONFIG.vad_min_speech_ms * self.sample_rate / 1000:
                    return audio
        return None
