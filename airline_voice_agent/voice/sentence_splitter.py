"""Turn streamed LLM text into speakable pieces as early as possible.

A piece is released at sentence end (. ? ! or the Hindi danda ।) followed by whitespace, or at a comma/semicolon once the
buffer is long, so TTS can start before Claude finishes. Abbreviations ("Rs.", "Dr.", "e.g.") and
decimals ("2.5") don't end a sentence.
"""
import re

ABBREVIATIONS = {"rs", "dr", "mr", "mrs", "ms", "no", "vs", "eg", "e.g", "ie", "i.e", "approx", "etc", "st", "inr", "nos"}
END_RE = re.compile(r"(?:[.?!]+|।)[\"')\]]*\s")  # "।" (danda) ends Hindi sentences
SOFT_RE = re.compile(r"[,;:]\s")
SOFT_SPLIT_CHARS = 120
FIRST_SOFT_SPLIT_CHARS = 60  # the first piece of a reply may be cut sooner, so speech starts quickly


class SentenceSplitter:
    def __init__(self, soft_split_chars: int = SOFT_SPLIT_CHARS, first_soft_split_chars: int = FIRST_SOFT_SPLIT_CHARS):
        self.buffer = ""
        self.soft_split_chars = soft_split_chars
        self.first_soft_split_chars = first_soft_split_chars
        self.released = 0

    def _is_abbreviation(self, end: int) -> bool:
        word = re.search(r"([A-Za-z.]+)\.$", self.buffer[: end + 1])
        return bool(word) and word.group(1).lower().rstrip(".") in ABBREVIATIONS

    def feed(self, text: str) -> list[str]:
        """Add streamed text; return any complete pieces."""
        self.buffer += text
        out = []
        while True:
            cut = None
            for m in END_RE.finditer(self.buffer):
                dot = m.start()
                if self.buffer[dot] == "." and self._is_abbreviation(dot):
                    continue
                cut = m.end()
                break
            limit, min_len = (self.first_soft_split_chars, 20) if self.released == 0 else (self.soft_split_chars, 40)
            if cut is None and len(self.buffer) > limit:
                soft = [m.end() for m in SOFT_RE.finditer(self.buffer) if m.end() > min_len]
                cut = soft[0] if self.released == 0 and soft else (soft[-1] if soft else None)
            if cut is None:
                return out
            piece, self.buffer = self.buffer[:cut].strip(), self.buffer[cut:]
            if piece:
                out.append(piece)
                self.released += 1

    def flush(self) -> list[str]:
        piece, self.buffer = self.buffer.strip(), ""
        return [piece] if piece else []
