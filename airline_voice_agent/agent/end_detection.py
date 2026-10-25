"""Decide when the caller is ending the call."""
import re

GOODBYE_RE = re.compile(
    r"\b(bye|goodbye|good bye|that'?s all|that is all|that will be all|nothing else|no more questions|"
    r"i'?m done|we'?re done|end the call|hang up|"
    r"bas itna hi|bas itna|bas yahi|aur kuch nahi|kuch nahi chahiye|dhanyavaad|dhanyawad|dhanyavad|shukriya|alvida)\b"
    r"|धन्यवाद|शुक्रिया|बस इतना",
    re.I,
)
QUESTION_WORDS = re.compile(
    r"\b(what|how|when|why|where|which|can|could|do|does|is|are|will|should|kya|kaise|kab|kitna|kitne|kyun|kahan|kaun)\b",
    re.I)


def is_goodbye(text: str) -> bool:
    """A short utterance with a farewell phrase and no new question.

    "That's all, thanks" -> True;  "That's all I wanted to ask, but what about my refund?" -> False.
    """
    text = text.strip()
    if not text or not GOODBYE_RE.search(text):
        return False
    if "?" in text:
        return False
    after = GOODBYE_RE.split(text, maxsplit=1)[-1]
    if QUESTION_WORDS.search(after):
        return False
    return len(text.split()) <= 12
