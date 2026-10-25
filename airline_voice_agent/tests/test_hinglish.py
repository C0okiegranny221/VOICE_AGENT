import pytest

from agent import caller_details
from agent.end_detection import is_goodbye
from agent.language import detect
from rag.tools import normalize_pnr
from voice.sentence_splitter import SentenceSplitter
from voice.tts import is_hindi, normalize


@pytest.mark.parametrize("text, lang", [
    ("Meri flight tin ghante late hai, mujhe kya milega?", "hinglish"),
    ("Refund kitne din mein aaega?", "hinglish"),
    ("मेरी फ्लाइट कैंसिल हो गई", "hinglish"),
    ("How much cabin baggage can I carry?", "english"),
    ("Can you tell me the baggage allowance to Delhi", "english"),
    ("Do I need to go to the main counter?", "english"),  # 'do', 'to', 'main' alone are not Hindi
])
def test_detect_language(text, lang):
    assert detect(text) == lang


@pytest.mark.parametrize("text, expected", [
    ("Meri flight tin ghante late hai", {"delay_hours": 3.0}),
    ("flight dedh ghanta late hai", {"delay_hours": 1.5}),
    ("Domastic flight mein kitna baggage allowed hai?", {"route_type": "domestic"}),
    ("videsh ja raha hoon", {"route_type": "international"}),
])
def test_hinglish_caller_details(text, expected):
    got = caller_details.extract(text)
    for k, v in expected.items():
        assert got.get(k) == v, got


def test_no_false_delay_from_word_endings():
    assert "delay_hours" not in caller_details.extract("kitna ghante ka flight hai")


@pytest.mark.parametrize("text, expected", [
    ("Bas itna hi, dhanyavaad", True), ("Theek hai, shukriya", True), ("धन्यवाद", True),
    ("Dhanyavaad, par refund kitne din mein aaega?", False),
])
def test_hinglish_goodbye(text, expected):
    assert is_goodbye(text) == expected


def test_glued_pnr():
    assert normalize_pnr("PNRM 4D8RT") == "M4D8RT"


def test_danda_splits_hindi_sentences():
    s = SentenceSplitter()
    out = s.feed("आपका बैगेज 15 किलो तक फ्री है। एक्स्ट्रा बैग के लिए 900 रुपये लगेंगे। ") + s.flush()
    assert out == ["आपका बैगेज 15 किलो तक फ्री है।", "एक्स्ट्रा बैग के लिए 900 रुपये लगेंगे।"]


def test_hindi_tts_normalization():
    assert is_hindi("बैगेज") and not is_hindi("baggage")
    assert normalize("एक्स्ट्रा बैग ₹1,000 और 7 kg") == "एक्स्ट्रा बैग 1000 रुपये और 7 किलो"
