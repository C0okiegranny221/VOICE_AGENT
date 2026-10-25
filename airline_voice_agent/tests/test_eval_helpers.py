import pytest

from eval.run_eval import contains_all, source_hit, words_to_digits
from eval.voice_benchmark import wer


@pytest.mark.parametrize("text, expected", [
    ("rupees eight thousand", "rupees 8000"), ("twenty one working days", "21 working days"),
    ("one thousand and five hundred", "1500"), ("a hundred millilitres", "100 millilitres"),
    ("INR 1,000 or 900", "inr 1000 or 900"),
])
def test_words_to_digits(text, expected):
    assert words_to_digits(text) == expected


def test_contains_all_accepts_digits_or_words():
    assert contains_all("You get fifteen kilograms and rupees nine hundred.", ["15", "900"]) == (True, [])
    assert contains_all("You get 20 kg.", ["15"]) == (False, ["15"])


def test_source_hit_by_passage_or_tool_table():
    turn = {"retrieved": [{"source": "dgca_car_m4_disruptions › 3. REQUIREMENTS › 3.4 Delays in Flight"}],
            "tool_calls": [{"name": "get_refund_timeline"}]}
    assert source_hit(["dgca :: 3.4"], turn) is True
    assert source_hit(["table :: refund_timelines"], turn) is True
    assert source_hit(["indigo_coc :: 11.2"], turn) is False
    assert source_hit([], turn) is None


def test_wer():
    assert wer("I booked a flight", "I booked a flight") == 0
    assert wer("I booked through an agent", "I book through an agent") == pytest.approx(0.2)
    assert wer("delayed by three hours", "delayed by 3 hours") == 0
