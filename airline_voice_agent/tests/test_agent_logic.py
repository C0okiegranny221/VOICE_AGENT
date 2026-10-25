import pytest

from agent import caller_details
from agent.end_detection import is_goodbye


@pytest.mark.parametrize(
    "text, expected",
    [
        ("My flight from Delhi to Dubai was delayed by 3 hours", {"route_type": "international", "delay_hours": 3.0}),
        ("I'm flying domestic, Mumbai to Goa", {"route_type": "domestic"}),
        ("we have a five hour delay", {"delay_hours": 5.0}),
        ("It's running 2.5 hours late", {"delay_hours": 2.5}),
        ("I booked through MakeMyTrip", {"payment_mode": "travel_agent"}),
        ("paid with my Amex card", {"payment_mode": "american_express"}),
        ("travelling with my baby", {"passenger_type": "infant"}),
        ("I have an IndiGo Stretch seat", {"product": "stretch"}),
        ("my PNR is ab12cd", {"pnr": "AB12CD"}),
    ],
)
def test_extract(text, expected):
    got = caller_details.extract(text)
    for k, v in expected.items():
        assert got.get(k) == v, (k, got)


def test_update_keeps_earlier_details_and_newest_wins():
    d = caller_details.update({}, "flying domestic tomorrow")
    d = caller_details.update(d, "paid by card")
    d = caller_details.update(d, "sorry, it's actually an international flight to Singapore")
    assert d == {"route_type": "international", "payment_mode": "card_or_wallet"}


@pytest.mark.parametrize(
    "text, expected",
    [
        ("That's all, thanks", True),
        ("Okay bye", True),
        ("No, nothing else. Thank you!", True),
        ("That's all I wanted, but what about my refund?", False),
        ("bye the way, how much is extra baggage", False),
        ("Can you tell me the baggage allowance", False),
        ("", False),
    ],
)
def test_is_goodbye(text, expected):
    assert is_goodbye(text) == expected
