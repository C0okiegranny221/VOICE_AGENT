import json

import pytest

from rag import tools
from rag.tools import get_booking, normalize_pnr


@pytest.mark.parametrize("spoken, pnr", [("K7Q2PX", "K7Q2PX"), ("k 7 q-2 p x", "K7Q2PX"), ("K seven Q two P X", "K7Q2PX")])
def test_normalize_pnr(spoken, pnr):
    assert normalize_pnr(spoken) == pnr


def test_lookup_needs_matching_last_name():
    assert get_booking("K7Q2PX", None)["need_more_information"]
    wrong_name, wrong_pnr = get_booking("K7Q2PX", "Kapoor"), get_booking("AAAAAA", "Mehta")
    assert not wrong_name["found"] and not wrong_pnr["found"]
    assert "Riya" not in json.dumps(wrong_name)  # nothing leaks on a mismatch
    assert get_booking("k7q2px", "mehta")["found"]


def test_derived_facts_feed_the_policy_tools():
    delayed = get_booking("M4D8RT", "Sharma")["segments"][0]
    assert delayed["route_type"] == "domestic" and delayed["block_time_hours"] == 1.25 and delayed["delay_hours"] == 3.17
    cancelled = get_booking("Z9F3LW", "Qureshi")["segments"][0]
    assert cancelled["route_type"] == "international"
    assert cancelled["block_time_hours"] == 3.92  # 21:30 IST -> 23:55 Dubai time
    assert cancelled["cancellation_notice_hours_before_departure"] == 11.5
    bumped = get_booking("P5V1CJ", "Das")["segments"][0]
    assert bumped["alternate_flight_departs_after_hours"] == 6.42


def test_booking_chain_gives_expected_compensation():
    seg = get_booking("P5V1CJ", "Das")["segments"][0]
    fare = get_booking("P5V1CJ", "Das")["booking"]["one_way_base_fare_plus_fuel_inr"]
    out, _ = tools.run_tool("get_denied_boarding_compensation",
                            {"alternate_flight_departs_after_hours": seg["alternate_flight_departs_after_hours"], "one_way_fare_inr": fare})
    assert json.loads(out)["computed"]["compensation_inr"] == 7600
    seg = get_booking("Z9F3LW", "Qureshi")["segments"][0]
    out, _ = tools.run_tool("get_cancellation_compensation", {"block_time_hours": seg["block_time_hours"], "one_way_fare_inr": 14200})
    assert json.loads(out)["computed"]["compensation_inr"] == 10000


def test_booking_tool_is_registered_and_strict():
    schema = next(t for t in tools.TOOL_SCHEMAS if t["name"] == "get_booking")
    assert schema["strict"] and "get_booking" in tools.TOOLS


def test_look_in_eligibility_is_computed():
    import datetime as dt

    from rag.tools import _load_bookings, _look_in

    b = _load_bookings()["T2H6NB"]  # international, booked 2026-10-15 20:30, departs 2026-10-20 23:45
    seg = get_booking("T2H6NB", "Iyer")["segments"][0]
    now = dt.datetime.fromisoformat("2026-10-16T09:00:00+05:30")
    r = _look_in(b, seg, now)
    assert r["look_in_free_cancellation_available"] is False
    assert r["failed_conditions"] == ["departure_at_least_15_days_after_booking"]
    # A domestic booking made an hour ago for a flight a month out would qualify.
    b2 = {**_load_bookings()["K7Q2PX"], "booked_at": "2026-10-16T08:00:00+05:30"}
    seg2 = {**get_booking("K7Q2PX", "Mehta")["segments"][0], "scheduled_departure": "2026-11-20T07:15:00+05:30"}
    assert _look_in(b2, seg2, now)["look_in_free_cancellation_available"] is True
    # Booked through a travel agent: never eligible.
    b3 = {**b2, "booked_via": "MakeMyTrip (travel agent)"}
    assert "booked_directly_with_indigo" in _look_in(b3, seg2, now)["failed_conditions"]


def test_booking_carries_computed_disruption_entitlements():
    delayed = get_booking("M4D8RT", "Sharma")["disruption_entitlements"]["computed"]
    assert any("meals" in e.lower() for e in delayed["entitlements"])
    assert get_booking("Z9F3LW", "Qureshi")["disruption_entitlements"]["computed"]["compensation_inr"] == 10000
    assert get_booking("P5V1CJ", "Das")["disruption_entitlements"]["computed"]["compensation_inr"] == 7600
    assert "disruption_entitlements" not in get_booking("K7Q2PX", "Mehta")  # on time
