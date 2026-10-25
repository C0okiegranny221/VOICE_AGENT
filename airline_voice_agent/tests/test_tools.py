import json

import pytest

from ingest.build_tables import check_quote
from rag import tools


def call(name, **args):
    out, is_error = tools.run_tool(name, args)
    assert not is_error, out
    return json.loads(out)


def test_domestic_economy_baggage():
    r = call("get_baggage_allowance", route_type="domestic", product="economy")
    checked = [x for x in r["results"] if x["baggage_type"] == "checked" and x["product"] == "economy"]
    assert checked[0]["max_kg"] == 15 and checked[0]["pieces"] == 1
    assert any(x["baggage_type"] == "hand" and x["max_kg"] == 7 for x in r["results"])
    assert r["sources"] and r["review_status"] == "pending human review"


def test_additional_piece_charges_by_route():
    dom = call("get_baggage_charges", route_type="domestic")["results"]
    intl = call("get_baggage_charges", route_type="international")["results"]
    assert {x["amount_inr"] for x in dom if x["item"] == "additional_piece"} == {900, 1000}
    assert {x["amount_inr"] for x in intl if x["item"] == "additional_piece"} == {1800, 2000}


@pytest.mark.parametrize(
    "block, delay, expect_meals",
    [(2.0, 2.0, True), (2.0, 1.5, False), (2.5, 2.0, True), (3.0, 2.5, False), (3.0, 3.0, True), (6.0, 3.5, False), (6.0, 4.0, True)],
)
def test_delay_meal_thresholds(block, delay, expect_meals):
    r = call("get_delay_entitlements", delay_hours=delay, block_time_hours=block, route_type="domestic", scheduled_departure_hhmm=None)
    meals = any("meals" in e.lower() for e in r["computed"]["entitlements"])
    assert meals == expect_meals


def test_delay_over_six_hours_domestic_and_night_hotel():
    r = call("get_delay_entitlements", delay_hours=7, block_time_hours=2, route_type="domestic", scheduled_departure_hhmm="22:30")
    ents = " ".join(r["computed"]["entitlements"]).lower()
    assert "alternate flight within 6 hours or a full refund" in ents
    assert "hotel" in ents
    intl = call("get_delay_entitlements", delay_hours=7, block_time_hours=2, route_type="international", scheduled_departure_hhmm="12:00")
    assert not any("refund" in e.lower() for e in intl["computed"]["entitlements"])


def test_delay_without_block_time_asks_for_it():
    r = call("get_delay_entitlements", delay_hours=3, block_time_hours=None, route_type="domestic", scheduled_departure_hhmm=None)
    assert "need_more_information" in r["computed"]


@pytest.mark.parametrize(
    "after, fare, band, amount",
    [(0.5, 5000, "within_1h", None), (10, 3000, "within_24h", 6000), (10, 8000, "within_24h", 10000),
     (30, 4000, "over_24h", 16000), (30, 9000, "over_24h", 20000), (-1, 4000, "declined", 16000)],
)
def test_denied_boarding_bands_and_caps(after, fare, band, amount):
    c = call("get_denied_boarding_compensation", alternate_flight_departs_after_hours=after, one_way_fare_inr=fare)["computed"]
    assert c["band"] == band
    assert c.get("compensation_inr") == amount


@pytest.mark.parametrize("block, fare, amount", [(1.0, 9000, 5000), (1.5, 9000, 7500), (2.5, 12000, 10000), (2.5, 9000, 9000), (2.5, 4000, 4000)])
def test_cancellation_compensation(block, fare, amount):
    assert call("get_cancellation_compensation", block_time_hours=block, one_way_fare_inr=fare)["computed"]["compensation_inr"] == amount


@pytest.mark.parametrize("route, km, pct", [("domestic", None, 75), ("international", 1200, 30), ("international", 2000, 50), ("international", 5000, 75)])
def test_downgrade(route, km, pct):
    c = call("get_downgrade_reimbursement", route_type=route, distance_km=km, ticket_cost_inr=10000)["computed"]
    assert c["percent_of_ticket_incl_taxes"] == pct and c["reimbursement_inr"] == pct * 100


def test_invalid_input_is_an_error_not_a_crash():
    out, is_error = tools.run_tool("get_baggage_allowance", {"route_type": "mars", "product": "economy"})
    assert is_error and "route_type" in out
    out, is_error = tools.run_tool("get_baggage_allowance", {"route_type": "domestic"})
    assert is_error and "missing" in out


def test_provenance_check_rejects_a_wrong_quote():
    row = {"source_doc": "indigo_coc_domestic", "section": "11.2", "source_quote": "free Checked-in Baggage up to 25kg"}
    assert check_quote(row) == "missing"
    row["source_quote"] = "IndiGo allows free Checked-in Baggage up to 15kg per Customer"
    assert check_quote(row) == "ok"
