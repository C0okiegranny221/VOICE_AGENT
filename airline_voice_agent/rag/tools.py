"""Policy-table tools for Claude. Each tool reads data/policy.db and returns rows plus their sources.

Disruption tools (delays, denied boarding, cancellation, downgrade) also *compute* the entitlement
from the caller's facts, so amounts come from the DGCA bands rather than from the model's arithmetic.
Every result includes `sources` (doc, section, page, quote) and `review_status`.
"""
import json
import sqlite3
from functools import lru_cache

from config import CONFIG

ROUTE = {"type": "string", "enum": ["domestic", "international"], "description": "Domestic (within India) or international flight."}


def _nullable_number(description: str) -> dict:
    return {"type": ["number", "null"], "description": description + " Use null if the caller has not said."}


TOOL_SCHEMAS = [
    {
        "name": "get_baggage_allowance",
        "description": "Free checked baggage, hand baggage, personal article and liquids limits for a route and fare product.",
        "input_schema": {
            "type": "object",
            "properties": {
                "route_type": ROUTE,
                "product": {"type": "string", "enum": ["economy", "stretch", "business", "lite_fare"],
                            "description": "Fare product. Use economy unless the caller mentions Stretch, business class or a Lite fare."},
            },
            "required": ["route_type", "product"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_baggage_charges",
        "description": "Charges for additional checked bags (pre-booked or at the airport) and call-centre surcharges.",
        "input_schema": {
            "type": "object",
            "properties": {"route_type": ROUTE},
            "required": ["route_type"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_change_cancellation_rules",
        "description": "Deadlines and rules for changing or cancelling a booking: minimum time before departure, check-in "
                       "no-show cutoff, 48-hour look-in, free name spelling correction window, call-centre fee.",
        "input_schema": {
            "type": "object",
            "properties": {"route_type": ROUTE},
            "required": ["route_type"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_refund_timeline",
        "description": "How long a refund takes, by how the ticket was paid for.",
        "input_schema": {
            "type": "object",
            "properties": {
                "payment_mode": {"type": "string", "enum": ["card_or_wallet", "american_express", "travel_agent", "cash_at_airport"]},
                "route_type": ROUTE,
            },
            "required": ["payment_mode", "route_type"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_delay_entitlements",
        "description": "What a passenger is owed for a flight delay under DGCA rules (meals, alternate flight or refund, hotel). "
                       "Computes the applicable entitlements from block time and delay length.",
        "input_schema": {
            "type": "object",
            "properties": {
                "delay_hours": {"type": "number", "description": "Expected or actual delay in hours."},
                "block_time_hours": _nullable_number("Scheduled flight duration in hours."),
                "route_type": ROUTE,
                "scheduled_departure_hhmm": {"type": ["string", "null"], "description": "Scheduled departure time as HH:MM (24h), or null."},
            },
            "required": ["delay_hours", "block_time_hours", "route_type", "scheduled_departure_hhmm"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_denied_boarding_compensation",
        "description": "DGCA compensation when a passenger with a confirmed booking is denied boarding because the flight is overbooked.",
        "input_schema": {
            "type": "object",
            "properties": {
                "alternate_flight_departs_after_hours": _nullable_number(
                    "Hours between the original departure and the alternate flight's departure. Use -1 if the passenger declined an alternate flight."),
                "one_way_fare_inr": _nullable_number("Booked one-way basic fare plus fuel charge in rupees."),
            },
            "required": ["alternate_flight_departs_after_hours", "one_way_fare_inr"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_cancellation_compensation",
        "description": "DGCA compensation when the airline cancels a flight without the required notice.",
        "input_schema": {
            "type": "object",
            "properties": {
                "block_time_hours": _nullable_number("Scheduled flight duration in hours."),
                "one_way_fare_inr": _nullable_number("Booked one-way basic fare plus fuel charge in rupees."),
            },
            "required": ["block_time_hours", "one_way_fare_inr"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_downgrade_reimbursement",
        "description": "DGCA reimbursement when a passenger is involuntarily moved to a lower class.",
        "input_schema": {
            "type": "object",
            "properties": {
                "route_type": ROUTE,
                "distance_km": _nullable_number("Flight distance in km (international only)."),
                "ticket_cost_inr": _nullable_number("Ticket cost including taxes in rupees."),
            },
            "required": ["route_type", "distance_km", "ticket_cost_inr"],
            "additionalProperties": False,
        },
    },
]
TOOL_SCHEMAS.append({
    "name": "get_booking",
    "description": "Look up a booking by PNR and the passenger's last name (both required; ask the caller for any "
                   "that is missing). Returns flights, times, status (on time / delayed / cancelled / denied boarding), "
                   "flight duration, delay, fare and how it was paid. Use these facts as inputs to the entitlement tools. "
                   "These are fictional demo bookings; nothing can be changed.",
    "input_schema": {
        "type": "object",
        "properties": {
            "pnr": {"type": "string", "description": "6-character booking reference, letters and digits, e.g. K7Q2PX."},
            "last_name": {"type": ["string", "null"], "description": "Passenger last name on the booking, or null if not given yet."},
        },
        "required": ["pnr", "last_name"],
        "additionalProperties": False,
    },
})
for _tool in TOOL_SCHEMAS:
    _tool["strict"] = True

INTERNAL_COLUMNS = {"row_id", "source_doc", "section", "page", "source_quote", "human_verified_on", "provenance"}


@lru_cache(maxsize=1)
def _db() -> sqlite3.Connection:
    con = sqlite3.connect(CONFIG.policy_db, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con


def _query(sql: str, params: tuple = ()) -> list[dict]:
    return [dict(r) for r in _db().execute(sql, params).fetchall()]


def _package(rows: list[dict], computed: dict | None = None, note: str | None = None) -> dict:
    sources, seen = [], set()
    for r in rows:
        key = (r["source_doc"], r["section"], r["source_quote"])
        if key not in seen:
            seen.add(key)
            sources.append({"doc": r["source_doc"], "section": r["section"], "page": r["page"], "quote": r["source_quote"]})
    result = {
        "found": bool(rows),
        "results": [{k: v for k, v in r.items() if k not in INTERNAL_COLUMNS and v is not None} for r in rows],
        "sources": sources,
        "review_status": "verified" if rows and all(r["human_verified_on"] for r in rows) else "pending human review",
    }
    if computed is not None:
        result["computed"] = computed
    if note:
        result["note"] = note
    if not rows:
        result["reason"] = "No matching rule in the policy tables."
    return result


def get_baggage_allowance(route_type: str, product: str) -> dict:
    rows = _query(
        "SELECT * FROM baggage_allowance WHERE route_type IN (?, 'any') AND product IN (?, 'any', 'multi_seat_addon') ORDER BY row_id",
        (route_type, product),
    )
    note = None
    if route_type == "international" and product in ("stretch", "business"):
        note = "The collected documents give no separate international Stretch/business allowance; the economy row is shown."
        rows = _query("SELECT * FROM baggage_allowance WHERE route_type IN (?, 'any') AND product IN ('economy', 'any', 'multi_seat_addon') "
                      "ORDER BY row_id", (route_type,))
    return _package(rows, note=note)


def get_baggage_charges(route_type: str) -> dict:
    return _package(_query("SELECT * FROM baggage_charges WHERE route_type IN (?, 'any') ORDER BY row_id", (route_type,)),
                    note="Per-kg excess baggage rates are not in the collected documents.")


def get_change_cancellation_rules(route_type: str) -> dict:
    return _package(_query("SELECT * FROM change_cancellation_rules WHERE route_type IN (?, 'any') ORDER BY row_id", (route_type,)))


def get_refund_timeline(payment_mode: str, route_type: str) -> dict:
    return _package(_query("SELECT * FROM refund_timelines WHERE payment_mode = ? AND route_type IN (?, 'any')",
                           (payment_mode, route_type)))


def _in_night_window(hhmm: str | None) -> bool:
    if not hhmm:
        return False
    try:
        h, m = (int(x) for x in hhmm.split(":"))
    except ValueError:
        return False
    minutes = h * 60 + m
    return minutes >= 20 * 60 or minutes <= 3 * 60


def get_delay_entitlements(delay_hours: float, block_time_hours: float | None, route_type: str,
                           scheduled_departure_hhmm: str | None) -> dict:
    rows = _query("SELECT * FROM delay_entitlements ORDER BY row_id")
    applies, needs = [], []
    for r in rows:
        if r["rule"] == "meals":
            if block_time_hours is None:
                continue
            lo, hi = r["block_time_min_h"] or 0, r["block_time_max_h"]
            in_band = (block_time_hours > lo or lo == 0) and (hi is None or block_time_hours <= hi)
            if in_band and delay_hours >= r["delay_min_h"]:
                applies.append(r)
        elif r["rule"] == "alternate_or_refund":
            if route_type == "domestic" and delay_hours > r["delay_min_h"]:
                applies.append(r)
        elif r["rule"] == "hotel":
            if r["departure_window"]:
                if delay_hours > r["delay_min_h"] and _in_night_window(scheduled_departure_hhmm):
                    applies.append(r)
            elif delay_hours > r["delay_min_h"]:
                applies.append(r)
    if block_time_hours is None:
        needs.append("block_time_hours (flight duration) to decide when meals are owed")
    computed = {
        "entitlements": sorted({r["entitlement"] for r in applies}),
        "conditions": "Only if the passenger checked in on time; not owed when the delay is caused by extraordinary "
                      "circumstances beyond the airline's control (weather, ATC, security, etc.).",
    }
    if needs:
        computed["need_more_information"] = needs
    # Return the applicable rows; if nothing applies yet, return all rows so the model can explain thresholds.
    return _package(applies or rows, computed)


def get_denied_boarding_compensation(alternate_flight_departs_after_hours: float | None, one_way_fare_inr: float | None) -> dict:
    rows = _query("SELECT * FROM denied_boarding_compensation ORDER BY row_id")
    h = alternate_flight_departs_after_hours
    if h is None:
        return _package(rows, {"need_more_information": ["when the alternate flight departs, or whether the passenger declined it"]})
    band = "declined" if h < 0 else "within_1h" if h <= 1 else "within_24h" if h <= 24 else "over_24h"
    row = next(r for r in rows if r["alternate_flight"] == band)
    computed = {"band": band, "percent_of_fare": row["percent_of_fare"], "cap_inr": row["cap_inr"],
                "plus_full_refund": row["plus_full_refund"] == "yes"}
    if one_way_fare_inr is not None and row["percent_of_fare"]:
        computed["compensation_inr"] = round(min(one_way_fare_inr * row["percent_of_fare"] / 100, row["cap_inr"]))
    elif row["percent_of_fare"]:
        computed["need_more_information"] = ["one-way basic fare plus fuel charge, to compute the exact amount"]
    return _package([row], computed)


def get_cancellation_compensation(block_time_hours: float | None, one_way_fare_inr: float | None) -> dict:
    rows = _query("SELECT * FROM cancellation_compensation ORDER BY row_id")
    if block_time_hours is None:
        return _package(rows, {"need_more_information": ["block_time_hours (flight duration)"]})
    row = next(r for r in rows if (block_time_hours > (r["block_time_min_h"] or 0) or r["block_time_min_h"] == 0)
               and (r["block_time_max_h"] is None or block_time_hours <= r["block_time_max_h"]))
    computed = {"max_compensation_inr": row["amount_inr"], "plus_full_refund": True}
    if one_way_fare_inr is not None:
        computed["compensation_inr"] = round(min(row["amount_inr"], one_way_fare_inr))
    return _package([row], computed)


def get_downgrade_reimbursement(route_type: str, distance_km: float | None, ticket_cost_inr: float | None) -> dict:
    rows = _query("SELECT * FROM downgrade_reimbursement WHERE route_type = ? ORDER BY row_id", (route_type,))
    if route_type == "international":
        if distance_km is None:
            return _package(rows, {"need_more_information": ["distance_km of the flight"]})
        rows = [r for r in rows if (distance_km > (r["distance_min_km"] or 0) or r["distance_min_km"] == 0)
                and (r["distance_max_km"] is None or distance_km <= r["distance_max_km"])]
    row = rows[0]
    computed = {"percent_of_ticket_incl_taxes": row["percent_of_ticket"]}
    if ticket_cost_inr is not None:
        computed["reimbursement_inr"] = round(ticket_cost_inr * row["percent_of_ticket"] / 100)
    return _package([row], computed)


def _load_bookings() -> dict:
    data = json.loads(CONFIG.mock_bookings.read_text())
    return {b["pnr"]: b for b in data["bookings"]}


_DIGIT_WORDS = {"zero": "0", "oh": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
                "six": "6", "seven": "7", "eight": "8", "nine": "9"}


def normalize_pnr(text: str) -> str:
    """'k 7 q-2 p x' / 'K seven Q two P X' -> 'K7Q2PX'."""
    import re

    parts = re.findall(r"[A-Za-z]+|\d", text)
    pnr = "".join(_DIGIT_WORDS.get(p.lower(), p) for p in parts).upper()
    # Speech recognition sometimes glues the word on: "PNRM4D8RT" -> "M4D8RT".
    return pnr[3:] if pnr.startswith("PNR") and len(pnr) == 9 else pnr


def _hours_between(a: str, b: str) -> float:
    import datetime as dt

    return round((dt.datetime.fromisoformat(b) - dt.datetime.fromisoformat(a)).total_seconds() / 3600, 2)


INDIAN_AIRPORTS = {"DEL", "BOM", "BLR", "HYD", "MAA", "CCU", "GAU", "PNQ", "GOI", "AMD", "COK", "JAI", "LKO"}


def get_booking(pnr: str, last_name: str | None) -> dict:
    bookings = _load_bookings()
    key = normalize_pnr(pnr)
    if not last_name:
        return {"found": False, "need_more_information": ["the passenger's last name on the booking"]}
    b = bookings.get(key)
    if b is None or not any(p["last_name"].lower() == last_name.strip().lower() for p in b["passengers"]):
        # Same answer for a wrong PNR and a wrong name, so the lookup can't be used to probe bookings.
        return {"found": False, "reason": f"No booking matches PNR {key} with last name {last_name}. Ask the caller to check both."}
    segments = []
    for seg in b["segments"]:
        out = {k: v for k, v in seg.items()}
        out["route_type"] = "domestic" if {seg["from"], seg["to"]} <= INDIAN_AIRPORTS else "international"
        out["block_time_hours"] = _hours_between(seg["scheduled_departure"], seg["scheduled_arrival"])
        if seg.get("delay_minutes"):
            out["delay_hours"] = round(seg["delay_minutes"] / 60, 2)
        if seg.get("cancellation_notified_at"):
            out["cancellation_notice_hours_before_departure"] = _hours_between(seg["cancellation_notified_at"], seg["scheduled_departure"])
        if seg.get("alternate_flight"):
            out["alternate_flight_departs_after_hours"] = _hours_between(seg["scheduled_departure"], seg["alternate_flight"]["departure"])
        segments.append(out)
    booking = {k: v for k, v in b.items() if k not in ("scenario", "segments")}
    booking["passengers"] = [{k: v for k, v in p.items() if k != "last_name"} for p in b["passengers"]]
    result = {"found": True, "booking": booking, "segments": segments,
              "free_cancellation": _look_in(b, segments[0]),
              "note": "Fictional demo booking. Times are local to each airport."}
    if entitlements := _disruption_entitlements(b, segments[0]):
        result["disruption_entitlements"] = entitlements
        result["note"] += " disruption_entitlements are already computed from the policy tables; use them directly."
    return result


def _disruption_entitlements(b: dict, seg: dict) -> dict | None:
    """For a delayed / cancelled / denied-boarding flight, run the matching policy tool on the booking's facts,
    so the agent needs no extra tool round (each round costs ~1-2 s on the free tier)."""
    fare = b.get("one_way_base_fare_plus_fuel_inr")
    if seg["status"] == "delayed" and seg.get("delay_hours"):
        hhmm = seg["scheduled_departure"][11:16]
        out = get_delay_entitlements(seg["delay_hours"], seg["block_time_hours"], seg["route_type"], hhmm)
    elif seg["status"] == "cancelled":
        notice = seg.get("cancellation_notice_hours_before_departure")
        if notice is not None and notice >= 24:
            return {"computed": {"compensation": "none - the passenger was informed at least 24 hours before departure; "
                                                 "they may choose an alternate flight or a refund"}}
        out = get_cancellation_compensation(seg["block_time_hours"], fare)
        out["computed"]["also"] = "or an alternate flight acceptable to the passenger, instead of the refund"
    elif seg["status"] == "denied_boarding":
        out = get_denied_boarding_compensation(seg.get("alternate_flight_departs_after_hours"), fare)
    else:
        return None
    return {"computed": out.get("computed"), "sources": out.get("sources")}


def _rule(route_type: str, rule: str) -> dict:
    return _query("SELECT * FROM change_cancellation_rules WHERE route_type = ? AND rule = ?", (route_type, rule))[0]


def _look_in(b: dict, seg: dict, now=None) -> dict:
    """Whether the 48-hour look-in (free change/cancellation) applies to this booking right now, with the reason.
    Thresholds come from the change_cancellation_rules table."""
    import datetime as dt

    route = seg["route_type"]
    window_h = _rule(route, "look_in_window")["value"]
    min_days = _rule(route, "look_in_min_days_booking_to_departure")["value"]
    cutoff_h = _rule(route, "min_time_before_departure_for_change_or_cancel")["value"]
    booked, departs = dt.datetime.fromisoformat(b["booked_at"]), dt.datetime.fromisoformat(seg["scheduled_departure"])
    now = now or dt.datetime.now(dt.timezone.utc)
    hours_since_booking = round((now - booked).total_seconds() / 3600, 1)
    days_booking_to_departure = round((departs - booked).total_seconds() / 86400, 1)
    direct = b["booked_via"].startswith("indigo_")
    checks = {
        "booked_directly_with_indigo": direct,
        f"within_{window_h}_hours_of_booking": hours_since_booking <= window_h,
        f"departure_at_least_{min_days}_days_after_booking": days_booking_to_departure >= min_days,
    }
    eligible = all(checks.values())
    reasons = [k for k, ok in checks.items() if not ok]
    why = {
        "booked_directly_with_indigo": "it was not booked directly on IndiGo's website or app",
        f"within_{window_h}_hours_of_booking": f"it was booked {hours_since_booking:.0f} hours ago, more than {window_h} hours",
        f"departure_at_least_{min_days}_days_after_booking":
            f"the flight departs {days_booking_to_departure:.0f} days after booking and {route} bookings need at least {min_days} days",
    }
    if eligible:
        verdict = (f"YES, free cancellation or change is available now under the {window_h}-hour look-in "
                   f"(any fare difference on a change is still payable).")
    else:
        verdict = (f"NO free cancellation: the {window_h}-hour look-in does not apply because "
                   + " and ".join(why[r] for r in reasons) + f". The fare type's cancellation fee applies; "
                   f"cancel at least {cutoff_h} hours before departure.")
    return {
        "verdict": verdict,
        "look_in_free_cancellation_available": eligible,
        "failed_conditions": reasons,
        "hours_since_booking": hours_since_booking,
        "days_from_booking_to_departure": days_booking_to_departure,
        "otherwise": f"the fare type's change/cancellation fee applies; changes or cancellations must be made at least "
                     f"{cutoff_h} hours before departure",
    }


TOOLS = {
    "get_booking": get_booking,
    "get_baggage_allowance": get_baggage_allowance,
    "get_baggage_charges": get_baggage_charges,
    "get_change_cancellation_rules": get_change_cancellation_rules,
    "get_refund_timeline": get_refund_timeline,
    "get_delay_entitlements": get_delay_entitlements,
    "get_denied_boarding_compensation": get_denied_boarding_compensation,
    "get_cancellation_compensation": get_cancellation_compensation,
    "get_downgrade_reimbursement": get_downgrade_reimbursement,
}
_SCHEMA_BY_NAME = {t["name"]: t["input_schema"] for t in TOOL_SCHEMAS}
_JSON_TYPES = {"string": str, "number": (int, float), "null": type(None), "boolean": bool}


def validate_input(name: str, args: dict) -> str | None:
    """Return an error message if args don't match the tool's schema, else None."""
    schema = _SCHEMA_BY_NAME.get(name)
    if schema is None:
        return f"unknown tool {name}"
    if not isinstance(args, dict):
        return "input must be an object"
    extra = set(args) - set(schema["properties"])
    missing = set(schema["required"]) - set(args)
    if extra or missing:
        return f"unexpected {sorted(extra)} / missing {sorted(missing)}"
    for key, spec in schema["properties"].items():
        types = spec["type"] if isinstance(spec["type"], list) else [spec["type"]]
        value = args[key]
        if not any(isinstance(value, _JSON_TYPES[t]) and not (t == "number" and isinstance(value, bool)) for t in types):
            return f"{key}: expected {types}"
        if "enum" in spec and value not in spec["enum"]:
            return f"{key}: must be one of {spec['enum']}"
    return None


def run_tool(name: str, args: dict) -> tuple[str, bool]:
    """Execute a tool call. Returns (json_result, is_error)."""
    error = validate_input(name, args)
    if error:
        return json.dumps({"error": error}), True
    try:
        return json.dumps(TOOLS[name](**args), ensure_ascii=False), False
    except Exception as e:  # a tool bug should not crash the call
        return json.dumps({"error": f"{type(e).__name__}: {e}"}), True
