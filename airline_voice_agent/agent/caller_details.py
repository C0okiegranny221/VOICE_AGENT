"""Pull call facts (route type, fare product, passenger type, payment mode, delay, PNR) out of what the
caller says, with plain keyword rules - no extra LLM call, so it adds no latency. Later mentions
overwrite earlier ones. The details feed retrieval filters and are shown to Claude each turn."""
import re

WORD_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "half an": 0.5, "an": 1, "a": 1,
    # romanized Hindi, as the Hinglish STT writes it
    "ek": 1, "do": 2, "teen": 3, "tin": 3, "char": 4, "chaar": 4, "paanch": 5, "panch": 5, "chhe": 6, "che": 6,
    "saat": 7, "aath": 8, "nau": 9, "das": 10, "dedh": 1.5, "dhai": 2.5, "aadha": 0.5,
}
FOREIGN_PLACES = (
    r"dubai|abu dhabi|sharjah|doha|muscat|kuwait|dammam|jeddah|riyadh|bahrain|singapore|bangkok|phuket|"
    r"krabi|kuala lumpur|bali|jakarta|hanoi|ho chi minh|da nang|colombo|male|maldives|kathmandu|dhaka|"
    r"istanbul|london|manchester|amsterdam|paris|copenhagen|athens|nairobi|tashkent|almaty|tbilisi|baku|"
    r"seoul|tokyo|guangzhou|hong kong|europe|uk|usa"
)
INDIAN_PLACES = (
    r"delhi|mumbai|bombay|bengaluru|bangalore|chennai|kolkata|calcutta|hyderabad|pune|ahmedabad|goa|"
    r"jaipur|lucknow|kochi|cochin|trivandrum|thiruvananthapuram|guwahati|patna|bhubaneswar|indore|"
    r"nagpur|chandigarh|srinagar|varanasi|amritsar|coimbatore|madurai|vizag|visakhapatnam|raipur|ranchi"
)
PATTERNS = {
    "route_type": [
        ("international", rf"\b(international|abroad|overseas|outside india|videsh|antarrashtriya|{FOREIGN_PLACES})\b"),
        ("domestic", rf"\b(domestic|domastic|within india|inside india|gharelu|desh ke andar|{INDIAN_PLACES})\b"),
    ],
    "product": [
        ("business", r"\bbusiness class\b|\bstretch ?(plus|\+)"),
        ("stretch", r"\bindigo stretch\b|\bstretch\b"),
        ("lite_fare", r"\blite fare\b|\blite\b"),
    ],
    "passenger_type": [
        ("infant", r"\b(infant|baby|toddler|newborn|shishu)\b"),
        ("child", r"\b(child|kid|son|daughter|minor)\b"),
        ("pregnant", r"\bpregnan|\bgarbhvati"),
        ("reduced_mobility", r"\b(wheelchair|disab|reduced mobility|cannot walk|can't walk)"),
        ("senior", r"\b(senior citizen|elderly|old age)\b"),
    ],
    "payment_mode": [
        ("american_express", r"\b(amex|american express)\b"),
        ("travel_agent", r"\b(travel agent|agent|makemytrip|make my trip|goibibo|cleartrip|yatra|easemytrip)\b"),
        ("cash_at_airport", r"\bcash\b"),
        ("card_or_wallet", r"\b(credit card|debit card|card|upi|wallet|paytm|phonepe|google pay|net ?banking)\b"),
    ],
}
NUMBER = r"(?<![A-Za-z])(\d+(?:\.\d+)?|" + "|".join(sorted(WORD_NUMBERS, key=len, reverse=True)) + r")(?![A-Za-z])"
DELAY_RE = re.compile(
    rf"(?:delay(?:ed)?\s+(?:by\s+|for\s+|of\s+)?(?:about\s+|around\s+|over\s+)?{NUMBER}\s*(?:hours?|hrs?)"
    rf"|{NUMBER}[\s-]*(?:hours?|hrs?)\s+(?:of\s+)?delay|{NUMBER}[\s-]*(?:hours?|hrs?)\s+late"
    rf"|{NUMBER}\s*(?:ghante|ghanta|ghanton)\s*(?:se\s+)?(?:late|der|delay|lat))",
    re.I,
)
PNR_RE = re.compile(r"\bpnr\s*(?:number\s*)?(?:is\s*)?[:\-]?\s*([a-z0-9]{6})\b", re.I)


def _to_number(token: str) -> float:
    token = token.lower()
    return float(WORD_NUMBERS[token]) if token in WORD_NUMBERS else float(token)


def extract(text: str) -> dict:
    """Details mentioned in one utterance."""
    found = {}
    lowered = text.lower()
    for field, rules in PATTERNS.items():
        for value, pattern in rules:
            if re.search(pattern, lowered):
                found[field] = value
                break
    if m := DELAY_RE.search(text):
        token = next(g for g in m.groups() if g)
        found["delay_hours"] = _to_number(token)
    if m := PNR_RE.search(text):
        found["pnr"] = m.group(1).upper()
    return found


def update(details: dict, text: str) -> dict:
    """Merge the details from a new utterance into the running call details (newest wins)."""
    return {**details, **extract(text)}
