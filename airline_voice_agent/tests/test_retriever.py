from rag.retriever import build_filter, build_query


def test_build_query_adds_previous_question_and_details():
    q = build_query("what about infants?", "checked baggage allowance", {"route_type": "international", "pnr": "X1"})
    assert q == "what about infants? checked baggage allowance international"


def test_filter_defaults_to_indian_rules():
    assert build_filter("my flight is delayed", None) == "jurisdiction IN ('india')"


def test_filter_adds_named_region_and_route():
    f = build_filter("flight from Bangkok was cancelled", "international")
    assert f == "jurisdiction IN ('india', 'thailand') AND route_scope IN ('international', 'both')"


def test_region_match_is_whole_word():
    assert "europe" not in build_filter("my reuse question", None)  # 'eu' inside 'reuse' must not match
