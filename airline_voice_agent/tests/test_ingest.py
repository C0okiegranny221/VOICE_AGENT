from ingest.chunk import approx_tokens, classify_topic, merge_small, split_long
from ingest.parse import _join, split_sections


def lines(*texts, marked=()):
    return [(1, t, i in marked) for i, t in enumerate(texts)]


def test_numbered_headings_follow_sequence_and_ignore_inner_lists():
    secs = split_sections(
        lines(
            "1. Introduction",
            "Some intro text.",
            "1.1 Scope",
            "Scope text.",
            "2. Baggage",
            "1. a numbered list item that is not a heading",
            "2.1 Hand Baggage",
            "Seven kilograms.",
        ),
        "Doc",
    )
    paths = [s["heading_path"] for s in secs]
    assert paths == [
        ["Doc", "1. Introduction"],
        ["Doc", "1. Introduction", "1.1 Scope"],
        ["Doc", "2. Baggage"],
        ["Doc", "2. Baggage", "2.1 Hand Baggage"],
    ]
    assert "numbered list item" in secs[2]["text"]


def test_level2_paragraph_number_without_title_keeps_body():
    secs = split_sections(lines("1. INTRODUCTION", "1.6 This CAR is issued under Rule 133 A.", "More."), "CAR")
    assert secs[0]["heading_path"][-1] == "1.6"
    assert secs[0]["text"].startswith("This CAR is issued")


def test_accordion_mode_uses_icon_marks_not_numbers():
    secs = split_sections(
        lines("10. Customer's Responsibility, Refusal and Limitation of Carriage and Special/Disability Assistance",
              "9. a list item", "11. Baggage", "Body.", marked={0, 2}),
        "CoC",
        mode="accordion",
    )
    assert [s["heading_path"][1] for s in secs] == [
        "10. Customer's Responsibility, Refusal and Limitation of Carriage and Special/Disability Assistance",
        "11. Baggage",
    ]


def test_join_unwraps_lines_but_keeps_sentence_breaks():
    assert _join(["IndiGo allows free", "baggage up to 15 kg.", "Excess is charged."]) == (
        "IndiGo allows free baggage up to 15 kg.\nExcess is charged."
    )


def test_split_long_respects_max_and_overlaps():
    # Paragraphs (~90 tokens) are bigger than the overlap budget, so overlap must come from sentences.
    para = " ".join(f"Sentence {{i}}.{j} has some words in it." for j in range(10))
    text = "\n".join(para.replace("{i}", str(i)) for i in range(12))
    pieces = split_long(text, max_tokens=200, overlap_tokens=40)
    assert len(pieces) > 1
    assert all(approx_tokens(p) <= 200 for p in pieces)
    for prev, nxt in zip(pieces, pieces[1:]):
        first_sentence = nxt.split(". ")[0]
        assert first_sentence in prev  # each piece starts by repeating the end of the previous one


def test_merge_small_only_within_same_clause():
    secs = [
        {"heading_path": ["D", "7. Refund", "7.1 A"], "page": 1, "text": "short"},
        {"heading_path": ["D", "7. Refund", "7.2 B"], "page": 1, "text": "short too"},
        {"heading_path": ["D", "8. Data", "8.1 C"], "page": 2, "text": "other clause"},
    ]
    merged = merge_small(secs, min_tokens=50, max_tokens=800)
    assert len(merged) == 2
    assert "7.2 B" in merged[0]["text"] and merged[0]["merged_headings"] == ["7.2 B"]


def test_topic_prefers_heading():
    assert classify_topic("11. Baggage", "refund refund refund") == "baggage"
    assert classify_topic("misc", "the refund and the refund policy") == "refund"
