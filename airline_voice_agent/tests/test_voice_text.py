from voice.sentence_splitter import SentenceSplitter
from voice.tts import normalize


def stream(splitter, text, step=3):
    out = []
    for i in range(0, len(text), step):
        out += splitter.feed(text[i : i + step])
    return out + splitter.flush()


def test_splits_sentences_while_streaming():
    s = SentenceSplitter()
    pieces = stream(s, "You can check in fifteen kilograms. Extra bags cost more! Anything else?")
    assert pieces == ["You can check in fifteen kilograms.", "Extra bags cost more!", "Anything else?"]


def test_abbreviations_and_decimals_do_not_split():
    s = SentenceSplitter()
    pieces = stream(s, "It costs Rs. 900 for a 2.5 hour flight, e.g. Delhi to Pune. Okay.")
    assert pieces == ["It costs Rs. 900 for a 2.5 hour flight, e.g. Delhi to Pune.", "Okay."]


def test_long_clause_soft_splits_at_comma():
    s = SentenceSplitter(soft_split_chars=60)
    first = s.feed("Under DGCA rules you are owed meals and refreshments while you wait, and if the delay goes on ")
    assert first and first[0].endswith(",")


def test_normalize_for_speech():
    assert normalize("An extra bag is ₹1,000 or INR 900.") == "An extra bag is 1000 rupees or 900 rupees."
    assert normalize("Hand bag 7 kg, 55 cm, DGCA rules, 75%") == "Hand bag 7 kilograms, 55 centimetres, D G C A rules, 75 percent"
    assert normalize("**Note**: see https://goindigo.in") == "Note : see the IndiGo website"


def test_first_piece_is_cut_early_at_a_comma():
    s = SentenceSplitter()
    first = s.feed("आपकी फ्लाइट सिक्स ई ज़ीरो सिक्स वन सेवन लेट है, और डीजीसीए नियमों के मुताबिक आपको फ्री मील्स मिलेंगे ")
    assert first == ["आपकी फ्लाइट सिक्स ई ज़ीरो सिक्स वन सेवन लेट है,"]
