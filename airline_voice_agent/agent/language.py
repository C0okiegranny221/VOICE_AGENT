"""Per-turn language: 'hinglish' (Hindi or Hindi-English mix, in Devanagari or Roman script) or 'english'.
The Hinglish STT writes everything in Roman script, so romanized Hindi function words are the signal."""
import re

DEVANAGARI = re.compile(r"[ऀ-ॿ]")
HINDI_WORDS = set("""
hai hain ho hoon hun tha thi the hoga hogi honge kya kyu kyun kaise kab kahan kaun kitna kitni kitne mera meri mere
mujhe mujhko main mai hum hamara hamari aap aapka aapki aapke tum tera teri ka ki ke ko se mein me par pe aur ya
nahi nahin na bhi toh to agar lekin kyunki chahiye milega milegi milenge mil sakta sakti sakte karna karo kar kiya
kiye diya di dena lena le ja jaana aana aaega aayega aaegi hua hui gaya gayi gaye wala wali wale abhi kal aaj din
ghante ghanta minute bahut thoda zyada kuch sab bas itna haan ji theek accha acha batao bataiye samjha yeh ye woh
wo iska uska unka koi kisi
""".split())
# Words that are also common English words; they only count alongside other Hindi words.
AMBIGUOUS = {"to", "me", "main", "par", "pe", "na", "do", "ho", "hi", "ya", "the", "kar", "le", "ja", "di", "bas"}


def detect(text: str) -> str:
    if DEVANAGARI.search(text):
        return "hinglish"
    words = re.findall(r"[a-z]+", text.lower())
    if not words:
        return "english"
    strong = sum(w in HINDI_WORDS and w not in AMBIGUOUS for w in words)
    weak = sum(w in AMBIGUOUS for w in words)
    return "hinglish" if strong >= 2 or (strong >= 1 and strong + weak >= 3) else "english"
