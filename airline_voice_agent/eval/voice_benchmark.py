"""Offline voice benchmark (no microphone needed), saved to report/08_voice/.

- TTS: Kokoro synthesis time, audio length and real-time factor for typical agent answers.
- STT: Whisper word error rate and latency on caller questions spoken by *other* Kokoro voices,
  clean and with added noise (15 dB SNR) as a stand-in for a laptop microphone.
- VAD: utterances found in a stream of speech separated by silences.
Synthetic speech is a proxy; the real-microphone test is the spoken-call milestone.

    python -m eval.voice_benchmark
"""
import re
import statistics
import time

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from config import CONFIG
from reporting import log_entry, md_table, save_artifact, stage_dir
from voice import stt, tts
from voice.vad import Utterances

AGENT_ANSWERS = [
    "On domestic flights you can check in one bag of up to 15 kg, plus one hand bag of up to 7 kg.",
    "Under DGCA rules, for a delay of three hours on a flight of that length, IndiGo must give you meals and refreshments.",
    "An extra checked bag costs INR 900 if you pre-book it, or ₹1,000 at the airport.",
    "Refunds to a credit card are processed within seven working days of your request.",
    "Is your flight domestic or international?",
    "I'm sorry, I don't have that information. I can connect you to an IndiGo agent.",
]
CALLER_QUESTIONS = [
    "How much cabin baggage can I carry on IndiGo?",
    "My flight from Delhi to Mumbai is delayed by three hours. What do I get?",
    "Can I carry a power bank in my checked bag?",
    "I booked through a travel agent. How long will my refund take?",
    "What compensation do I get if I am denied boarding?",
    "Is web check-in free for a Saver fare?",
    "I need a wheelchair at the airport in Bengaluru.",
    "There is a spelling mistake in my name on the ticket.",
    "My bag was damaged on the flight, how do I claim?",
    "What was the fee you mentioned for the extra bag?",
]
CALLER_VOICES = ["am_michael", "bf_emma"]
# Hinglish callers: Devanagari text spoken by Kokoro's Hindi voices; reference = Roman-script Hinglish as a person
# would write it. Romanization varies (teen/tin, mein/me), so WER uses a small spelling-variant map (HINGLISH_VARIANTS).
HINGLISH_QUESTIONS = [
    ("मेरी फ्लाइट तीन घंटे लेट है, मुझे क्या मिलेगा?", "Meri flight teen ghante late hai, mujhe kya milega?"),
    ("डोमेस्टिक फ्लाइट में कितना बैगेज अलाउड है?", "Domestic flight mein kitna baggage allowed hai?"),
    ("रिफंड कितने दिन में आएगा?", "Refund kitne din mein aayega?"),
    ("क्या मैं पावर बैंक ले जा सकता हूँ?", "Kya main power bank le ja sakta hoon?"),
    ("मेरी फ्लाइट कैंसिल हो गई, कम्पेन्सेशन मिलेगा क्या?", "Meri flight cancel ho gayi, compensation milega kya?"),
    ("टिकट पे नाम की स्पेलिंग गलत है।", "Ticket pe naam ki spelling galat hai."),
    ("हैंड बैगेज कितने किलो तक ले जा सकते हैं?", "Hand baggage kitne kilo tak le ja sakte hain?"),
    ("बस इतना ही, धन्यवाद।", "Bas itna hi, dhanyavaad."),
]
HINGLISH_VOICES = ["hm_omega", "hf_beta"]
HINGLISH_VARIANTS = {"tin": "teen", "me": "mein", "mai": "main", "aaega": "aayega", "aega": "aayega", "domastic": "domestic",
                     "gai": "gayi", "gaya": "gayi", "hu": "hoon", "hun": "hoon", "hai": "hai", "hain": "hain", "kilo": "kilo",
                     "dhanyawad": "dhanyavaad", "dhanyavad": "dhanyavaad", "galt": "galat", "bhai": "bhi"}
NUM_WORDS = {w: str(i) for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen".split())}


def _words(text: str, variants: dict | None = None) -> list[str]:
    text = re.sub(r"[^a-z0-9' ]", " ", text.lower().replace("-", " "))
    words = [NUM_WORDS.get(w, w) for w in text.split()]
    return [variants.get(w, w) for w in words] if variants else words


def wer(ref: str, hyp: str, variants: dict | None = None) -> float:
    r, h = _words(ref, variants), _words(hyp, variants)
    d = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        prev, d[0] = d[0], i
        for j, hw in enumerate(h, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (rw != hw))
    return d[len(h)] / max(1, len(r))


def to_16k(audio: np.ndarray) -> np.ndarray:
    return resample_poly(audio, 2, 3).astype(np.float32)  # 24 kHz -> 16 kHz


def add_noise(audio: np.ndarray, snr_db: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    power = np.mean(audio**2)
    noise = rng.standard_normal(len(audio)) * np.sqrt(power / (10 ** (snr_db / 10)))
    return (audio + noise).astype(np.float32)


def main() -> None:
    audio_dir = stage_dir("voice") / "audio"
    audio_dir.mkdir(exist_ok=True)
    warm = {"tts_load_and_first_s": round(tts.warmup(), 2), "stt_load_and_first_s": round(stt.warmup(), 2)}
    print("warmup", warm)

    tts_rows = []
    for i, text in enumerate(AGENT_ANSWERS, 1):
        t0 = time.perf_counter()
        audio = tts.synthesize(text)
        ms = (time.perf_counter() - t0) * 1000
        seconds = len(audio) / CONFIG.tts_rate
        sf.write(audio_dir / f"agent_{i:02d}.wav", audio, CONFIG.tts_rate)
        tts_rows.append({"#": i, "text": text, "spoken_as": tts.normalize(text), "synth_ms": round(ms),
                         "audio_s": round(seconds, 2), "RTF": round(ms / 1000 / seconds, 3)})
    print("tts done")

    stt_rows = []
    for v, voice in enumerate(CALLER_VOICES):
        for i, text in enumerate(CALLER_QUESTIONS, 1):
            clean = to_16k(tts.synthesize(text, voice=voice))
            if v == 0:
                sf.write(audio_dir / f"caller_{i:02d}_{voice}.wav", clean, 16000)
            for condition, audio in (("clean", clean), ("noise_15dB", add_noise(clean, 15, seed=i))):
                t0 = time.perf_counter()
                hyp = stt.transcribe(audio)
                ms = (time.perf_counter() - t0) * 1000
                stt_rows.append({"voice": voice, "condition": condition, "reference": text, "transcript": hyp,
                                 "WER": round(wer(text, hyp), 3), "stt_ms": round(ms), "audio_s": round(len(audio) / 16000, 2)})
    print("stt done")

    hinglish_rows = []
    for voice in HINGLISH_VOICES:
        for i, (devanagari, reference) in enumerate(HINGLISH_QUESTIONS, 1):
            clean = to_16k(tts.synthesize(devanagari, voice=voice))
            for condition, audio in (("clean", clean), ("noise_15dB", add_noise(clean, 15, seed=100 + i))):
                t0 = time.perf_counter()
                hyp = stt.transcribe(audio)
                ms = (time.perf_counter() - t0) * 1000
                hinglish_rows.append({"voice": voice, "condition": condition, "spoken": devanagari, "reference": reference,
                                      "transcript": hyp, "WER": round(wer(reference, hyp, HINGLISH_VARIANTS), 3),
                                      "stt_ms": round(ms), "audio_s": round(len(audio) / 16000, 2)})
    print("hinglish stt done")

    # VAD: three questions separated by 1.2 s silences, with 0.8 s lead-in and tail, fed in mic-sized blocks.
    silence = np.zeros(int(1.2 * 16000), dtype=np.float32)
    parts = [np.zeros(int(0.8 * 16000), dtype=np.float32)]
    for text in CALLER_QUESTIONS[:3]:
        parts += [to_16k(tts.synthesize(text, voice="am_michael")), silence]
    stream = add_noise(np.concatenate(parts), 30, seed=99)
    detector, found, t0 = Utterances(), [], time.perf_counter()
    for start in range(0, len(stream), 512):
        u = detector.feed(stream[start : start + 512])
        if u is not None:
            found.append(round(len(u) / 16000, 2))
    vad_ms_per_s = (time.perf_counter() - t0) * 1000 / (len(stream) / 16000)
    vad = {"utterances_expected": 3, "utterances_found": len(found), "durations_s": found,
           "processing_ms_per_audio_second": round(vad_ms_per_s, 2)}
    print("vad", vad)

    def agg(cond):
        rows = [r for r in stt_rows if r["condition"] == cond]
        return {"condition": cond, "n": len(rows), "mean_WER": round(statistics.mean(r["WER"] for r in rows), 3),
                "exact_match": f"{sum(r['WER'] == 0 for r in rows)}/{len(rows)}",
                "median_stt_ms": round(statistics.median(r["stt_ms"] for r in rows)),
                "max_stt_ms": max(r["stt_ms"] for r in rows)}

    stt_summary = [agg("clean"), agg("noise_15dB")]

    def agg_h(cond):
        rows = [r for r in hinglish_rows if r["condition"] == cond]
        return {"condition": f"hinglish {cond}", "n": len(rows), "mean_WER": round(statistics.mean(r["WER"] for r in rows), 3),
                "exact_match": f"{sum(r['WER'] == 0 for r in rows)}/{len(rows)}",
                "median_stt_ms": round(statistics.median(r["stt_ms"] for r in rows)),
                "max_stt_ms": max(r["stt_ms"] for r in rows)}

    stt_summary += [agg_h("clean"), agg_h("noise_15dB")]
    tts_summary = {"median_synth_ms": round(statistics.median(r["synth_ms"] for r in tts_rows)),
                   "max_synth_ms": max(r["synth_ms"] for r in tts_rows),
                   "median_RTF": round(statistics.median(r["RTF"] for r in tts_rows), 3)}
    body = (
        f"# Voice benchmark\n\nSTT engine `{CONFIG.stt_engine}` "
        f"(`{CONFIG.stt_hinglish_model if CONFIG.stt_engine == 'hinglish' else CONFIG.stt_model}`) · TTS Kokoro-82M voices "
        f"`{CONFIG.tts_voice}` / `{CONFIG.tts_voice_hindi}` · VAD Silero "
        f"(silence {CONFIG.vad_silence_ms} ms). Caller speech is synthetic (Kokoro voices {CALLER_VOICES}); noise is white "
        f"noise at 15 dB SNR. Audio samples are in `08_voice/audio/`.\n\n"
        f"## Model warm-up (first call, includes loading)\n\n{md_table([warm])}\n"
        f"## TTS\n\n{md_table([tts_summary])}\n{md_table(tts_rows)}\n"
        f"## STT\n\n{md_table(stt_summary)}\nHinglish WER is against a Roman-script reference after mapping common spelling "
        f"variants (`HINGLISH_VARIANTS`); remaining differences are often spelling, not recognition errors.\n\n"
        f"{md_table(stt_rows)}\n### Hinglish\n\n{md_table(hinglish_rows)}\n"
        f"## VAD\n\n{md_table([vad])}"
    )
    path = save_artifact("voice", "voice_benchmark", body)
    save_artifact("voice", "voice_benchmark", {"warmup": warm, "tts": tts_rows, "stt": stt_rows, "stt_hinglish": hinglish_rows, "vad": vad})
    log_entry("voice", "Voice benchmark (synthetic speech)",
              md_table(stt_summary) + "\n" + md_table([tts_summary]) + "\n" + md_table([vad]), [path])
    print(md_table(stt_summary), md_table([tts_summary]))


if __name__ == "__main__":
    main()
