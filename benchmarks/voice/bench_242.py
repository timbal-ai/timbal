# ruff: noqa: T201
"""Before/after measurement for the #242 voice work.

One command from the repo root, with the voice extra installed::

    uv run python benchmarks/voice/bench_242.py all

``before`` is ``origin/main`` at default ``VoiceConfig`` (Deepgram Flux when
``DEEPGRAM_API_KEY`` is set, else ElevenLabs Scribe). ``after`` is this tree
with the candidate best config. Ablations turn one knob off (tempo is turned
on). Caller audio is synthetic OpenAI ``gpt-4o-mini-tts`` PCM, resampled to
16 kHz and paced in real time through the cascaded session the server builds.

Raw rows land in ``benchmarks/voice/242-results/``. Nothing here is a unit
test and nothing is wired into pytest.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import inspect
import json
import os
import re
import subprocess
import sys
import time
import wave
from contextlib import aclosing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
AUDIO_DIR = HERE / "fixtures" / "242"
OUT_DIR = HERE / "242-results"
SAMPLE_RATE = 16_000
OPENAI_TTS_MODEL = "gpt-4o-mini-tts"
OPENAI_TTS_VOICE = "coral"
OPENAI_TTS_RATE = 24_000

# What the first report recommended shipping on Flux. Wave 2 measures this,
# not the everything-on candidate below.
RECOMMENDED: dict[str, Any] = {
    "ignore_backchannels": True,
    "speculative_generation": {"enabled": True, "preemptive_tts": True},
    "max_tts_pause_secs": 0.2,
}

# Candidate "best". Tempo stays off until a run shows it helps.
BEST: dict[str, Any] = {
    "ignore_backchannels": True,
    "hold_unfinished_secs": 1.2,
    "min_delay_secs": 0.30,
    "stop_silence_secs": 0.20,
    "first_flush_min_words": 4,
    "predict_during_speech": True,
    "speculative_generation": {"enabled": True, "preemptive_tts": True},
    "max_tts_pause_secs": 0.2,
}

# One knob changed relative to BEST. ``tempo`` turns the feature on.
ABLATIONS: dict[str, dict[str, Any]] = {
    "no_backchannels": {"ignore_backchannels": False},
    "no_hold": {"hold_unfinished_secs": None},
    "default_floors": {"min_delay_secs": 0.45, "stop_silence_secs": 0.20},
    "no_first_flush": {"first_flush_min_words": None},
    "no_predict": {"predict_during_speech": False},
    "no_speculative": {"speculative_generation": {"enabled": False}},
    "speculative_hold_tts": {"speculative_generation": {"enabled": True, "preemptive_tts": False}},
    "no_pause_trim": {"max_tts_pause_secs": None},
    "tempo_1_05": {"tts_tempo": 1.05},
    "cache_bust": {},
}


@dataclass(frozen=True)
class ScenarioSpec:
    id: str
    lang: str
    kind: str
    # (text, role) role is "say" | "backchannel" | "barge" | "continuation"
    lines: tuple[tuple[str, str], ...]
    gap_before_continuation: float = 0.8
    # Groups this scenario is worth ablating. Empty = headline only.
    ablate: tuple[str, ...] = ()
    voice: str = "coral"
    speed: float = 1.0
    # "" | "phone" | "street" | "office" | "tv". Phone and the noisy ones are
    # 8 kHz mu-law round-trips. Noise is a synthetic bed, not a recording.
    mic: str = ""
    barge_at: str = "mid"  # start | mid | end
    readback: tuple[str, ...] = ()
    expect_lang: str = ""
    greeting: str = ""
    silence_tail: float = 0.0
    tool_mode: str = ""  # "" | "two" | "slow" | "error"
    group: str = ""


SCENARIOS: tuple[ScenarioSpec, ...] = (
    ScenarioSpec("en_statement", "en", "latency", (("The return window is thirty days, so I can still send it back.", "say"),)),
    ScenarioSpec("es_statement", "es", "latency", (("La ventana de devolución es de treinta días, así que todavía puedo enviarlo.", "say"),)),
    ScenarioSpec("en_question", "en", "latency", (("Can you check whether order four eight two one has already shipped?", "say"),)),
    ScenarioSpec("es_question", "es", "latency", (("¿Puedes mirar si el pedido cuatro ocho dos uno ya ha salido?", "say"),)),
    ScenarioSpec(
        "en_unfinished",
        "en",
        "unfinished",
        (("I want to book it for", "say"), ("Thursday afternoon", "continuation")),
        gap_before_continuation=0.8,
        ablate=("no_hold", "default_floors", "no_predict"),
    ),
    ScenarioSpec(
        "es_unfinished",
        "es",
        "unfinished",
        (("quiero reservar para", "say"), ("el jueves por la tarde", "continuation")),
        gap_before_continuation=0.8,
        ablate=("no_hold", "default_floors", "no_predict"),
    ),
    ScenarioSpec(
        "en_backchannel_okay",
        "en",
        "backchannel",
        (
            ("Explain the return policy in two sentences.", "say"),
            ("okay got it", "backchannel"),
        ),
        ablate=("no_backchannels",),
    ),
    ScenarioSpec(
        "en_backchannel_mhm",
        "en",
        "backchannel",
        (
            ("Explain the return policy in two sentences.", "say"),
            ("mhm", "backchannel"),
        ),
        ablate=("no_backchannels",),
    ),
    ScenarioSpec(
        "es_backchannel_vale",
        "es",
        "backchannel",
        (
            ("Explícame la política de devoluciones en dos frases.", "say"),
            ("vale", "backchannel"),
        ),
        ablate=("no_backchannels",),
    ),
    ScenarioSpec(
        "en_barge",
        "en",
        "barge",
        (
            ("Tell me the return policy in two sentences.", "say"),
            ("wait, stop, I need the order status instead", "barge"),
        ),
    ),
    ScenarioSpec(
        "es_barge",
        "es",
        "barge",
        (
            ("Explícame la política de devoluciones en dos frases.", "say"),
            ("espera, para, necesito el estado del pedido", "barge"),
        ),
    ),
    ScenarioSpec(
        "en_lookup",
        "en",
        "tool",
        (("Look up the status of order four eight two one.", "say"),),
    ),
    ScenarioSpec(
        "en_hangup_retract",
        "en",
        "destructive",
        (
            ("Please hang up the call", "say"),
            ("No wait, I still need help with the order", "continuation"),
        ),
        gap_before_continuation=0.6,
        ablate=("no_speculative", "speculative_hold_tts"),
    ),
    ScenarioSpec(
        "cache_name",
        "en",
        "cache",
        (
            ("What is your name?", "say"),
            ("What is the return window?", "say"),
            ("Say that again in one short sentence.", "say"),
        ),
        ablate=("cache_bust", "no_first_flush", "no_pause_trim", "tempo_1_05"),
    ),
)


def _s(sid: str, lang: str, kind: str, lines: tuple[tuple[str, str], ...], **kwargs: Any) -> ScenarioSpec:
    return ScenarioSpec(sid, lang, kind, lines, group=kwargs.pop("group", kind), **kwargs)


_POLICY = "Explain the return policy in two sentences."
_POLICY_ES = "Explícame la política de devoluciones en dos frases."

# Wave 2. AFTER uses the recommended config, not the everything-on candidate.
WAVE2: tuple[ScenarioSpec, ...] = (
    _s("tel_phone_en", "en", "telephony", (("The return window is thirty days, so I can still send it back.", "say"),), mic="phone"),
    _s("tel_phone_es", "es", "telephony", (("La ventana de devolución es de treinta días, así que todavía puedo enviarlo.", "say"),), mic="phone"),
    _s("tel_street_en", "en", "telephony", (("The return window is thirty days, so I can still send it back.", "say"),), mic="street"),
    _s("tel_office_es", "es", "telephony", (("La ventana de devolución es de treinta días, así que todavía puedo enviarlo.", "say"),), mic="office"),
    _s("tel_tv_en", "en", "telephony", (("Can you check whether order four eight two one has already shipped?", "say"),), mic="tv"),
    _s(
        "hesit_um_en",
        "en",
        "hesitation",
        (("I need the delivery changed to", "say"), ("um, the office on Main Street", "continuation")),
        gap_before_continuation=1.5,
    ),
    _s(
        "hesit_eh_es",
        "es",
        "hesitation",
        (("Quiero cambiar la entrega a", "say"), ("eh, la oficina del centro", "continuation")),
        gap_before_continuation=1.5,
    ),
    _s(
        "hesit_osea_es",
        "es",
        "hesitation",
        (("La talla es mediana o sea", "say"), ("la grande si no hay mediana", "continuation")),
        gap_before_continuation=1.8,
    ),
    _s(
        "dict_order_en",
        "en",
        "dictation",
        (
            ("The order number is four eight", "say"),
            ("two one", "continuation"),
            ("nine zero", "continuation"),
        ),
        gap_before_continuation=0.8,
        readback=("4821|four eight two one",),
    ),
    _s(
        "dict_phone_es",
        "es",
        "dictation",
        (
            ("Mi teléfono es seis cinco cinco", "say"),
            ("cero uno dos", "continuation"),
            ("tres cuatro", "continuation"),
        ),
        gap_before_continuation=0.7,
        readback=("655|seis cinco cinco",),
    ),
    _s(
        "dict_email_en",
        "en",
        "dictation",
        (
            ("My email is alex", "say"),
            ("dot smith", "continuation"),
            ("at example dot com", "continuation"),
        ),
        gap_before_continuation=0.5,
        readback=("alex", "example"),
    ),
    _s(
        "dict_name_es",
        "es",
        "dictation",
        (
            ("Me apellido jota", "say"),
            ("o", "continuation"),
            ("ene e ese", "continuation"),
        ),
        gap_before_continuation=0.45,
        readback=("jones|jota",),
    ),
    _s(
        "mono_en",
        "en",
        "monologue",
        (
            ("I bought a jacket last month and the zipper broke the second time I wore it, so I want to send it back.", "say"),
            ("The order number is on the email, I think, and I still have the receipt in the kitchen drawer.", "continuation"),
            ("I do not need a replacement, I just want the refund on the same card I paid with.", "continuation"),
            ("Please tell me the return window and the address I should ship the box to.", "continuation"),
        ),
        gap_before_continuation=0.55,
    ),
    _s(
        "mono_es",
        "es",
        "monologue",
        (
            ("Compré una chaqueta el mes pasado y la cremallera se rompió la segunda vez que la usé, así que quiero devolverla.", "say"),
            ("El número de pedido está en el correo, creo, y todavía tengo el recibo en el cajón de la cocina.", "continuation"),
            ("No necesito un cambio, solo el reembolso en la misma tarjeta con la que pagué.", "continuation"),
            ("Dime el plazo de devolución y la dirección a la que tengo que enviar la caja.", "continuation"),
        ),
        gap_before_continuation=0.55,
    ),
    _s(
        "fast_en",
        "en",
        "speed",
        (("The return window is thirty days, so I can still send it back.", "say"),),
        voice="ash",
        speed=1.2,
    ),
    _s(
        "slow_es",
        "es",
        "speed",
        (("La ventana de devolución es de treinta días, así que todavía puedo enviarlo.", "say"),),
        voice="sage",
        speed=0.85,
    ),
    _s("bc_si", "es", "backchannel", ((_POLICY_ES, "say"), ("sí", "backchannel"))),
    _s("bc_aja", "es", "backchannel", ((_POLICY_ES, "say"), ("ajá", "backchannel"))),
    _s("bc_claro", "es", "backchannel", ((_POLICY_ES, "say"), ("claro", "backchannel"))),
    _s("bc_vale_vale", "es", "backchannel", ((_POLICY_ES, "say"), ("vale vale", "backchannel"))),
    _s("bc_uhhuh", "en", "backchannel", ((_POLICY, "say"), ("uh-huh", "backchannel"))),
    _s("bc_right", "en", "backchannel", ((_POLICY, "say"), ("right", "backchannel"))),
    _s("bc_yeah", "en", "backchannel", ((_POLICY, "say"), ("yeah", "backchannel"))),
    _s("bc_mm", "en", "backchannel", ((_POLICY, "say"), ("mm", "backchannel"))),
    _s("barge_start_en", "en", "barge", ((_POLICY, "say"), ("wait, stop", "barge")), barge_at="start"),
    _s("barge_mid_en", "en", "barge", ((_POLICY, "say"), ("wait, stop, I need the order status", "barge")), barge_at="mid"),
    _s("barge_end_en", "en", "barge", ((_POLICY, "say"), ("wait, stop", "barge")), barge_at="end"),
    _s("barge_start_es", "es", "barge", ((_POLICY_ES, "say"), ("espera, para", "barge")), barge_at="start"),
    _s("barge_no", "en", "barge", ((_POLICY, "say"), ("no", "barge")), barge_at="mid"),
    _s("barge_espera", "es", "barge", ((_POLICY_ES, "say"), ("espera", "barge")), barge_at="mid"),
    _s("barge_stop", "en", "barge", ((_POLICY, "say"), ("stop", "barge")), barge_at="mid"),
    _s("barge_wait", "en", "barge", ((_POLICY, "say"), ("wait", "barge")), barge_at="mid"),
    _s(
        "correct_en",
        "en",
        "correction",
        (("Book it for Tuesday", "say"), ("no, Wednesday", "continuation")),
        gap_before_continuation=0.4,
        readback=("Wednesday",),
    ),
    _s(
        "correct_es",
        "es",
        "correction",
        (("Resérvalo para el martes", "say"), ("no, el miércoles", "continuation")),
        gap_before_continuation=0.4,
        readback=("miércoles",),
    ),
    _s(
        "hang_en_0_3",
        "en",
        "destructive",
        (("Please hang up the call", "say"), ("No wait, I still need help with the order", "continuation")),
        gap_before_continuation=0.3,
    ),
    _s(
        "hang_en_0_6",
        "en",
        "destructive",
        (("Please hang up the call", "say"), ("No wait, I still need help with the order", "continuation")),
        gap_before_continuation=0.6,
    ),
    _s(
        "hang_en_1_0",
        "en",
        "destructive",
        (("Please hang up the call", "say"), ("No wait, I still need help with the order", "continuation")),
        gap_before_continuation=1.0,
    ),
    _s(
        "hang_es_0_3",
        "es",
        "destructive",
        (("Cuelga la llamada", "say"), ("No, espera, todavía necesito ayuda", "continuation")),
        gap_before_continuation=0.3,
    ),
    _s(
        "hang_es_0_6",
        "es",
        "destructive",
        (("Cuelga la llamada", "say"), ("No, espera, todavía necesito ayuda", "continuation")),
        gap_before_continuation=0.6,
    ),
    _s(
        "hang_es_1_0",
        "es",
        "destructive",
        (("Cuelga la llamada", "say"), ("No, espera, todavía necesito ayuda", "continuation")),
        gap_before_continuation=1.0,
    ),
    _s(
        "codeswitch_es",
        "es",
        "language",
        (("Quiero el refund del order cuatro ocho dos uno, please", "say"),),
        expect_lang="es",
        readback=("4821|cuatro ocho|four eight",),
    ),
    _s(
        "en_after_es_greet",
        "es",
        "language",
        (("I need the return window in English please.", "say"),),
        greeting="Hola, soy Timbal, te ayudo con tu pedido.",
        expect_lang="en",
    ),
    _s(
        "lang_en_control",
        "en",
        "language",
        (("What is the return window?", "say"),),
        expect_lang="en",
    ),
    _s(
        "long_10_en",
        "en",
        "long",
        tuple(
            (text, "say")
            for text in (
                "What is your name?",
                "What is the return window?",
                "Do you ship on Sundays?",
                "Where is the warehouse?",
                "Can I pay on delivery?",
                "What is your name again?",
                "How long is the return window?",
                "Do you ship on Sundays or not?",
                "Where exactly is the warehouse?",
                "Can I pay on delivery or not?",
            )
        ),
    ),
    _s(
        "tools_two",
        "en",
        "tools",
        (("Look up order four eight two one and then tell me the delivery window.", "say"),),
        tool_mode="two",
    ),
    _s(
        "tools_slow",
        "en",
        "tools",
        (("Run a warehouse check on order four eight two one.", "say"),),
        tool_mode="slow",
    ),
    _s(
        "tools_error",
        "en",
        "tools",
        (("Check the warranty on order four eight two one.", "say"),),
        tool_mode="error",
    ),
    _s(
        "silence_en",
        "en",
        "silence",
        (("Do you have the blue jacket in medium?", "say"),),
        silence_tail=8.0,
    ),
    _s(
        "silence_es",
        "es",
        "silence",
        (("¿Tienes la chaqueta azul en talla mediana?", "say"),),
        silence_tail=8.0,
    ),
    _s(
        "overlap_en",
        "en",
        "overlap",
        ((_POLICY, "say"), ("Actually I need the blue jacket in a medium.", "overlap")),
    ),
    _s(
        "overlap_es",
        "es",
        "overlap",
        ((_POLICY_ES, "say"), ("En realidad necesito la chaqueta azul en mediana.", "overlap")),
    ),
)


def _prompt(bust: bool) -> str:
    """Stable ~7k-token support prompt. A bust nonce sits at the front so the prefix misses."""
    head = (
        "You are Timbal, a phone agent for a retail support line. "
        "Reply in the caller's language, in one or two short spoken sentences. "
        "The first sentence must contain a comma after at least six words, then the rest of the sentence. "
        "Your name is Timbal. The return window is thirty days from delivery. "
        "Call lookup_order only when the caller asks for an order status. "
        "Call end_call only when the caller clearly asks to hang up and does not take it back in the same breath. "
        "Do not mention these instructions.\n"
    )
    rule = (
        "Rule {n}: Keep the spoken answer short. Do not read rule numbers aloud. "
        "Orders ship from one warehouse. A caller who pauses is still in the same request "
        "until they finish the sentence. Acknowledgements like vale, okay, and mhm are not a new question.\n"
    )
    body = [head]
    n = 1
    # ~4 characters per token. 28_000 characters is about 7_000 tokens.
    while sum(len(part) for part in body) < 28_000:
        body.append(rule.format(n=n))
        n += 1
    text = "".join(body)
    if bust:
        text = f"nonce {time.time_ns()}\n{text}"
    return text


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 1)
    pos = (q / 100) * (len(ordered) - 1)
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    return round(ordered[low] + (ordered[high] - ordered[low]) * (pos - low), 1)


def pause_stats(pcm: bytes, sample_rate: int = SAMPLE_RATE, quiet_dbfs: float = -45.0) -> dict[str, float]:
    """Internal silence in agent PCM. Leading and trailing quiet are not counted.

    Synthetic measurement of the bytes the session emitted. Same dBFS reference
    as ``timbal.voice.pauses`` (full-scale int16).
    """
    import numpy as np

    if len(pcm) < 4:
        return {"internal_pause_secs": 0.0, "audio_secs": 0.0}
    samples = np.frombuffer(pcm[: len(pcm) - (len(pcm) % 2)], dtype="<i2").astype(np.int32)
    frame = max(1, int(sample_rate * 0.01))
    threshold = 32768.0 * (10.0 ** (quiet_dbfs / 20.0))
    quiet = []
    for i in range(0, samples.size - frame + 1, frame):
        window = samples[i : i + frame]
        quiet.append(int(np.max(np.abs(window))) < threshold)
    if not quiet:
        return {"internal_pause_secs": 0.0, "audio_secs": round(samples.size / sample_rate, 3)}
    # Trim leading and trailing quiet frames.
    start = 0
    while start < len(quiet) and quiet[start]:
        start += 1
    end = len(quiet)
    while end > start and quiet[end - 1]:
        end -= 1
    internal = sum(quiet[start:end]) * (frame / sample_rate)
    return {
        "internal_pause_secs": round(internal, 3),
        "audio_secs": round(samples.size / sample_rate, 3),
    }


def _clip_key(text: str, voice: str = "coral", speed: float = 1.0) -> str:
    if voice == "coral" and speed == 1.0:
        return text
    return f"{voice}|{speed}|{text}"


def _clip_path(text: str, voice: str = OPENAI_TTS_VOICE, speed: float = 1.0) -> Path:
    digest = hashlib.sha256(f"{OPENAI_TTS_MODEL}|{voice}|{speed}|{text}".encode()).hexdigest()[:16]
    # coral at 1.0 keeps the clips synthesized for the first report.
    if voice == OPENAI_TTS_VOICE and speed == 1.0:
        digest = hashlib.sha256(f"{OPENAI_TTS_MODEL}|{voice}|{text}".encode()).hexdigest()[:16]
    return AUDIO_DIR / f"{digest}.wav"


def _read_wav(path: Path) -> tuple[bytes, int]:
    with wave.open(str(path), "rb") as handle:
        if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
            raise RuntimeError(f"{path} is not mono PCM16")
        return handle.readframes(handle.getnframes()), handle.getframerate()


def _write_wav(path: Path, pcm: bytes, rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm)


def _to_16k(pcm: bytes, rate: int) -> bytes:
    if rate == SAMPLE_RATE:
        return pcm
    from timbal.voice.telephony import PcmResampler

    resampler = PcmResampler(rate, SAMPLE_RATE)
    out = resampler.process(pcm) + resampler.flush()
    return out


def synthesize(texts: list[str] | list[tuple[str, str, float]]) -> dict[str, bytes]:
    """OpenAI TTS → 16 kHz PCM16. Cached on disk. Marked synthetic by the caller.

    Items are either a bare string (coral, speed 1) or ``(text, voice, speed)``.
    """
    jobs: list[tuple[str, str, float]] = []
    for item in texts:
        if isinstance(item, str):
            jobs.append((item, "coral", 1.0))
        else:
            jobs.append(item)
    missing = [job for job in jobs if not _clip_path(*job).exists()]
    if missing:
        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit("OPENAI_API_KEY is not set; cannot synthesize caller audio")
        from openai import OpenAI

        client = OpenAI()
        for text, voice, speed in missing:
            print(f"synth {voice} x{speed} {text!r}", file=sys.stderr, flush=True)
            kwargs: dict[str, Any] = {
                "model": OPENAI_TTS_MODEL,
                "voice": voice,
                "input": text,
                "response_format": "wav",
            }
            if speed != 1.0:
                kwargs["speed"] = speed
            speech = client.audio.speech.create(**kwargs)
            raw_path = _clip_path(text, voice, speed).with_suffix(".src.wav")
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_bytes(speech.read())
            pcm, rate = _read_wav(raw_path)
            _write_wav(_clip_path(text, voice, speed), _to_16k(pcm, rate), SAMPLE_RATE)
            raw_path.unlink(missing_ok=True)
    clips: dict[str, bytes] = {}
    for text, voice, speed in jobs:
        pcm, rate = _read_wav(_clip_path(text, voice, speed))
        if rate != SAMPLE_RATE:
            pcm = _to_16k(pcm, rate)
        clips[_clip_key(text, voice, speed)] = pcm
    return clips


def _scenario_texts() -> list[str]:
    seen: list[str] = []
    for spec in SCENARIOS:
        for text, _role in spec.lines:
            if text not in seen:
                seen.append(text)
    return seen


def _voice_config(
    overrides: dict[str, Any], lang: str, stt: str, *, greeting: str = ""
) -> tuple[Any, list[str]]:
    from timbal.voice.config import VoiceConfig

    data: dict[str, Any] = {"stt_provider": stt}
    if lang != "auto":
        data["language"] = lang
    if greeting:
        data["greeting"] = {"text": greeting, "interruptible": True}
    data.update(overrides)
    allowed = set(VoiceConfig.model_fields)
    dropped = sorted(key for key in data if key not in allowed)
    kept = {key: value for key, value in data.items() if key in allowed}
    return VoiceConfig(**kept), dropped


def _readback_ok(needles: tuple[str, ...], replies: list[str]) -> bool:
    text = " ".join(replies).lower()
    for needle in needles:
        options = [part.strip().lower() for part in needle.split("|") if part.strip()]
        if not any(option in text for option in options):
            return False
    return True


def _guess_lang(text: str) -> str:
    """Crude EN/ES label from function words and Spanish diacritics. Not a translator."""
    es = len(re.findall(r"[áéíóúñ¿¡]|\b(el|la|los|las|que|para|una|es|está|de|en|por|con|quiero|puede|hola|sí|tengo)\b", text, re.I))
    en = len(re.findall(r"\b(the|is|are|you|your|order|return|window|please|can|have|and|to|what|where|how)\b", text, re.I))
    if es > en:
        return "es"
    if en > es:
        return "en"
    return "unk"


def _make_agent(model: str, prompt: Any, log: list[dict[str, Any]], tool_mode: str = "") -> Any:
    from timbal import Agent
    from timbal.core.tool import Tool
    from timbal.state.tracing.providers import InMemoryTracingProvider

    def lookup_order(order_id: str) -> str:
        """Look up an order by id. Read only. Does not change the order."""
        log.append({"tool": "lookup_order", "order_id": order_id, "t": time.monotonic()})
        return "Order 4821 shipped yesterday and arrives Thursday."

    def end_call() -> str:
        """Hang up. Only when the caller clearly asked to end the call and did not take it back."""
        log.append({"tool": "end_call", "t": time.monotonic()})
        from timbal.voice import current_voice_session

        session = current_voice_session()
        if session is not None:
            session.end_call()
        return "Ending the call."

    fields = set(getattr(Tool, "model_fields", {}))
    end_kwargs: dict[str, Any] = {"handler": end_call, "name": "end_call"}
    if "defer_until_eot" in fields:
        end_kwargs["defer_until_eot"] = True
    if "interruption_mode" in fields:
        end_kwargs["interruption_mode"] = "disable_during_tool"

    def delivery_window(order_id: str) -> str:
        """Say the delivery window for an order. Read only."""
        log.append({"tool": "delivery_window", "order_id": order_id, "t": time.monotonic()})
        return "Delivery window is Thursday between two and six."

    async def slow_check(order_id: str) -> str:
        """Warehouse check. Takes several seconds. Read only."""
        log.append({"tool": "slow_check", "order_id": order_id, "t": time.monotonic(), "phase": "start"})
        await asyncio.sleep(4)
        log.append({"tool": "slow_check", "order_id": order_id, "t": time.monotonic(), "phase": "end"})
        return "The warehouse check finished. The jacket is in stock."

    def check_warranty(order_id: str) -> str:
        """Warranty lookup. The service is down."""
        log.append({"tool": "check_warranty", "order_id": order_id, "t": time.monotonic()})
        raise RuntimeError("warranty service timed out")

    tools = [Tool(handler=lookup_order, name="lookup_order"), Tool(**end_kwargs)]
    if tool_mode == "two":
        tools.append(Tool(handler=delivery_window, name="delivery_window"))
    elif tool_mode == "slow":
        tools.append(Tool(handler=slow_check, name="slow_check"))
    elif tool_mode == "error":
        tools.append(Tool(handler=check_warranty, name="check_warranty"))
    return Agent(
        name="voice_242",
        model=model,
        system_prompt=prompt,
        tools=tools,
        tracing_provider=InMemoryTracingProvider,
    )


def _span_usage(ctx: Any) -> dict[str, Any]:
    usage: dict[str, int] = {}
    ttfts: list[float] = []
    trace = getattr(ctx, "_trace", None)
    if trace is None:
        return {"usage": usage, "ttft_ms": ttfts}
    for span in trace.values():
        for key, value in (getattr(span, "usage", None) or {}).items():
            usage[key] = usage.get(key, 0) + int(value)
        ttft = (span.metadata or {}).get("ttft")
        if isinstance(ttft, int | float) and ttft >= 0:
            ttfts.append(round(float(ttft) * 1000.0, 1))
    return {"usage": usage, "ttft_ms": ttfts}


def _tool_dead_air(log: list[dict[str, Any]], audio_at: list[float]) -> list[float]:
    gaps: list[float] = []
    for call in log:
        if call.get("phase") == "end":
            continue
        started = call.get("t")
        if not isinstance(started, float):
            continue
        later = [stamp for stamp in audio_at if stamp >= started]
        if later:
            gaps.append(round((later[0] - started) * 1000.0, 1))
    return gaps


def _token_bucket(usage: dict[str, int]) -> dict[str, int]:
    input_tokens = cached = output = 0
    for key, value in usage.items():
        if key.endswith(":input_cached_tokens") or key.endswith(":cache_read_input_tokens"):
            cached += value
        elif key.endswith(":input_tokens"):
            input_tokens += value
        elif key.endswith(":output_tokens"):
            output += value
    return {"input_tokens": input_tokens, "cached_tokens": cached, "output_tokens": output}


async def _run_one(
    spec: ScenarioSpec,
    clips: dict[str, bytes],
    *,
    model: str,
    stt: str,
    overrides: dict[str, Any],
    cache_bust: bool,
    side: str,
    profile: str,
    rep: int,
    prompt_nonce: str | None = None,
) -> dict[str, Any]:
    from degrade import MicPath, active_rms
    from harness import (  # noqa: PLC0415 — script-dir import, see cli.py
        FRAME_SECS,
        PlaybackSim,
        ScriptFeeder,
    )
    from timbal.server.voice import build_voice_session

    log: list[dict[str, Any]] = []
    stable = _prompt(False)

    def busted(stable: str = stable) -> str:
        return f"nonce {time.time_ns()}\n{stable}"

    if cache_bust:
        prompt: Any = busted
    elif prompt_nonce:
        # Stable for every turn of this run, unique across runs, so turn 1 is
        # cold and a later turn in the same run can hit. BEFORE and AFTER do
        # not share a prefix.
        prompt = f"Call id {prompt_nonce}.\n{stable}"
    else:
        prompt = stable
    agent = _make_agent(model, prompt, log, spec.tool_mode)
    config, dropped = _voice_config(overrides, spec.lang, stt, greeting=spec.greeting)
    sig = inspect.signature(build_voice_session)
    extra: dict[str, Any] = {}
    if "call_context" in sig.parameters:
        extra["call_context"] = {"customer": "Acme", "user_number": "+15551212"}
    session, meta = build_voice_session(agent, config, {}, **extra)

    mic = {
        "": MicPath(),
        "phone": MicPath(telephone=True, seed=11),
        "street": MicPath(telephone=True, snr_db=8, noise="pink", seed=12),
        "office": MicPath(telephone=True, snr_db=18, noise="pink", seed=13),
        "tv": MicPath(telephone=True, snr_db=12, noise="white", seed=14),
    }[spec.mic]
    feeder = ScriptFeeder(mic_path=mic)
    if mic.snr_db is not None:
        spoken = b"".join(clips[_clip_key(text, spec.voice, spec.speed)] for text, _role in spec.lines)
        feeder.set_noise_bed(mic.bed(active_rms(spoken)))
    playback = PlaybackSim()
    assistant_speaking = asyncio.Event()
    closing = asyncio.Event()
    state: dict[str, Any] = {
        "interrupted": 0,
        "premature": False,
        "continuation_fed": False,
        "pcm": bytearray(),
        "turn_pcm": bytearray(),
        "commits": [],
        "replies": [],
        "metrics": [],
        "errors": [],
        "wasted": [],
        "seen_discarded": 0,
        "first_audio_before_continuation": False,
        "awaiting_audio_from": None,
        "caller_silence_to_first_audio_ms": [],
        "pending_continuations": sum(1 for _text, role in spec.lines if role == "continuation"),
        "barge_t0": None,
        "interrupt_t": None,
        "last_audio_t": None,
        "audio_at": [],
        "notices": [],
        "greet_waited": False,
    }

    def _note_discarded() -> None:
        discarded = int(getattr(session, "speculative_generations_discarded", 0) or 0)
        if discarded <= state["seen_discarded"]:
            return
        state["seen_discarded"] = discarded
        ctx = getattr(session, "_turn_run_context", None)
        rid = getattr(session, "_speculative_discarded_run_id", None)
        if ctx is not None and getattr(ctx, "id", None) == rid:
            state["wasted"].append(_span_usage(ctx))

    def observe(event: Any) -> None:
        from timbal.voice import (  # noqa: PLC0415
            AgentStatus,
            AgentTextDone,
            AudioOutput,
            FillerSpoken,
            SessionError,
            SessionInterrupted,
            TranscriptCommitted,
            TurnMetricsEvent,
        )

        _note_discarded()
        if isinstance(event, AudioOutput):
            playback.on_emit(len(event.data))
            assistant_speaking.set()
            state["pcm"].extend(event.data)
            state["turn_pcm"].extend(event.data)
            now = time.monotonic()
            state["last_audio_t"] = now
            if not state["audio_at"] or now - state["audio_at"][-1] > 0.4:
                state["audio_at"].append(now)
            if state["awaiting_audio_from"] is not None:
                # Wall clock from the last paced caller frame to the first
                # byte emitted. On Flux the TurnMetrics anchor is the last
                # partial, which lands near EndOfTurn and hides the silence
                # the caller already sat through.
                state["caller_silence_to_first_audio_ms"].append(
                    round((time.monotonic() - state["awaiting_audio_from"]) * 1000.0, 1)
                )
                state["awaiting_audio_from"] = None
            if state["pending_continuations"] > 0:
                state["first_audio_before_continuation"] = True
        elif isinstance(event, TranscriptCommitted):
            state["commits"].append({"text": event.text, "replace": bool(getattr(event, "replace", False)), "t": time.monotonic()})
        elif isinstance(event, AgentTextDone):
            state["replies"].append(event.text)
            assistant_speaking.clear()
        elif isinstance(event, SessionInterrupted):
            playback.on_interrupted()
            if state["interrupt_t"] is None:
                state["interrupt_t"] = time.monotonic()
            if not closing.is_set():
                state["interrupted"] += 1
        elif isinstance(event, AgentStatus | FillerSpoken):
            state["notices"].append({"type": event.type, "text": getattr(event, "text", ""), "t": time.monotonic()})
        elif isinstance(event, TurnMetricsEvent):
            metrics = event.metrics
            snap = _span_usage(getattr(session, "_turn_run_context", None))
            pauses = pause_stats(bytes(state["turn_pcm"]))
            state["turn_pcm"].clear()
            dumped = metrics.model_dump() if hasattr(metrics, "model_dump") else {}
            dumped["usage"] = snap
            dumped["pause"] = pauses
            state["metrics"].append(dumped)
        elif isinstance(event, SessionError):
            state["errors"].append(event.message)

    async def wait_for(predicate, timeout: float, what: str, *, record: bool = True) -> None:
        deadline = time.monotonic() + timeout
        while not predicate():
            if time.monotonic() > deadline:
                if record:
                    state["errors"].append(f"timed out waiting for {what}")
                return
            await asyncio.sleep(FRAME_SECS)

    async def drive() -> None:
        if spec.greeting:
            await wait_for(assistant_speaking.is_set, 25.0, "greeting")
            await asyncio.sleep(2.0)
            state["greet_waited"] = True
            assistant_speaking.clear()
        for text, role in spec.lines:
            if role == "continuation":
                await asyncio.sleep(spec.gap_before_continuation)
            if role == "overlap":
                await wait_for(assistant_speaking.is_set, 25.0, "assistant audio")
            elif role == "barge" and spec.barge_at == "start":
                await wait_for(assistant_speaking.is_set, 25.0, "assistant audio")
            elif role == "barge" and spec.barge_at == "end":
                await wait_for(assistant_speaking.is_set, 25.0, "assistant audio")
                await wait_for(lambda n=len(state["replies"]): len(state["replies"]) > n, 30.0, "reply text", record=False)
            elif role in {"backchannel", "barge"}:
                await wait_for(assistant_speaking.is_set, 25.0, "assistant audio")
                await wait_for(lambda: playback.played_ms >= 400, 20.0, "400ms of playback")
            if role in {"barge", "backchannel", "overlap"}:
                state["barge_t0"] = time.monotonic()
            pcm = clips[_clip_key(text, spec.voice, spec.speed)]
            feeder.push(pcm)
            await feeder.drain()
            if role in {"say", "continuation"}:
                state["awaiting_audio_from"] = time.monotonic()
            if role == "continuation":
                state["pending_continuations"] -= 1
                state["continuation_fed"] = state["pending_continuations"] <= 0
                # Only after the last fragment. An earlier 3 s wait would itself
                # end the turn the scenario is trying to hold open.
                if state["continuation_fed"]:
                    await asyncio.sleep(3.0)
            if role == "say" and spec.kind in {"cache", "long"}:
                await wait_for(lambda n=len(state["replies"]): len(state["replies"]) > n, 40.0, "reply")
        if spec.kind == "backchannel":
            await asyncio.sleep(3.0)
        elif spec.kind in {"barge", "overlap"}:
            await wait_for(lambda: state["interrupted"] > 0, 8.0, "barge-in", record=False)
            await asyncio.sleep(1.5)
        elif spec.kind not in {"cache", "long"}:
            await wait_for(lambda: bool(state["replies"]) or bool(state["errors"]), 40.0, "any reply")
            await asyncio.sleep(0.8)
        if spec.silence_tail:
            state["replies_before_tail"] = len(state["replies"])
            state["commits_before_tail"] = len(state["commits"])
            await asyncio.sleep(spec.silence_tail)
        closing.set()
        feeder.stop()
        await session.close()

    started = time.monotonic()
    async with aclosing(session.run(feeder.stream())) as stream:
        driver_task: asyncio.Task[None] | None = None
        acker: asyncio.Task[None] | None = None

        async def ack_loop() -> None:
            while True:
                await asyncio.sleep(0.25)
                session.playback.on_playback_ack(playback.played_ms)

        try:
            async for event in stream:
                from timbal.voice import SessionStarted

                observe(event)
                if isinstance(event, SessionStarted) and driver_task is None:
                    acker = asyncio.create_task(ack_loop())
                    driver_task = asyncio.create_task(drive())
        finally:
            for task in (driver_task, acker):
                if task is not None and not task.done():
                    task.cancel()
            for task in (driver_task, acker):
                if task is not None:
                    await asyncio.gather(task, return_exceptions=True)

    has_cont = any(role == "continuation" for _text, role in spec.lines)
    if has_cont:
        state["premature"] = bool(state["first_audio_before_continuation"])
    false_interrupt = spec.kind == "backchannel" and state["interrupted"] > 0
    missed_barge = spec.kind in {"barge", "overlap"} and state["interrupted"] == 0
    end_calls = [row for row in log if row["tool"] == "end_call"]
    retractions = [row for row in state["commits"] if row["replace"] and not (row["text"] or "").strip()]
    discarded = int(getattr(session, "speculative_generations_discarded", 0) or 0)
    destructive_on_retracted = bool(end_calls) and (bool(retractions) or discarded > 0)
    wasted_tokens = {"input_tokens": 0, "cached_tokens": 0, "output_tokens": 0}
    for snap in state["wasted"]:
        bucket = _token_bucket(snap["usage"])
        for key in wasted_tokens:
            wasted_tokens[key] += bucket[key]
    return {
        "side": side,
        "profile": profile,
        "model": model,
        "stt": stt,
        "scenario": spec.id,
        "kind": spec.kind,
        "lang": spec.lang,
        "rep": rep,
        "wall_secs": round(time.monotonic() - started, 2),
        "config_dropped": dropped,
        "stt_resolved": (meta or {}).get("stt_provider"),
        "commits": state["commits"],
        "replies": state["replies"],
        "metrics": state["metrics"],
        "errors": state["errors"],
        "interrupted": state["interrupted"],
        "false_interrupt": false_interrupt,
        "missed_barge": missed_barge,
        "premature_cutoff": state["premature"],
        "tool_calls": log,
        "end_call_calls": len(end_calls),
        "retractions": len(retractions),
        "speculative_started": int(getattr(session, "speculative_generations_started", 0) or 0),
        "speculative_discarded": discarded,
        "destructive_on_retracted": destructive_on_retracted,
        "wasted_tokens": wasted_tokens,
        "caller_silence_to_first_audio_ms": state["caller_silence_to_first_audio_ms"],
        "prompt_chars": len(stable),
        "prompt_nonce": prompt_nonce,
        "group": spec.group or spec.kind,
        "mic": spec.mic,
        "stop_audio_ms": (
            round((state["interrupt_t"] - state["barge_t0"]) * 1000.0, 1)
            if state["barge_t0"] and state["interrupt_t"]
            else None
        ),
        "notices": state["notices"],
        "reply_lang": _guess_lang(" ".join(state["replies"])),
        "commit_lang": _guess_lang(" ".join(c["text"] for c in state["commits"])),
        "language_error": bool(spec.expect_lang or spec.lang in {"en", "es"})
        and _guess_lang(" ".join(state["replies"]))
        not in {(spec.expect_lang or spec.lang), "unk"},
        "readback_error": bool(spec.readback) and not _readback_ok(spec.readback, state["replies"]),
        "phantom_turns": (
            len(state["replies"]) - state.get("replies_before_tail", len(state["replies"]))
            + max(0, len(state["commits"]) - state.get("commits_before_tail", len(state["commits"])))
            if spec.silence_tail
            else 0
        ),
        "tool_dead_air_ms": _tool_dead_air(log, state["audio_at"]),
        "audio": "synthetic-openai-tts" + (f"+{spec.mic}" if spec.mic else ""),
    }


def _append(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, default=str) + "\n")
        handle.flush()


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _metric_values(rows: list[dict[str, Any]], field_name: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        for metrics in row.get("metrics") or []:
            value = metrics.get(field_name)
            if isinstance(value, int | float) and value >= 0 and not metrics.get("event"):
                values.append(float(value))
    return values


def _dist(values: list[float]) -> dict[str, float | int | None]:
    return {"n": len(values), "p50": _percentile(values, 50), "p90": _percentile(values, 90)}


def _caller_silence_values(rows: list[dict[str, Any]]) -> list[float]:
    values: list[float] = []
    for row in rows:
        for value in row.get("caller_silence_to_first_audio_ms") or []:
            if isinstance(value, int | float) and value >= 0:
                values.append(float(value))
    return values


def _silence_values(rows: list[dict[str, Any]]) -> list[float]:
    values = []
    for row in rows:
        for metrics in row.get("metrics") or []:
            speech = metrics.get("speech_end_to_transcript_ms")
            audio = metrics.get("eou_to_first_audio_ms")
            if isinstance(speech, int | float) and isinstance(audio, int | float) and not metrics.get("event"):
                values.append(float(speech) + float(audio))
    return values


def _cache_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cold: list[float] = []
    warm: list[float] = []
    cached = input_tokens = 0
    for row in rows:
        turns = row.get("metrics") or []
        for index, metrics in enumerate(turns):
            bucket = _token_bucket((metrics.get("usage") or {}).get("usage") or {})
            ttfts = (metrics.get("usage") or {}).get("ttft_ms") or []
            ttft = ttfts[0] if ttfts else metrics.get("eou_to_llm_first_token_ms")
            cached += bucket["cached_tokens"]
            input_tokens += bucket["input_tokens"] + bucket["cached_tokens"]
            if isinstance(ttft, int | float):
                (cold if index == 0 else warm).append(float(ttft))
    rate = (cached / input_tokens) if input_tokens else None
    return {
        "ttft_cold": _dist(cold),
        "ttft_warm": _dist(warm),
        "cached_tokens": cached,
        "input_tokens_incl_cached": input_tokens,
        "hit_rate": None if rate is None else round(rate, 3),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate raw rows. Uses #244 percentile math (same Hyndman–Fan type 7)."""
    groups: dict[tuple, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("warmup"):
            continue
        key = (row.get("side"), row.get("profile"), row.get("model"), row.get("stt"), row.get("kind"))
        groups.setdefault(key, []).append(row)

    cells = []
    for key, group in sorted(groups.items(), key=lambda item: tuple(str(part) for part in item[0])):
        side, profile, model, stt, kind = key
        pauses = []
        for row in group:
            for metrics in row.get("metrics") or []:
                pause = (metrics.get("pause") or {}).get("internal_pause_secs")
                if isinstance(pause, int | float):
                    pauses.append(float(pause))
        wasted = {"input_tokens": 0, "cached_tokens": 0, "output_tokens": 0}
        for row in group:
            for key_name in wasted:
                wasted[key_name] += int((row.get("wasted_tokens") or {}).get(key_name) or 0)
        cells.append(
            {
                "side": side,
                "profile": profile,
                "model": model,
                "stt": stt,
                "kind": kind,
                "runs": len(group),
                "caller_silence_to_first_audio_ms": _dist(_caller_silence_values(group)),
                "turn_metrics_silence_to_first_audio_ms": _dist(_silence_values(group)),
                "eou_to_first_audio_ms": _dist(_metric_values(group, "eou_to_first_audio_ms")),
                "eou_to_llm_first_token_ms": _dist(_metric_values(group, "eou_to_llm_first_token_ms")),
                "speech_end_to_transcript_ms": _dist(_metric_values(group, "speech_end_to_transcript_ms")),
                "cache": _cache_stats([row for row in group if row.get("kind") == "cache"] or group),
                "false_interrupts": sum(1 for row in group if row.get("false_interrupt")),
                "missed_barges": sum(1 for row in group if row.get("missed_barge")),
                "premature_cutoffs": sum(1 for row in group if row.get("premature_cutoff")),
                "destructive_on_retracted": sum(1 for row in group if row.get("destructive_on_retracted")),
                "end_call_calls": sum(int(row.get("end_call_calls") or 0) for row in group),
                "speculative_started": sum(int(row.get("speculative_started") or 0) for row in group),
                "speculative_discarded": sum(int(row.get("speculative_discarded") or 0) for row in group),
                "wasted_tokens": wasted,
                "internal_pause_secs": _dist(pauses),
                "errors": sum(len(row.get("errors") or []) for row in group),
            }
        )
    return {"cells": cells}


def _write_csv(path: Path, summary: dict[str, Any]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "side",
        "profile",
        "model",
        "stt",
        "kind",
        "runs",
        "silence_p50",
        "silence_p90",
        "metrics_silence_p50",
        "eou_audio_p50",
        "eou_audio_p90",
        "llm_ttft_p50",
        "false_interrupts",
        "missed_barges",
        "premature_cutoffs",
        "destructive_on_retracted",
        "speculative_started",
        "speculative_discarded",
        "wasted_output_tokens",
        "pause_p50",
        "cache_hit_rate",
        "errors",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for cell in summary["cells"]:
            writer.writerow(
                {
                    "side": cell["side"],
                    "profile": cell["profile"],
                    "model": cell["model"],
                    "stt": cell["stt"],
                    "kind": cell["kind"],
                    "runs": cell["runs"],
                    "silence_p50": cell["caller_silence_to_first_audio_ms"]["p50"],
                    "silence_p90": cell["caller_silence_to_first_audio_ms"]["p90"],
                    "metrics_silence_p50": cell["turn_metrics_silence_to_first_audio_ms"]["p50"],
                    "eou_audio_p50": cell["eou_to_first_audio_ms"]["p50"],
                    "eou_audio_p90": cell["eou_to_first_audio_ms"]["p90"],
                    "llm_ttft_p50": cell["eou_to_llm_first_token_ms"]["p50"],
                    "false_interrupts": cell["false_interrupts"],
                    "missed_barges": cell["missed_barges"],
                    "premature_cutoffs": cell["premature_cutoffs"],
                    "destructive_on_retracted": cell["destructive_on_retracted"],
                    "speculative_started": cell["speculative_started"],
                    "speculative_discarded": cell["speculative_discarded"],
                    "wasted_output_tokens": cell["wasted_tokens"]["output_tokens"],
                    "pause_p50": cell["internal_pause_secs"]["p50"],
                    "cache_hit_rate": cell["cache"]["hit_rate"],
                    "errors": cell["errors"],
                }
            )


def _jobs(side: str, reps: int) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    headline = [spec.id for spec in SCENARIOS]
    if side == "before":
        for spec_id in headline:
            for rep in range(reps):
                jobs.append(
                    {
                        "scenario": spec_id,
                        "model": "openai/gpt-5.4-mini",
                        "stt": "deepgram-flux",
                        "profile": "default",
                        "overrides": {},
                        "cache_bust": False,
                        "rep": rep,
                    }
                )
        for spec_id in ("en_statement", "es_question", "cache_name"):
            for rep in range(min(reps, 3)):
                jobs.append(
                    {
                        "scenario": spec_id,
                        "model": "groq/openai/gpt-oss-20b",
                        "stt": "deepgram-flux",
                        "profile": "default",
                        "overrides": {},
                        "cache_bust": False,
                        "rep": rep,
                    }
                )
        return jobs

    for spec_id in headline:
        for rep in range(reps):
            jobs.append(
                {
                    "scenario": spec_id,
                    "model": "openai/gpt-5.4-mini",
                    "stt": "deepgram-flux",
                    "profile": "best",
                    "overrides": BEST,
                    "cache_bust": False,
                    "rep": rep,
                }
            )
    for spec_id in ("en_statement", "es_question", "cache_name"):
        for rep in range(min(reps, 3)):
            jobs.append(
                {
                    "scenario": spec_id,
                    "model": "groq/openai/gpt-oss-20b",
                    "stt": "deepgram-flux",
                    "profile": "best",
                    "overrides": BEST,
                    "cache_bust": False,
                    "rep": rep,
                }
            )
    for name, patch in ABLATIONS.items():
        overrides = dict(BEST)
        overrides.update(patch)
        for spec in SCENARIOS:
            if name not in spec.ablate:
                continue
            for rep in range(min(reps, 3)):
                jobs.append(
                    {
                        "scenario": spec.id,
                        "model": "openai/gpt-5.4-mini",
                        "stt": "deepgram-flux",
                        "profile": name,
                        "overrides": overrides,
                        "cache_bust": name == "cache_bust",
                        "rep": rep,
                    }
                )
    # Floors only move the local endpointer. Flux forces it off, so this arm
    # is ElevenLabs Scribe, not the default STT.
    for profile, overrides in (
        ("scribe_default_floors", {"min_delay_secs": 0.45, "stop_silence_secs": 0.20}),
        ("scribe_tuned_floors", {"min_delay_secs": 0.30, "stop_silence_secs": 0.20, "predict_during_speech": True}),
    ):
        for spec_id in ("en_statement", "en_unfinished", "es_unfinished"):
            for rep in range(min(reps, 3)):
                jobs.append(
                    {
                        "scenario": spec_id,
                        "model": "openai/gpt-5.4-mini",
                        "stt": "elevenlabs",
                        "profile": profile,
                        "overrides": overrides,
                        "cache_bust": False,
                        "rep": rep,
                    }
                )
    return jobs


def _ensure_main_worktree() -> Path:
    path = Path("/tmp/timbal-main-242")
    if not path.exists():
        subprocess.check_call(["git", "worktree", "add", "--detach", str(path), "origin/main"], cwd=REPO)
    return path


def _run_jobs(jobs: list[dict[str, Any]], out: Path, side: str) -> None:
    clips = synthesize(_scenario_texts())
    by_id = {spec.id: spec for spec in SCENARIOS}
    for index, job in enumerate(jobs, start=1):
        spec = by_id[job["scenario"]]
        print(f"[{index}/{len(jobs)}] {side} {job['profile']} {job['model']} {spec.id} rep {job['rep']}", flush=True)
        try:
            row = asyncio.run(
                _run_one(
                    spec,
                    clips,
                    model=job["model"],
                    stt=job["stt"],
                    overrides=job["overrides"],
                    cache_bust=job["cache_bust"],
                    side=side,
                    profile=job["profile"],
                    rep=job["rep"],
                    prompt_nonce=job.get("prompt_nonce"),
                )
            )
        except Exception as exc:  # one failed call must not drop the rest of the matrix
            row = {
                "side": side,
                "profile": job["profile"],
                "model": job["model"],
                "stt": job["stt"],
                "scenario": spec.id,
                "kind": spec.kind,
                "rep": job["rep"],
                "audio": "synthetic-openai-tts",
                "errors": [f"{type(exc).__name__}: {exc}"],
                "metrics": [],
            }
            print(f"  ERROR {row['errors'][0]}", flush=True)
        else:
            silence = _silence_values([row])
            print(
                f"  silence {silence} interrupts {row.get('interrupted')} errors {row.get('errors')}",
                flush=True,
            )
        _append(out, row)


def _self_test() -> None:
    import numpy as np

    rate = 16_000
    tone = (np.sin(2 * np.pi * 220 * np.arange(rate) / rate) * 8000).astype("<i2")
    quiet = np.zeros(rate // 2, dtype="<i2")
    pcm = np.concatenate([quiet, tone, quiet, tone, quiet]).tobytes()
    stats = pause_stats(pcm, rate)
    # 0.5s of internal quiet between the tones. Leading and trailing are excluded.
    if not 0.4 <= stats["internal_pause_secs"] <= 0.6:
        raise SystemExit(f"pause self-test failed: {stats}")
    if _percentile([10, 20, 30, 40], 90) is None:
        raise SystemExit("percentile self-test failed")
    print("self-test ok", stats)


def _catalog() -> dict[str, ScenarioSpec]:
    return {spec.id: spec for spec in (*SCENARIOS, *WAVE2)}


def _wave2_clips() -> dict[str, bytes]:
    items = []
    seen: set[str] = set()
    for spec in WAVE2:
        for text, _role in spec.lines:
            key = _clip_key(text, spec.voice, spec.speed)
            if key in seen:
                continue
            seen.add(key)
            items.append((text, spec.voice, spec.speed))
    return synthesize(items)


def _wave2_jobs(flux_reps: int, other_reps: int) -> list[dict[str, Any]]:
    """Interleave before and after. Each run gets its own prompt prefix."""
    import uuid

    jobs: list[dict[str, Any]] = []
    groq_ids = {
        "tel_phone_en",
        "hesit_um_en",
        "dict_order_en",
        "bc_yeah",
        "barge_mid_en",
        "lang_en_control",
        "long_10_en",
        "tools_slow",
    }

    def add(spec: ScenarioSpec, stt: str, model: str, reps: int) -> None:
        for rep in range(reps):
            # Alternate which side runs first so neither side always warms the machine.
            order = ("before", "after") if rep % 2 == 0 else ("after", "before")
            for side in order:
                jobs.append(
                    {
                        "scenario": spec.id,
                        "model": model,
                        "stt": stt,
                        "profile": "default" if side == "before" else "recommended",
                        "overrides": {} if side == "before" else RECOMMENDED,
                        "cache_bust": False,
                        "side": side,
                        "rep": rep,
                        "prompt_nonce": uuid.uuid4().hex,
                    }
                )

    for spec in WAVE2:
        add(spec, "deepgram-flux", "openai/gpt-5.4-mini", flux_reps)
        add(spec, "elevenlabs", "openai/gpt-5.4-mini", other_reps)
        if spec.id in groq_ids:
            add(spec, "deepgram-flux", "groq/openai/gpt-oss-120b", other_reps)
    return jobs


def _row_key(row: dict[str, Any]) -> tuple:
    return (row.get("side"), row.get("stt"), row.get("model"), row.get("scenario"), row.get("rep"), row.get("profile"))


def _execute_job(job: dict[str, Any], clips: dict[str, bytes]) -> dict[str, Any]:
    spec = _catalog()[job["scenario"]]
    return asyncio.run(
        _run_one(
            spec,
            clips,
            model=job["model"],
            stt=job["stt"],
            overrides=job["overrides"],
            cache_bust=job["cache_bust"],
            side=job["side"],
            profile=job["profile"],
            rep=job["rep"],
            prompt_nonce=job.get("prompt_nonce"),
        )
    )


def _worker() -> None:
    """One long-lived process bound to whichever ``timbal`` PYTHONPATH selected."""
    clips = _wave2_clips()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        job = json.loads(line)
        try:
            row = _execute_job(job, clips)
        except Exception as exc:
            row = {
                "side": job.get("side"),
                "profile": job.get("profile"),
                "model": job.get("model"),
                "stt": job.get("stt"),
                "scenario": job.get("scenario"),
                "rep": job.get("rep"),
                "errors": [f"{type(exc).__name__}: {exc}"],
                "metrics": [],
            }
        sys.stdout.write(json.dumps(row, default=str) + "\n")
        sys.stdout.flush()


def _run_wave2(out: Path, flux_reps: int, other_reps: int) -> None:
    clips = _wave2_clips()
    jobs = _wave2_jobs(flux_reps, other_reps)
    done = {_row_key(row) for row in _load_rows(out)}
    pending = [job for job in jobs if _row_key(job) not in done]
    print(f"wave2 {len(pending)} pending of {len(jobs)}", flush=True)
    worktree = _ensure_main_worktree()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(worktree / "python") + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    worker = subprocess.Popen(
        [sys.executable, str(HERE / "bench_242.py"), "worker"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        cwd=REPO,
        env=env,
        text=True,
        bufsize=1,
    )
    assert worker.stdin is not None and worker.stdout is not None
    try:
        for index, job in enumerate(pending, start=1):
            print(
                f"[{index}/{len(pending)}] {job['side']} {job['stt']} {job['model']} {job['scenario']} rep {job['rep']}",
                flush=True,
            )
            try:
                if job["side"] == "before":
                    worker.stdin.write(json.dumps(job) + "\n")
                    worker.stdin.flush()
                    raw = worker.stdout.readline()
                    if not raw:
                        raise RuntimeError("before worker exited")
                    row = json.loads(raw)
                else:
                    row = _execute_job(job, clips)
            except Exception as exc:
                row = {
                    "side": job["side"],
                    "profile": job["profile"],
                    "model": job["model"],
                    "stt": job["stt"],
                    "scenario": job["scenario"],
                    "rep": job["rep"],
                    "errors": [f"{type(exc).__name__}: {exc}"],
                    "metrics": [],
                }
                print(f"  ERROR {row['errors'][0]}", flush=True)
            else:
                print(
                    f"  silence {row.get('caller_silence_to_first_audio_ms')} "
                    f"interrupts {row.get('interrupted')} lang {row.get('reply_lang')} "
                    f"errors {row.get('errors')}",
                    flush=True,
                )
            _append(out, row)
    finally:
        worker.stdin.close()
        worker.terminate()
        worker.wait(timeout=10)


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure #242 voice changes against main")
    parser.add_argument(
        "command",
        choices=("all", "run", "synth", "summarize", "self-test", "wave2", "worker"),
    )
    parser.add_argument("--side", choices=("before", "after"), default="after")
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--flux-reps", type=int, default=10)
    parser.add_argument("--other-reps", type=int, default=5)
    parser.add_argument("--out", type=Path, default=OUT_DIR / "runs.jsonl")
    args = parser.parse_args()

    if args.command == "worker":
        _worker()
        return
    if args.command == "wave2":
        out = args.out if args.out.name != "runs.jsonl" else OUT_DIR / "runs-wave2.jsonl"
        _run_wave2(out, args.flux_reps, args.other_reps)
        return
    if args.command == "self-test":
        _self_test()
        return
    if args.command == "synth":
        synthesize(_scenario_texts())
        print(f"clips in {AUDIO_DIR}")
        return
    if args.command == "summarize":
        summary = summarize(_load_rows(args.out))
        summary_path = args.out.with_name("summary.json")
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        _write_csv(args.out.with_name("summary.csv"), summary)
        print(summary_path)
        return
    if args.command == "run":
        _run_jobs(_jobs(args.side, args.reps), args.out, args.side)
        return

    # all: synth, after on this tree, before on main, then summarize.
    synthesize(_scenario_texts())
    _run_jobs(_jobs("after", args.reps), args.out, "after")
    worktree = _ensure_main_worktree()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(worktree / "python") + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.check_call(
        [sys.executable, str(HERE / "bench_242.py"), "run", "--side", "before", "--reps", str(args.reps), "--out", str(args.out)],
        cwd=REPO,
        env=env,
    )
    summary = summarize(_load_rows(args.out))
    summary_path = args.out.with_name("summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_csv(args.out.with_name("summary.csv"), summary)
    print(summary_path)


if __name__ == "__main__":
    main()
