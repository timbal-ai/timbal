"""Listening sounds in a real call: where they fall, and what they cost.

A caller (a synthesized voice, rendered as one sentence and cut at its pauses,
so every group keeps the intonation of a sentence still going) talks to an
agent over real STT, a real model and real TTS, once with
``VoiceConfig.backchannel`` off and once with it on. The scripts:

- ``dni``: an ID number dictated in groups, a pause after each from the second on;
- ``dni1``: the same with a pause after the first group too. One number word after a
  sentence ("es el doce") is not a number being dictated for ``entity_patience``, so the
  endpointer may take that pause for the end of the turn, listening sounds or not;
- ``long``: an explanation the caller stops in the middle of twice, mid-sentence;
- ``sentences``: an explanation in three clauses, each one complete, a pause after each.
  An endpointer that ends the turn at a finished clause does so here, listening sounds or
  not: a sound is for a caller the session is waiting for, and it is not waiting;
- ``fluent``: the same ID number with no pauses;
- ``question``: a short question (no sound is due).

Per call: each listening sound with the pause it fell in and how long after
the caller went quiet; whether any played while the caller was speaking;
interruptions (a caller who goes on over the sound must not be one); the
agent's turns; whether what was committed has every digit; and the time from
the caller's last word to the reply's first audio.

    uv run python benchmarks/voice/backchannel_call.py
    uv run python benchmarks/voice/backchannel_call.py --scripts dni --runs 5 --wav /tmp/calls

Needs ``DEEPGRAM_API_KEY`` and ``ELEVENLABS_API_KEY``, and the key of ``--model``'s provider.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from contextlib import aclosing
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from accept_duplex import CallState, PacedFeeder, PlaybackSim, observe  # noqa: E402
from degrade import noise_bed  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from synth import (  # noqa: E402
    BYTES_PER_SECOND,
    SAMPLE_RATE,
    USER_VOICE_ID,
    part_key,
    silence,
    synthesize_fluent,
    write_wav,
)
from timbal import Agent  # noqa: E402
from timbal.server.voice import build_voice_session  # noqa: E402
from timbal.state.tracing.providers import InMemoryTracingProvider  # noqa: E402
from timbal.voice import AudioOutput, BackchannelSpoken, SessionStarted, TurnMetricsEvent, VoiceConfig  # noqa: E402

PROMPT = (
    "Eres Alba, agente telefónica de una compañía eléctrica. Hablas español de España, con frases cortas. "
    "Si el cliente te dicta su DNI, espera a que termine y repíteselo una vez, cifra a cifra, para confirmarlo. "
    "Si te cuenta un problema, dile en una frase que lo vas a mirar. No tienes herramientas."
)
GREETING = "Buenas, te atiende Alba. ¿En qué te puedo ayudar?"

DNI = ("Sí, mira, es el doce,", "treinta y cuatro,", "cincuenta y seis,", "setenta y ocho,", "letra zeta.")
SCRIPTS: dict[str, dict[str, Any]] = {
    "dni": {"parts": (" ".join(DNI[:2]), *DNI[2:]), "pause": 1.0, "digits": True},
    "dni1": {"parts": DNI, "pause": 1.0, "digits": True},
    "long": {
        "parts": (
            "Pues mira, te llamo porque ayer me llegó la factura de la luz y resulta que",
            "pone un importe que es casi el doble de lo que pago normalmente, y además",
            "me aparece un concepto nuevo que no había visto nunca.",
        ),
        "pause": 0.9,
    },
    "sentences": {
        "parts": (
            "Pues mira, te llamo porque ayer me llegó la factura de la luz y no la entiendo,",
            "porque pone un importe que es casi el doble de lo que pago normalmente,",
            "y además me aparece un concepto nuevo que no había visto nunca.",
        ),
        "pause": 0.9,
    },
    "fluent": {"parts": (" ".join(DNI),), "pause": 0.0, "digits": True},
    "question": {"parts": ("¿A qué hora abrís mañana por la mañana?",), "pause": 0.0},
}
TAIL_SECS = 0.3
GROUPS = (("doce", "12"), ("treinta y cuatro", "34"), ("cincuenta y seis", "56"), ("setenta y ocho", "78"))


def heard_all(text: str) -> bool:
    """Whether ``text`` has every group of the ID number, as words or as digits."""
    lowered = text.lower()
    digits = "".join(ch for ch in lowered if ch.isdigit())
    return all(words in lowered or number in digits for words, number in GROUPS)


def on_a_line(pcm: bytes, noise_dbfs: float | None, seed: int) -> bytes:
    """``pcm`` with the noise of a line under it (digital silence between words is not a phone call)."""
    if noise_dbfs is None:
        return pcm
    from array import array

    samples = array("h")
    samples.frombytes(pcm[: len(pcm) - len(pcm) % 2])
    bed = array("h")
    bed.frombytes(noise_bed(len(samples) / SAMPLE_RATE + 0.1, 32768 * 10 ** (noise_dbfs / 20), kind="pink", seed=seed))
    for i in range(len(samples)):
        samples[i] = max(-32768, min(32767, samples[i] + bed[i]))
    return samples.tobytes()


async def caller_audio(name: str, args: argparse.Namespace) -> tuple[bytes, list[tuple[float, float]]]:
    """The caller's side of one script, and where each pause starts and ends in it (seconds)."""
    script = SCRIPTS[name]
    parts = script["parts"]
    utterance = " ".join(parts)
    clips = await synthesize_fluent([(utterance, parts)], voice_id=args.caller_voice, model=args.caller_model)
    audio = bytearray()
    pauses: list[tuple[float, float]] = []
    for index in range(len(parts)):
        audio += clips[part_key(utterance, index)]
        if index < len(parts) - 1:
            start = len(audio) / BYTES_PER_SECOND
            audio += silence(script["pause"])
            pauses.append((start, len(audio) / BYTES_PER_SECOND))
    audio += silence(TAIL_SECS)
    return on_a_line(bytes(audio), args.noise_dbfs, seed=len(utterance)), pauses


def build_session(args: argparse.Namespace, *, backchannel: bool):
    agent = Agent(
        name="backchannel_call",
        model=args.model,
        system_prompt=PROMPT,
        tools=[],
        max_tokens=200,
        tracing_provider=InMemoryTracingProvider,
    )
    config = VoiceConfig(
        language="es",
        reply_language="es",
        stt_provider=args.stt,
        tts_model=args.tts_model,
        entity_patience=True,
        ignore_backchannels=True,
        greeting={"text": GREETING},
        backchannel={"rate": args.rate} if backchannel else None,
        **({"voice": args.voice} if args.voice else {}),
    )
    session, _ = build_voice_session(agent, config, {})
    return session


async def one_call(name: str, audio: bytes, pauses: list[tuple[float, float]], args, *, backchannel: bool) -> dict:
    state = CallState(name=name, t0=time.monotonic())
    feeder = PacedFeeder()
    playback = PlaybackSim()
    session = build_session(args, backchannel=backchannel)
    sounds: list[dict[str, Any]] = []
    chunks: list[dict[str, Any]] = []  # every agent audio chunk: when, and whether it was a listening sound
    marks: dict[str, float] = {}
    turn_ends: list[float] = []

    async def acks() -> None:
        while True:
            await asyncio.sleep(0.1)
            session.playback.on_playback_ack(playback.played_ms)

    async def drive() -> None:
        try:
            # The greeting, heard to its end, and the clips ready (a real caller takes longer to start).
            while not chunks or time.monotonic() < playback._playing_until + 0.4:
                await asyncio.sleep(0.05)
            if backchannel:
                for _ in range(100):
                    if session._backchannel_clips.get("es"):
                        break
                    await asyncio.sleep(0.1)
            marks["start"] = time.monotonic()
            feeder.push(audio)
            await feeder.drain()
            marks["end"] = end = time.monotonic() - TAIL_SECS
            # The turn that answers the whole of it: one that ended after the caller did.
            deadline = time.monotonic() + 25
            while time.monotonic() < deadline and not any(t >= end for t in turn_ends):
                await asyncio.sleep(0.05)
            # Its reply, heard to the end.
            while time.monotonic() < min(playback._playing_until + 0.5, deadline + 20):
                await asyncio.sleep(0.05)
        finally:
            feeder.stop()
            if not session.closed:
                await session.close()

    driver = acker = None
    try:
        async with asyncio.timeout(120):
            async with aclosing(session.run(feeder.stream())) as stream:
                async for event in stream:
                    observe(event, state, playback)
                    now = time.monotonic()
                    if isinstance(event, AudioOutput):
                        chunks.append({"t": now, "secs": len(event.data) / BYTES_PER_SECOND, "sound": False})
                    elif isinstance(event, BackchannelSpoken):
                        chunks[-1]["sound"] = True
                        sounds.append({"text": event.text, "reason": event.reason, "t": chunks[-1]["t"]})
                    elif isinstance(event, TurnMetricsEvent):
                        turn_ends.append(now)
                    elif isinstance(event, SessionStarted) and driver is None:
                        acker = asyncio.create_task(acks())
                        driver = asyncio.create_task(drive())
    except TimeoutError:
        state.timeouts.append("call 120s")
    finally:
        feeder.stop()
        for task in (driver, acker):
            if task is not None and not task.done():
                task.cancel()
        await asyncio.gather(*(t for t in (driver, acker) if t is not None), return_exceptions=True)
        if not session.closed:
            await session.close()

    start, end = marks.get("start"), marks.get("end")
    for sound in sounds:
        at = sound.pop("t") - (start or 0.0)
        sound["at"] = round(at, 2)
        inside = [i for i, (a, b) in enumerate(pauses) if a <= at < b]
        sound["pause"] = inside[0] + 1 if inside else None
        sound["after_quiet"] = round(at - pauses[inside[0]][0], 2) if inside else None
        sound["after_end"] = at >= (end or 0.0) - (start or 0.0)
        sound["over_speech"] = not inside and not sound["after_end"]
    reply = [c["t"] for c in chunks if not c["sound"] and end is not None and c["t"] >= end]
    after_start = [i for i in state.interrupts if start is not None and state.t0 + i["t"] >= start]
    commits = [c["text"] for c in state.commits if start is not None and state.t0 + c["t"] >= start]
    if args.wav:
        Path(args.wav).mkdir(parents=True, exist_ok=True)
        write_wav(Path(args.wav) / f"{name}-{'on' if backchannel else 'off'}-{int(state.t0)}.wav", bytes(state.audio))
    return {
        "script": name,
        "backchannel": backchannel,
        "sounds": sounds,
        "interruptions": len(after_start),
        "turns": len(state.metrics),
        "turns_before_end": sum(1 for c in state.commits if end is not None and state.t0 + c["t"] < end - 0.2),
        "commits": commits,
        "digits_ok": heard_all(" ".join(commits[-1:])) if SCRIPTS[name].get("digits") else None,
        "reply_ms": round((reply[0] - end) * 1000) if reply else None,
        "reply": state.dones[-1]["text"] if state.dones else "",
        "counted": [m.get("backchannels") for m in state.metrics],
        "errors": state.errors,
        "timeouts": state.timeouts,
    }


def summarize(rows: list[dict]) -> list[str]:
    lines = []
    for name in dict.fromkeys(row["script"] for row in rows):
        for on in (False, True):
            calls = [row for row in rows if row["script"] == name and row["backchannel"] is on]
            if not calls:
                continue
            sounds = [sound for call in calls for sound in call["sounds"]]
            delays = [sound["after_quiet"] for sound in sounds if sound["after_quiet"] is not None]
            replies = [call["reply_ms"] for call in calls if call["reply_ms"] is not None]
            digits = [call["digits_ok"] for call in calls if call["digits_ok"] is not None]
            lines.append(
                f"{name:<9} {'on ' if on else 'off'} calls {len(calls)}"
                f" | sounds {[len(call['sounds']) for call in calls]}"
                f" in pauses {sorted(sound['pause'] for sound in sounds if sound['pause'])}"
                f" after the end {sum(sound['after_end'] for sound in sounds)}"
                f" over speech {sum(sound['over_speech'] for sound in sounds)}"
                + (
                    f" | after quiet {statistics.median(delays):.2f} s ({min(delays):.2f}-{max(delays):.2f})"
                    if delays
                    else ""
                )
                + f" | interruptions {sum(call['interruptions'] for call in calls)}"
                f" | turns {[call['turns'] for call in calls]}"
                f" early commits {sum(call['turns_before_end'] for call in calls)}"
                + (f" | digits {sum(digits)}/{len(digits)}" if digits else "")
                + (
                    f" | reply {round(statistics.median(replies))} ms ({min(replies)}-{max(replies)})"
                    if replies
                    else ""
                )
                + (
                    f" | FAILED {sum(bool(call['errors'] or call['timeouts']) for call in calls)}"
                    if any(call["errors"] or call["timeouts"] for call in calls)
                    else ""
                )
            )
    return lines


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scripts", default=",".join(SCRIPTS), help="comma-separated: " + ", ".join(SCRIPTS))
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--rate", type=float, default=1.0, help="backchannel.rate (1 = every pause that qualifies)")
    parser.add_argument("--model", default="google/gemini-2.5-flash")
    parser.add_argument("--stt", default="deepgram-flux")
    parser.add_argument("--tts-model", default="eleven_v3_conversational")
    parser.add_argument("--voice", default=None, help="the agent's voice (the provider's default without it)")
    parser.add_argument("--caller-voice", default=USER_VOICE_ID)
    parser.add_argument("--caller-model", default="eleven_multilingual_v2")
    parser.add_argument("--noise-dbfs", type=float, default=-55.0)
    parser.add_argument("--no-noise", dest="noise_dbfs", action="store_const", const=None)
    parser.add_argument("--only", choices=("on", "off"), default=None)
    parser.add_argument("--wav", default=None, help="directory for what the caller heard, one file per call")
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    load_dotenv()

    rows: list[dict] = []
    for name in [item.strip() for item in args.scripts.split(",") if item.strip()]:
        audio, pauses = await caller_audio(name, args)
        for on in (False, True):
            if args.only and (args.only == "on") is not on:
                continue
            for _ in range(args.runs):
                row = await one_call(name, audio, pauses, args, backchannel=on)
                rows.append(row)
                print(json.dumps(row, ensure_ascii=False), flush=True)  # noqa: T201
    print()  # noqa: T201
    for line in summarize(rows):
        print(line)  # noqa: T201
    if args.json:
        Path(args.json).write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
