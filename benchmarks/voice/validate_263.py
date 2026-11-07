"""Bounded live checks for the #263 voice changes. Not a percentile bench.

Analysis is assistant text events and Flux committed transcripts. WAV files are
emitted PCM. Nothing here listens, and the WAV is not sent back through a
second recognizer.

    TIMBAL_VOICE_VALIDATE_263=1 uv run python benchmarks/voice/validate_263.py

Missing OpenAI, Deepgram, or ElevenLabs keys exit 2 and name the key.
Groq and Anthropic are not called.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import sys
import time
from contextlib import aclosing
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import accept_duplex as gate
from degrade import MicPath, active_rms, telephone_state
from synth import SAMPLE_RATE, silence, synthesize_clips
from timbal.voice import (
    AudioInputConfig,
    AudioOutputConfig,
    SessionStarted,
    SpeculativeGeneration,
    VoiceSession,
    current_voice_session,
    resolve_stt,
)
from timbal.voice.elevenlabs import ElevenLabsStreamTTS

FLAG = "TIMBAL_VOICE_VALIDATE_263"
ARTIFACT_DIR = Path("/opt/cursor/artifacts/voice-validate-263")
PAUSES = (0.6, 1.0, 1.5)

_ES = frozenset(
    [
        "el",
        "la",
        "los",
        "las",
        "un",
        "una",
        "que",
        "qué",
        "hoy",
        "tiempo",
        "hace",
        "esta",
        "está",
        "reloj",
        "clima",
        "tengo",
    ]
)
_CA = frozenset(
    ["quina", "hora", "és", "es", "avui", "temps", "rellotge", "tinc", "fa", "ara", "mateix", "són", "les", "ho", "sé"]
)
_EN = frozenset(["the", "time", "clock", "weather", "today", "don't", "dont", "have", "not", "i"])


def judge_reply_language(text: str) -> str:
    """Lexical guess from the assistant's text. Not ASR and not listening."""
    words = re.findall(r"[^\W\d_]+", text.casefold(), flags=re.UNICODE)
    scores = {
        "es": sum(word in _ES for word in words),
        "ca": sum(word in _CA for word in words),
        "en": sum(word in _EN for word in words),
    }
    best = max(scores, key=scores.get)
    if scores[best] == 0 or list(scores.values()).count(scores[best]) > 1:
        return "unknown"
    return best


def recall_digits(transcript: str) -> float:
    """Digit recall. ``oh`` and ``zero`` count as the same word."""
    words = re.findall(r"[a-z0-9']+", transcript.casefold())
    folded = ["zero" if word in {"oh", "o", "zero"} else word for word in words]
    wanted = ("four", "one", "five", "five", "five", "zero", "one", "nine", "nine")
    blob = " ".join(folded)
    hits = 0
    cursor = 0
    for word in wanted:
        at = blob.find(word, cursor)
        if at < 0:
            continue
        hits += 1
        cursor = at + len(word)
    return round(hits / len(wanted), 3)


def phone_pcm(pcm: bytes) -> bytes:
    """Band-limit, 8 kHz round trip, back at 16 kHz. Same length as the input."""
    path = MicPath(telephone=True)
    state = telephone_state()
    frames = []
    width = int(SAMPLE_RATE * 2 * 0.02)
    padded = pcm + b"\x00" * ((width - len(pcm) % width) % width)
    for offset in range(0, len(padded), width):
        frames.append(path.apply(padded[offset : offset + width], state))
    out = b"".join(frames)
    return out[: len(pcm) - (len(pcm) % 2)]


def mix_snr(pcm: bytes, snr_db: float = 8.0) -> bytes:
    import array

    speech = active_rms(pcm) or 1.0
    bed = MicPath(snr_db=snr_db, noise="pink", seed=8).bed(speech)
    usable = len(pcm) - (len(pcm) % 2)
    out = array.array("h")
    out.frombytes(pcm[:usable])
    other = array.array("h")
    other.frombytes(bed[:usable])
    for i in range(min(len(out), len(other))):
        mixed = int(out[i]) + int(other[i])
        out[i] = 32767 if mixed > 32767 else (max(mixed, -32768))
    return out.tobytes()


def _session(
    name: str,
    prompt: str,
    tools: list[Any],
    *,
    language: str = "en",
    reply_language: str | None = None,
    speculative: bool = False,
    extra: dict[str, Any] | None = None,
) -> VoiceSession:
    return VoiceSession(
        agent=gate._agent(name, prompt, tools),
        stt=resolve_stt("deepgram-flux"),
        tts=ElevenLabsStreamTTS(),
        audio_input=AudioInputConfig(language=language, sample_rate=SAMPLE_RATE, extra=dict(extra or {})),
        audio_output=AudioOutputConfig(model=gate.TTS_MODEL, voice=gate.ASSISTANT_VOICE_ID, sample_rate=SAMPLE_RATE),
        turn_detector="provider",
        ignore_backchannels=True,
        speculative_generation=SpeculativeGeneration(enabled=speculative, preemptive_tts=speculative),
        reply_language=reply_language,
    )


async def _drive(name: str, session: VoiceSession, steps, out_dir: Path) -> dict[str, Any]:
    state = gate.CallState(name=name, t0=time.monotonic())
    feeder = gate.PacedFeeder()
    playback = gate.PlaybackSim()
    state.stt_name = type(session.stt).__name__
    state.tts_name = type(session.tts).__name__

    async def ack_loop() -> None:
        while True:
            await asyncio.sleep(0.25)
            session.playback.on_playback_ack(playback.played_ms)

    async def drive() -> None:
        try:
            await steps(state, feeder, playback)
            state.script_finished = True
        finally:
            feeder.stop()
            if not session.closed:
                await session.close()

    driver = None
    acker = None
    try:
        async with asyncio.timeout(70):
            async with aclosing(session.run(feeder.stream())) as stream:
                async for event in stream:
                    gate.observe(event, state, playback)
                    if isinstance(event, SessionStarted) and driver is None:
                        acker = asyncio.create_task(ack_loop())
                        driver = asyncio.create_task(drive())
    except TimeoutError:
        state.timeouts.append("wall 70s")
    finally:
        feeder.stop()
        for task in (driver, acker):
            if task is not None and not task.done():
                task.cancel()
        for task in (driver, acker):
            if task is not None:
                await asyncio.gather(task, return_exceptions=True)
        if not session.closed:
            await session.close()
    wav = out_dir / f"{name}.wav"
    gate.write_wav(wav, bytes(state.audio))
    return {
        "name": name,
        "assistant_spoken": state.spoken(),
        "commits": state.commits,
        "interrupts": state.interrupts,
        "metrics": state.metrics,
        "timeouts": state.timeouts,
        "errors": state.errors,
        "wav": str(wav),
        "hint": session.audio_input.language,
        "reply_language": session.reply_language,
        "effective_reply": session.effective_reply_language(),
    }


def _effects() -> tuple[list[str], Any]:
    seen: list[str] = []

    async def note_account(account: str = "") -> str:
        """Record an account number. Call only after the caller finishes."""
        seen.append(account)
        session = current_voice_session()
        speculative = bool(session and session._speculative_attempt and not session._speculative_attempt.settled)
        return f"noted {account}" if not speculative else "held"

    return seen, note_account


async def run_validation(out_dir: Path) -> dict[str, Any]:
    missing = gate.missing_provider_keys()
    if missing:
        raise RuntimeError(f"{FLAG}=1 but missing {', '.join(missing)}")
    texts = {
        "es": "¿Qué tiempo hace hoy?",
        "en": "What time is it?",
        "note": "Please note account one.",
        "note2": "No, account two.",
        "story": "Tell me a long story about a red train.",
        "weather": "What is the weather?",
        "yeah": "Yeah.",
        "num_a": "My number is four one five.",
        "num_b": "Five five five oh one nine nine.",
        "letters": "J. O. N. E. S.",
    }
    clips = await synthesize_clips(texts.values())
    out_dir.mkdir(parents=True, exist_ok=True)
    lang_prompt = "Answer the question in one short spoken sentence."
    language_rows = []
    cases = (
        ("unset_pin_en", None, "es", "en"),
        ("follow_spanish", "follow", "es", "es"),
        ("pin_ca_hint_en", "ca", "en", "ca"),
    )
    for name, reply, clip_key, expect in cases:
        session = _session(name, lang_prompt, [], language="en", reply_language=reply)

        async def steps(state, feeder, playback, pcm=clips[texts[clip_key]]):
            del playback
            await gate._say(feeder, pcm)
            await gate._wait(lambda: bool(state.dones), 25, "reply", state)

        row = await _drive(name, session, steps, out_dir)
        judged = judge_reply_language(row["assistant_spoken"])
        row["judged_language"] = judged
        row["expected_language"] = expect
        row["language_ok"] = judged == expect
        row["analysis"] = "lexical classifier on assistant text events"
        language_rows.append(row)

    resume_rows = []
    for pause in PAUSES:
        seen, tool = _effects()
        name = f"resume_{pause:.1f}".replace(".", "_")
        session = _session(
            name,
            "When they ask you to note an account, call note_account once, after they finish. One short sentence.",
            [tool],
            speculative=True,
            extra={"eager_eot_threshold": "0.3", "eot_threshold": "0.7", "eot_timeout_ms": "2500"},
        )

        async def steps(state, feeder, playback, pause=pause):
            del playback
            await gate._say(feeder, clips[texts["note"]])
            await asyncio.sleep(pause)
            await gate._say(feeder, clips[texts["note2"]])
            await gate._wait(lambda: bool(state.dones), 30, "reply", state)

        row = await _drive(name, session, steps, out_dir)
        started = max((m.get("speculative_generations_started") or 0) for m in row["metrics"]) if row["metrics"] else 0
        discarded = (
            max((m.get("speculative_generations_discarded") or 0) for m in row["metrics"]) if row["metrics"] else 0
        )
        row["tool_effects"] = list(seen)
        row["speculative_started"] = started
        row["speculative_discarded"] = discarded
        row["effects_escaped"] = bool(discarded and seen)
        resume_rows.append(row)

    story = (
        "When asked for a story, tell a long story of at least six sentences about a red train. "
        "If they then ask about the weather, answer only the weather in one sentence. "
        "Never say the word SECRET-CANARY."
    )

    async def story_then(state, feeder, playback, follow: bytes, label: str):
        await gate._say(feeder, clips[texts["story"]])
        await gate._wait(lambda: len(state.audio) > 8000, 25, "story audio", state)
        start = playback.played_ms
        await gate._wait(lambda: playback.played_ms - start >= 500, 15, "playback", state)
        state.marks[label] = time.monotonic()
        before = len(state.dones)
        await gate._say(feeder, follow)
        await gate._wait(lambda: len(state.dones) > before or bool(state.interrupts), 20, "after interrupt", state)
        await asyncio.sleep(0.6)

    recovery = []
    for name, follow, label in (
        ("genuine_weather", clips[texts["weather"]], "weather"),
        ("backchannel_yeah", clips[texts["yeah"]], "yeah"),
    ):
        session = _session(name, story, [], speculative=False)
        row = await _drive(
            name,
            session,
            lambda state, feeder, playback, follow=follow, label=label: story_then(
                state, feeder, playback, follow, label
            ),
            out_dir,
        )
        heard = ""
        if row["interrupts"]:
            heard = row["interrupts"][0].get("heard_text") or ""
        spoken = row["assistant_spoken"]
        row["heard_prefix"] = heard
        row["canary_spoken"] = "SECRET-CANARY" in spoken
        row["heard_is_prefix"] = (not heard) or spoken.startswith(heard) or heard in spoken
        row["weather_addressed"] = (
            "weather" in spoken.casefold() or "lluv" in spoken.casefold() or "sol" in spoken.casefold()
        )
        recovery.append(row)

    number = clips[texts["num_a"]] + silence(1.8) + clips[texts["num_b"]]
    letter_clips = await synthesize_clips(["J.", "O.", "N.", "E.", "S."])
    spelled = b""
    for piece in ("J.", "O.", "N.", "E.", "S."):
        spelled += letter_clips[piece] + silence(0.55)
    clean_number = clips[texts["num_a"]] + silence(0.2) + clips[texts["num_b"]]
    dictation = []
    for name, pcm in (
        ("long_hesitation", number),
        ("spelled_pauses", spelled),
        ("phone_number", phone_pcm(clean_number)),
        ("clean_number", clean_number),
        ("noisy_number", mix_snr(clean_number, 8.0)),
    ):
        session = _session(
            name,
            "Read the whole number or the whole spelled name back once, after they finish. One sentence.",
            [],
        )

        async def steps(state, feeder, playback, pcm=pcm):
            del playback
            await gate._say(feeder, pcm)
            await gate._wait(lambda: bool(state.metrics), 30, "metrics", state)

        row = await _drive(name, session, steps, out_dir)
        text = " ".join(item["text"] for item in row["commits"])
        row["transcript"] = text
        row["digit_recall_oh_zero"] = recall_digits(text) if "spell" not in name else None
        if row["metrics"]:
            row["speech_end_to_transcript_ms"] = row["metrics"][0].get("speech_end_to_transcript_ms")
            row["eou_to_first_audio_ms"] = row["metrics"][0].get("eou_to_first_audio_ms")
        dictation.append(row)

    report = {
        "listening": False,
        "analysis": "assistant text events and Flux committed transcripts; wavs are emitted PCM and were not re-recognized",
        "n_note": "each live cell is n=1 except the three resume pauses, which are one call each",
        "language": language_rows,
        "turn_resumed": {
            "reproduced": any(row["speculative_discarded"] for row in resume_rows),
            "effects_escaped": any(row["effects_escaped"] for row in resume_rows),
            "runs": resume_rows,
        },
        "recovery": recovery,
        "dictation": dictation,
    }
    (out_dir / "index.json").write_text(json.dumps(report, indent=2) + "\n")
    target = ARTIFACT_DIR / out_dir.name
    target.mkdir(parents=True, exist_ok=True)
    for path in out_dir.iterdir():
        if path.suffix in {".json", ".wav"}:
            shutil.copy2(path, target / path.name)
    report["artifact_dir"] = str(target)
    (target / "index.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


async def run_followup(out_dir: Path) -> dict[str, Any]:
    """Two calls the first batch did not finish: weather after a real barge, and a long pause with patience on."""
    clips = await synthesize_clips(
        ["Tell me a long story about a red train.", "What is the weather?", "My number is four one five.", "Five five five oh one nine nine."]
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    story = (
        "When asked for a story, tell a long story of at least four sentences about a red train. "
        "If they then ask about the weather, answer only the weather in one sentence."
    )
    session = _session("weather_after_barge", story, [])

    async def steps(state, feeder, playback):
        await gate._say(feeder, clips["Tell me a long story about a red train."])
        await gate._wait(lambda: len(state.audio) > 8000, 25, "story", state)
        start = playback.played_ms
        await gate._wait(lambda: playback.played_ms - start >= 500, 12, "play", state)
        dones = len(state.dones)
        await gate._say(feeder, clips["What is the weather?"])
        await gate._wait(lambda: any("weather" in c["text"].casefold() for c in state.commits), 15, "weather commit", state)
        await gate._wait(lambda: len(state.dones) > dones, 20, "weather reply", state)

    weather = await _drive("weather_after_barge", session, steps, out_dir)
    number = clips["My number is four one five."] + silence(1.8) + clips["Five five five oh one nine nine."]
    patient = _session(
        "hesitation_patience",
        "Read the whole number back once, after they finish. One sentence.",
        [],
    )
    patient.entity_patience = True

    async def slow(state, feeder, playback):
        del playback
        await gate._say(feeder, number)
        await gate._wait(lambda: bool(state.metrics), 30, "metrics", state)

    held = await _drive("hesitation_patience", patient, slow, out_dir)
    text = " ".join(item["text"] for item in held["commits"])
    report = {
        "listening": False,
        "analysis": "assistant text events and Flux commits; not a second ASR and not listening",
        "weather_spoken": weather["assistant_spoken"],
        "weather_commits": [c["text"] for c in weather["commits"]],
        "weather_heard": [c.get("heard_text") for c in weather["interrupts"]],
        "canary": "SECRET-CANARY" in weather["assistant_spoken"],
        "patience_transcript": text,
        "patience_recall": recall_digits(text),
        "patience_eot_ms": (held["metrics"][0].get("speech_end_to_transcript_ms") if held["metrics"] else None),
        "patience_timeouts": held["timeouts"],
    }
    (out_dir / "followup.json").write_text(json.dumps(report, indent=2) + "\n")
    target = ARTIFACT_DIR / out_dir.name
    target.mkdir(parents=True, exist_ok=True)
    for path in out_dir.iterdir():
        if path.suffix in {".json", ".wav"}:
            shutil.copy2(path, target / path.name)
    report["artifact_dir"] = str(target)
    (target / "followup.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> int:
    if os.environ.get("TIMBAL_VOICE_VALIDATE_263_FOLLOWUP") == "1":
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        dest = gate.RESULTS_DIR.parent / "validate-263" / f"{stamp}-followup"
        report = asyncio.run(run_followup(dest))
        print(json.dumps(report, indent=2))  # noqa: T201
        return 0
    if os.environ.get(FLAG) != "1":
        print(json.dumps({"skipped": f"set {FLAG}=1"}))  # noqa: T201
        return 0
    missing = gate.missing_provider_keys()
    if missing:
        print(json.dumps({"passed": False, "missing_keys": missing}))  # noqa: T201
        return 2
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    dest = gate.RESULTS_DIR.parent / "validate-263" / stamp
    report = asyncio.run(run_validation(dest))
    summary = {
        "artifact_dir": report["artifact_dir"],
        "language": [
            {
                "name": row["name"],
                "judged": row["judged_language"],
                "ok": row["language_ok"],
                "text": row["assistant_spoken"][:180],
            }
            for row in report["language"]
        ],
        "turn_resumed_reproduced": report["turn_resumed"]["reproduced"],
        "effects_escaped": report["turn_resumed"]["effects_escaped"],
        "resume": [
            {
                "name": row["name"],
                "started": row["speculative_started"],
                "discarded": row["speculative_discarded"],
                "tools": row["tool_effects"],
            }
            for row in report["turn_resumed"]["runs"]
        ],
        "recovery": [
            {
                "name": row["name"],
                "heard": (row.get("heard_prefix") or "")[:80],
                "canary": row["canary_spoken"],
                "weather": row["weather_addressed"],
                "interrupts": len(row["interrupts"]),
            }
            for row in report["recovery"]
        ],
        "dictation": [
            {
                "name": row["name"],
                "recall": row.get("digit_recall_oh_zero"),
                "eot_ms": row.get("speech_end_to_transcript_ms"),
                "audio_ms": row.get("eou_to_first_audio_ms"),
                "text": (row.get("transcript") or "")[:120],
            }
            for row in report["dictation"]
        ],
    }
    print(json.dumps(summary, indent=2))  # noqa: T201
    lang_ok = all(row["language_ok"] for row in report["language"])
    escaped = report["turn_resumed"]["effects_escaped"]
    canary = any(row["canary_spoken"] for row in report["recovery"])
    return 0 if lang_ok and not escaped and not canary else 1


if __name__ == "__main__":
    raise SystemExit(main())
