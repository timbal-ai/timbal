"""Listening sounds while the caller talks (``VoiceConfig.backchannel``)."""

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import aclosing
from pathlib import Path

import pytest
from timbal import Agent
from timbal.core.test_model import TestModel
from timbal.server.voice import CLIENT_SETTABLE_VOICE_FIELDS, build_voice_session, event_to_payloads
from timbal.voice import (
    AudioInputConfig,
    AudioOutputConfig,
    BackchannelSpoken,
    TranscriptEvent,
    VoiceConfig,
    VoiceSession,
)
from timbal.voice.backchannel import (
    DEFAULT_BACKCHANNELS,
    MAX_CLIP_SECS,
    Backchanneler,
    prepare_clip,
    scale_pcm,
    trim_silence,
)
from timbal.voice.events import AudioOutput

from .test_mic_activity import silence, talk, tone
from .test_session import DelayedMockSTT
from .test_tts_pace import SR, _PcmTTS, _session

# ---------------------------------------------------------------------------
# When, and which
# ---------------------------------------------------------------------------


def due(chan: Backchanneler, **overrides) -> tuple[str | None, str]:
    """A pause that qualifies in every way, with ``overrides`` on top."""
    values = {
        "now": 100.0,
        "pause": 1,
        "quiet_secs": 0.5,
        "speech_secs": 3.0,
        "utterance_secs": 3.0,
        "reason": "unfinished",
        "choices": ["Ajá.", "Vale."],
    }
    values.update(overrides)
    return chan.decide(**values)


def sure(**kwargs) -> Backchanneler:
    return Backchanneler(rate=1.0, seed=1, **kwargs)


def test_a_pause_that_qualifies_gets_a_sound():
    phrase, why = due(sure())
    assert phrase in ("Ajá.", "Vale.")
    assert why == "unfinished"


@pytest.mark.parametrize(
    "overrides,why",
    [
        ({"quiet_secs": 0.3}, "no_pause"),  # still between two words
        ({"quiet_secs": 1.5}, "no_pause"),  # the moment has passed
        ({"quiet_secs": None}, "no_pause"),  # no mic audio
        ({"reason": None}, "no_reason"),
        ({"speech_secs": 1.0}, "little_said"),
        ({"choices": []}, "no_clip"),
        ({"reason": "long"}, "not_long"),
    ],
)
def test_a_pause_that_does_not(overrides, why):
    assert due(sure(), **overrides) == (None, why)


def test_a_long_explanation_gets_one_without_any_other_sign():
    assert due(sure(), reason="long", utterance_secs=7.0)[1] == "long"
    assert due(sure(long_speech_secs=None), reason="long", utterance_secs=60.0) == (None, "not_long")


def test_a_group_of_a_number_needs_less_speech_than_a_phrase():
    assert due(sure(), reason="entity", speech_secs=0.9)[1] == "entity"
    assert due(sure(), reason="unfinished", speech_secs=0.9) == (None, "little_said")
    assert due(sure(min_speech_secs=0.5), reason="entity", speech_secs=0.6)[1] == "entity"
    assert due(sure(), reason="entity", speech_secs=0.5) == (None, "little_said")


def test_one_pause_gets_one_decision():
    chan = sure()
    assert due(chan, pause=7)[0] is not None
    assert due(chan, pause=7, now=110.0) == (None, "decided")


def test_a_draw_that_says_no_is_not_asked_again_for_the_same_pause():
    chan = Backchanneler(rate=0.5, seed=3, max_per_turn=40)
    answers = [due(chan, pause=n, now=100.0 + 10 * n) for n in range(40)]
    assert 8 <= len([phrase for phrase, _ in answers if phrase]) <= 32
    assert {why for phrase, why in answers if phrase is None} == {"draw"}
    while due(chan, pause=41, now=900.0)[0] is not None:
        chan = Backchanneler(rate=0.5, seed=chan.spoken + 100, max_per_turn=40)
    # Asked again while the same pause lasts: no second draw.
    assert all(due(chan, pause=41, now=901.0 + n) == (None, "decided") for n in range(20))


def test_two_sounds_are_never_closer_than_the_gap():
    chan = sure(min_gap_secs=3.0)
    assert due(chan, pause=1, now=100.0)[0]
    assert due(chan, pause=2, now=102.0) == (None, "too_soon")
    assert due(chan, pause=2, now=103.5)[0]


def test_there_is_a_limit_before_the_agents_next_reply():
    chan = sure(max_per_turn=2)
    assert due(chan, pause=1, now=100.0)[0]
    assert due(chan, pause=2, now=110.0)[0]
    assert due(chan, pause=3, now=120.0) == (None, "turn_limit")
    chan.agent_replied()
    assert due(chan, pause=4, now=130.0)[0]
    assert chan.spoken == 3


def test_the_same_sound_is_not_said_twice_in_a_row():
    chan = sure(max_per_turn=12)
    said = [due(chan, pause=n, now=100.0 + 10 * n)[0] for n in range(12)]
    assert all(a != b for a, b in zip(said, said[1:], strict=False))
    only = sure(max_per_turn=12)
    assert [due(only, pause=n, now=100.0 + 10 * n, choices=["Ajá."])[0] for n in range(3)] == ["Ajá."] * 3


def test_a_rate_of_zero_never_speaks():
    chan = Backchanneler(rate=0.0, seed=1)
    assert all(due(chan, pause=n, now=100.0 + 10 * n)[0] is None for n in range(20))


def test_the_sounds_of_a_language():
    chan = Backchanneler()
    assert chan.phrases("es-ES") == DEFAULT_BACKCHANNELS["es"]
    assert chan.phrases("en") == DEFAULT_BACKCHANNELS["en"]
    assert chan.phrases("de") == ("Mm-hm.",)
    assert chan.phrases(None) == ("Mm-hm.",)
    assert Backchanneler(["Ya."]).phrases("en") == ("Ya.",)
    by_language = Backchanneler({"es": ["Ya."], "*": ["Hm."]})
    assert by_language.phrases("es") == ("Ya.",)
    assert by_language.phrases("fr") == ("Hm.",)
    assert Backchanneler({"es": ["Ya."]}).phrases("en") == DEFAULT_BACKCHANNELS["en"]


@pytest.mark.parametrize("phrases", DEFAULT_BACKCHANNELS.values())
def test_no_default_sound_can_be_taken_for_an_answer(phrases):
    assert not {"sí.", "si.", "yes.", "no."} & {phrase.lower() for phrase in phrases}


# ---------------------------------------------------------------------------
# Clips
# ---------------------------------------------------------------------------


def test_the_silence_around_a_clip_is_trimmed():
    clip = trim_silence(silence(0.3) + tone(0.4) + silence(0.5), SR)
    assert len(clip) / 2 / SR == pytest.approx(0.44, abs=0.02)
    assert trim_silence(silence(1.0), SR) == b""
    assert trim_silence(b"", SR) == b""


def test_a_clip_is_played_quieter():
    loud = tone(0.1, dbfs=-12.0)
    quiet = scale_pcm(loud, 0.5)
    assert len(quiet) == len(loud)
    peak = max(abs(int.from_bytes(loud[i : i + 2], "little", signed=True)) for i in range(0, len(loud), 2))
    quiet_peak = max(abs(int.from_bytes(quiet[i : i + 2], "little", signed=True)) for i in range(0, len(quiet), 2))
    assert quiet_peak == pytest.approx(peak / 2, abs=2)
    assert scale_pcm(loud, 1.0) == loud
    louder = scale_pcm(tone(0.05, dbfs=-6.0), 3.0)  # clipped, not wrapped
    assert max(int.from_bytes(louder[i : i + 2], "little", signed=True) for i in range(0, len(louder), 2)) == 32767


def test_a_take_that_is_not_a_short_sound_is_not_a_clip():
    assert prepare_clip(silence(0.2) + tone(0.4) + silence(0.2), SR) is not None
    assert prepare_clip(tone(MAX_CLIP_SECS + 0.5), SR) is None
    assert prepare_clip(silence(1.0), SR) is None
    assert prepare_clip(b"", SR) is None


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_off_by_default_and_true_is_the_defaults():
    assert VoiceConfig().backchannel is None
    assert VoiceConfig(backchannel=False).backchannel is None
    assert VoiceConfig(backchannel={}).backchannel is None
    block = VoiceConfig(backchannel=True).backchannel
    assert (block.rate, block.pause_secs, block.max_per_turn, block.volume) == (0.7, 0.45, 3, 0.6)
    assert block.enabled
    assert not VoiceConfig(backchannel={"rate": 0}).backchannel.enabled


def test_phrases_are_cleaned_and_keyed_by_language():
    block = VoiceConfig(backchannel={"phrases": {"es-ES": [" Ajá. ", "Ajá.", ""], "*": ["Mm."]}}).backchannel
    assert block.phrases == {"es": ["Ajá."], "*": ["Mm."]}
    assert VoiceConfig(backchannel={"phrases": ["Ya.", "Ya."]}).backchannel.phrases == ["Ya."]
    assert VoiceConfig(backchannel={"phrases": []}).backchannel.phrases is None


@pytest.mark.parametrize(
    "value",
    [
        {"rate": 1.5},
        {"rate": True},
        {"pause_secs": 0.1},
        {"min_gap_secs": 0},
        {"max_per_turn": 0},
        {"volume": 0},
        {"long_speech_secs": 1.0},
        {"phrases": "Ajá."},
        {"phrases": ["x" * 25]},
        {"phrases": [f"s{i}" for i in range(13)]},
        {"phrases": {"": ["Ajá."]}},
        {"unknown": 1},
        "yes",
    ],
)
def test_bad_values_are_rejected(value):
    with pytest.raises(ValueError):
        VoiceConfig(backchannel=value)


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

CLIP_SOUND = silence(0.1) + tone(0.3) + silence(0.2)


def make_session(tmp_path: Path, *, sound: bytes = CLIP_SOUND, backchannel=None, **kwargs) -> VoiceSession:
    settings = {"rate": 1.0, "seed": 1, "volume": 1.0}
    settings.update(backchannel or {})
    session, recorder = _session(
        _PcmTTS(sound), tmp_path, audio_input=AudioInputConfig(language="es"), backchannel=settings, **kwargs
    )
    recorder.close()
    session._recorder = None
    session._start_mic_activity()
    return session


async def clips_ready(session: VoiceSession) -> dict[str, bytes]:
    clips = session._backchannel_clips_for("es")
    await session._backchannel_clip_tasks["es"]
    return clips


def caller_pauses(session: VoiceSession, said: str, *, speech_secs: float = 2.0, quiet_secs: float = 0.5) -> None:
    """The caller talks for ``speech_secs``, the caption reads ``said``, and they go quiet."""
    session._mic_activity.push(talk(speech_secs) + silence(quiet_secs))
    session._latest_partial_text = said
    session._last_partial_at = time.monotonic()


def emitted(session: VoiceSession) -> list:
    events = []
    while not session._event_queue.empty():
        events.append(session._event_queue.get_nowait())
    return events


def sounds(events: list) -> list[tuple[str, str]]:
    return [(event.text, event.reason) for event in events if isinstance(event, BackchannelSpoken)]


async def test_the_clips_are_the_agents_voice_trimmed(tmp_path: Path):
    session = make_session(tmp_path)
    clips = await clips_ready(session)
    assert list(clips) == list(DEFAULT_BACKCHANNELS["es"])
    assert all(len(clip) / 2 / SR == pytest.approx(0.34, abs=0.02) for clip in clips.values())


async def test_a_voice_that_cannot_say_it_briefly_never_says_it(tmp_path: Path):
    session = make_session(tmp_path, sound=tone(2.0))
    assert await clips_ready(session) == {}
    caller_pauses(session, "Mi DNI es el 12 34")
    await session._maybe_backchannel()
    assert sounds(emitted(session)) == []


async def test_a_pause_while_dictating_a_number_gets_a_sound(tmp_path: Path):
    session = make_session(tmp_path)
    clips = await clips_ready(session)
    caller_pauses(session, "Mi DNI es el 12 34")
    await session._maybe_backchannel()
    events = emitted(session)
    (text, reason), *rest = sounds(events)
    assert (reason, rest) == ("entity", [])
    audio = [event.data for event in events if isinstance(event, AudioOutput)]
    assert audio == [clips[text]]
    assert session.output_audio == b"" and session.transcript == []


async def test_it_is_not_the_assistant_speaking(tmp_path: Path):
    session = make_session(tmp_path)
    await clips_ready(session)
    caller_pauses(session, "Mi DNI es el 12 34")
    await session._maybe_backchannel()
    assert session.playback.is_playing
    assert not session._assistant_audio_playing
    assert not session._assistant_active
    assert not session._turn_state().assistant_active
    # The caller goes on over it: nothing to interrupt.
    await session.interrupt()
    assert session.playback.is_playing


async def test_a_reply_after_it_is_the_assistant_again(tmp_path: Path):
    session = make_session(tmp_path)
    await clips_ready(session)
    caller_pauses(session, "Mi DNI es el 12 34")
    await session._maybe_backchannel()
    await session._emit_pcm(tone(0.2), ["hola", 0], turnless=True)
    assert session._assistant_audio_playing


@pytest.mark.parametrize(
    "said,speech_secs,reason",
    [
        ("Mi teléfono es el seis cero cero", 2.0, "entity"),
        ("Pues mira, te llamo porque", 2.0, "unfinished"),
        ("Quería cambiar la cuenta y", 2.0, "unfinished"),
        ("Pues resulta que ayer me llegó una factura que no era mía, la miré", 7.0, "long"),
        ("Pues resulta que ayer me llegó una factura, la miré y no sé...", 7.0, "long"),
    ],
)
async def test_what_says_the_caller_is_not_done(tmp_path: Path, said, speech_secs, reason):
    session = make_session(tmp_path)
    await clips_ready(session)
    caller_pauses(session, said, speech_secs=speech_secs)
    await session._maybe_backchannel()
    assert [why for _, why in sounds(emitted(session))] == [reason]


@pytest.mark.parametrize(
    "said,speech_secs",
    [
        ("Quería cambiar la cuenta del banco.", 2.0),  # a finished phrase, not long
        ("¿Y entonces cuánto tengo que pagar este mes de la luz, me lo puedes decir?", 7.0),  # a question
        ("Pues resulta que ayer me llegó una factura que no era mía.", 7.0),  # long, and finished
        ("Pues resulta que ayer me llegó una factura que no era mía!", 7.0),
        ("Pues mira, te llamo porque", 1.0),  # hardly started
        ("", 3.0),  # no caption
    ],
)
async def test_what_does_not(tmp_path: Path, said, speech_secs):
    session = make_session(tmp_path)
    await clips_ready(session)
    caller_pauses(session, said, speech_secs=speech_secs)
    await session._maybe_backchannel()
    assert sounds(emitted(session)) == []


async def test_a_commit_held_for_the_rest_of_the_phrase_gets_one(tmp_path: Path):
    session = make_session(tmp_path)
    await clips_ready(session)
    session._mic_activity.push(talk(2.0) + silence(0.5))
    session._held_user_text = "Quería saber si"
    await session._maybe_backchannel()
    assert [why for _, why in sounds(emitted(session))] == ["held"]


async def test_a_caption_older_than_the_last_commit_is_not_read(tmp_path: Path):
    session = make_session(tmp_path)
    await clips_ready(session)
    caller_pauses(session, "Mi DNI es el 12 34")
    session._last_commit_at = time.monotonic() + 1
    await session._maybe_backchannel()
    assert sounds(emitted(session)) == []


@pytest.mark.parametrize(
    "busy",
    ["_is_speaking", "_greeting_speaking", "_idle_prompt_speaking", "_hangup_open", "_closed"],
)
async def test_never_while_the_agent_has_the_line(tmp_path: Path, busy):
    session = make_session(tmp_path)
    await clips_ready(session)
    caller_pauses(session, "Mi DNI es el 12 34")
    setattr(session, busy, True)
    await session._maybe_backchannel()
    assert sounds(emitted(session)) == []


async def test_never_over_audio_still_playing_or_a_reply_being_prepared(tmp_path: Path):
    session = make_session(tmp_path)
    await clips_ready(session)
    caller_pauses(session, "Mi DNI es el 12 34")
    session.playback.on_audio_emitted(SR)  # half a second of a reply still to play
    await session._maybe_backchannel()
    assert sounds(emitted(session)) == []
    session.playback.on_interrupted()
    session._commits_in_flight = 1
    await session._maybe_backchannel()
    session._commits_in_flight = 0
    session._speculative_attempt = object()
    await session._maybe_backchannel()
    assert sounds(emitted(session)) == []
    session._speculative_attempt = None
    await session._maybe_backchannel()
    assert len(sounds(emitted(session))) == 1


async def test_one_sound_per_pause_and_the_next_needs_more_speech_and_time(tmp_path: Path):
    session = make_session(tmp_path, backchannel={"min_gap_secs": 0.5})
    await clips_ready(session)
    caller_pauses(session, "Mi DNI es el 12 34")
    await session._maybe_backchannel()
    await session._maybe_backchannel()
    assert len(sounds(emitted(session))) == 1
    session.playback.on_interrupted()
    # They say one more digit and pause again: too little, too soon.
    caller_pauses(session, "Mi DNI es el 12 34 5", speech_secs=0.3)
    await session._maybe_backchannel()
    assert sounds(emitted(session)) == []
    await asyncio.sleep(0.55)
    caller_pauses(session, "Mi DNI es el 12 34 56 78", speech_secs=1.2)
    await session._maybe_backchannel()
    assert len(sounds(emitted(session))) == 1


async def test_the_turn_that_answers_counts_them_and_the_count_starts_again(tmp_path: Path):
    session = make_session(tmp_path, backchannel={"min_gap_secs": 0.5})
    await clips_ready(session)
    caller_pauses(session, "Mi DNI es el 12 34")
    await session._maybe_backchannel()
    session.playback.on_interrupted()
    session._note_caller_utterance("Mi DNI es el 12 34 56 78 Z")
    await session._run_turn("Mi DNI es el 12 34 56 78 Z")
    assert session.metrics[-1].backchannels == 1
    assert session._backchanneler.spoken_this_turn == 0
    session._note_caller_utterance("Gracias")
    await session._run_turn("Gracias")
    assert session.metrics[-1].backchannels == 0


async def test_a_turn_that_starts_under_it_walks_past_its_audio(tmp_path: Path):
    session = make_session(tmp_path)
    await clips_ready(session)
    caller_pauses(session, "Mi DNI es el 12 34")
    await session._maybe_backchannel()
    pending = session._backchannel_pending_bytes()
    assert 0 < pending <= int(0.34 * SR) * 2
    await session._run_turn("Mi DNI es el 12 34 56 78 Z")
    first = session._turn_tts_segment_records[0]
    assert first[0] == "" and 0 < first[1] <= pending


async def test_the_sound_is_recorded_as_agent_audio(tmp_path: Path):
    session = make_session(tmp_path, record_audio=True)
    clips = await clips_ready(session)
    caller_pauses(session, "Mi DNI es el 12 34")
    await session._maybe_backchannel()
    ((text, _),) = sounds(emitted(session))
    assert session.output_audio == clips[text]


async def test_without_the_option_nothing_watches(tmp_path: Path):
    session, recorder = _session(_PcmTTS(CLIP_SOUND), tmp_path)
    recorder.close()
    session._start_mic_activity()
    assert session._mic_activity is None
    assert session._start_backchannel() is None
    await session._maybe_backchannel()
    assert emitted(session) == []


async def test_audio_that_is_not_pcm_turns_it_off(tmp_path: Path):
    session, recorder = _session(
        _PcmTTS(CLIP_SOUND),
        tmp_path,
        backchannel=True,
        audio_output=AudioOutputConfig(encoding="mulaw", sample_rate=SR),
    )
    recorder.close()
    session._start_mic_activity()
    assert session._start_backchannel() is None
    session, recorder = _session(
        _PcmTTS(CLIP_SOUND), tmp_path, backchannel=True, audio_input=AudioInputConfig(encoding="mulaw")
    )
    recorder.close()
    session._start_mic_activity()
    assert session._start_backchannel() is None


async def test_through_a_call_the_caller_hears_it_and_the_turn_goes_on():
    """Through ``run()``: the watcher hears the pause, plays the sound, and the turn still happens."""
    stt = DelayedMockSTT()
    session = VoiceSession(
        agent=Agent(name="t", model=TestModel(responses=["Perfecto, ya lo tengo."]), tools=[]),
        stt=stt,
        tts=_PcmTTS(CLIP_SOUND),
        turn_detector="heuristic",
        backchannel={"rate": 1.0, "seed": 1},
        audio_input=AudioInputConfig(language="es", sample_rate=SR),
        audio_output=AudioOutputConfig(sample_rate=SR),
    )
    events: list = []
    spoke = asyncio.Event()
    paused = asyncio.Event()

    async def mic() -> AsyncIterator[bytes]:
        await spoke.wait()
        audio = talk(2.0) + silence(0.5)
        for start in range(0, len(audio), 640):
            yield audio[start : start + 640]
        paused.set()
        await asyncio.Event().wait()

    async def drive() -> None:
        while "es" not in session._backchannel_clips or not session._backchannel_clips["es"]:
            await asyncio.sleep(0.01)
        await stt.inject(TranscriptEvent(type="partial", text="Mi DNI es el 12 34"))
        await asyncio.sleep(0.05)
        spoke.set()
        await paused.wait()
        while not any(isinstance(event, BackchannelSpoken) for event in events):
            await asyncio.sleep(0.01)
        await stt.inject(TranscriptEvent(type="committed", text="Mi DNI es el 12 34 56 78 Z."))
        while not session.metrics:
            await asyncio.sleep(0.01)
        await stt.finish()

    driver = asyncio.create_task(drive())
    async with aclosing(session.run(mic())) as stream:
        async for event in stream:
            events.append(event)
    await asyncio.wait_for(driver, 5)
    assert [why for _, why in sounds(events)] == ["entity"]
    assert session.metrics[-1].backchannels == 1
    assert [entry.role for entry in session.transcript] == ["user", "assistant"]


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


def test_the_agent_sets_it_and_a_caller_hello_cannot():
    assert "backchannel" not in CLIENT_SETTABLE_VOICE_FIELDS
    agent = Agent(name="desk", model=TestModel(responses=["hi"]), tools=[])
    session, _ = build_voice_session(agent, VoiceConfig(stt_provider="elevenlabs", backchannel={"rate": 0.4}), {})
    assert session.backchannel.rate == 0.4
    assert session._backchanneler.rate == 0.4
    session, _ = build_voice_session(agent, VoiceConfig(stt_provider="elevenlabs"), {"backchannel": True})
    assert session.backchannel is None


def test_the_client_is_told():
    assert event_to_payloads(BackchannelSpoken(text="Ajá.", reason="entity"), None, {}) == [
        {"type": "backchannel", "text": "Ajá.", "reason": "entity"}
    ]
