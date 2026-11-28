"""Laughs and sighs written as ``[laughs]`` (``VoiceConfig.audio_tags``)."""

import pytest
from timbal import Agent
from timbal.core.test_model import TestModel
from timbal.server.voice import CLIENT_SETTABLE_VOICE_FIELDS, build_voice_session
from timbal.voice import AgentTextDelta, AudioInputConfig, AudioOutputConfig, VoiceConfig, VoiceSession
from timbal.voice.audio_tags import DEFAULT_AUDIO_TAGS, AudioTagPolicy, audio_tags_note, strip_audio_tags
from timbal.voice.elevenlabs import ElevenLabsStreamTTS

from .test_session import MockSTT, MockTTS, StreamingMockTTS

# ---------------------------------------------------------------------------
# The policy
# ---------------------------------------------------------------------------


def policy(**kwargs) -> AudioTagPolicy:
    made = AudioTagPolicy(DEFAULT_AUDIO_TAGS, **kwargs)
    made.start_reply()
    return made


@pytest.mark.parametrize(
    "text",
    [
        "[laughs] Claro que sí, ahora mismo te lo miro.",
        "Uy, [laughs] perdona, me he liado yo sola.",
        "[clears throat] Perdona. Te decía que el cargo es del día veinte.",
        "[ Clears  Throat ] Perdona.",
        "[SIGHS] Vaya.",
    ],
)
def test_a_voice_that_performs_a_listed_mark_gets_it(text):
    assert policy().voiced(text, performed=True) == (text, 1)


@pytest.mark.parametrize(
    "text,spoken",
    [
        ("[laughs] Claro que sí.", "Claro que sí."),
        ("Uy, [laughs] perdona, me he liado.", "Uy, perdona, me he liado."),
        (" [sighs] Vaya, lo siento.", " Vaya, lo siento."),
        ("Claro[laughs] que sí.", "Claro que sí."),
        ("Vale. [chuckles]", "Vale. "),
        ("[laughs]", ""),
    ],
)
def test_a_voice_that_would_read_it_aloud_gets_the_text_without_it(text, spoken):
    assert policy().voiced(text, performed=False) == (spoken, 0)


@pytest.mark.parametrize("text", ["[whispers] Es un secreto.", "[risas] Claro.", "[excited] ¡Genial!"])
def test_a_mark_that_is_not_listed_is_never_spoken(text):
    assert "[" not in policy().voiced(text, performed=True)[0]
    assert "[" not in policy().voiced(text, performed=False)[0]


@pytest.mark.parametrize(
    "text", ["La opción [1] es la buena.", "Lista: [a, b].", "Sin marcas.", "[", "[]", "Código [A1]."]
)
def test_brackets_that_are_not_a_word_are_left_alone(text):
    assert policy().voiced(text, performed=True) == (text, 0)
    assert policy().voiced(text, performed=False) == (text, 0)


def test_one_mark_a_reply():
    limiter = policy()
    assert limiter.voiced("[laughs] Claro. [laughs] Ahora te lo miro.", performed=True) == (
        "[laughs] Claro. Ahora te lo miro.",
        1,
    )
    # The next piece of the same reply.
    assert limiter.voiced(" [sighs] Un momento.", performed=True) == (" Un momento.", 0)


def test_more_marks_a_reply_when_asked():
    limiter = policy(max_per_reply=2)
    text = "[laughs] Claro. [sighs] Ya. [laughs] Vale."
    assert limiter.voiced(text, performed=True) == ("[laughs] Claro. [sighs] Ya. Vale.", 2)


def test_the_reply_after_one_with_a_mark_goes_without():
    limiter = policy()
    heard = []
    for _ in range(5):
        heard.append(limiter.voiced("[laughs] Claro.", performed=True)[1])
        limiter.start_reply()
    assert heard == [1, 0, 1, 0, 1]


def test_a_reply_without_a_mark_does_not_start_a_rest():
    limiter = policy(rest_replies=2)
    replies = ["Claro.", "[laughs] Claro.", "[laughs] Sí.", "[laughs] Sí.", "[laughs] Sí.", "Vale.", "[sighs] Ya."]
    heard = []
    for reply in replies:
        heard.append(limiter.voiced(reply, performed=True)[1])
        limiter.start_reply()
    assert heard == [0, 1, 0, 0, 1, 0, 0]


def test_no_rest_when_set_to_zero():
    limiter = policy(rest_replies=0)
    heard = []
    for _ in range(3):
        heard.append(limiter.voiced("[laughs] Claro.", performed=True)[1])
        limiter.start_reply()
    assert heard == [1, 1, 1]


def test_a_line_the_application_wrote_is_not_limited_and_does_not_count():
    limiter = policy()
    greeting = "[laughs] Hola. [sighs] Qué día."
    assert limiter.voiced(greeting, performed=True, limited=False) == (greeting, 2)
    assert limiter.voiced("[laughs] Claro.", performed=True) == ("[laughs] Claro.", 1)
    # Still without the marks nobody listed, and without any for a voice that reads them.
    assert limiter.voiced("[whispers] Hola.", performed=True, limited=False) == ("Hola.", 0)
    assert limiter.voiced(greeting, performed=False, limited=False) == ("Hola. Qué día.", 0)


def test_a_reply_nobody_heard_is_rewound():
    limiter = policy()
    before = limiter.state()
    limiter.start_reply()
    assert limiter.voiced("[laughs] Claro.", performed=True)[1] == 1
    limiter.restore(before)
    limiter.start_reply()
    assert limiter.voiced("[laughs] Claro.", performed=True)[1] == 1


def test_strip_audio_tags_is_the_text_in_words():
    assert strip_audio_tags("[laughs] Claro, [sighs] ya lo miro. [whatever]") == "Claro, ya lo miro. "
    assert strip_audio_tags("La opción [1].") == "La opción [1]."


def test_the_note_lists_the_marks():
    note = audio_tags_note(["laughs", "clears throat"])
    assert "[laughs], [clears throat]" in note
    assert "\n" not in note


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_off_by_default_true_is_the_default_marks_and_a_list_names_them():
    assert VoiceConfig().audio_tags is None
    assert VoiceConfig(audio_tags=False).audio_tags is None
    assert VoiceConfig(audio_tags={}).audio_tags is None
    assert VoiceConfig(audio_tags=True).audio_tags.tags == list(DEFAULT_AUDIO_TAGS)
    assert VoiceConfig(audio_tags=["[Laughs]", " sighs ", "laughs"]).audio_tags.tags == ["laughs", "sighs"]
    assert VoiceConfig(audio_tags={"max_per_reply": 2}).audio_tags.tags == list(DEFAULT_AUDIO_TAGS)
    assert not VoiceConfig(audio_tags={"tags": []}).audio_tags.enabled


@pytest.mark.parametrize(
    "value",
    [
        ["laughs1"],
        ["a" * 31],
        ["x"] * 0 + [f"tag {chr(97 + i)}" for i in range(13)],
        {"tags": ["laughs"], "max_per_reply": 0},
        {"tags": ["laughs"], "rest_replies": -1},
        {"tags": ["laughs"], "unknown": 1},
        "laughs",
    ],
)
def test_bad_values_are_rejected(value):
    with pytest.raises(ValueError):
        VoiceConfig(audio_tags=value)


# ---------------------------------------------------------------------------
# Which voices perform them
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model,performs",
    [
        ("eleven_v3", True),
        ("eleven_v3_conversational", True),
        ("eleven_v4_turbo", True),
        ("eleven_flash_v2_5", False),
        ("eleven_multilingual_v2", False),
        ("eleven_turbo_v2_5", False),
    ],
)
def test_the_elevenlabs_dialogue_models_perform_them(model, performs):
    tts = ElevenLabsStreamTTS(api_key="test")
    assert tts.performs_audio_tags(AudioOutputConfig(model=model)) is performs


def test_any_other_voice_does_not_by_default():
    assert MockTTS().performs_audio_tags(AudioOutputConfig()) is False


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


class PerformingTTS(StreamingMockTTS):
    def performs_audio_tags(self, config: AudioOutputConfig) -> bool:  # noqa: ARG002
        return True


class PerformingPlainTTS(MockTTS):
    def performs_audio_tags(self, config: AudioOutputConfig) -> bool:  # noqa: ARG002
        return True


def make_session(responses, *, tts, **kwargs) -> VoiceSession:
    return VoiceSession(
        agent=Agent(name="desk", model=TestModel(responses=responses), tools=[], system_prompt="You are Laura."),
        stt=MockSTT(),
        tts=tts,
        audio_input=AudioInputConfig(language="es"),
        turn_detector="heuristic",
        **kwargs,
    )


def fed(tts) -> str:
    return "".join(text for stream in tts.streams for text in stream.fed)


def drain_events(session: VoiceSession) -> list:
    events = []
    while not session._event_queue.empty():
        events.append(session._event_queue.get_nowait())
    return events


REPLY = "[laughs] Claro que sí, ahora mismo te lo miro."


async def test_a_performing_voice_is_fed_the_mark_and_the_turn_counts_it():
    tts = PerformingTTS()
    session = make_session([REPLY], tts=tts, audio_tags=True)
    await session._run_turn("¿me lo miras?")
    assert fed(tts) == REPLY
    assert session.metrics[-1].audio_tags == 1
    assert session.transcript[-1].text == REPLY


async def test_a_voice_that_reads_marks_aloud_is_fed_the_words_only():
    tts = StreamingMockTTS()
    session = make_session([REPLY], tts=tts, audio_tags=True)
    await session._run_turn("¿me lo miras?")
    assert fed(tts) == "Claro que sí, ahora mismo te lo miro."
    assert session.metrics[-1].audio_tags == 0
    events = drain_events(session)
    assert "".join(e.text for e in events if isinstance(e, AgentTextDelta)) == REPLY
    assert session.transcript[-1].text == REPLY
    assert session._last_run_context.root_span().memory[-1].collect_text() == REPLY


async def test_without_the_option_the_text_reaches_the_voice_as_written():
    tts = StreamingMockTTS()
    session = make_session(["[whispers] Claro. [laughs] Ya."], tts=tts)
    await session._run_turn("hola")
    assert fed(tts) == "[whispers] Claro. [laughs] Ya."


async def test_a_plain_voice_gets_the_same_treatment():
    performing, reading = PerformingPlainTTS(), MockTTS()
    await make_session([REPLY], tts=performing, audio_tags=True)._run_turn("hola")
    await make_session([REPLY], tts=reading, audio_tags=True)._run_turn("hola")
    assert performing.synthesized_texts == [REPLY]
    assert reading.synthesized_texts == ["Claro que sí, ahora mismo te lo miro."]


async def test_a_reply_that_is_only_a_mark_says_nothing_on_a_voice_that_reads_it():
    tts = MockTTS()
    session = make_session(["[laughs]"], tts=tts, audio_tags=True)
    await session._run_turn("hola")
    assert tts.synthesized_texts == []
    assert session.transcript[-1].text == "[laughs]"


async def test_the_limit_runs_across_the_turns_of_a_call():
    tts = PerformingTTS()
    session = make_session(
        [REPLY, "[sighs] Vaya, lo siento mucho.", "[laughs] Pues ya está hecho."], tts=tts, audio_tags=True
    )
    for prompt in ("uno", "dos", "tres"):
        await session._run_turn(prompt)
    assert [stream.fed for stream in tts.streams] == [
        [REPLY],
        ["Vaya, lo siento mucho."],
        ["[laughs] Pues ya está hecho."],
    ]
    assert [m.audio_tags for m in session.metrics] == [1, 0, 1]


async def test_a_greeting_keeps_its_marks_on_a_performing_voice_and_loses_them_on_another():
    performing, reading = PerformingPlainTTS(), MockTTS()
    await make_session(["hi"], tts=performing, audio_tags=True)._speak("[chuckles] Hola, soy Laura.", greeting=True)
    await make_session(["hi"], tts=reading, audio_tags=True)._speak("[chuckles] Hola, soy Laura.", greeting=True)
    assert performing.synthesized_texts == ["[chuckles] Hola, soy Laura."]
    assert reading.synthesized_texts == ["Hola, soy Laura."]


async def test_the_model_is_told_about_the_marks_only_when_the_voice_performs_them():
    for tts, told in ((PerformingTTS(), True), (StreamingMockTTS(), False)):
        session = make_session(["Vale."], tts=tts, audio_tags=["laughs", "sighs"])
        prompt = session.prompt_for_turn("You are Laura.")
        assert prompt.static == "You are Laura."
        assert ("[laughs], [sighs]" in prompt.dynamic) is told
    session = make_session(["Vale."], tts=PerformingTTS())
    assert "[laughs]" not in session.prompt_for_turn("You are Laura.").dynamic


async def test_the_note_comes_before_the_lines_that_change():
    session = make_session(["Vale."], tts=PerformingTTS(), audio_tags=True, call_context={"customer": "Marta"})
    session.prompt_call_context_keys = ("customer",)
    dynamic = session.prompt_for_turn("You are Laura.").dynamic
    assert dynamic.index("[laughs]") < dynamic.index("customer: Marta") < dynamic.index("Current date and time")


async def test_a_mark_and_a_hesitation_sound_do_not_stack():
    tts = PerformingTTS()
    session = make_session([REPLY], tts=tts, audio_tags=True, hesitation={"rate": 1.0, "openers": ["eh..."]})
    await session._run_turn("hola")
    assert fed(tts) == REPLY
    assert session.metrics[-1].hesitations == 0


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


def test_the_agent_sets_it_and_a_caller_hello_cannot():
    assert "audio_tags" not in CLIENT_SETTABLE_VOICE_FIELDS
    agent = Agent(name="desk", model=TestModel(responses=["hi"]), tools=[])
    session, _ = build_voice_session(agent, VoiceConfig(stt_provider="elevenlabs", audio_tags=["laughs"]), {})
    assert session.audio_tags.tags == ["laughs"]
    session, _ = build_voice_session(agent, VoiceConfig(stt_provider="elevenlabs"), {"audio_tags": True})
    assert session.audio_tags is None
