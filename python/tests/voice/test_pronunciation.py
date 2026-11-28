"""A spoken form for terms the voice reads badly (``VoiceConfig.pronunciations``)."""

import pytest
from timbal import Agent
from timbal.core.test_model import TestModel
from timbal.server.voice import CLIENT_SETTABLE_VOICE_FIELDS, build_voice_session
from timbal.voice import AgentTextDelta, AgentTextDone, AudioInputConfig, VoiceConfig, VoiceSession
from timbal.voice.config import PRONUNCIATION_TERM_MAX_CHARS, PRONUNCIATIONS_MAX
from timbal.voice.pronunciation import Pronouncer

from .test_session import MockSTT, MockTTS, StreamingMockTTS

SPANISH = {"DNI": "deneí", "NIE": "níe"}


# ---------------------------------------------------------------------------
# Which text is replaced
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,spoken",
    [
        ("¿Me dices tu DNI, por favor?", "¿Me dices tu deneí, por favor?"),
        ("Necesito el DNI o el NIE.", "Necesito el deneí o el níe."),
        ("DNI", "deneí"),
        ("(DNI)", "(deneí)"),
        ("Tu DNI: 12345678Z", "Tu deneí: 12345678Z"),
    ],
)
def test_a_whole_term_is_replaced(text, spoken):
    assert Pronouncer(SPANISH).apply(text, "es") == spoken


@pytest.mark.parametrize(
    "text", ["Los DNIs de los dos.", "El DNIE electrónico.", "dni", "Dni", "ADNI", "DNI_2", "DNI2"]
)
def test_a_term_inside_a_longer_word_or_in_another_case_is_left_alone(text):
    assert Pronouncer(SPANISH).apply(text, "es") == text


def test_the_longest_term_wins_where_two_overlap():
    pronouncer = Pronouncer({"DNI": "deneí", "DNI electrónico": "deneí electrónico, el del chip"})
    assert pronouncer.apply("Trae el DNI electrónico.", "es") == "Trae el deneí electrónico, el del chip."
    assert pronouncer.apply("Trae el DNI.", "es") == "Trae el deneí."


def test_a_term_with_punctuation_in_it_is_matched_as_written():
    pronouncer = Pronouncer({"C++": "ce más más", "24/7": "veinticuatro horas", "a.m.": "de la mañana"})
    assert pronouncer.apply("Soporte de C++ 24/7, desde las 9 a.m.", "es") == (
        "Soporte de ce más más veinticuatro horas, desde las 9 de la mañana"
    )


def test_a_replacement_is_not_replaced_again():
    assert Pronouncer({"A": "B", "B": "C"}).apply("A y B", "es") == "B y C"


def test_empty_text_and_no_terms_pass_through():
    assert Pronouncer(SPANISH).apply("", "es") == ""
    assert Pronouncer(None).apply("El DNI.", "es") == "El DNI."
    assert not Pronouncer(None)
    assert not Pronouncer({})
    assert Pronouncer(SPANISH)


# ---------------------------------------------------------------------------
# Which language
# ---------------------------------------------------------------------------


def test_a_flat_mapping_serves_every_language():
    pronouncer = Pronouncer(SPANISH)
    for language in ("es", "es-ES", "ca", "en", None, "follow"):
        assert pronouncer.apply("El DNI.", language) == "El deneí."


def test_a_mapping_by_language_follows_the_reply_language():
    pronouncer = Pronouncer({"es": {"DNI": "deneí"}, "ca": {"DNI": "de-ena-i"}, "*": {"SMS": "ese eme ese"}})
    assert pronouncer.apply("El DNI i un SMS.", "ca-ES") == "El de-ena-i i un ese eme ese."
    assert pronouncer.apply("El DNI y un SMS.", "es_ES") == "El deneí y un ese eme ese."
    assert pronouncer.apply("The DNI and an SMS.", "en") == "The DNI and an ese eme ese."
    assert pronouncer.apply("El DNI y un SMS.", None) == "El DNI y un ese eme ese."


def test_a_language_entry_wins_over_the_one_for_every_language():
    pronouncer = Pronouncer({"*": {"IBAN": "iban"}, "en": {"IBAN": "eye-ban"}})
    assert pronouncer.terms("en") == {"IBAN": "eye-ban"}
    assert pronouncer.terms("es") == {"IBAN": "iban"}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_off_by_default_and_an_empty_mapping_is_unset():
    assert VoiceConfig().pronunciations is None
    assert VoiceConfig(pronunciations={}).pronunciations is None
    assert VoiceConfig(pronunciations={"es": {}}).pronunciations is None


def test_terms_are_trimmed_and_languages_merged():
    assert VoiceConfig(pronunciations={" DNI ": "  de  ne í "}).pronunciations == {"DNI": "de ne í"}
    by_language = VoiceConfig(pronunciations={"es-ES": {"DNI": "deneí"}, "ES": {"NIE": "níe"}, " * ": {"SMS": "s"}})
    assert by_language.pronunciations == {"es": {"DNI": "deneí", "NIE": "níe"}, "*": {"SMS": "s"}}


@pytest.mark.parametrize(
    "value",
    [
        "DNI",
        ["DNI"],
        {"DNI": 3},
        {"DNI": ""},
        {" ": "deneí"},
        {"DNI": "deneí", "es": {"NIE": "níe"}},
        {"": {"DNI": "deneí"}},
        {"x" * (PRONUNCIATION_TERM_MAX_CHARS + 1): "x"},
        {f"T{i}": "t" for i in range(PRONUNCIATIONS_MAX + 1)},
    ],
)
def test_bad_values_are_rejected(value):
    with pytest.raises(ValueError):
        VoiceConfig(pronunciations=value)


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


def make_session(responses, *, tts=None, language="es", **kwargs) -> VoiceSession:
    return VoiceSession(
        agent=Agent(name="desk", model=TestModel(responses=responses), tools=[]),
        stt=MockSTT(),
        tts=tts or StreamingMockTTS(),
        audio_input=AudioInputConfig(language=language),
        turn_detector="heuristic",
        **kwargs,
    )


def drain_events(session: VoiceSession) -> list:
    events = []
    while not session._event_queue.empty():
        events.append(session._event_queue.get_nowait())
    return events


async def test_the_voice_gets_the_spoken_form_and_every_text_keeps_the_term():
    reply = "Perfecto. ¿Me dices tu DNI o tu NIE?"
    tts = StreamingMockTTS()
    session = make_session([reply], tts=tts, pronunciations=SPANISH)
    await session._run_turn("quiero ver mi factura")
    assert "".join(text for stream in tts.streams for text in stream.fed) == "Perfecto. ¿Me dices tu deneí o tu níe?"
    events = drain_events(session)
    assert "".join(e.text for e in events if isinstance(e, AgentTextDelta)) == reply
    assert [e.text for e in events if isinstance(e, AgentTextDone)] == [reply]
    assert session.transcript[-1].text == reply
    assert session._last_run_context.root_span().memory[-1].collect_text() == reply
    assert "".join(str(record[0]) for record in session._turn_tts_segment_records) == reply


async def test_a_voice_without_a_streaming_context_and_lines_outside_a_turn_get_it_too():
    tts = MockTTS()
    session = make_session(["Tu DNI acaba en zeta."], tts=tts, pronunciations=SPANISH)
    await session._run_turn("¿cuál es?")
    await session._speak("Luzia Energía, buenas. Ten a mano el DNI.", greeting=True)
    await session._speak("Un momento, busco tu DNI.", filler=True)
    assert tts.synthesized_texts == [
        "Tu deneí acaba en zeta.",
        "Luzia Energía, buenas. Ten a mano el deneí.",
        "Un momento, busco tu deneí.",
    ]


async def test_the_reply_language_picks_the_terms():
    tts = StreamingMockTTS()
    session = make_session(
        ["The DNI is on file."],
        tts=tts,
        language="es",
        reply_language="en",
        pronunciations={"es": {"DNI": "deneí"}, "en": {"DNI": "D N I"}},
    )
    await session._run_turn("is my id on file")
    assert tts.streams[0].fed == ["The D N I is on file."]


BY_LANGUAGE = {"es": {"DNI": "deneí"}, "ca": {"DNI": "de-ena-i"}, "*": {"SMS": "esemese"}}


async def test_a_reply_that_follows_the_caller_gets_the_terms_of_the_language_they_speak():
    tts = StreamingMockTTS()
    session = make_session(
        ["Tu DNI y un SMS.", "El teu DNI i un SMS."],
        tts=tts,
        language="ca",
        reply_language="follow",
        pronunciations=BY_LANGUAGE,
    )
    session.set_spoken_language("es")
    await session._run_turn("hola, quiero pedir cita para el coche")
    session.set_spoken_language("ca-ES")
    await session._run_turn("ara t'ho dic en català, si us plau")
    assert ["".join(stream.fed) for stream in tts.streams] == [
        "Tu deneí y un esemese.",
        "El teu de-ena-i i un esemese.",
    ]


async def test_the_language_the_transcriber_reported_picks_the_terms_too():
    tts = StreamingMockTTS()
    session = make_session(["Tu DNI."], tts=tts, language=None, reply_language="follow", pronunciations=BY_LANGUAGE)
    session._note_heard_language("es", "hola, quiero pedir cita para el coche")
    await session._run_turn("hola, quiero pedir cita para el coche")
    assert tts.streams[0].fed == ["Tu deneí."]


async def test_until_the_caller_has_spoken_it_is_the_transcribers_language_and_with_none_only_the_shared_terms():
    tts = MockTTS()
    hinted = make_session(["hi"], tts=tts, language="ca", reply_language="follow", pronunciations=BY_LANGUAGE)
    await hinted._speak("Bona tarda. Tingues a mà el DNI i el SMS.", greeting=True)
    unknown = make_session(["hi"], tts=tts, language=None, reply_language="follow", pronunciations=BY_LANGUAGE)
    await unknown._speak("Bona tarda. Tingues a mà el DNI i el SMS.", greeting=True)
    assert tts.synthesized_texts == [
        "Bona tarda. Tingues a mà el de-ena-i i el esemese.",
        "Bona tarda. Tingues a mà el DNI i el esemese.",
    ]


async def test_a_pinned_reply_language_is_not_moved_by_what_the_caller_speaks():
    tts = StreamingMockTTS()
    session = make_session(["Tu DNI."], tts=tts, language="es", reply_language="es", pronunciations=BY_LANGUAGE)
    session.set_spoken_language("ca")
    await session._run_turn("bon dia, vull demanar hora")
    assert tts.streams[0].fed == ["Tu deneí."]


async def test_a_hesitation_sound_and_a_spoken_form_go_together():
    tts = StreamingMockTTS()
    session = make_session(
        ["DNI y nombre del titular, por favor."],
        tts=tts,
        pronunciations=SPANISH,
        hesitation={"rate": 1.0, "openers": ["eh..."]},
    )
    await session._run_turn("quiero darme de alta")
    assert tts.streams[0].fed == ["Eh... deneí y nombre del titular, por favor."]
    assert session.transcript[-1].text == "DNI y nombre del titular, por favor."


def test_session_is_off_by_default():
    assert make_session(["hi"])._pronouncer is None
    assert make_session(["hi"], pronunciations={})._pronouncer is None
    with pytest.raises(ValueError):
        make_session(["hi"], pronunciations={"DNI": ""})


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


def test_the_agent_sets_it_and_a_caller_hello_cannot():
    assert "pronunciations" not in CLIENT_SETTABLE_VOICE_FIELDS
    agent = Agent(name="desk", model=TestModel(responses=["hi"]), tools=[])
    defaults = VoiceConfig(stt_provider="elevenlabs", pronunciations=SPANISH)
    session, _ = build_voice_session(agent, defaults, {"pronunciations": {"DNI": "hacked"}})
    assert session.pronunciations == SPANISH
    session, _ = build_voice_session(agent, VoiceConfig(stt_provider="elevenlabs"), {"pronunciations": SPANISH})
    assert session.pronunciations is None
