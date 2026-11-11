"""A hesitation sound at the opening of some replies (``VoiceConfig.hesitation``)."""

import asyncio

import pytest
from structlog.testing import capture_logs
from timbal import Agent
from timbal.core.test_model import TestModel
from timbal.types.content import TextContent, ToolUseContent
from timbal.types.message import Message
from timbal.voice import (
    AgentTextDelta,
    AgentTextDone,
    AudioInputConfig,
    HesitationConfig,
    SpeculativeGeneration,
    TranscriptEvent,
    VoiceConfig,
    VoiceSession,
)
from timbal.voice.config import DEFAULT_HESITATION_RATE, HESITATION_OPENER_MAX_CHARS, HESITATION_OPENERS_MAX
from timbal.voice.hesitation import DEFAULT_OPENERS, SOUND_AUDIO_WEIGHT, Hesitator
from timbal.voice.playback import map_played_bytes_to_text

from .test_session import MockSTT, MockTTS, StreamingMockTTS
from .test_speculative import ScriptedSTT, ScriptedTTS, _Drive, _until

ALWAYS = {"rate": 1.0, "openers": ["eh..."]}
PLAIN = "Son diez con sesenta."


def consider(text: str, *, language: str | None = "es", openers=("eh...",)):
    return Hesitator(1.0, list(openers)).consider(text, language)


# ---------------------------------------------------------------------------
# Which openings get a sound
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        PLAIN,
        "Una Brasa Clásica, ¿sola o en menú?",
        "Te paso con un compañero de operaciones.",
        "That comes to ten sixty.",
        # The comma inside a figure does not close the first clause.
        "Son 10,60 euros en total.",
    ],
)
def test_an_opening_of_three_words_or_more_gets_the_sound(text):
    made, reason = consider(text)
    assert reason == "added"
    assert (made.offset, made.opener) == (0, "Eh... ")
    assert made.spoken == "Eh... " + text[0].lower() + text[1:]


@pytest.mark.parametrize(
    "text",
    [
        "Vale, te paso con un compañero.",
        "De acuerdo. Te lo miro ahora mismo.",
        "Hasta luego, que vaya bien.",
        "Venga, adiós, que tengas buen día.",
        "Pedro, ya te he encontrado.",
        "Dime, ¿de qué se trata?",
        "Of course, let me check that for you.",
        "Pedido confirmado.",
        "Claro.",
    ],
)
def test_a_one_or_two_word_opening_is_said_clean(text):
    assert consider(text) == (None, "short_opening")


@pytest.mark.parametrize(
    "text",
    [
        "¿Me das tu documento, por favor?",
        "¡Qué bien que hayas llamado hoy!",
        "10 euros con sesenta en total.",
        "[laughs] That is a good one.",
        '"Brasa" es el nombre de la casa.',
    ],
)
def test_an_opening_that_does_not_start_with_a_letter_is_said_clean(text):
    assert consider(text) == (None, "not_a_word")


@pytest.mark.parametrize(
    "text",
    [
        "Eh... son diez con sesenta.",
        "eh, son diez con sesenta",
        "Mm, a ver, déjame ver.",
        "Ehhh... son diez con sesenta.",
        "Hmm. Let me look at that order.",
        "Um, I can check that for you.",
        "Uh, I can check that for you.",
        "Er... I can check that for you.",
        "Ah, ya veo lo que ha pasado.",
        "Ehm, déjame que lo mire.",
        "Mmm — déjame que lo mire.",
    ],
)
def test_a_sound_the_model_wrote_is_not_doubled(text):
    assert consider(text) == (None, "already")


def test_a_bare_sound_needs_a_comma_a_full_stop_an_ellipsis_or_a_dash():
    # "Um?" asks the caller to repeat. It is a one-word opening, said clean, and not a hesitation.
    assert consider("Um? I did not catch that.") == (None, "short_opening")
    # A configured opener is matched by its letters, whatever follows them.
    assert consider("Um? I did not catch that.", openers=("um,",)) == (None, "already")


def test_an_opening_that_starts_with_a_configured_opener_is_not_doubled():
    openers = ("a ver,", "pues...")
    assert consider("A ver, una Brasa XL en menú.", openers=openers) == (None, "already")
    assert consider("Pues son diez con sesenta.", openers=openers) == (None, "already")
    # The same letters at the start of a longer word are not the opener.
    assert consider("A Veracruz se llega en tres horas.", openers=openers)[1] == "added"
    assert consider("Puesto que lo pides, te lo cambio.", openers=openers)[1] == "added"


@pytest.mark.parametrize(
    "text,language",
    [("Um café com leite custa dois euros.", "pt"), ("Er kommt morgen um drei Uhr.", "de")],
)
def test_a_word_that_spells_like_a_sound_is_a_word(text, language):
    made, reason = consider(text, language=language)
    assert reason == "added"
    assert made.spoken == "Eh... " + text[0].lower() + text[1:]


@pytest.mark.parametrize(
    "text", ["DNI y nombre del titular, por favor.", "A las nueve tienes mesa.", "I can check that for you."]
)
def test_an_acronym_and_a_one_letter_word_keep_their_capital(text):
    made, _ = consider(text)
    assert made.spoken == "Eh... " + text


def test_the_sound_goes_after_leading_whitespace():
    made, _ = consider(" Tienes mesa a las nueve.")
    assert made.offset == 1
    assert made.spoken == " Eh... tienes mesa a las nueve."


@pytest.mark.parametrize(
    "text", [PLAIN, " Tienes mesa a las nueve.", "DNI y nombre del titular, por favor.", "Él ya lo sabe todo."]
)
def test_spoken_is_the_original_plus_the_sound(text):
    made, _ = consider(text)
    without = made.spoken[: made.offset] + made.spoken[made.offset + len(made.opener) :]
    assert len(without) == len(text)
    assert without.lower() == text.lower()
    assert sum(a != b for a, b in zip(without, text, strict=True)) <= 1


def test_blank_text_is_left_alone():
    hesitator = Hesitator(1.0, ["eh..."])
    before = hesitator.state()
    assert hesitator.consider("   ", "es") == (None, "not_a_word")
    assert hesitator.state() == before


# ---------------------------------------------------------------------------
# How often
# ---------------------------------------------------------------------------


def test_the_opening_after_a_hesitated_one_is_clean():
    hesitator = Hesitator(1.0, ["eh...", "mm,"], seed=1)
    reasons = [hesitator.consider(PLAIN, "es")[1] for _ in range(6)]
    assert reasons == ["added", "after_hesitation"] * 3


def test_a_sound_the_model_wrote_counts_as_a_hesitation():
    hesitator = Hesitator(1.0, ["eh..."])
    reasons = [hesitator.consider(text, "es")[1] for text in ("Mm, déjame ver un momento.", PLAIN, PLAIN)]
    assert reasons == ["already", "after_hesitation", "added"]


def test_only_the_opening_right_after_is_held_back():
    hesitator = Hesitator(1.0, ["eh..."])
    reasons = [hesitator.consider(text, "es")[1] for text in (PLAIN, "Vale, ahora lo miro.", PLAIN)]
    assert reasons == ["added", "short_opening", "added"]


@pytest.mark.parametrize("rate", [0.15, 0.5, 1.0])
def test_share_is_rate_over_one_plus_rate_when_every_opening_can_take_one(rate):
    hesitator = Hesitator(rate, seed=7)
    total = 40_000
    added = sum(hesitator.consider(PLAIN, "es")[0] is not None for _ in range(total))
    assert added / total == pytest.approx(rate / (1 + rate), abs=0.01)


def test_rate_zero_never_adds():
    hesitator = Hesitator(0.0, seed=7)
    assert {hesitator.consider(PLAIN, "es")[1] for _ in range(500)} == {"draw"}


def test_the_same_sound_is_not_used_twice_running():
    hesitator = Hesitator(1.0, ["eh...", "mm,", "pues..."], seed=3)
    sounds = [made.opener for _ in range(200) if (made := hesitator.consider(PLAIN, "es")[0])]
    assert len(sounds) == 100
    assert all(a != b for a, b in zip(sounds, sounds[1:], strict=False))
    assert set(sounds) == {"Eh... ", "Mm, ", "Pues... "}


def test_a_single_sound_repeats():
    hesitator = Hesitator(1.0, ["eh..."])
    sounds = [made.opener for _ in range(6) if (made := hesitator.consider(PLAIN, "es")[0])]
    assert sounds == ["Eh... "] * 3


def test_seed_fixes_the_draws():
    def run(seed):
        hesitator = Hesitator(0.5, seed=seed)
        return [made.opener if (made := hesitator.consider(PLAIN, "es")[0]) else None for _ in range(40)]

    assert run(11) == run(11)
    assert run(11) != run(12)


def test_restore_forgets_the_openings_considered_since():
    hesitator = Hesitator(1.0, ["eh..."])
    before = hesitator.state()
    assert hesitator.consider(PLAIN, "es")[1] == "added"
    hesitator.restore(before)
    assert hesitator.consider(PLAIN, "es")[1] == "added"


# ---------------------------------------------------------------------------
# Which sounds
# ---------------------------------------------------------------------------


def test_built_in_sounds_follow_the_reply_language():
    hesitator = Hesitator(1.0)
    assert hesitator.openers("es-ES") == DEFAULT_OPENERS["es"]
    assert hesitator.openers("EN_us") == DEFAULT_OPENERS["en"]
    assert hesitator.openers("ca") == DEFAULT_OPENERS["ca"]
    assert hesitator.openers("fr") == ()
    assert hesitator.openers(None) == ()


def test_a_language_with_no_sounds_gets_none_and_keeps_the_state():
    hesitator = Hesitator(1.0)
    assert hesitator.consider(PLAIN, "es")[1] == "added"
    before = hesitator.state()
    assert hesitator.consider("Cela fait dix euros soixante.", "fr") == (None, "no_openers")
    assert hesitator.state() == before


def test_a_list_serves_every_language():
    hesitator = Hesitator(1.0, ["hm,"])
    assert hesitator.openers("fr") == hesitator.openers("es") == hesitator.openers(None) == ("hm,",)


def test_a_dict_is_keyed_by_language_with_a_star_for_the_rest():
    keyed = Hesitator(1.0, {"fr": ["euh..."], "*": ["hm,"]})
    assert keyed.openers("fr-CA") == ("euh...",)
    assert keyed.openers("es") == keyed.openers(None) == ("hm,",)
    # Without "*", a language the dict leaves out keeps the built-in list.
    partial = Hesitator(1.0, {"fr": ["euh..."]})
    assert partial.openers("es") == DEFAULT_OPENERS["es"]
    assert partial.openers("de") == ()


def test_built_in_sounds_pass_the_config_rules():
    openers = {language: list(sounds) for language, sounds in DEFAULT_OPENERS.items()}
    assert HesitationConfig(openers=openers).openers == openers


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_off_by_default():
    assert VoiceConfig().hesitation is None


@pytest.mark.parametrize("value", [None, False, {}, HesitationConfig()])
def test_an_empty_or_false_block_is_off(value):
    assert VoiceConfig(hesitation=value).hesitation is None


def test_true_is_the_default_rate():
    config = VoiceConfig(hesitation=True).hesitation
    assert config.rate == DEFAULT_HESITATION_RATE == 0.15
    assert config.openers is None
    assert config.enabled


def test_a_block_without_a_rate_uses_the_default_rate():
    assert VoiceConfig(hesitation={"openers": ["hm,"]}).hesitation.rate == DEFAULT_HESITATION_RATE


def test_rate_zero_is_kept_and_is_off():
    config = VoiceConfig(hesitation={"rate": 0}).hesitation
    assert config.rate == 0
    assert not config.enabled


def test_openers_are_trimmed_merged_by_language_and_deduplicated():
    config = HesitationConfig(
        openers={"es-ES": [" eh... ", "", "mm,"], "ES": ["mm,", "pues…"], "*": ["hm,"], "fr": [" "]}
    )
    assert config.openers == {"es": ["eh...", "mm,", "pues…"], "*": ["hm,"]}
    assert HesitationConfig(openers=[" eh... ", "eh...", ""]).openers == ["eh..."]
    assert HesitationConfig(openers=["", " "]).openers is None


@pytest.mark.parametrize(
    "block",
    [
        {"rate": -0.1},
        {"rate": 1.5},
        {"rate": True},
        {"rate": float("nan")},
        # Read as the first word of the reply.
        {"openers": ["eh"]},
        {"openers": ["eh."]},
        {"openers": ["...eh,"]},
        {"openers": ["m" * HESITATION_OPENER_MAX_CHARS + ","]},
        {"openers": [f"{'m' * (n + 2)}," for n in range(HESITATION_OPENERS_MAX + 1)]},
        {"openers": {"es": ["eh"]}},
        {"sounds": ["eh..."]},
    ],
)
def test_bad_values_are_rejected(block):
    with pytest.raises(ValueError):
        HesitationConfig(**block)


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


def make_session(responses, *, tts=None, tools=None, language="es", **kwargs) -> VoiceSession:
    return VoiceSession(
        agent=Agent(name="desk", model=TestModel(responses=responses), tools=tools or []),
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


def fed(tts: StreamingMockTTS) -> str:
    return "".join(text for stream in tts.streams for text in stream.fed)


def test_session_is_off_by_default_and_takes_a_block_a_model_or_a_boolean():
    def hesitation(value):
        return make_session(["hi"], hesitation=value).hesitation

    assert make_session(["hi"]).hesitation is None
    assert make_session(["hi"])._hesitator is None
    for off in (None, False, {}, HesitationConfig(), {"rate": 0}):
        assert hesitation(off) is None
    assert hesitation(True).rate == DEFAULT_HESITATION_RATE
    assert hesitation({"rate": 0.4}).rate == 0.4
    assert hesitation(HesitationConfig(rate=0.3)).rate == 0.3
    for bad in ("natural", 1, {"rate": 2}):
        with pytest.raises(ValueError):
            hesitation(bad)


async def test_off_leaves_the_reply_as_the_model_wrote_it():
    reply = "Son diez con sesenta. Pasa a la ventanilla de pago."
    tts = StreamingMockTTS()
    session = make_session([reply], tts=tts)
    await session._run_turn("¿cuánto es?")
    assert fed(tts) == reply
    assert session.metrics[-1].hesitations == 0


async def test_the_sound_is_spoken_and_is_in_no_text():
    reply = "Son diez con sesenta. Pasa a la ventanilla de pago."
    tts = StreamingMockTTS()
    session = make_session([reply], tts=tts, hesitation=ALWAYS)
    await session._run_turn("¿cuánto es?")
    assert fed(tts) == "Eh... son diez con sesenta. Pasa a la ventanilla de pago."
    events = drain_events(session)
    assert "".join(e.text for e in events if isinstance(e, AgentTextDelta)) == reply
    assert [e.text for e in events if isinstance(e, AgentTextDone)] == [reply]
    assert session.transcript[-1].text == reply
    assert session._last_run_context.root_span().memory[-1].collect_text() == reply
    assert "".join(str(record[0]) for record in session._turn_tts_segment_records) == reply
    assert session.metrics[-1].hesitations == 1


async def test_a_voice_without_a_streaming_context_is_given_the_sound_too():
    reply = "Son diez con sesenta. Pasa a la ventanilla de pago."
    tts = MockTTS()
    session = make_session([reply], tts=tts, hesitation=ALWAYS)
    await session._run_turn("¿cuánto es?")
    assert "".join(tts.synthesized_texts) == "Eh... son diez con sesenta. Pasa a la ventanilla de pago."
    first = session._turn_tts_segment_records[0]
    assert "".join(str(record[0]) for record in session._turn_tts_segment_records) == reply
    assert session._turn_tts_added == {id(first): (0, len("Eh... "))}
    assert session.transcript[-1].text == reply


async def test_built_in_sounds_are_used_for_the_call_language():
    tts = StreamingMockTTS()
    session = make_session([PLAIN], tts=tts, language="es-ES", hesitation={"rate": 1.0, "seed": 4})
    await session._run_turn("¿cuánto es?")
    spoken = fed(tts)
    sound = next(s for s in DEFAULT_OPENERS["es"] if spoken.lower().startswith(s))
    assert spoken == sound[0].upper() + sound[1:] + " son diez con sesenta."


async def test_reply_language_picks_the_sounds_over_the_stt_language():
    tts = StreamingMockTTS()
    session = make_session(
        ["That comes to ten sixty."],
        tts=tts,
        language="es",
        reply_language="en",
        hesitation={"rate": 1.0, "openers": {"en": ["um,"], "es": ["eh..."]}},
    )
    await session._run_turn("how much is it")
    assert fed(tts) == "Um, that comes to ten sixty."


@pytest.mark.parametrize("language,reply_language", [("fr", None), (None, None), ("es", "follow")])
async def test_a_reply_language_with_no_sounds_is_spoken_clean(language, reply_language):
    tts = StreamingMockTTS()
    session = make_session([PLAIN], tts=tts, language=language, reply_language=reply_language, hesitation=True)
    session._hesitator.rate = 1.0
    await session._run_turn("¿cuánto es?")
    assert fed(tts) == PLAIN
    assert session.metrics[-1].hesitations == 0


async def test_each_sound_is_logged_and_a_skipped_opening_says_why():
    session = make_session([PLAIN, "Vale, pasa a la ventanilla."], hesitation=ALWAYS)
    with capture_logs() as logs:
        await session._run_turn("¿cuánto es?")
        await session._run_turn("vale")
    (added,) = [entry for entry in logs if entry["event"] == "tts_hesitation"]
    assert (added["log_level"], added["opener"], added["text_preview"]) == ("info", "Eh...", PLAIN)
    (skipped,) = [entry for entry in logs if entry["event"] == "tts_hesitation_skipped"]
    assert (skipped["log_level"], skipped["reason"]) == ("debug", "short_opening")


async def test_a_language_with_no_sounds_is_reported_once():
    session = make_session([PLAIN, PLAIN, PLAIN], language="fr", hesitation=True)
    with capture_logs() as logs:
        await session._run_turn("combien")
        await session._run_turn("combien")
        await session.set_language("de")
        await session._run_turn("wie viel")
    reported = [entry for entry in logs if entry["event"] == "tts_hesitation_unavailable"]
    assert [(entry["log_level"], entry["language"]) for entry in reported] == [("info", "fr"), ("info", "de")]


def tool_step(text: str) -> Message:
    return Message(
        role="assistant",
        content=[TextContent(text=text), ToolUseContent(id="c1", name="lookup", input={})],
        stop_reason="tool_use",
    )


async def lookup() -> str:
    return "ok"


async def test_the_first_speech_after_a_tool_call_can_get_one():
    tts = StreamingMockTTS()
    session = make_session(
        [tool_step("Vale, lo miro."), "Tienes mesa a las nueve. ¿Te la reservo ahora mismo?"],
        tts=tts,
        tools=[lookup],
        hesitation=ALWAYS,
    )
    await session._run_turn("¿hay mesa?")
    assert fed(tts) == "Vale, lo miro. Eh... tienes mesa a las nueve. ¿Te la reservo ahora mismo?"
    assert session.transcript[-1].text == "Vale, lo miro. Tienes mesa a las nueve. ¿Te la reservo ahora mismo?"
    assert session.metrics[-1].hesitations == 1


async def test_a_turn_does_not_hesitate_before_and_after_the_same_tool_call():
    tts = StreamingMockTTS()
    session = make_session(
        [tool_step("Ahora mismo lo miro."), "Tienes mesa a las nueve."],
        tts=tts,
        tools=[lookup],
        hesitation=ALWAYS,
    )
    await session._run_turn("¿hay mesa?")
    assert fed(tts) == "Eh... ahora mismo lo miro. Tienes mesa a las nueve."
    assert session.metrics[-1].hesitations == 1


async def test_the_next_turn_opens_clean_after_a_hesitated_one():
    tts = StreamingMockTTS()
    session = make_session([PLAIN, "Pasa a la ventanilla de pago.", PLAIN], tts=tts, hesitation=ALWAYS)
    for _ in range(3):
        await session._run_turn("¿cuánto es?")
    assert [stream.fed for stream in tts.streams] == [
        ["Eh... son diez con sesenta."],
        ["Pasa a la ventanilla de pago."],
        ["Eh... son diez con sesenta."],
    ]
    assert [m.hesitations for m in session.metrics] == [1, 0, 1]


async def test_only_the_first_segment_of_a_step_is_considered():
    segments = ["Son diez con sesenta.", " Pasa a la ventanilla de pago.", " Luego te doy el tique allí."]
    tts = StreamingMockTTS()
    session = make_session(["hi"], tts=tts, hesitation=ALWAYS)
    session._turn_hesitation_due = True
    for segment in segments:
        session._schedule_tts(segment)
    await session._tts_tail
    try:
        assert tts.streams[0].fed == ["Eh... son diez con sesenta.", *segments[1:]]
        assert session._turn_hesitations == 1
    finally:
        await session._abort_tts_stream()


@pytest.mark.parametrize("flags", [{"greeting": True}, {"filler": True}, {"idle": True}, {}])
async def test_a_line_spoken_as_written_is_never_hesitated(flags):
    # {} is how the hangup farewell, the timeout fallback and a resumed tail are spoken.
    tts = MockTTS()
    session = make_session(["hi"], tts=tts, hesitation=ALWAYS)
    session._turn_hesitation_due = True
    await session._speak("Bienvenido a Brasa, dime qué te pongo.", **flags)
    assert tts.synthesized_texts == ["Bienvenido a Brasa, dime qué te pongo."]
    assert session._turn_tts_added == {}


# ---------------------------------------------------------------------------
# What the caller heard when a hesitated reply is cut short
# ---------------------------------------------------------------------------


async def test_a_cut_inside_the_sound_means_no_word_was_heard():
    text = "Son diez con sesenta en total."
    tts = StreamingMockTTS(bytes_per_char=10)
    session = make_session(["hi"], tts=tts, hesitation=ALWAYS)
    session._turn_hesitation_due = True
    session._schedule_tts(text)
    await session._tts_tail
    spoken_bytes = len("Eh... " + text) * 10
    while session._turn_audio_bytes < spoken_bytes:
        await asyncio.sleep(0)
    try:
        assert session._turn_tts_segment_records == [[text, spoken_bytes]]
        sound_chars = len("Eh... ") * SOUND_AUDIO_WEIGHT
        sound_bytes = spoken_bytes * sound_chars // (len(text) + sound_chars)
        words_bytes = spoken_bytes - sound_bytes
        assert session._heard_segments() == [("", sound_bytes), (text, words_bytes)]

        def heard(played: int) -> str:
            return map_played_bytes_to_text(session._heard_segments(), played)

        assert heard(sound_bytes // 2) == ""
        assert heard(sound_bytes) == ""
        # The record as it stands would credit the caller with a word.
        assert map_played_bytes_to_text([(text, spoken_bytes)], sound_bytes) == "Son"
        assert heard(sound_bytes + words_bytes // 2) == "Son diez con"
        assert heard(spoken_bytes) == text

        session._turn_assistant_text = text
        session._turn_heard_bytes = sound_bytes
        session._apply_interruption_truncation(None)
        assert session._last_interruption_heard_text == ""
        assert session._unheard_tail == text
        assert session.transcript == []
    finally:
        await session._abort_tts_stream()


def test_a_sound_after_a_tool_call_keeps_the_space_between_the_two_steps():
    session = make_session(["hi"], hesitation=ALWAYS)
    before, after = ["Vale, lo miro.", 140], [" Tienes mesa a las nueve.", 300]
    session._turn_tts_segment_records = [before, after]
    session._turn_tts_added = {id(after): (1, len("Eh... "))}
    sound_chars = len("Eh... ") * SOUND_AUDIO_WEIGHT
    sound_bytes = 300 * sound_chars // (len(after[0]) + sound_chars)
    assert session._heard_segments() == [
        ("Vale, lo miro.", 140),
        ("", sound_bytes),
        (" Tienes mesa a las nueve.", 300 - sound_bytes),
    ]
    assert map_played_bytes_to_text(session._heard_segments(), 140 + sound_bytes) == "Vale, lo miro."
    assert map_played_bytes_to_text(session._heard_segments(), 440) == "Vale, lo miro. Tienes mesa a las nueve."


def test_records_without_a_sound_map_as_before():
    session = make_session(["hi"], hesitation=ALWAYS)
    records = [["Checking…", 10], ["", 40], [" Order has shipped tomorrow.", 30]]
    session._turn_tts_segment_records = records
    assert session._heard_segments() == [(text, num_bytes) for text, num_bytes in records]
    # A sound whose audio never arrived takes no share.
    session._turn_tts_added = {id(records[1]): (0, 6)}
    records[1][1] = 0
    assert session._heard_segments() == [("Checking…", 10), ("", 0), (" Order has shipped tomorrow.", 30)]


# ---------------------------------------------------------------------------
# Speculative generation
# ---------------------------------------------------------------------------


def speculative_session() -> tuple[VoiceSession, ScriptedSTT, ScriptedTTS]:
    def reply(messages):
        return "The red one is in stock." if "red" in str(messages[-1]) else "The blue one is in stock."

    stt, tts = ScriptedSTT(), ScriptedTTS()
    session = VoiceSession(
        agent=Agent(name="speculative_voice", model=TestModel(handler=reply)),
        stt=stt,
        tts=tts,
        turn_detector="heuristic",
        speculative_generation=SpeculativeGeneration(enabled=True, preemptive_tts=True),
        hesitation={"rate": 1.0, "openers": ["um,"]},
    )
    return session, stt, tts


async def test_a_dropped_speculative_reply_does_not_use_up_the_hesitation():
    session, stt, tts = speculative_session()
    async with _Drive(session, stt) as drive:
        await stt.inject(TranscriptEvent(type="eager", text="which one"))
        await _until(lambda: tts.synthesized_texts)
        assert tts.synthesized_texts == ["Um, the blue one is in stock."]
        await stt.inject(TranscriptEvent(type="resumed", text=""))
        await _until(lambda: session.speculative_generations_discarded == 1)
        await stt.inject(TranscriptEvent(type="committed", text="which one do you have in red"))
        await _until(lambda: len(tts.synthesized_texts) == 2)
        await _until(lambda: drive.audio())
    # Nobody heard the first one, so the reply that is heard still hesitates.
    assert tts.synthesized_texts[1] == "Um, the red one is in stock."


async def test_a_confirmed_speculative_reply_counts():
    session, stt, tts = speculative_session()
    async with _Drive(session, stt) as drive:
        await stt.inject(TranscriptEvent(type="eager", text="which one"))
        await _until(lambda: tts.synthesized_texts)
        await stt.inject(TranscriptEvent(type="committed", text="which one"))
        await _until(lambda: drive.audio())
        await _until(lambda: len(session.metrics) == 1)
        await stt.inject(TranscriptEvent(type="committed", text="and in red"))
        await _until(lambda: len(tts.synthesized_texts) == 2)
        await _until(lambda: len(session.metrics) == 2)
    assert tts.synthesized_texts == ["Um, the blue one is in stock.", "The red one is in stock."]
    assert [m.hesitations for m in session.metrics] == [1, 0]
