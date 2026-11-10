"""Per-agent transcription vocabulary and language hints. No live audio."""

from __future__ import annotations

import json
from urllib.parse import parse_qs, urlparse

import pytest
from timbal.voice import AudioInputConfig, VoiceConfig
from timbal.voice.config import validate_keyterms
from timbal.voice.deepgram import DeepgramFluxSTT, DeepgramNovaSTT
from timbal.voice.elevenlabs import ElevenLabsRealtimeSTT
from timbal.voice.munsit import MunsitStreamSTT
from timbal.voice.session import VoiceSession

from .test_session import MockSTT, MockTTS


def _flux_query(config: AudioInputConfig) -> dict[str, list[str]]:
    return parse_qs(urlparse(DeepgramFluxSTT()._build_uri(config)).query)


def test_keyterms_reject_weights_blanks_and_the_101st_phrase() -> None:
    assert validate_keyterms(["Norte Market", "NM1001"]) == ["Norte Market", "NM1001"]
    with pytest.raises(ValueError, match="weighted"):
        VoiceConfig(keyterms=["sku:0.15"])
    with pytest.raises(ValueError, match="blank"):
        VoiceConfig(keyterms=["  "])
    with pytest.raises(TypeError, match="list"):
        VoiceConfig(keyterms="Norte Market")
    with pytest.raises(ValueError, match="100"):
        VoiceConfig(keyterms=[f"t{i}" for i in range(101)])


def test_scribe_keyterm_limits_fail_an_elevenlabs_agent_at_config_time() -> None:
    """Realtime Scribe takes 50 terms of 20 characters. Past that it closes the socket."""
    fifty = [f"t{i}" for i in range(50)]
    assert VoiceConfig(stt_provider="elevenlabs", keyterms=fifty).keyterms == fifty
    with pytest.raises(ValueError, match="at most 50 keyterms"):
        VoiceConfig(stt_provider="elevenlabs", keyterms=[*fifty, "one more"])
    with pytest.raises(ValueError, match="at most 20 characters"):
        VoiceConfig(stt_provider="elevenlabs", keyterms=["a phrase of twenty-five ch"])
    # Flux has neither limit. If it falls back to Scribe, the adapter trims.
    hundred = [f"a longer phrase number {i}" for i in range(100)]
    assert VoiceConfig(keyterms=hundred).keyterms == hundred


def test_scribe_turn_commit_threshold_is_checked_at_config_time() -> None:
    ok = VoiceConfig(stt_provider="elevenlabs", stt_extra={"turn_commit_threshold": 0.75})
    assert ok.stt_extra["turn_commit_threshold"] == 0.75
    with pytest.raises(ValueError, match="not 'turn_prediction'"):
        VoiceConfig(
            stt_provider="elevenlabs",
            stt_extra={"turn_commit_threshold": 0.75, "commit_strategy": "turn_prediction"},
        )
    for bad in (0, 1.5, "0.75", True):
        with pytest.raises(ValueError, match="between 0 and 1"):
            VoiceConfig(stt_provider="elevenlabs", stt_extra={"turn_commit_threshold": bad})


async def test_session_ignores_a_diarized_transcript() -> None:
    """Scribe v3 Live repeats each commit split by speaker. It is not a second turn."""
    from timbal.voice.providers import SpeakerSegment, TranscriptEvent

    from .test_session import _collect_events, _make_session

    session, _, _ = _make_session(
        stt_script=[
            TranscriptEvent(type="committed", text="Hi there"),
            TranscriptEvent(
                type="diarized",
                text="Hi there",
                speakers=[SpeakerSegment(speaker_id="speaker_0", text="Hi there", start=0.1, end=0.6)],
            ),
        ],
        responses=["Hello!", "A second reply would be a bug."],
    )
    await _collect_events(session)
    assert [(entry.role, entry.text) for entry in session.transcript if entry.role == "user"] == [("user", "Hi there")]


def test_reply_language_stays_independent_of_stt_hints() -> None:
    cfg = VoiceConfig(language="es", reply_language="follow", stt_languages=["es", "en"], keyterms=["Ana Ruiz"])
    assert cfg.reply_language == "follow"
    assert cfg.language == "es"
    assert cfg.stt_languages == ["es", "en"]
    assert cfg.keyterms == ["Ana Ruiz"]


def test_catalan_and_english_flux_and_other_providers_fail_clearly() -> None:
    with pytest.raises(ValueError, match="Catalan|not a Deepgram Flux"):
        VoiceConfig(language="ca")
    with pytest.raises(ValueError, match="not a Deepgram Flux"):
        VoiceConfig(stt_languages=["ca"])
    with pytest.raises(ValueError, match="flux-general-multi"):
        VoiceConfig(stt_model="flux-general-en", stt_languages=["en"])
    with pytest.raises(ValueError, match="does not accept keyterms"):
        VoiceConfig(stt_provider="openai", keyterms=["Acme"])
    with pytest.raises(ValueError, match="does not accept keyterms"):
        VoiceConfig(stt_provider="munsit", keyterms=["Acme"])
    with pytest.raises(ValueError, match="Nova-2"):
        VoiceConfig(stt_provider="deepgram-nova", stt_model="nova-2", keyterms=["Acme"])
    with pytest.raises(ValueError, match="stt_extra"):
        VoiceConfig(keyterms=["Acme"], stt_extra={"keyterm": "Acme"})


def test_defaults_still_omit_vocabulary() -> None:
    cfg = VoiceConfig()
    assert cfg.keyterms == []
    assert cfg.stt_languages is None
    q = _flux_query(AudioInputConfig())
    assert "keyterm" not in q
    assert "language_hint" not in q


def test_flux_connect_payload_uses_repeated_hints_and_terms() -> None:
    q = _flux_query(AudioInputConfig(language="es", languages=["es", "en"], keyterms=["Norte Market", "NM1001"]))
    assert q["language_hint"] == ["es", "en"]
    assert q["keyterm"] == ["Norte Market", "NM1001"]
    with pytest.raises(ValueError, match="flux-general-en"):
        _flux_query(AudioInputConfig(model="flux-general-en", languages=["en"]))


def test_nova3_connect_repeats_terms_and_nova2_fails() -> None:
    nova = parse_qs(
        urlparse(DeepgramNovaSTT()._build_uri(AudioInputConfig(model="nova-3", keyterms=["Ana Ruiz", "NM1001"]))).query
    )
    assert nova["keyterm"] == ["Ana Ruiz", "NM1001"]
    with pytest.raises(ValueError, match="Nova-2"):
        DeepgramNovaSTT()._build_uri(AudioInputConfig(model="nova-2", keyterms=["Ana Ruiz"]))
    with pytest.raises(ValueError, match="does not accept"):
        MunsitStreamSTT()._build_uri(AudioInputConfig(keyterms=["Acme"]))


async def test_scribe_connect_repeats_keyterms(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, str] = {}

    async def _connect(uri: str, **_kwargs: object) -> object:
        captured["uri"] = uri

        class _Socket:
            def __aiter__(self) -> object:
                return self

            async def __anext__(self) -> str:
                raise StopAsyncIteration

        return _Socket()

    monkeypatch.setattr("timbal.voice.elevenlabs.ws_connect", _connect)
    monkeypatch.setattr("timbal.voice.elevenlabs._resolve_api_key", lambda *_args: "test-key")
    await ElevenLabsRealtimeSTT().connect(AudioInputConfig(language="es", keyterms=["Norte Market", "NM1001"]))
    query = parse_qs(urlparse(captured["uri"]).query)
    assert query["keyterms"] == ["Norte Market", "NM1001"]
    assert query["language_code"] == ["es"]


def test_hello_cannot_set_vocabulary() -> None:
    from timbal.server.voice import merge_client_voice_overrides

    merged = merge_client_voice_overrides(
        VoiceConfig(keyterms=["Norte Market"]),
        {"keyterms": ["attacker"], "stt_languages": ["en"], "stt_extra": {"keyterm": "attacker"}},
    )
    assert merged.keyterms == ["Norte Market"]
    assert merged.stt_languages is None
    assert "keyterm" not in merged.stt_extra


class _Ws:
    def __init__(self) -> None:
        self.sent: list[str] = []

    async def send(self, raw: str) -> None:
        self.sent.append(raw)


async def test_flux_replaces_keyterms_and_does_not_touch_another_session() -> None:
    agent_terms = ["Norte Market"]
    first = AudioInputConfig(keyterms=list(agent_terms))
    second = AudioInputConfig(keyterms=list(agent_terms))
    stt = DeepgramFluxSTT()
    stt._ws = _Ws()
    stt._input_config = first
    stt.patient_thresholds_live = True
    assert await stt.update_keyterms(["Ana Ruiz", "Jordi Puig"])
    sent = json.loads(stt._ws.sent[-1])
    assert sent == {"type": "Configure", "keyterms": ["Ana Ruiz", "Jordi Puig"]}
    assert stt._input_config.keyterms == ["Ana Ruiz", "Jordi Puig"]
    assert first.keyterms == ["Norte Market"]
    assert second.keyterms == ["Norte Market"]
    assert agent_terms == ["Norte Market"]
    assert stt.patient_thresholds_live is True
    assert await stt.update_keyterms([])
    # Flux takes one Configure at a time: the clear goes out after the ack.
    assert len(stt._ws.sent) == 1
    assert stt._input_config.keyterms == []
    await stt._handle_message({"type": "ConfigureSuccess"})
    assert stt.patient_thresholds_live is True
    assert json.loads(stt._ws.sent[-1]) == {"type": "Configure", "keyterms": []}


async def test_session_set_keyterms_copies_and_rejects_scribe() -> None:
    terms = ["NM1001"]
    stt = DeepgramFluxSTT()
    stt._ws = _Ws()
    session = VoiceSession(
        _agent(),
        stt,
        MockTTS(),
    )
    session.audio_input = AudioInputConfig(keyterms=["old"])
    stt._input_config = session.audio_input
    await session.set_keyterms(terms)
    terms.append("mutated-by-caller")
    assert session.audio_input.keyterms == ["NM1001"]
    assert json.loads(stt._ws.sent[-1])["keyterms"] == ["NM1001"]
    scribe = VoiceSession(_agent(), ElevenLabsRealtimeSTT(), MockTTS())
    with pytest.raises(ValueError, match="cannot replace"):
        await scribe.set_keyterms(["Acme"])


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("keyterm", "Acme"),
        ("keyterms", ["Acme"]),
        ("keywords", "Acme:2"),
        ("hotwords", "Acme"),
        ("keyterm", None),
        ("keyterm", ""),
        ("keyterm", []),
    ],
)
def test_nonempty_keyterms_conflict_even_when_the_extra_value_is_empty(key: str, value: object) -> None:
    with pytest.raises(ValueError, match="stt_extra"):
        VoiceConfig(keyterms=["Acme"], stt_extra={key: value})


@pytest.mark.parametrize("terms", [None, []])
def test_empty_keyterms_keep_a_legacy_extra_term(terms: list[str] | None) -> None:
    cfg = VoiceConfig(keyterms=terms, stt_extra={"keyterm": "Acme"})
    assert cfg.keyterms == []
    query = _flux_query(AudioInputConfig(extra={"keyterm": ["Acme"]}))
    assert query["keyterm"] == ["Acme"]


def test_reply_language_catalan_does_not_change_the_flux_hint() -> None:
    cfg = VoiceConfig(language="es", reply_language="ca")
    assert cfg.language == "es"
    assert cfg.reply_language == "ca"
    for blocked in ("ca", "ar", "zh"):
        with pytest.raises(ValueError, match="not a Deepgram Flux"):
            VoiceConfig(language=blocked)


def test_hello_keeps_a_valid_flux_hint_and_a_supported_update_does_not_replace_stt_languages() -> None:
    from timbal.server.voice import merge_client_voice_overrides

    base = VoiceConfig(language="es", reply_language="ca", stt_languages=["es", "en"], keyterms=["Norte Market"])
    blocked = merge_client_voice_overrides(base, {"language": "ca"})
    assert blocked.language == "es"
    assert blocked.reply_language == "ca"
    assert blocked.stt_languages == ["es", "en"]
    assert blocked.keyterms == ["Norte Market"]
    hints = _flux_query(
        AudioInputConfig(
            language=blocked.language, languages=list(blocked.stt_languages), keyterms=list(blocked.keyterms)
        )
    )
    assert hints["language_hint"] == ["es", "en"]

    updated = merge_client_voice_overrides(base, {"language": "fr"})
    assert updated.language == "fr"
    assert updated.reply_language == "ca"
    assert updated.stt_languages == ["es", "en"]
    kept = _flux_query(AudioInputConfig(language=updated.language, languages=list(updated.stt_languages)))
    assert kept["language_hint"] == ["es", "en"]

    for bad in ("", "   ", "english", "en-", "zh", "ar", 42):
        assert merge_client_voice_overrides(base, {"language": bad}).language == "es"


def test_hello_can_select_catalan_on_scribe_but_not_by_abandoning_flux_hints() -> None:
    from timbal.server.voice import merge_client_voice_overrides

    scribe = merge_client_voice_overrides(VoiceConfig(stt_provider="elevenlabs", language="en"), {"language": "ca"})
    assert scribe.stt_provider == "elevenlabs"
    assert scribe.language == "ca"

    moved = merge_client_voice_overrides(VoiceConfig(language="es"), {"stt_provider": "elevenlabs", "language": "ca"})
    assert moved.stt_provider == "elevenlabs"
    assert moved.language == "ca"

    kept = merge_client_voice_overrides(
        VoiceConfig(language="es", stt_languages=["es", "en"]),
        {"stt_provider": "elevenlabs", "stt_model": "scribe_v2_realtime", "language": "ca"},
    )
    assert kept.stt_provider == "deepgram"
    assert kept.stt_model == "flux-general-multi"
    assert kept.language == "es"
    assert kept.stt_languages == ["es", "en"]

    english_only = merge_client_voice_overrides(
        VoiceConfig(language="es", stt_languages=["es", "en"]),
        {"stt_model": "flux-general-en"},
    )
    assert english_only.stt_model == "flux-general-multi"
    assert english_only.stt_languages == ["es", "en"]


async def test_set_language_rejects_catalan_on_flux_before_any_change() -> None:
    config = AudioInputConfig(model="flux-general-multi", language="es", languages=["es", "en"])
    stt = DeepgramFluxSTT()
    stt._ws = _Ws()
    stt._input_config = config
    session = VoiceSession(_agent(), stt, MockTTS(), reply_language="ca")
    session.audio_input = config

    with pytest.raises(ValueError, match="not a Deepgram Flux"):
        await session.set_language("ca")

    assert session.audio_input.language == "es"
    assert session.audio_input.languages == ["es", "en"]
    assert session.reply_language == "ca"
    assert stt._ws.sent == []
    assert await stt.update_language("zh") is False
    assert stt._ws.sent == []

    assert await session.set_language("fr") is True
    assert session.audio_input.language == "fr"
    assert session.audio_input.languages == ["fr"]
    assert session.reply_language == "ca"
    assert json.loads(stt._ws.sent[-1]) == {"type": "Configure", "language_hints": ["fr"]}
    assert _flux_query(session.audio_input)["language_hint"] == ["fr"]


async def test_closed_flux_socket_does_not_store_an_unsupported_hint_or_terms() -> None:
    stt = DeepgramFluxSTT()
    session = VoiceSession(_agent(), stt, MockTTS(), reply_language="ca")
    session.audio_input = AudioInputConfig(
        model="flux-general-multi", language="es", languages=["es"], keyterms=["old"]
    )
    with pytest.raises(ValueError, match="not a Deepgram Flux"):
        await session.set_language("ar")
    assert session.audio_input.language == "es"
    assert session.audio_input.languages == ["es"]
    with pytest.raises(ValueError, match="cannot replace"):
        await session.set_keyterms(["new"])
    assert session.audio_input.keyterms == ["old"]


async def test_other_providers_still_accept_catalan_as_a_reply_hint() -> None:
    session = VoiceSession(_agent(), MockSTT(), MockTTS(), reply_language="follow")
    session.audio_input = AudioInputConfig(language="en")
    assert await session.set_language("ca") is False
    assert session.audio_input.language == "ca"
    assert session.reply_language == "follow"

    english_only = AudioInputConfig(model="flux-general-en", language="en")
    stt = DeepgramFluxSTT()
    stt._ws = _Ws()
    stt._input_config = english_only
    pinned = VoiceSession(_agent(), stt, MockTTS())
    pinned.audio_input = english_only
    assert await pinned.set_language("ca") is False
    assert pinned.audio_input.language == "ca"
    assert stt._ws.sent == []


def _agent():
    from timbal import Agent
    from timbal.core.test_model import TestModel

    return Agent(name="vocab", model=TestModel(responses=["ok"]))
