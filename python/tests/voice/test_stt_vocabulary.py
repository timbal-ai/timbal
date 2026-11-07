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

from .test_session import MockTTS


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
    assert json.loads(stt._ws.sent[-1]) == {"type": "Configure", "keyterms": []}
    assert stt._input_config.keyterms == []


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


def _agent():
    from timbal import Agent
    from timbal.core.test_model import TestModel

    return Agent(name="vocab", model=TestModel(responses=["ok"]))
