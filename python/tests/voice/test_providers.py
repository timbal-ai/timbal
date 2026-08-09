import pytest
from timbal import Agent
from timbal.core.test_model import TestModel
from timbal.server.voice import build_voice_session
from timbal.voice.config import DEFAULT_VOICE_ID, VoiceConfig
from timbal.voice.elevenlabs import (
    _DEFAULT_TTS_MODEL as ELEVENLABS_DEFAULT_TTS_MODEL,
)
from timbal.voice.elevenlabs import (
    ElevenLabsStreamTTS,
    effective_tts_model,
)
from timbal.voice.fish_audio import FishAudioStreamTTS
from timbal.voice.munsit import MunsitStreamTTS
from timbal.voice.providers import AudioOutputConfig, resolve_tts


class TestResolveTts:
    def test_default_is_elevenlabs(self) -> None:
        assert isinstance(resolve_tts(None), ElevenLabsStreamTTS)
        assert isinstance(resolve_tts(""), ElevenLabsStreamTTS)
        assert isinstance(resolve_tts("  "), ElevenLabsStreamTTS)

    def test_explicit_providers_and_aliases(self) -> None:
        assert isinstance(resolve_tts("elevenlabs"), ElevenLabsStreamTTS)
        assert isinstance(resolve_tts("11labs"), ElevenLabsStreamTTS)
        assert isinstance(resolve_tts("munsit"), MunsitStreamTTS)
        assert isinstance(resolve_tts("faseeh"), MunsitStreamTTS)
        assert isinstance(resolve_tts("fishaudio"), FishAudioStreamTTS)
        assert isinstance(resolve_tts("fish-audio"), FishAudioStreamTTS)
        assert isinstance(resolve_tts("fish"), FishAudioStreamTTS)
        # Case-insensitive, whitespace-tolerant (client hello strings).
        assert isinstance(resolve_tts(" FishAudio "), FishAudioStreamTTS)

    def test_unknown_provider_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown TTS provider"):
            resolve_tts("polly")

    def test_provider_ids_match_config_values(self) -> None:
        # ``build_voice_session`` reports ``tts.provider_id`` to clients/logs;
        # it must round-trip through resolve_tts.
        for provider_id in ("elevenlabs", "munsit", "fishaudio"):
            assert resolve_tts(provider_id).provider_id == provider_id


class TestElevenLabsModelGuard:
    """ElevenLabs' half of the cross-provider model guard.

    Munsit and Fish each swap foreign model ids out; without the mirror here
    ElevenLabs was the one provider that would put another vendor's id in its
    stream-input query string.
    """

    def test_eleven_ids_pass_through(self) -> None:
        assert effective_tts_model(AudioOutputConfig(model="eleven_flash_v2_5")) == "eleven_flash_v2_5"
        assert effective_tts_model(AudioOutputConfig(model="eleven_turbo_v2_5")) == "eleven_turbo_v2_5"

    def test_foreign_and_empty_ids_fall_back(self) -> None:
        for model in (None, "", "  ", "faseeh-v1-preview", "s2.1-pro", "scribe_v2_realtime"):
            assert effective_tts_model(AudioOutputConfig(model=model)) == ELEVENLABS_DEFAULT_TTS_MODEL


def _session_for(**voice_config_kwargs):
    agent = Agent(name="t", model=TestModel(responses=["ok"]), tools=[])
    defaults = VoiceConfig(turn_detector="heuristic", **voice_config_kwargs)
    return build_voice_session(agent, defaults, {})


class TestUnknownProviderFallback:
    """An unrecognized ``tts_provider`` degrades to a *working* ElevenLabs session.

    The failure this pins: the fallback swapped the provider but kept the
    requested model and voice, so the session reported ElevenLabs while putting
    a Munsit model id and voice on its wire — TTS died on a config that looked
    correct in the logs.
    """

    def test_foreign_model_and_voice_are_dropped(self) -> None:
        session, meta = _session_for(
            tts_provider="polly",
            tts_model="faseeh-v1-preview",
            voice="ar-najdi-male-2",
        )
        assert meta["tts_provider"] == "elevenlabs"
        assert session.audio_output.model is None
        # Not merely non-Munsit: the voice is a required path segment, so it
        # has to be a real ElevenLabs id rather than None/"".
        assert session.audio_output.voice == DEFAULT_VOICE_ID

    def test_fallback_config_yields_an_elevenlabs_model_on_the_wire(self) -> None:
        session, _ = _session_for(tts_provider="polly", tts_model="s2.1-pro")
        assert effective_tts_model(session.audio_output) == ELEVENLABS_DEFAULT_TTS_MODEL

    def test_known_provider_keeps_its_own_model_and_voice(self) -> None:
        session, meta = _session_for(
            tts_provider="munsit",
            tts_model="faseeh-v1-preview",
            voice="ar-najdi-male-2",
        )
        assert meta["tts_provider"] == "munsit"
        assert session.audio_output.model == "faseeh-v1-preview"
        assert session.audio_output.voice == "ar-najdi-male-2"
