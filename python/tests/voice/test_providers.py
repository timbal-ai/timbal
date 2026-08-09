import pytest
from timbal.voice.elevenlabs import ElevenLabsStreamTTS
from timbal.voice.fish_audio import FishAudioStreamTTS
from timbal.voice.munsit import MunsitStreamTTS
from timbal.voice.providers import resolve_tts


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
