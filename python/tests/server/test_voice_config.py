"""Tests for ``/voice`` STT/TTS config: env defaults, runnable merge, client overrides, lifespan."""

from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError
from timbal import __version__ as timbal_version
from timbal.server import voice as voice_routes
from timbal.server.http import create_app, lifespan
from timbal.utils import ImportSpec
from timbal.voice.config import DEFAULT_VOICE_ID, FillerConfig, RecordingConfig, VoiceConfig

from .voice_env import VOICE_ENV_KEYS


@pytest.mark.usefixtures("clear_voice_env")
class TestVoiceWarmupIntended:
    """Non-voice deployments must not pre-load ONNX models at server boot.

    Regression: warmup used to run for every Agent app whenever the
    timbal[voice] extra was installed (e.g. platform images from timbal[all]),
    downloading and loading Smart Turn + Namo + Silero for nothing.
    """

    class _Runnable:
        voice_config = None

    def _clear_warmup_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("TIMBAL_VOICE_WARMUP", raising=False)
        for k in list(os.environ):
            if k.startswith("TIMBAL_VOICE_"):
                monkeypatch.delenv(k, raising=False)

    def test_no_voice_signals_means_no_warmup(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._clear_warmup_env(monkeypatch)
        assert voice_routes.voice_warmup_intended(self._Runnable()) is False

    def test_runnable_voice_config_enables_warmup(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._clear_warmup_env(monkeypatch)
        r = self._Runnable()
        r.voice_config = {"stt_provider": "elevenlabs"}
        assert voice_routes.voice_warmup_intended(r) is True

    def test_voice_env_enables_warmup(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._clear_warmup_env(monkeypatch)
        monkeypatch.setenv("TIMBAL_VOICE_LANGUAGE", "en")
        assert voice_routes.voice_warmup_intended(self._Runnable()) is True

    def test_elevenlabs_voice_id_enables_warmup(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._clear_warmup_env(monkeypatch)
        monkeypatch.setenv("ELEVENLABS_VOICE_ID", "abc")
        assert voice_routes.voice_warmup_intended(self._Runnable()) is True

    def test_env_override_forces_warmup(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The playground launcher sets TIMBAL_VOICE_WARMUP=1 for its children."""
        self._clear_warmup_env(monkeypatch)
        monkeypatch.setenv("TIMBAL_VOICE_WARMUP", "1")
        assert voice_routes.voice_warmup_intended(self._Runnable()) is True

    def test_env_override_forces_no_warmup(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._clear_warmup_env(monkeypatch)
        monkeypatch.setenv("TIMBAL_VOICE_WARMUP", "0")
        r = self._Runnable()
        r.voice_config = {"stt_provider": "elevenlabs"}  # even a voice app
        assert voice_routes.voice_warmup_intended(r) is False


class TestVoiceOnnxWarmupIntended:
    """Flux / provider EOU must not pull Smart Turn + Namo + Silero at boot."""

    def test_flux_skips_onnx(self) -> None:
        cfg = VoiceConfig(stt_provider="deepgram", stt_model="flux-general-multi")
        assert voice_routes.voice_onnx_warmup_intended(cfg) is False

    def test_elevenlabs_loads_onnx(self) -> None:
        cfg = VoiceConfig(
            stt_provider="elevenlabs",
            stt_model="scribe_v2_realtime",
            turn_detector="local",
        )
        assert voice_routes.voice_onnx_warmup_intended(cfg) is True

    def test_flux_with_explicit_local_still_skips(self) -> None:
        # Session setup overrides local → provider for Flux; warmup must match.
        cfg = VoiceConfig(
            stt_provider="deepgram",
            stt_model="flux-general-multi",
            turn_detector="local",
        )
        assert voice_routes.voice_onnx_warmup_intended(cfg) is False

    def test_explicit_heuristic_skips_onnx(self) -> None:
        cfg = VoiceConfig(stt_provider="elevenlabs", turn_detector="heuristic")
        assert voice_routes.voice_onnx_warmup_intended(cfg) is False

    def test_nova_loads_onnx(self) -> None:
        cfg = VoiceConfig(
            stt_provider="deepgram-nova",
            stt_model="nova-3",
            turn_detector="local",
        )
        assert voice_routes.voice_onnx_warmup_intended(cfg) is True

    def test_explicit_provider_skips_onnx(self) -> None:
        cfg = VoiceConfig(stt_provider="elevenlabs", turn_detector="provider")
        assert voice_routes.voice_onnx_warmup_intended(cfg) is False


@pytest.mark.usefixtures("clear_voice_env")
class TestDefaultVoiceConfigFromEnv:
    def test_defaults_when_unset(self) -> None:
        cfg = voice_routes.default_voice_config_from_env()
        assert cfg.stt_provider == "elevenlabs"
        assert cfg.stt_model == "scribe_v2_realtime"
        assert cfg.tts_model == "eleven_flash_v2_5"
        assert cfg.voice == DEFAULT_VOICE_ID
        assert cfg.language is None  # provider auto-detect
        assert cfg.sample_rate == 16_000
        assert cfg.stt_extra["commit_strategy"] == "vad"
        assert cfg.tts_extra["auto_mode"] is True

    def test_env_overrides(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TIMBAL_STT_PROVIDER", "deepgram")
        monkeypatch.setenv("TIMBAL_STT_MODEL", "custom_stt")
        monkeypatch.setenv("TIMBAL_TTS_MODEL", "custom_tts")
        monkeypatch.setenv("TIMBAL_VOICE_LANGUAGE", "en")
        cfg = voice_routes.default_voice_config_from_env()
        assert cfg.stt_provider == "deepgram"
        assert cfg.stt_model == "custom_stt"
        assert cfg.tts_model == "custom_tts"
        assert cfg.language == "en"

    def test_elevenlabs_voice_id_precedence(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ELEVENLABS_VOICE_ID", raising=False)
        monkeypatch.delenv("TIMBAL_VOICE_ID", raising=False)
        monkeypatch.setenv("TIMBAL_VOICE_ID", "from_timbal")
        assert voice_routes.default_voice_config_from_env().voice == "from_timbal"
        monkeypatch.setenv("ELEVENLABS_VOICE_ID", "from_el")
        assert voice_routes.default_voice_config_from_env().voice == "from_el"


@pytest.mark.usefixtures("clear_voice_env")
class TestMergeVoiceConfig:
    def test_no_runnable_voice_config_uses_env_only(self) -> None:
        class R:
            pass

        merged = voice_routes.merge_voice_config(R())
        assert merged.language is None
        assert merged.voice == DEFAULT_VOICE_ID

    def test_dict_overrides_top_level(self) -> None:
        class R:
            voice_config = {"voice": "v1", "language": "pt"}

        merged = voice_routes.merge_voice_config(R())
        assert merged.voice == "v1"
        assert merged.language == "pt"
        assert merged.stt_model == "scribe_v2_realtime"

    def test_callable_voice_config(self) -> None:
        class R:
            @staticmethod
            def voice_config():
                return {"voice": "callable_v"}

        merged = voice_routes.merge_voice_config(R())
        assert merged.voice == "callable_v"

    def test_voice_config_instance(self) -> None:
        class R:
            voice_config = VoiceConfig(voice="typed_v", language="de")

        merged = voice_routes.merge_voice_config(R())
        assert merged.voice == "typed_v"
        assert merged.language == "de"
        assert merged.stt_model == "scribe_v2_realtime"

    def test_stt_extra_deep_merge(self) -> None:
        class R:
            voice_config = {"stt_extra": {"vad_threshold": 0.99}}

        merged = voice_routes.merge_voice_config(R())
        assert merged.stt_extra["commit_strategy"] == "vad"
        assert merged.stt_extra["vad_threshold"] == 0.99

    def test_tts_extra_deep_merge(self) -> None:
        class R:
            voice_config = {"tts_extra": {"auto_mode": False}}

        merged = voice_routes.merge_voice_config(R())
        assert merged.tts_extra["auto_mode"] is False

    def test_none_values_in_runnable_dict_skipped(self) -> None:
        class R:
            voice_config = {"voice": None, "language": "it"}

        merged = voice_routes.merge_voice_config(R())
        assert merged.voice == DEFAULT_VOICE_ID
        assert merged.language == "it"

    def test_unknown_key_fails_fast(self) -> None:
        """A typo'd voice_config key must raise (at server boot), not silently no-op."""

        class R:
            voice_config = {"languag": "en"}

        with pytest.raises(ValidationError, match="languag"):
            voice_routes.merge_voice_config(R())

    def test_recording_dict_is_validated(self) -> None:
        class R:
            voice_config = {"recording": {"dir": "/tmp/rec", "layout": "sideways"}}

        with pytest.raises(ValidationError, match="layout"):
            voice_routes.merge_voice_config(R())

    def test_turn_detector_instance_passes_through(self) -> None:
        sentinel = object()

        class R:
            voice_config = {"turn_detector": sentinel}

        assert voice_routes.merge_voice_config(R()).turn_detector is sentinel

    def test_filler_deep_merges_over_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Agent's partial filler dict must not drop platform TIMBAL_VOICE_FILLER_* settings."""
        monkeypatch.setenv("TIMBAL_VOICE_FILLER_SYSTEM_PROMPT", "PLATFORM PROMPT")
        monkeypatch.setenv("TIMBAL_VOICE_FILLER_MODEL", "openai/gpt-4o-mini")

        class R:
            voice_config = {"filler": {"delay_secs": 2.0}}

        merged = voice_routes.merge_voice_config(R())
        assert merged.filler.system_prompt == "PLATFORM PROMPT"
        assert merged.filler.model == "openai/gpt-4o-mini"
        assert merged.filler.delay_secs == 2.0

    def test_empty_filler_dict_keeps_env_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TIMBAL_VOICE_FILLER_SYSTEM_PROMPT", "PLATFORM PROMPT")

        class R:
            voice_config = {"filler": {}}

        assert voice_routes.merge_voice_config(R()).filler.system_prompt == "PLATFORM PROMPT"

    def test_filler_via_voice_config_instance_merges_sparsely(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Unset FillerConfig fields (e.g. the default prompt) must not clobber env values."""
        monkeypatch.setenv("TIMBAL_VOICE_FILLER_SYSTEM_PROMPT", "PLATFORM PROMPT")

        class R:
            voice_config = VoiceConfig(filler=FillerConfig(delay_secs=2.0))

        merged = voice_routes.merge_voice_config(R())
        assert merged.filler.system_prompt == "PLATFORM PROMPT"
        assert merged.filler.delay_secs == 2.0

    def test_filler_without_env_uses_agent_values(self) -> None:
        class R:
            voice_config = {"filler": {"delay_secs": 0.5}}

        merged = voice_routes.merge_voice_config(R())
        assert merged.filler.delay_secs == 0.5

    def test_invalid_filler_key_fails_fast(self) -> None:
        class R:
            voice_config = {"filler": {"phrases": ["nope"]}}

        with pytest.raises(ValidationError, match="phrases"):
            voice_routes.merge_voice_config(R())


class TestMergeClientVoiceOverrides:
    def test_overlay(self) -> None:
        base = VoiceConfig(voice="a", language="es")
        out = voice_routes.merge_client_voice_overrides(base, {"language": "en", "voice": "b"})
        assert out.language == "en"
        assert out.voice == "b"
        assert out.sample_rate == 16_000

    def test_none_skipped(self) -> None:
        base = VoiceConfig(voice="a", language="es")
        out = voice_routes.merge_client_voice_overrides(base, {"language": None})
        assert out.language == "es"

    def test_non_allowlisted_keys_ignored(self) -> None:
        base = VoiceConfig(recording=RecordingConfig(dir="/srv/rec"))
        out = voice_routes.merge_client_voice_overrides(
            base,
            {"recording": {"dir": "/tmp/evil"}, "turn_detector": "heuristic", "bogus": 1, "voice": "b"},
        )
        assert out.recording is not None and out.recording.dir == "/srv/rec"
        assert out.voice == "b"

    def test_allowlist_is_subset_of_model_fields(self) -> None:
        assert voice_routes.CLIENT_SETTABLE_VOICE_FIELDS <= set(VoiceConfig.model_fields)
        assert "recording" not in voice_routes.CLIENT_SETTABLE_VOICE_FIELDS


_HOSTILE_HELLO = {
    "stt_extra": {"stt_host": "evil.example", "callback": "https://evil.example/t", "eot_threshold": 0.9},
    "tts_extra": {"tts_host": "evil.example", "speed": 1.1},
}


class TestClientExtrasAllowlist:
    """A caller must never pick the provider WebSocket host.

    The adapters read ``stt_host`` / ``tts_host`` off the extras and send the
    API key in that handshake; Deepgram Nova and ElevenLabs forward any other
    key into the query string. Client extras are therefore allow-listed to
    tuning knobs and layered *over* the server's extras, never replacing them.
    """

    def test_hostile_hello_loses_host_and_callback(self) -> None:
        out = voice_routes.merge_client_voice_overrides(VoiceConfig(), _HOSTILE_HELLO)
        assert "stt_host" not in out.stt_extra
        assert "callback" not in out.stt_extra
        assert "tts_host" not in out.tts_extra
        # The tuning knobs on the same hello still apply.
        assert out.stt_extra["eot_threshold"] == 0.9
        assert out.tts_extra["speed"] == 1.1

    def test_adapters_resolve_their_default_host_after_the_merge(self) -> None:
        from urllib.parse import urlparse

        from timbal.voice import AudioInputConfig, AudioOutputConfig
        from timbal.voice.deepgram import DeepgramFluxSTT, DeepgramNovaSTT
        from timbal.voice.fish_audio import _DEFAULT_HOST as FISH_HOST
        from timbal.voice.munsit import _DEFAULT_HOST as MUNSIT_HOST
        from timbal.voice.munsit import MunsitStreamSTT

        out = voice_routes.merge_client_voice_overrides(VoiceConfig(), _HOSTILE_HELLO)
        stt_cfg = AudioInputConfig(extra=out.stt_extra)
        for stt in (DeepgramFluxSTT(), DeepgramNovaSTT()):
            assert urlparse(stt._build_uri(stt_cfg)).hostname == "api.deepgram.com"
        assert urlparse(MunsitStreamSTT()._build_uri(stt_cfg)).hostname == MUNSIT_HOST
        # No Nova query param may carry the callback either.
        assert "callback" not in urlparse(DeepgramNovaSTT()._build_uri(stt_cfg)).query
        tts_cfg = AudioOutputConfig(extra=out.tts_extra)
        assert tts_cfg.extra.get("tts_host", MUNSIT_HOST) == MUNSIT_HOST
        assert tts_cfg.extra.get("tts_host", FISH_HOST) == FISH_HOST

    def test_server_pin_survives_client_tuning(self) -> None:
        base = VoiceConfig(tts_extra={"dialect": "emirati"}, stt_extra={"hotwords": "Timbal"})
        out = voice_routes.merge_client_voice_overrides(
            base, {"tts_extra": {"speed": 1.1}, "stt_extra": {"endpointing": 400}}
        )
        assert out.tts_extra == {"dialect": "emirati", "speed": 1.1}
        assert out.stt_extra == {"hotwords": "Timbal", "endpointing": 400}

    def test_server_set_host_still_works(self) -> None:
        """Self-hosted Munsit / on-prem Deepgram: the *server* keeps choosing the host."""
        base = VoiceConfig(stt_extra={"stt_host": "munsit.internal"}, tts_extra={"tts_host": "munsit.internal"})
        out = voice_routes.merge_client_voice_overrides(base, {"stt_extra": {"endpointing": 400}, "tts_extra": {}})
        assert out.stt_extra["stt_host"] == "munsit.internal"
        assert out.tts_extra["tts_host"] == "munsit.internal"
        # And a client cannot move it either.
        out = voice_routes.merge_client_voice_overrides(base, _HOSTILE_HELLO)
        assert out.stt_extra["stt_host"] == "munsit.internal"
        assert out.tts_extra["tts_host"] == "munsit.internal"

    def test_client_cannot_clear_a_server_key(self) -> None:
        base = VoiceConfig(tts_extra={"dialect": "emirati"})
        out = voice_routes.merge_client_voice_overrides(base, {"tts_extra": {"dialect": None}})
        assert out.tts_extra["dialect"] == "emirati"

    @pytest.mark.parametrize("bad", ["stt_host=evil", 7, ["stt_host", "evil"]])
    def test_non_object_extra_keeps_servers(self, bad) -> None:
        base = VoiceConfig(stt_extra={"vad_threshold": 0.9})
        out = voice_routes.merge_client_voice_overrides(base, {"stt_extra": bad})
        assert out.stt_extra == {"vad_threshold": 0.9}

    def test_allowlists_carry_no_host_url_or_callback(self) -> None:
        for allowed in (voice_routes.CLIENT_TUNING_STT_EXTRA, voice_routes.CLIENT_TUNING_TTS_EXTRA):
            for key in allowed:
                assert not key.endswith(("_host", "_url")), key
                assert key not in ("callback", "metadata", "correlation_id"), key

    def test_phone_tuned_sip_extras_are_allowlisted(self) -> None:
        """The LiveKit SIP path injects its PSTN VAD tuning as *client* extras."""
        from timbal.server.livekit_sip import _PHONE_TUNED_STT_EXTRA

        assert set(_PHONE_TUNED_STT_EXTRA) <= voice_routes.CLIENT_TUNING_STT_EXTRA

    def test_livekit_hello_path(self) -> None:
        """Dial ``client_config`` (browser config forwarded by the platform) + data-channel hello."""
        from timbal.server.livekit_session import merge_client_config

        base = VoiceConfig(tts_extra={"dialect": "emirati"})
        config = merge_client_config(
            json.dumps({"stt_extra": {"stt_host": "evil.example"}}),
            {"tts_extra": {"tts_host": "evil.example", "speed": 1.1}},
        )
        out = voice_routes.merge_client_voice_overrides(base, config)
        assert "stt_host" not in out.stt_extra
        assert out.tts_extra == {"dialect": "emirati", "speed": 1.1}

    def test_telephony_parameters_cannot_carry_extras(self) -> None:
        """``<Parameter>`` values are strings; the start-frame allowlist has no extras at all."""
        from timbal.server.telephony import _CONFIG_PARAM_KEYS

        assert "stt_extra" not in _CONFIG_PARAM_KEYS
        assert "tts_extra" not in _CONFIG_PARAM_KEYS

    def test_build_voice_session_hostile_hello(self) -> None:
        """End to end through the WS/RTC entry point: the session's adapter configs carry no host."""
        from timbal import Agent
        from timbal.core.test_model import TestModel

        agent = Agent(name="voice_test", model=TestModel(responses=["hi"]), tools=[])
        defaults = VoiceConfig(turn_detector="heuristic", tts_extra={"dialect": "emirati"})
        session, _ = voice_routes.build_voice_session(agent, defaults, _HOSTILE_HELLO)
        assert "stt_host" not in session.audio_input.extra
        assert "callback" not in session.audio_input.extra
        assert "tts_host" not in session.audio_output.extra
        assert session.audio_output.extra["dialect"] == "emirati"
        assert session.audio_output.extra["speed"] == 1.1


class TestLifespanVoiceState:
    @pytest.mark.asyncio
    async def test_lifespan_sets_voice_config_from_runnable(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        mod = tmp_path / "runnable_voice.py"
        mod.write_text(
            'class R:\n'
            '    voice_config = {"language": "nl", "voice": "nl_voice"}\n'
            "agent = R()\n",
        )
        monkeypatch.setenv("TIMBAL_RUNNABLE", f"{mod.resolve()}::agent")
        for k in VOICE_ENV_KEYS:
            monkeypatch.delenv(k, raising=False)

        app = FastAPI()
        spec = ImportSpec.from_fqn(os.environ["TIMBAL_RUNNABLE"])
        async with lifespan(app, spec):
            assert app.state.voice_config.language == "nl"
            assert app.state.voice_config.voice == "nl_voice"
            assert app.state.voice_config.stt_extra["commit_strategy"] == "vad"

    @pytest.mark.asyncio
    async def test_lifespan_merge_with_env(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        mod = tmp_path / "plain.py"
        mod.write_text("class T: pass\nrunnable = T()\n")
        monkeypatch.setenv("TIMBAL_RUNNABLE", f"{mod.resolve()}::runnable")
        for k in VOICE_ENV_KEYS:
            monkeypatch.delenv(k, raising=False)
        monkeypatch.setenv("TIMBAL_VOICE_ID", "env_only_voice")

        app = FastAPI()
        spec = ImportSpec.from_fqn(os.environ["TIMBAL_RUNNABLE"])
        async with lifespan(app, spec):
            assert app.state.voice_config.voice == "env_only_voice"


class TestCreateAppVoiceIntegration:
    def test_testclient_startup_sets_voice_config(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        mod = tmp_path / "app.py"
        mod.write_text(
            'class R:\n'
            '    voice_config = {"language": "sv"}\n'
            "x = R()\n",
        )
        monkeypatch.setenv("TIMBAL_RUNNABLE", f"{mod.resolve()}::x")
        for k in VOICE_ENV_KEYS:
            monkeypatch.delenv(k, raising=False)

        app = create_app()
        with TestClient(app) as client:
            r = client.get("/healthcheck")
            assert r.status_code == 204
            assert app.state.voice_config.language == "sv"

    def test_voice_page_injects_runnable_meta(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        mod = tmp_path / "named.py"
        mod.write_text(
            "from timbal import Agent\n"
            'r = Agent(name="voice_demo", model="timbal/TestModel", tools=[])\n',
        )
        monkeypatch.setenv("TIMBAL_RUNNABLE", f"{mod.resolve()}::r")
        for k in VOICE_ENV_KEYS:
            monkeypatch.delenv(k, raising=False)

        app = create_app()
        with TestClient(app) as client:
            r = client.get("/voice/")
            assert r.status_code == 200
            assert "voice_demo" in r.text
            assert "Agent" in r.text
            meta_match = re.search(
                r'id="timbal-voice-runnable-meta">([^<]+)</script>',
                r.text,
            )
            assert meta_match is not None
            meta = json.loads(meta_match.group(1))
            assert Path(meta["import_spec"].split("::")[0]).resolve() == mod.resolve()
            assert timbal_version in r.text


class TestVoiceServerScript:
    def test_main_sets_timbal_runnable_and_calls_cli(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        script = tmp_path / "voice_server.py"
        script.write_text(
            "import os\n"
            "import sys\n"
            "from pathlib import Path\n"
            "from timbal.server.http import run_server_cli\n"
            "agent = object()\n"
            "def main():\n"
            '    os.environ["TIMBAL_RUNNABLE"] = f"{Path(__file__).resolve()}::agent"\n'
            "    run_server_cli(sys.argv[1:])\n"
        )
        argv_captured: list[list[str]] = []

        def fake_run_server_cli(argv: list[str] | None = None) -> None:
            argv_captured.append(list(argv) if argv is not None else [])

        monkeypatch.setattr("timbal.server.http.run_server_cli", fake_run_server_cli)

        spec = importlib.util.spec_from_file_location("_voice_server_under_test", script)
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        monkeypatch.setattr(sys, "argv", [str(script), "--port", "9999"])
        mod.main()

        assert len(argv_captured) == 1
        assert argv_captured[0] == ["--port", "9999"]
        runn = os.environ["TIMBAL_RUNNABLE"]
        assert runn.endswith("::agent")
        assert "voice_server.py" in runn
