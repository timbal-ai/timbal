"""Tests for ``/voice`` STT/TTS config: env defaults, runnable merge, client overrides, lifespan."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from timbal import __version__ as timbal_version
from timbal.server import voice as voice_routes
from timbal.server.http import create_app, lifespan
from timbal.utils import ImportSpec

from .voice_env import VOICE_ENV_KEYS


@pytest.mark.usefixtures("clear_voice_env")
class TestDefaultVoiceConfigFromEnv:
    def test_defaults_when_unset(self) -> None:
        cfg = voice_routes.default_voice_config_from_env()
        assert cfg["stt_model"] == "scribe_v2_realtime"
        assert cfg["tts_model"] == "eleven_flash_v2_5"
        assert cfg["voice"] == voice_routes._DEFAULT_VOICE_ID
        assert cfg["language"] == "es"
        assert cfg["sample_rate"] == 16_000
        assert cfg["stt_extra"]["commit_strategy"] == "vad"
        assert cfg["tts_extra"]["auto_mode"] is True

    def test_env_overrides(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TIMBAL_STT_MODEL", "custom_stt")
        monkeypatch.setenv("TIMBAL_TTS_MODEL", "custom_tts")
        monkeypatch.setenv("TIMBAL_VOICE_LANGUAGE", "en")
        cfg = voice_routes.default_voice_config_from_env()
        assert cfg["stt_model"] == "custom_stt"
        assert cfg["tts_model"] == "custom_tts"
        assert cfg["language"] == "en"

    def test_elevenlabs_voice_id_precedence(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ELEVENLABS_VOICE_ID", raising=False)
        monkeypatch.delenv("TIMBAL_VOICE_ID", raising=False)
        monkeypatch.setenv("TIMBAL_VOICE_ID", "from_timbal")
        assert voice_routes.default_voice_config_from_env()["voice"] == "from_timbal"
        monkeypatch.setenv("ELEVENLABS_VOICE_ID", "from_el")
        assert voice_routes.default_voice_config_from_env()["voice"] == "from_el"


@pytest.mark.usefixtures("clear_voice_env")
class TestMergeVoiceConfig:
    def test_no_runnable_voice_config_uses_env_only(self) -> None:
        class R:
            pass

        merged = voice_routes.merge_voice_config(R())
        assert merged["language"] == "es"
        assert merged["voice"] == voice_routes._DEFAULT_VOICE_ID

    def test_dict_overrides_top_level(self) -> None:
        class R:
            voice_config = {"voice": "v1", "language": "pt"}

        merged = voice_routes.merge_voice_config(R())
        assert merged["voice"] == "v1"
        assert merged["language"] == "pt"
        assert merged["stt_model"] == "scribe_v2_realtime"

    def test_callable_voice_config(self) -> None:
        class R:
            @staticmethod
            def voice_config():
                return {"voice": "callable_v"}

        merged = voice_routes.merge_voice_config(R())
        assert merged["voice"] == "callable_v"

    def test_stt_extra_deep_merge(self) -> None:
        class R:
            voice_config = {"stt_extra": {"vad_threshold": 0.99}}

        merged = voice_routes.merge_voice_config(R())
        assert merged["stt_extra"]["commit_strategy"] == "vad"
        assert merged["stt_extra"]["vad_threshold"] == 0.99

    def test_tts_extra_deep_merge(self) -> None:
        class R:
            voice_config = {"tts_extra": {"auto_mode": False}}

        merged = voice_routes.merge_voice_config(R())
        assert merged["tts_extra"]["auto_mode"] is False

    def test_none_values_in_runnable_dict_skipped(self) -> None:
        class R:
            voice_config = {"voice": None, "language": "it"}

        merged = voice_routes.merge_voice_config(R())
        assert merged["voice"] == voice_routes._DEFAULT_VOICE_ID
        assert merged["language"] == "it"


class TestMergeClientVoiceOverrides:
    def test_overlay(self) -> None:
        base = {"voice": "a", "language": "es", "sample_rate": 16_000}
        out = voice_routes.merge_client_voice_overrides(base, {"language": "en", "voice": "b"})
        assert out["language"] == "en"
        assert out["voice"] == "b"
        assert out["sample_rate"] == 16_000

    def test_none_skipped(self) -> None:
        base = {"voice": "a", "language": "es"}
        out = voice_routes.merge_client_voice_overrides(base, {"language": None})
        assert out["language"] == "es"


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
            assert app.state.voice_config["language"] == "nl"
            assert app.state.voice_config["voice"] == "nl_voice"
            assert "stt_extra" in app.state.voice_config

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
            assert app.state.voice_config["voice"] == "env_only_voice"


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
            assert app.state.voice_config["language"] == "sv"

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
            assert str(mod.resolve()) in r.text
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
