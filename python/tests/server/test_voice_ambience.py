"""Ambient background audio: config validation, lazy CDN download, HTTP routes."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from timbal.server import voice as voice_routes
from timbal.server.http import create_app
from timbal.voice import ambience
from timbal.voice.ambience import PRESETS, ensure_ambient_source, validate_ambient_source
from timbal.voice.config import AmbientAudioConfig, VoiceConfig

from .voice_env import VOICE_ENV_KEYS

FAKE_WAV = b"RIFF" + b"\x00" * 64
FAKE_SHA = hashlib.sha256(FAKE_WAV).hexdigest()


@pytest.fixture
def fake_cdn(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[str]:
    """Isolated cache dir + stubbed CDN serving FAKE_WAV for every preset.

    Returns the list of fetched URLs so tests can assert download counts.
    """
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    calls: list[str] = []

    def _fetch(url: str) -> bytes:
        calls.append(url)
        return FAKE_WAV

    monkeypatch.setattr(ambience, "_fetch", _fetch)
    for name in PRESETS:
        monkeypatch.setitem(PRESETS, name, FAKE_SHA)
    return calls


class TestAmbientAudioConfig:
    def test_preset_source_defaults(self) -> None:
        cfg = AmbientAudioConfig(source="office")
        assert cfg.volume == 0.3

    def test_unknown_preset_fails_at_validation(self) -> None:
        with pytest.raises(ValidationError, match="neither a preset"):
            AmbientAudioConfig(source="rainforest")

    def test_custom_file_source(self, tmp_path: Path) -> None:
        f = tmp_path / "track.wav"
        f.write_bytes(b"RIFF")
        assert AmbientAudioConfig(source=str(f)).source == str(f)

    def test_volume_bounds(self) -> None:
        with pytest.raises(ValidationError):
            AmbientAudioConfig(source="office", volume=1.5)
        with pytest.raises(ValidationError):
            AmbientAudioConfig(source="office", volume=-0.1)

    def test_nested_in_voice_config(self) -> None:
        cfg = VoiceConfig(ambient={"source": "cafe", "volume": 0.1})
        assert cfg.ambient is not None
        assert cfg.ambient.source == "cafe"

    def test_voice_config_default_is_off(self) -> None:
        assert VoiceConfig().ambient is None


class TestValidateAmbientSource:
    """Validation is offline — a preset name never touches the network."""

    def test_preset_case_insensitive(self) -> None:
        validate_ambient_source("OFFICE")

    def test_custom_path(self, tmp_path: Path) -> None:
        f = tmp_path / "x.wav"
        f.write_bytes(b"RIFF")
        validate_ambient_source(str(f))

    def test_missing_path_raises(self) -> None:
        with pytest.raises(ValueError, match="neither a preset"):
            validate_ambient_source("/nope/missing.wav")

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="preset name or file path"):
            validate_ambient_source("  ")


class TestEnsureAmbientSource:
    def test_preset_downloads_once_then_caches(self, fake_cdn: list[str]) -> None:
        first = ensure_ambient_source("office")
        second = ensure_ambient_source("OFFICE")
        assert first == second
        assert first.read_bytes() == FAKE_WAV
        assert fake_cdn == [f"{ambience.base_url()}/office.wav"]

    def test_custom_path_never_fetches(self, fake_cdn: list[str], tmp_path: Path) -> None:
        f = tmp_path / "x.wav"
        f.write_bytes(b"RIFF")
        assert ensure_ambient_source(str(f)) == f
        assert fake_cdn == []

    @pytest.mark.usefixtures("fake_cdn")
    def test_checksum_mismatch_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(PRESETS, "office", "0" * 64)
        with pytest.raises(RuntimeError, match="checksum"):
            ensure_ambient_source("office")
        assert not (ambience.cache_dir() / "office.wav").exists()

    def test_base_url_env_override(self, fake_cdn: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TIMBAL_AMBIENCE_BASE_URL", "https://cdn.example.com/amb/")
        ensure_ambient_source("cafe")
        assert fake_cdn == ["https://cdn.example.com/amb/cafe.wav"]


@pytest.mark.usefixtures("clear_voice_env")
class TestAmbientEnvDefaults:
    def test_unset_means_off(self) -> None:
        assert voice_routes.default_voice_config_from_env().ambient is None

    def test_env_source_and_volume(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TIMBAL_VOICE_AMBIENT_SOURCE", "office")
        monkeypatch.setenv("TIMBAL_VOICE_AMBIENT_VOLUME", "0.15")
        cfg = voice_routes.default_voice_config_from_env()
        assert cfg.ambient is not None
        assert cfg.ambient.source == "office"
        assert cfg.ambient.volume == 0.15

    def test_bad_env_source_fails_fast(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TIMBAL_VOICE_AMBIENT_SOURCE", "not-a-preset")
        with pytest.raises(ValidationError):
            voice_routes.default_voice_config_from_env()


class TestClientCannotSetAmbient:
    def test_allowlist_excludes_ambient(self) -> None:
        assert "ambient" not in voice_routes.CLIENT_SETTABLE_VOICE_FIELDS

    def test_client_override_ignored(self) -> None:
        base = VoiceConfig()
        out = voice_routes.merge_client_voice_overrides(base, {"ambient": {"source": "/etc/passwd"}})
        assert out.ambient is None


class TestAmbienceRoutes:
    def _client(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        voice_config: str = "",
    ) -> TestClient:
        mod = tmp_path / "app.py"
        body = f"    voice_config = {voice_config}\n" if voice_config else "    pass\n"
        mod.write_text(f"class R:\n{body}x = R()\n")
        monkeypatch.setenv("TIMBAL_RUNNABLE", f"{mod.resolve()}::x")
        for k in VOICE_ENV_KEYS:
            monkeypatch.delenv(k, raising=False)
        return TestClient(create_app())

    def test_index_lists_presets(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        with self._client(monkeypatch, tmp_path) as client:
            r = client.get("/voice/ambience")
            assert r.status_code == 200
            assert r.json() == {"presets": sorted(PRESETS)}

    @pytest.mark.usefixtures("fake_cdn")
    def test_preset_is_served(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        with self._client(monkeypatch, tmp_path) as client:
            r = client.get("/voice/ambience/typing")
            assert r.status_code == 200
            assert r.content == FAKE_WAV

    def test_preset_cached_across_requests(
        self, fake_cdn: list[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with self._client(monkeypatch, tmp_path) as client:
            assert client.get("/voice/ambience/city").status_code == 200
            assert client.get("/voice/ambience/city").status_code == 200
        assert len(fake_cdn) == 1

    def test_unknown_preset_404(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        with self._client(monkeypatch, tmp_path) as client:
            assert client.get("/voice/ambience/nope").status_code == 404

    def test_no_path_traversal(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """The preset route is a membership check — encoded traversal never reaches the filesystem."""
        with self._client(monkeypatch, tmp_path) as client:
            r = client.get("/voice/ambience/..%2F..%2Fserver%2Fvoice.html")
            assert r.status_code == 404

    @pytest.mark.usefixtures("fake_cdn")
    def test_download_failure_502(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        def _boom(_url: str) -> bytes:
            raise RuntimeError("cdn down")

        monkeypatch.setattr(ambience, "_fetch", _boom)
        with self._client(monkeypatch, tmp_path) as client:
            assert client.get("/voice/ambience/office").status_code == 502

    def test_current_404_when_off(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        with self._client(monkeypatch, tmp_path) as client:
            assert client.get("/voice/ambience/current").status_code == 404

    def test_current_serves_custom_file(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        track = tmp_path / "custom.wav"
        track.write_bytes(b"RIFF" + b"\x00" * 64)
        cfg = f"{{'ambient': {{'source': {str(track)!r}, 'volume': 0.2}}}}"
        with self._client(monkeypatch, tmp_path, voice_config=cfg) as client:
            r = client.get("/voice/ambience/current")
            assert r.status_code == 200
            assert r.content == track.read_bytes()

    @pytest.mark.usefixtures("fake_cdn")
    def test_current_serves_configured_preset(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        with self._client(monkeypatch, tmp_path, voice_config="{'ambient': {'source': 'office'}}") as client:
            r = client.get("/voice/ambience/current")
            assert r.status_code == 200
            assert r.content == FAKE_WAV
