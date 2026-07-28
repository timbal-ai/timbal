"""Platform push for call recordings (option C of the recording handoff).

Contract under test: PUT multipart via ``platform.utils._request`` (auth and
host from ``resolve_platform_config``) to
``orgs/{org}/projects/{project}/voice-recordings/{session_id}``, delete
local files only on 2xx, keep them on 4xx (no retry) and on exhausted
5xx/network retries.
"""

# ruff: noqa: ARG001 — mock handlers/uploaders must match real signatures
from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import pytest
import timbal.state.config_loader as config_loader
from timbal.server.recording_upload import (
    _upload_tasks,
    platform_recording_upload_hook,
    upload_recording,
)
from timbal.state.config_loader import resolve_platform_config

PLATFORM_ENV = {
    "TIMBAL_API_HOST": "api.timbal.test",
    "TIMBAL_ORG_ID": "org1",
    "TIMBAL_PROJECT_ID": "proj1",
    "TIMBAL_API_TOKEN": "tok-123",
}

PATH = "orgs/org1/projects/proj1/sessions/sess1"
URL = f"https://api.timbal.test/{PATH}"


@pytest.fixture(autouse=True)
def _platform_env(monkeypatch: pytest.MonkeyPatch):
    """Point platform config at the test platform; restore the module cache after."""
    saved = (config_loader._cached_default_config, config_loader._default_config_resolved)
    for k in ("TIMBAL_API_KEY", "TIMBAL_APP_ID"):
        monkeypatch.delenv(k, raising=False)
    for k, v in PLATFORM_ENV.items():
        monkeypatch.setenv(k, v)
    resolve_platform_config(force_refresh=True)
    yield
    config_loader._cached_default_config, config_loader._default_config_resolved = saved


def _files(tmp_path: Path, session_id: str = "sess1") -> tuple[Path, Path]:
    audio = tmp_path / f"{session_id}.mp3"
    manifest = tmp_path / f"{session_id}.json"
    audio.write_bytes(b"\xff\xfbmp3data")
    manifest.write_text('{"session_id": "sess1"}')
    return audio, manifest


def _mock_http(monkeypatch: pytest.MonkeyPatch, statuses: list[int]) -> list[httpx.Request]:
    """Route all httpx clients through a MockTransport replying with *statuses* in order."""
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        status = statuses[min(len(requests) - 1, len(statuses) - 1)]
        if isinstance(status, int):
            return httpx.Response(status)
        raise status  # an exception instance → simulate a network error

    transport = httpx.MockTransport(handler)
    real_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real_client(**{**kw, "transport": transport}))
    return requests


class TestUploadRecording:
    async def test_2xx_uploads_multipart_and_deletes_files(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        audio, manifest = _files(tmp_path)
        requests = _mock_http(monkeypatch, [200])
        ok = await upload_recording(audio, manifest, path=PATH)
        assert ok
        assert not audio.exists() and not manifest.exists()

        (req,) = requests
        assert req.method == "PUT"
        assert str(req.url) == URL
        assert req.headers["authorization"] == "Bearer tok-123"  # from resolve_platform_config
        assert req.headers["content-type"].startswith("multipart/form-data")
        body = req.read()
        assert b'name="manifest"' in body and b"application/json" in body
        assert b'name="audio"' in body and b"audio/mpeg" in body
        assert b"\xff\xfbmp3data" in body

    @pytest.mark.parametrize("status", [400, 401, 403, 404, 409, 413])
    async def test_4xx_keeps_files_and_never_retries(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, status: int
    ) -> None:
        """The contract's permanent failures: bad parts / auth / project / ownership / size."""
        audio, manifest = _files(tmp_path)
        requests = _mock_http(monkeypatch, [status])
        ok = await upload_recording(audio, manifest, path=PATH)
        assert not ok
        assert audio.exists() and manifest.exists()
        assert len(requests) == 1

    async def test_5xx_retries_with_backoff_then_succeeds(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        audio, manifest = _files(tmp_path)
        requests = _mock_http(monkeypatch, [500, 502, 201])
        ok = await upload_recording(audio, manifest, path=PATH, backoff=lambda _: 0.01)
        assert ok
        assert len(requests) == 3
        assert not audio.exists() and not manifest.exists()

    async def test_network_errors_are_retryable(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        audio, manifest = _files(tmp_path)
        requests = _mock_http(monkeypatch, [httpx.ConnectError("boom"), 200])  # type: ignore[list-item]
        ok = await upload_recording(audio, manifest, path=PATH, backoff=lambda _: 0.01)
        assert ok
        assert len(requests) == 2

    async def test_exhausted_retries_give_up_and_keep_files(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        audio, manifest = _files(tmp_path)
        requests = _mock_http(monkeypatch, [500])
        ok = await upload_recording(audio, manifest, path=PATH, max_retries=2, backoff=lambda _: 0.01)
        assert not ok
        assert audio.exists() and manifest.exists()
        assert len(requests) == 3  # initial + 2 retries

    async def test_missing_files_abort_without_request(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        requests = _mock_http(monkeypatch, [200])
        ok = await upload_recording(tmp_path / "gone.mp3", tmp_path / "gone.json", path=PATH)
        assert not ok and not requests


class TestPlatformHookFactory:
    def test_incomplete_platform_config_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("TIMBAL_PROJECT_ID", raising=False)  # org present, project missing
        assert platform_recording_upload_hook() is None

    async def test_hook_builds_path_from_platform_config_and_schedules_upload(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        captured: dict = {}

        async def _fake_upload(audio_path, manifest_path, *, path, **kwargs):
            captured.update(audio=audio_path, manifest=manifest_path, path=path)
            return True

        monkeypatch.setattr("timbal.server.recording_upload.upload_recording", _fake_upload)
        hook = platform_recording_upload_hook()
        assert hook is not None

        audio, manifest = _files(tmp_path, session_id="abc42")

        class _Result:
            audio_path = audio
            manifest_path = manifest

        await hook(_Result())  # returns immediately — the upload is a background task
        await asyncio.gather(*_upload_tasks)

        assert captured["path"] == "orgs/org1/projects/proj1/sessions/abc42"
        assert captured["audio"] == audio and captured["manifest"] == manifest

    async def test_hook_skips_when_manifest_missing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """mp3-without-json = crashed call: leave it for the sweeper, don't push."""
        called = False

        async def _fake_upload(*args, **kwargs):
            nonlocal called
            called = True

        monkeypatch.setattr("timbal.server.recording_upload.upload_recording", _fake_upload)
        hook = platform_recording_upload_hook()

        class _Result:
            audio_path = tmp_path / "x.mp3"
            manifest_path = None

        await hook(_Result())
        await asyncio.gather(*_upload_tasks)
        assert not called
