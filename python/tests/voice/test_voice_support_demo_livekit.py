"""Preflight for the LiveKit Norte Market demo. No real credentials or rooms."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
_SPEC = importlib.util.spec_from_file_location(
    "voice_support_demo_livekit",
    ROOT / "examples" / "voice_support_demo_livekit.py",
)
assert _SPEC and _SPEC.loader
live = importlib.util.module_from_spec(_SPEC)
sys.modules["voice_support_demo_livekit"] = live
_SPEC.loader.exec_module(live)


class _Body:
    def __init__(self, status: int, body: bytes) -> None:
        self.status = status
        self._body = body

    def read(self, _n: int) -> bytes:
        return self._body

    def __enter__(self) -> _Body:
        return self

    def __exit__(self, *_args: object) -> bool:
        return False


def _ok_opener(req: urllib.request.Request, timeout: float = 0) -> _Body:  # noqa: ARG001
    url = req.full_url
    if url.endswith("/twirp/livekit.RoomService/ListRooms"):
        assert (req.get_header("Authorization") or "").startswith("Bearer ")
        return _Body(200, b'{"rooms":[{},{"name":"do-not-print"}]}')
    if url.endswith("/twirp/livekit.RoomService/DeleteRoom"):
        assert b"pg-abc" in (req.data or b"")
        return _Body(200, b"{}")
    if url.startswith("https://api.openai.com/"):
        return _Body(200, b'{"data":[{"id":"gpt-5.4-mini"},{"id":"other-model"}]}')
    if url.startswith("https://api.deepgram.com/"):
        return _Body(200, b"{}")
    if url.startswith("https://api.elevenlabs.io/"):
        return _Body(200, b'{"email":"hidden@example.com"}')
    raise AssertionError(url)


@pytest.fixture
def _env():
    saved = dict(os.environ)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved)


def _file_env() -> dict[str, str]:
    return {
        "OPENAI_API_KEY": "file-openai",
        "DEEPGRAM_API_KEY": "file-deepgram",
        "ELEVENLABS_API_KEY": "file-eleven",
        "LIVEKIT_API_KEY": "file-lk-key",
        "LIVEKIT_API_SECRET": "file-lk-secret",
        "TIMBAL_LIVEKIT_URL": "wss://super-secret-host.example",
    }


def test_file_wins_and_report_is_redacted(_env, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYTHONPATH", "/tmp/other-timbal")
    monkeypatch.setenv("OPENAI_API_KEY", "shell-openai-stale")
    monkeypatch.setenv("LIVEKIT_INFERENCE_API_KEY", "inference-secret")
    code, lines = live.run_checks(
        file_env=_file_env(),
        dotenv_present=True,
        opener=_ok_opener,
        importable=lambda: True,
        join=lambda: (True, "worker_join=voice_livekit_joined room=pg-abc", "pg-abc"),
    )
    text = "\n".join(lines)
    assert code == 0
    assert "ready" in lines
    assert "OPENAI_API_KEY both_differ file_wins" in lines
    assert "TIMBAL_LIVEKIT_URL scheme=wss host=present" in lines
    assert "LIVEKIT_URL absent_ok" in lines
    assert "LIVEKIT_INFERENCE_API_KEY unused" in lines
    assert "openai=ok model=listed" in lines
    assert "list_rooms=ok count=2" in lines
    assert "delete_room=ok" in lines
    assert "dial=" not in text
    for secret in (
        "file-openai",
        "shell-openai-stale",
        "file-lk-secret",
        "super-secret-host",
        "do-not-print",
        "hidden@example.com",
        "inference-secret",
    ):
        assert secret not in text
    assert os.environ["OPENAI_API_KEY"] == "file-openai"
    assert "PYTHONPATH" not in os.environ
    assert os.environ["UV_FROZEN"] == "1"
    assert os.environ["UV_NO_SYNC"] == "1"


def test_public_ws_url_blocks_before_the_join(_env) -> None:
    env = _file_env()
    env["TIMBAL_LIVEKIT_URL"] = "ws://super-secret-host.example:7880"
    code, lines = live.run_checks(
        file_env=env,
        dotenv_present=True,
        opener=_ok_opener,
        importable=lambda: True,
        join=lambda: (_ for _ in ()).throw(AssertionError("join")),
    )
    text = "\n".join(lines)
    assert code == 1
    assert "scheme=ws host=public_rejected" in text
    assert "blocked=config_package_or_provider" in lines
    assert "super-secret-host" not in text


def test_missing_livekit_package_still_checks_providers(_env) -> None:
    code, lines = live.run_checks(
        file_env=_file_env(),
        dotenv_present=True,
        opener=_ok_opener,
        importable=lambda: False,
        join=lambda: (_ for _ in ()).throw(AssertionError("join")),
    )
    assert code == 1
    assert "livekit_package=missing" in lines
    assert "openai=ok model=listed" in lines
    assert any(line.startswith("run: uv sync --frozen") for line in lines)
    assert "ready" not in lines


def test_room_service_auth_is_not_a_join(_env) -> None:
    def opener(req: urllib.request.Request, timeout: float = 0) -> _Body:  # noqa: ARG001
        if "livekit.RoomService" in req.full_url:
            raise urllib.error.HTTPError(req.full_url, 401, "no", hdrs=None, fp=None)
        return _ok_opener(req, timeout)

    code, lines = live.run_checks(
        file_env=_file_env(),
        dotenv_present=True,
        opener=opener,
        importable=lambda: True,
        join=lambda: (_ for _ in ()).throw(AssertionError("join")),
    )
    text = "\n".join(lines)
    assert code == 1
    assert "list_rooms=auth status=401" in lines
    assert "blocked=livekit_room_service" in lines
    assert "super-secret-host" not in text


def test_openai_auth_is_distinct_from_transport(_env) -> None:
    def opener(req: urllib.request.Request, timeout: float = 0) -> _Body:  # noqa: ARG001
        if "api.openai.com" in req.full_url:
            raise urllib.error.HTTPError(req.full_url, 401, "no", hdrs=None, fp=None)
        if "api.deepgram.com" in req.full_url:
            raise urllib.error.URLError("dns")
        return _ok_opener(req, timeout)

    _code, lines = live.run_checks(
        file_env=_file_env(),
        dotenv_present=True,
        opener=opener,
        importable=lambda: True,
        join=lambda: (_ for _ in ()).throw(AssertionError("join")),
    )
    assert "openai=auth status=401" in lines
    assert "deepgram=transport" in lines
    assert "blocked=config_package_or_provider" in lines


def test_join_event_and_redaction_hide_the_dial() -> None:
    line = "event voice_livekit_joined url=wss://super-secret-host.example token=eyJhbGciOiJIUzI1NiJ9.eyJyb29tIjoicGctMSJ9.c2ln"
    assert live.first_join_event([line]) == "voice_livekit_joined"
    hidden = live.redact(line, ["wss://super-secret-host.example"])
    assert "super-secret-host" not in hidden
    assert "eyJ" not in hidden
    assert live.dial_report({"url": "wss://super-secret-host.example", "token": "a.b.c", "room": "pg-1"}) == (
        "dial url=present token=present room=pg-1"
    )


def test_room_list_grant_is_not_a_join_token() -> None:
    from timbal.server.playground import mint_livekit_jwt

    token = mint_livekit_jwt(
        api_key="APIxxx",
        api_secret="supersecret",
        identity="timbal-demo-preflight",
        room="",
        video={"roomList": True},
    )
    payload = token.split(".")[1]
    payload += "=" * (-len(payload) % 4)
    import base64

    claims = json.loads(base64.urlsafe_b64decode(payload))
    assert claims["video"] == {"roomList": True}
    assert "supersecret" not in token


def test_launcher_spawn_returns_a_dial(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from http.server import ThreadingHTTPServer

    from timbal.server.playground import ChildServer, _make_handler

    agent = tmp_path / "agent.py"
    agent.write_text("agent = None\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    for name in (
        "TIMBAL_LIVEKIT_URL",
        "LIVEKIT_URL",
        "LIVEKIT_API_KEY",
        "TIMBAL_LIVEKIT_API_KEY",
        "LIVEKIT_API_SECRET",
        "TIMBAL_LIVEKIT_API_SECRET",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TIMBAL_LIVEKIT_URL", "wss://super-secret-host.example")
    monkeypatch.setenv("LIVEKIT_API_KEY", "file-lk-key")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "file-lk-secret")

    def fake_popen(cmd, cwd=None, env=None, **kwargs):  # noqa: ARG001
        class _Proc:
            pid = 1
            stdout = iter(())

            def poll(self) -> None:
                return None

            def terminate(self) -> None:
                return None

            def kill(self) -> None:
                return None

            def wait(self, timeout: float | None = None) -> int:  # noqa: ARG002
                return 0

        assert "--extra" in cmd and "voice-livekit" in cmd
        assert env is not None and env.get("TIMBAL_LIVEKIT_TOKEN")
        return _Proc()

    monkeypatch.setattr("timbal.server.playground.shutil.which", lambda _name: "/usr/bin/uv")
    monkeypatch.setattr("timbal.server.playground.subprocess.Popen", fake_popen)
    monkeypatch.setattr("timbal.server.playground._healthy", lambda _port: False)

    child = ChildServer()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(child, ""))
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        body = json.dumps({"import_spec": f"{agent}::agent", "transport": "livekit"}).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/launcher/spawn",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            payload = json.load(resp)
    finally:
        httpd.shutdown()
        child.stop()
        httpd.server_close()

    dial = payload["child"]["livekit"]
    report = live.dial_report(dial)
    assert report.startswith("dial url=present token=present room=pg-")
    hidden = live.redact(json.dumps(payload), ["file-lk-key", "file-lk-secret", "wss://super-secret-host.example"])
    assert "file-lk-secret" not in hidden
    assert "super-secret-host" not in hidden
    assert "eyJ" not in hidden
