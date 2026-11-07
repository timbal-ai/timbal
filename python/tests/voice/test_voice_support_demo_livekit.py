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


_ROOM = "pg-0123456789ab"


def _video_grant(req: urllib.request.Request) -> dict:
    token = (req.get_header("Authorization") or "").split(" ", 1)[-1]
    payload = token.split(".")[1]
    payload += "=" * (-len(payload) % 4)
    import base64

    claims = json.loads(base64.urlsafe_b64decode(payload))
    video = claims.get("video")
    return video if isinstance(video, dict) else {}


def _ok_opener(req: urllib.request.Request, timeout: float = 0) -> _Body:  # noqa: ARG001
    """LiveKit's real split: list needs roomList, delete needs roomCreate, else 401."""
    url = req.full_url
    if url.endswith("/twirp/livekit.RoomService/ListRooms"):
        if _video_grant(req).get("roomList") is not True:
            raise urllib.error.HTTPError(url, 401, "unauthenticated", hdrs=None, fp=None)
        return _Body(200, b'{"rooms":[{},{"name":"do-not-print"}]}')
    if url.endswith("/twirp/livekit.RoomService/DeleteRoom"):
        # roomAdmin is not enough. EnsureCreatePermission requires roomCreate.
        if _video_grant(req).get("roomCreate") is not True:
            raise urllib.error.HTTPError(url, 401, "unauthenticated", hdrs=None, fp=None)
        assert json.loads(req.data or b"{}")["room"] == _ROOM
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
        join=lambda: (True, f"worker_join=voice_livekit_joined room={_ROOM}", _ROOM),
    )
    text = "\n".join(lines)
    assert code == 0
    assert "checks=ok" in lines
    assert "ready" not in lines
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
    assert "checks=ok" not in lines
    assert "ready" not in lines


def test_room_admin_grant_is_not_delete_permission() -> None:
    from timbal.server.playground import mint_livekit_jwt

    token = mint_livekit_jwt(
        api_key="file-lk-key",
        api_secret="file-lk-secret",
        identity="timbal-demo-preflight",
        room=_ROOM,
        video={"room": _ROOM, "roomAdmin": True},
    )

    class _Req:
        full_url = "https://sfu.example/twirp/livekit.RoomService/DeleteRoom"
        data = json.dumps({"room": _ROOM}).encode()

        def get_header(self, name: str) -> str:  # noqa: ARG002
            return f"Bearer {token}"

    with pytest.raises(urllib.error.HTTPError) as raised:
        _ok_opener(_Req())
    assert raised.value.code == 401
    assert "file-lk-secret" not in token


def test_unrelated_room_is_not_sent(_env) -> None:
    def opener(req: urllib.request.Request, timeout: float = 0) -> _Body:  # noqa: ARG001
        raise AssertionError(req.full_url)

    assert live.delete_room("wss://super-secret-host.example", "k", "s", "customer-room", opener) == (
        "delete_room=refused"
    )
    assert "super-secret-host" not in "delete_room=refused"


def test_cleanup_auth_failure_is_not_checks_ok(_env) -> None:
    def opener(req: urllib.request.Request, timeout: float = 0) -> _Body:
        if req.full_url.endswith("/DeleteRoom"):
            raise urllib.error.HTTPError(req.full_url, 401, "unauthenticated", hdrs=None, fp=None)
        return _ok_opener(req, timeout)

    code, lines = live.run_checks(
        file_env=_file_env(),
        dotenv_present=True,
        opener=opener,
        importable=lambda: True,
        join=lambda: (True, f"worker_join=voice_livekit_joined room={_ROOM}", _ROOM),
    )
    assert code == 1
    assert "delete_room=auth status=401" in lines
    assert "blocked=delete_room" in lines
    assert "checks=ok" not in lines
    assert "ready" not in lines


def test_delete_404_is_already_gone(_env) -> None:
    def opener(req: urllib.request.Request, timeout: float = 0) -> _Body:
        if req.full_url.endswith("/DeleteRoom"):
            assert _video_grant(req).get("roomCreate") is True
            raise urllib.error.HTTPError(req.full_url, 404, "not found", hdrs=None, fp=None)
        return _ok_opener(req, timeout)

    code, lines = live.run_checks(
        file_env=_file_env(),
        dotenv_present=True,
        opener=opener,
        importable=lambda: True,
        join=lambda: (True, f"worker_join=voice_livekit_joined room={_ROOM}", _ROOM),
    )
    assert code == 0
    assert "delete_room=ok absent" in lines
    assert "checks=ok" in lines


def test_existing_launcher_is_not_a_browser_transport(_env) -> None:
    status = {
        "launcher": {"version": "0", "default_import_spec": "examples/voice_support_demo.py::agent"},
        "child": {
            "state": "running",
            "transport": "webrtc",
            "livekit": {"url": "wss://super-secret-host.example", "token": "eyJhbGciOiJIUzI1NiJ9.e30.sig"},
            "logs": ["voice_rtc_connected"],
        },
    }
    lines = live.describe_existing(status)
    text = "\n".join(lines)
    assert lines[0] == "playground=http://localhost:7777/?transport=livekit existing"
    assert "browser_transport=not_observable" in lines
    assert "child_transport=webrtc" in lines
    assert "child_logs=launcher_memory" in lines
    assert "super-secret-host" not in text
    assert "eyJ" not in text
    assert "voice_rtc_connected" not in text
    assert "ready" not in lines


def test_page_query_beats_saved_transport() -> None:
    html = (ROOT / "python" / "timbal" / "server" / "voice.html").read_text(encoding="utf-8")
    query_at = html.index("transportSelect.value = transportFromUrl || storedTransport || 'webrtc'")
    fallback_at = html.index("transportSelect.value = 'ws'")
    assert query_at < fallback_at
    assert "localStorage.setItem(TRANSPORT_STORAGE_KEY, transportFromUrl)" in html


def test_child_terminal_line_keeps_the_event_and_hides_the_url() -> None:
    from timbal.server.playground import playground_page_url, redact_child_log

    raw = (
        '{"event": "voice_livekit_joined", "url": "wss://super-secret-host.example"} '
        "eyJhbGciOiJIUzI1NiJ9.eyJyb29tIjoicGcifQ.sig"
    )
    text = redact_child_log(raw)
    assert "voice_livekit_joined" in text
    assert "super-secret-host" not in text
    assert "eyJ" not in text
    assert playground_page_url("127.0.0.1", 7777, "livekit") == "http://localhost:7777/?transport=livekit"


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


def test_thinking_datetime_tool_is_only_for_a_date_question() -> None:
    spec = importlib.util.spec_from_file_location(
        "voice_playground_agents",
        ROOT / "examples" / "voice_playground_agents.py",
    )
    assert spec and spec.loader
    agents = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(agents)
    prompt = agents.thinking.system_prompt
    assert "only when the caller asked for the date or the time" in prompt
    assert "Do not call it for any other question." in prompt
    assert [tool.name for tool in agents.thinking.tools] == ["get_datetime"]
    assert agents.thinking.voice_config["filler"]["model"] == "groq/qwen/qwen3.8-27b"
    if not os.environ.get("TIMBAL_VOICE_DEMO_MODEL"):
        assert agents.thinking.model == "groq/qwen/qwen3.8-27b"


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
