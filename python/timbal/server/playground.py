"""Standalone voice playground launcher: ``python -m timbal.server.playground``.

Serves ``voice.html`` *without* a running agent underneath. The page detects it
was served raw (no injected runnable meta) and switches to standalone mode,
where you pick a target:

* **Local server** — pick an agent file (``path/to/agent.py::object``) and
  press Start: the page asks *this* launcher to spawn ``uv run python -m
  timbal.server --import_spec … --port …`` from the agent file's directory (so
  ``uv`` resolves that project's environment and ``.env``), waits for the
  healthcheck, and dials it. The port is picked automatically unless fixed in
  the form. Changing the agent or port respawns on the next Start. Transport
  **LiveKit** is local-only: the launcher mints a room + caller JWT (stdlib
  HMAC, no PyJWT) from ``LIVEKIT_API_KEY`` / ``LIVEKIT_API_SECRET`` /
  ``TIMBAL_LIVEKIT_URL`` (or ``LIVEKIT_URL``) in the env or the agent's
  ``.env``. Use ``wss://lk.timbal.ai`` — port 7880 is VPC-only. The child
  gets ``TIMBAL_VOICE_SINGLE_SESSION=1`` so Stop/Start respawns a fresh
  driver. Agent extra: ``timbal[voice-livekit]``.
* **Platform** — a deployed workforce through ``api.timbal.ai`` /
  ``api.dev.timbal.ai``. The page talks to the platform directly (ticket mint
  for WS, bearer-authenticated POST for RTC); the launcher is not involved.

Deliberately stdlib-only: the launcher must work even when no agent project
(and therefore no fastapi/uvicorn environment) is set up yet.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import os
import shutil
import socket
import subprocess
import threading
import time
import uuid
import webbrowser
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.request import urlopen

from .. import __version__

_HTML_PATH = Path(__file__).parent / "voice.html"

_LOG_LINES = 400
_STOP_GRACE_SECS = 5.0

def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
        return value[1:-1]
    return value


def _read_dotenv(path: Path) -> dict[str, str]:
    """Minimal KEY=VALUE reader — launcher stays stdlib-only."""
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, _, value = line.partition("=")
        key = key.strip()
        if key:
            out[key] = _unquote(value.strip())
    return out


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def mint_livekit_jwt(
    *,
    api_key: str,
    api_secret: str,
    identity: str,
    room: str,
    ttl_secs: int = 3600,
    name: str | None = None,
) -> str:
    """HS256 LiveKit access token. Grant keys are camelCase, matching livekit-server."""
    now = int(time.time())
    header = _b64url(json.dumps({"alg": "HS256", "typ": "JWT"}, separators=(",", ":")).encode())
    payload = _b64url(
        json.dumps(
            {
                "iss": api_key,
                "sub": identity,
                "name": name or identity,
                "nbf": now,
                "iat": now,
                "exp": now + ttl_secs,
                "jti": uuid.uuid4().hex,
                "video": {
                    "roomJoin": True,
                    "room": room,
                    "canPublish": True,
                    "canSubscribe": True,
                    "canPublishData": True,
                },
            },
            separators=(",", ":"),
        ).encode()
    )
    sig = hmac.new(api_secret.encode(), f"{header}.{payload}".encode(), hashlib.sha256).digest()
    return f"{header}.{payload}.{_b64url(sig)}"


def livekit_creds(project_dir: Path) -> tuple[str, str, str]:
    """``(url, api_key, api_secret)``. Process env wins over the agent's ``.env``."""
    file_env = _read_dotenv(project_dir / ".env")

    def pick(*names: str) -> str:
        for n in names:
            v = os.environ.get(n)
            if v:
                return v
        for n in names:
            v = file_env.get(n)
            if v:
                return v
        return ""

    return (
        pick("TIMBAL_LIVEKIT_URL", "LIVEKIT_URL"),
        pick("LIVEKIT_API_KEY", "TIMBAL_LIVEKIT_API_KEY"),
        pick("LIVEKIT_API_SECRET", "TIMBAL_LIVEKIT_API_SECRET"),
    )


_LIVEKIT_CHILD_ENV_KEYS = (
    "TIMBAL_VOICE_TRANSPORT",
    "TIMBAL_LIVEKIT_URL",
    "TIMBAL_LIVEKIT_TOKEN",
    "TIMBAL_LIVEKIT_ROOM",
    "TIMBAL_LIVEKIT_CALLER_IDENTITY",
    "TIMBAL_VOICE_SINGLE_SESSION",
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _healthy(port: int) -> bool:
    try:
        with urlopen(f"http://127.0.0.1:{port}/healthcheck", timeout=0.5) as resp:
            return resp.status in (200, 204)
    except Exception:
        return False


class ChildServer:
    """The single ``timbal.server`` subprocess this launcher manages."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._port: int | None = None
        self._import_spec: str | None = None
        self._requested_spec: str | None = None
        """The spec exactly as the client sent it — echoed in status so the page
        can tell whether the running child matches its form without having to
        replicate the launcher's path resolution."""
        self._logs: deque[str] = deque(maxlen=_LOG_LINES)
        self._transport: str | None = None
        self._livekit: dict[str, str] | None = None

    def _reader(self, proc: subprocess.Popen) -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            self._logs.append(line.rstrip("\n"))

    def spawn(self, import_spec: str, port: int | None = None, transport: str = "ws") -> dict:
        spec = import_spec.strip()
        parts = spec.split("::")
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise ValueError("Import spec must look like 'path/to/agent.py::object_name'.")
        spec_path = Path(parts[0]).expanduser()
        if not spec_path.is_absolute():
            spec_path = Path.cwd() / spec_path
        spec_path = spec_path.resolve()
        if not spec_path.is_file():
            raise ValueError(f"No such file: {spec_path}")
        if shutil.which("uv") is None:
            raise ValueError("'uv' was not found on PATH — install uv or start the server manually.")
        resolved_spec = f"{spec_path}::{parts[1]}"
        kind = (transport or "ws").strip().lower()
        livekit_plan: tuple[str, str, str] | None = None
        if kind == "livekit":
            livekit_plan = livekit_creds(spec_path.parent)
            if not all(livekit_plan):
                raise ValueError(
                    "LiveKit needs TIMBAL_LIVEKIT_URL (or LIVEKIT_URL) plus "
                    "LIVEKIT_API_KEY and LIVEKIT_API_SECRET in the environment "
                    "or the agent's .env. Browser clients must use wss://… "
                    "(port 7880 is VPC-only)."
                )

        with self._lock:
            self._stop_locked()
            child_port = port or _free_port()
            cmd = [
                "uv",
                "run",
                "python",
                "-m",
                "timbal.server",
                "--import_spec",
                resolved_spec,
                "--host",
                "127.0.0.1",
                "--port",
                str(child_port),
            ]
            self._logs.clear()
            self._logs.append(f"$ {' '.join(cmd)}  (cwd: {spec_path.parent})")
            # cwd = the agent file's directory: `uv run` walks up from there to
            # find the agent project's pyproject/venv, and load_dotenv picks up
            # that project's .env — not the launcher's.
            # TIMBAL_VOICE_WARMUP=1: playground children pre-load the voice
            # stack so picking "Smart Turn" on first Start doesn't eat the
            # ONNX/HuggingFace cold path; production servers gate warmup on
            # actual voice intent (see server.voice.voice_warmup_intended).
            env = {**os.environ, "TIMBAL_VOICE_WARMUP": "1"}
            for k in _LIVEKIT_CHILD_ENV_KEYS:
                env.pop(k, None)
            livekit: dict[str, str] | None = None
            if livekit_plan is not None:
                url, api_key, api_secret = livekit_plan
                room = f"pg-{uuid.uuid4().hex[:12]}"
                agent_id = "agent"
                caller_id = "playground"
                env.update(
                    {
                        "TIMBAL_VOICE_TRANSPORT": "livekit",
                        "TIMBAL_LIVEKIT_URL": url,
                        "TIMBAL_LIVEKIT_TOKEN": mint_livekit_jwt(
                            api_key=api_key,
                            api_secret=api_secret,
                            identity=agent_id,
                            room=room,
                        ),
                        "TIMBAL_LIVEKIT_ROOM": room,
                        "TIMBAL_LIVEKIT_CALLER_IDENTITY": caller_id,
                        "TIMBAL_VOICE_SINGLE_SESSION": "1",
                    }
                )
                livekit = {
                    "url": url,
                    "token": mint_livekit_jwt(
                        api_key=api_key,
                        api_secret=api_secret,
                        identity=caller_id,
                        room=room,
                    ),
                    "identity": caller_id,
                    "room": room,
                }
                self._logs.append(f"livekit room {room} @ {url}")
            proc = subprocess.Popen(  # noqa: S603
                cmd,
                cwd=spec_path.parent,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            self._proc = proc
            self._port = child_port
            self._import_spec = resolved_spec
            self._requested_spec = spec
            self._transport = kind
            self._livekit = livekit
            threading.Thread(target=self._reader, args=(proc,), daemon=True).start()
            return self.status_locked()

    def stop(self) -> dict:
        with self._lock:
            self._stop_locked()
            return self.status_locked()

    def _stop_locked(self) -> None:
        proc = self._proc
        if proc is None:
            return
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=_STOP_GRACE_SECS)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=_STOP_GRACE_SECS)
        self._proc = None
        self._port = None
        self._import_spec = None
        self._requested_spec = None
        self._transport = None
        self._livekit = None

    def status(self) -> dict:
        with self._lock:
            return self.status_locked()

    def status_locked(self) -> dict:
        proc = self._proc
        if proc is None:
            # Keep the last logs around so a crash is inspectable after stop.
            return {"state": "stopped", "logs": list(self._logs)}
        extra: dict = {}
        if self._transport:
            extra["transport"] = self._transport
        if self._livekit:
            extra["livekit"] = self._livekit
        exit_code = proc.poll()
        if exit_code is not None:
            return {
                "state": "exited",
                "exit_code": exit_code,
                "port": self._port,
                "import_spec": self._import_spec,
                "requested_import_spec": self._requested_spec,
                "logs": list(self._logs),
                **extra,
            }
        state = "running" if _healthy(self._port) else "starting"
        return {
            "state": state,
            "pid": proc.pid,
            "port": self._port,
            "import_spec": self._import_spec,
            "requested_import_spec": self._requested_spec,
            "logs": list(self._logs),
            **extra,
        }


def _make_handler(child: ChildServer, default_import_spec: str) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = f"timbal-playground/{__version__}"

        def log_message(self, format: str, *args) -> None:  # noqa: A002
            pass  # keep the terminal usable; the page shows child logs itself

        def _send_json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0].rstrip("/") or "/"
            if path in ("/", "/voice"):
                body = _HTML_PATH.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if path == "/launcher/status":
                self._send_json(
                    200,
                    {
                        "launcher": {"version": __version__, "default_import_spec": default_import_spec},
                        "child": child.status(),
                    },
                )
                return
            self._send_json(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0].rstrip("/")
            length = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                self._send_json(400, {"error": "invalid JSON body"})
                return
            if path == "/launcher/spawn":
                import_spec = str(body.get("import_spec") or "").strip()
                if not import_spec:
                    self._send_json(400, {"error": "import_spec is required"})
                    return
                port = body.get("port")
                transport = str(body.get("transport") or "ws").strip().lower()
                try:
                    status = child.spawn(
                        import_spec,
                        port=int(port) if port else None,
                        transport=transport,
                    )
                except ValueError as e:
                    self._send_json(400, {"error": str(e)})
                    return
                except OSError as e:
                    self._send_json(500, {"error": f"spawn failed: {e}"})
                    return
                self._send_json(200, {"child": status})
                return
            if path == "/launcher/stop":
                self._send_json(200, {"child": child.stop()})
                return
            self._send_json(404, {"error": "not found"})

    return Handler


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Standalone Timbal voice playground.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the playground to.")
    parser.add_argument("--port", type=int, default=7777, help="Port to bind the playground to.")
    parser.add_argument(
        "--import_spec",
        "--import-spec",
        dest="import_spec",
        type=str,
        default="",
        help="Default import spec prefilled in the page's spawn field (path/to/agent.py::object).",
    )
    parser.add_argument("--no-open", action="store_true", help="Don't open the browser automatically.")
    args = parser.parse_args(argv)

    child = ChildServer()
    handler = _make_handler(child, args.import_spec)
    httpd = ThreadingHTTPServer((args.host, args.port), handler)
    url = f"http://{'localhost' if args.host in ('0.0.0.0', '127.0.0.1') else args.host}:{args.port}/"
    print(f"Voice playground at {url}")  # noqa: T201
    if not args.no_open:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        child.stop()
        httpd.server_close()


if __name__ == "__main__":
    main()
