"""Preflight and launch for the Norte Market demo on the LiveKit SFU.

The shop stays the in-memory simulation in ``voice_support_demo.py``.
Audio uses the production playground path: this process becomes
``python -m timbal.server.playground``, Start mints a room, and the
spawned ``timbal.server`` joins that room with ``livekit.rtc``. There is
no separate LiveKit Agents worker and no phone trunk.

Run from the repository root, with no ``PYTHONPATH``::

    uv sync --frozen --extra server --extra voice --extra voice-livekit
    uv run --frozen python examples/voice_support_demo_livekit.py

``--frozen`` does not rewrite ``uv.lock``. A root ``.env`` value wins
over a same-named shell export. The report prints names and statuses,
never values, URLs, or tokens.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / ".env"
IMPORT_SPEC = "examples/voice_support_demo.py::agent"
PLAYGROUND_PORT = 7777
JOIN_TIMEOUT_SECS = 45.0
HTTP_TIMEOUT_SECS = 10.0

# Read by the playground launcher. LIVEKIT_URL is the alias, not required
# when TIMBAL_LIVEKIT_URL is set. LIVEKIT_INFERENCE_* is a different product.
FILE_WINS = (
    "OPENAI_API_KEY",
    "DEEPGRAM_API_KEY",
    "ELEVENLABS_API_KEY",
    "LIVEKIT_API_KEY",
    "LIVEKIT_API_SECRET",
    "TIMBAL_LIVEKIT_URL",
    "LIVEKIT_URL",
    "ELEVENLABS_VOICE_ID",
    "TIMBAL_VOICE_ID",
)
REQUIRED = (
    "OPENAI_API_KEY",
    "DEEPGRAM_API_KEY",
    "ELEVENLABS_API_KEY",
    "LIVEKIT_API_KEY",
    "LIVEKIT_API_SECRET",
    "TIMBAL_LIVEKIT_URL",
)
UNUSED = ("LIVEKIT_INFERENCE_API_KEY", "LIVEKIT_INFERENCE_API_SECRET")
_DROP = (
    "PYTHONPATH",
    "TIMBAL_VOICE_TRANSPORT",
    "TIMBAL_LIVEKIT_TOKEN",
    "TIMBAL_LIVEKIT_ROOM",
    "TIMBAL_LIVEKIT_CALLER_IDENTITY",
    "TIMBAL_LIVEKIT_AGENT_IDENTITY",
)
_JWT = re.compile(r"eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+")
_JOIN_EVENTS = (
    "voice_livekit_missing_extra",
    "voice_livekit_missing_env",
    "voice_livekit_join_failed",
    "voice_livekit_startup_failed",
    "voice_livekit_joined",
)


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
        return value[1:-1]
    return value


def read_dotenv(path: Path) -> dict[str, str]:
    """KEY=VALUE reader. Matches the playground launcher, stdlib only."""
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


def classify(name: str, file_env: dict[str, str], environ: dict[str, str]) -> str:
    """How a name is set. Does not return the value."""
    file_set = bool(file_env.get(name))
    shell_set = bool(environ.get(name))
    if file_set and shell_set:
        return "both_same" if file_env[name] == environ[name] else "both_differ"
    if file_set:
        return "file"
    if shell_set:
        return "shell"
    return "missing"


def overlay_env(file_env: dict[str, str], base: dict[str, str] | None = None) -> dict[str, str]:
    """Copy ``base`` and let a nonempty root ``.env`` value replace the shell.

    Drops ``PYTHONPATH`` and any boot-env LiveKit dial left in the shell.
    ``UV_FROZEN`` and ``UV_NO_SYNC`` stop the spawned server from rewriting
    ``uv.lock`` or pruning the extras this script just checked.
    """
    out = dict(os.environ if base is None else base)
    for key, value in file_env.items():
        if value:
            out[key] = value
    for key in _DROP:
        out.pop(key, None)
    out["UV_FROZEN"] = "1"
    out["UV_NO_SYNC"] = "1"
    return out


def apply_overlay(overlay: dict[str, str]) -> None:
    for key in list(os.environ):
        if key not in overlay:
            del os.environ[key]
    os.environ.update(overlay)


def url_status(url: str) -> tuple[bool, str]:
    """Scheme check. The host is not included in the text."""
    from urllib.parse import urlparse

    parsed = urlparse(url)
    scheme = (parsed.scheme or "").lower()
    host = (parsed.hostname or "").lower()
    if not host:
        return False, "scheme=absent host=missing"
    local = host in {"localhost", "127.0.0.1", "::1"}
    if scheme == "wss":
        return True, "scheme=wss host=present"
    if scheme == "ws" and local:
        return True, "scheme=ws host=local"
    if scheme == "ws":
        return False, "scheme=ws host=public_rejected"
    return False, f"scheme={scheme or 'absent'} host=present"


def redact(text: str, secrets: list[str]) -> str:
    out = text
    for secret in sorted((s for s in secrets if s), key=len, reverse=True):
        out = out.replace(secret, "[redacted]")
    return _JWT.sub("[redacted]", out)


def http_origin(ws_url: str) -> str:
    from urllib.parse import urlparse

    parsed = urlparse(ws_url)
    scheme = "https" if parsed.scheme == "wss" else "http"
    return f"{scheme}://{parsed.netloc}"


def _request(
    url: str,
    *,
    headers: dict[str, str],
    body: bytes | None,
    opener: Any,
) -> tuple[str, int, bytes]:
    req = urllib.request.Request(url, data=body, headers=headers, method="POST" if body is not None else "GET")
    try:
        with opener(req, timeout=HTTP_TIMEOUT_SECS) as resp:
            return "ok", int(resp.status), resp.read(1_000_000)
    except urllib.error.HTTPError as exc:
        kind = "auth" if exc.code in (401, 403) else "http"
        return kind, int(exc.code), b""
    except (urllib.error.URLError, TimeoutError, OSError):
        return "transport", 0, b""


def list_rooms(url: str, api_key: str, api_secret: str, opener: Any = urllib.request.urlopen) -> str:
    """Documented RoomService ListRooms. Does not create or change a room."""
    from timbal.server.playground import mint_livekit_jwt

    token = mint_livekit_jwt(
        api_key=api_key,
        api_secret=api_secret,
        identity="timbal-demo-preflight",
        room="",
        video={"roomList": True},
    )
    kind, status, raw = _request(
        http_origin(url) + "/twirp/livekit.RoomService/ListRooms",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        body=b"{}",
        opener=opener,
    )
    if kind == "ok":
        try:
            rooms = json.loads(raw.decode() or "{}").get("rooms") or []
            count = len(rooms) if isinstance(rooms, list) else "unknown"
        except json.JSONDecodeError:
            count = "unparsed"
        return f"list_rooms=ok count={count}"
    if kind == "auth":
        return f"list_rooms=auth status={status}"
    if kind == "http":
        return f"list_rooms=http status={status}"
    return "list_rooms=transport"


def delete_room(url: str, api_key: str, api_secret: str, room: str, opener: Any = urllib.request.urlopen) -> str:
    """Delete only the preflight room this process created."""
    from timbal.server.playground import mint_livekit_jwt

    token = mint_livekit_jwt(
        api_key=api_key,
        api_secret=api_secret,
        identity="timbal-demo-preflight",
        room=room,
        video={"room": room, "roomAdmin": True},
    )
    kind, status, _raw = _request(
        http_origin(url) + "/twirp/livekit.RoomService/DeleteRoom",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        body=json.dumps({"room": room}).encode(),
        opener=opener,
    )
    if kind == "ok":
        return "delete_room=ok"
    if kind == "auth":
        return f"delete_room=auth status={status}"
    if kind == "http":
        return f"delete_room=http status={status}"
    return "delete_room=transport"


def check_openai(key: str, opener: Any = urllib.request.urlopen) -> str:
    kind, status, raw = _request(
        "https://api.openai.com/v1/models",
        headers={"Authorization": f"Bearer {key}"},
        body=None,
        opener=opener,
    )
    if kind != "ok":
        return f"openai={kind}" + (f" status={status}" if status else "")
    listed = "absent"
    try:
        data = json.loads(raw.decode() or "{}").get("data") or []
        ids = {item.get("id") for item in data if isinstance(item, dict)}
        if "gpt-5.4-mini" in ids:
            listed = "listed"
    except json.JSONDecodeError:
        listed = "unparsed"
    return f"openai=ok model={listed}"


def check_deepgram(key: str, opener: Any = urllib.request.urlopen) -> str:
    kind, status, _raw = _request(
        "https://api.deepgram.com/v1/projects",
        headers={"Authorization": f"Token {key}"},
        body=None,
        opener=opener,
    )
    if kind == "ok":
        return "deepgram=ok"
    return f"deepgram={kind}" + (f" status={status}" if status else "")


def check_elevenlabs(key: str, opener: Any = urllib.request.urlopen) -> str:
    kind, status, _raw = _request(
        "https://api.elevenlabs.io/v1/user",
        headers={"xi-api-key": key},
        body=None,
        opener=opener,
    )
    if kind == "ok":
        return "elevenlabs=ok"
    return f"elevenlabs={kind}" + (f" status={status}" if status else "")


def timbal_from_checkout() -> bool:
    import timbal

    return ROOT.resolve() in Path(timbal.__file__).resolve().parents


def livekit_importable() -> bool:
    uv = shutil.which("uv")
    if uv is None:
        return False
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["UV_FROZEN"] = "1"
    env["UV_NO_SYNC"] = "1"
    proc = subprocess.run(  # noqa: S603
        [uv, "run", "--frozen", "--no-sync", "python", "-c", "from livekit import rtc; assert hasattr(rtc, 'Room')"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    return proc.returncode == 0


def first_join_event(lines: list[str]) -> str | None:
    for line in lines:
        for event in _JOIN_EVENTS:
            if event in line:
                return event
    return None


def dial_report(livekit: dict[str, str] | None) -> str:
    if not livekit:
        return "dial=missing"
    room = str(livekit.get("room") or "")
    return (
        f"dial url={'present' if livekit.get('url') else 'missing'}"
        f" token={'present' if livekit.get('token') else 'missing'}"
        f" room={room}"
    )


def prove_worker_join(child: Any | None = None) -> tuple[bool, str, str]:
    """Spawn the real server into a fresh room and wait for ``voice_livekit_joined``.

    The caller microphone is not published, so Flux and ElevenLabs are not
    opened. Returns ``(ok, line, room)``.
    """
    from timbal.server.playground import ChildServer

    child = ChildServer() if child is None else child
    status = child.spawn(str(ROOT / "examples" / "voice_support_demo.py") + "::agent", transport="livekit")
    room = str((status.get("livekit") or {}).get("room") or "")
    deadline = time.monotonic() + JOIN_TIMEOUT_SECS
    event = first_join_event(status.get("logs") or [])
    try:
        while event is None and time.monotonic() < deadline:
            state = status.get("state")
            if state == "exited":
                return False, f"worker_join=exited code={status.get('exit_code')}", room
            if state == "stopped":
                return False, "worker_join=stopped", room
            time.sleep(0.25)
            status = child.status()
            event = first_join_event(status.get("logs") or [])
        if event == "voice_livekit_joined":
            return True, f"worker_join=voice_livekit_joined room={room}", room
        if event:
            return False, f"worker_join={event} room={room}", room
        return False, f"worker_join=timeout room={room}", room
    finally:
        child.stop()


def key_lines(file_env: dict[str, str], environ: dict[str, str]) -> tuple[list[str], dict[str, str]]:
    lines: list[str] = []
    chosen: dict[str, str] = {}
    for name in FILE_WINS:
        state = classify(name, file_env, environ)
        if name not in REQUIRED and state == "missing":
            if name == "LIVEKIT_URL":
                lines.append("LIVEKIT_URL absent_ok")
            continue
        suffix = " file_wins" if state == "both_differ" else ""
        lines.append(f"{name} {state}{suffix}")
        if file_env.get(name):
            chosen[name] = file_env[name]
        elif environ.get(name):
            chosen[name] = environ[name]
    for name in UNUSED:
        lines.append(f"{name} {'unused' if file_env.get(name) or environ.get(name) else 'absent_ok'}")
    return lines, chosen


def _provider_blocked(results: list[str]) -> bool:
    return any("=ok" not in line.split()[0] for line in results)


def run_checks(
    *,
    file_env: dict[str, str] | None = None,
    dotenv_present: bool | None = None,
    opener: Any = urllib.request.urlopen,
    importable: Any = livekit_importable,
    join: Any = prove_worker_join,
) -> tuple[int, list[str]]:
    lines: list[str] = []
    if os.environ.get("PYTHONPATH"):
        lines.append("PYTHONPATH dropped")
    if not timbal_from_checkout():
        lines.append("timbal_import=outside_checkout")
        lines.append("run: uv run --frozen python examples/voice_support_demo_livekit.py")
        return 1, lines
    if not (ROOT / "uv.lock").is_file() or not (ROOT / "pyproject.toml").is_file():
        lines.append("checkout=missing_project_file")
        return 1, lines

    loaded = read_dotenv(ENV_PATH) if file_env is None else file_env
    present = ENV_PATH.is_file() if dotenv_present is None else dotenv_present
    lines.append(f"dotenv={'present' if present else 'absent'}")
    key_report, chosen = key_lines(loaded, dict(os.environ))
    lines.extend(key_report)
    apply_overlay(overlay_env(loaded))

    missing = [name for name in REQUIRED if not chosen.get(name)]
    url = chosen.get("TIMBAL_LIVEKIT_URL", "")
    url_ok, url_line = url_status(url) if url else (False, "scheme=absent host=missing")
    lines.append(f"TIMBAL_LIVEKIT_URL {url_line}")

    providers: list[str] = []
    if all(chosen.get(name) for name in ("OPENAI_API_KEY", "DEEPGRAM_API_KEY", "ELEVENLABS_API_KEY")):
        providers = [
            check_openai(chosen["OPENAI_API_KEY"], opener),
            check_deepgram(chosen["DEEPGRAM_API_KEY"], opener),
            check_elevenlabs(chosen["ELEVENLABS_API_KEY"], opener),
        ]
        lines.extend(providers)
    else:
        lines.append("providers=skipped")

    package_ok = bool(importable())
    lines.append("livekit_package=ok" if package_ok else "livekit_package=missing")
    if not package_ok:
        lines.append("run: uv sync --frozen --extra server --extra voice --extra voice-livekit")

    if missing or not url_ok or not package_ok or not providers or _provider_blocked(providers):
        lines.append("blocked=config_package_or_provider")
        return 1, lines

    rooms = list_rooms(url, chosen["LIVEKIT_API_KEY"], chosen["LIVEKIT_API_SECRET"], opener)
    lines.append(rooms)
    if not rooms.startswith("list_rooms=ok"):
        lines.append("blocked=livekit_room_service")
        return 1, lines

    ok, join_line, room = join()
    lines.append(join_line)
    if room:
        lines.append(delete_room(url, chosen["LIVEKIT_API_KEY"], chosen["LIVEKIT_API_SECRET"], room, opener))
    if not ok:
        lines.append("blocked=worker_join")
        return 1, lines
    lines.append("ready")
    return 0, lines


def _port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def launch_playground() -> None:
    uv = shutil.which("uv")
    if uv is None:
        print("uv=missing", file=sys.stderr)  # noqa: T201
        raise SystemExit(1)
    port = PLAYGROUND_PORT
    if not _port_free(port):
        print(f"playground_port={port} in_use", file=sys.stderr)  # noqa: T201
        raise SystemExit(1)
    print(f"playground=http://127.0.0.1:{port}/")  # noqa: T201
    os.chdir(ROOT)
    os.execvpe(  # noqa: S606
        uv,
        [
            uv,
            "run",
            "--frozen",
            "--no-sync",
            "python",
            "-m",
            "timbal.server.playground",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--import_spec",
            IMPORT_SPEC,
        ],
        os.environ,
    )


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    code, lines = run_checks()
    secrets = [os.environ.get(name, "") for name in FILE_WINS]
    for line in lines:
        print(redact(line, secrets))  # noqa: T201
    if code != 0 or "--check" in args:
        return code
    launch_playground()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
