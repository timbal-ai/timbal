"""A real server killed with SIGKILL mid-tool, then started again: the session comes back on its own,
the dead turn is recovered by a read-only turn told what may have run, and nothing runs twice."""

import os
import signal
import socket
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import httpx
import pytest

AGENT = textwrap.dedent(
    """
    import asyncio, os
    from pathlib import Path
    from timbal import Agent, Tool
    from timbal.core.test_model import TestModel
    from timbal.state.tracing.providers import SqliteTracingProvider
    from timbal.types.content import ToolUseContent
    from timbal.types.message import Message

    DIR = Path(os.environ["E2E_DIR"])

    async def charge() -> str:
        with open(DIR / "charges.log", "a") as f:
            f.write("charged\\n")  # the side effect lands; the process dies before the result does
        while (DIR / "hang").exists():
            await asyncio.sleep(0.05)
        return "charged"

    def handler(messages):
        last = messages[-1]
        text = next((m.collect_text() for m in reversed(messages) if m.role == "user"), "")
        if "cut off by a server restart" in text:
            (DIR / "notice.txt").write_text(text)
            return "The server restarted while I was charging you; it may have gone through. I'll check first."
        if last.role == "tool":
            return "done"
        if text == "pay":
            return Message(role="assistant", content=[ToolUseContent(id="c1", name="charge", input={})],
                           stop_reason="tool_use")
        return f"echo: {text}"

    agent = Agent(
        name="agent",
        model=TestModel(handler=handler),
        tools=[Tool(name="charge", handler=charge)],
        tracing_provider=SqliteTracingProvider.configured(_path=DIR / "timbal.db"),
    )
    """
)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def until(predicate, timeout=20.0, what="condition"):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            value = predicate()
        except httpx.HTTPError:
            value = None
        if value:
            return value
        time.sleep(0.1)
    raise AssertionError(f"timed out waiting for {what}")


class Server:
    def __init__(self, workdir: Path):
        self.workdir = workdir
        self.port = free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.proc: subprocess.Popen | None = None

    def start(self):
        env = {**os.environ, "E2E_DIR": str(self.workdir), "TIMBAL_LOG_LEVEL": "WARNING"}
        env.pop("TIMBAL_CHAT_STORE", None)
        log = open(self.workdir / f"server-{time.time_ns()}.log", "w")  # noqa: SIM115
        self.proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "timbal.server",
                "--import_spec",
                f"{self.workdir / 'agent.py'}::agent",
                "--port",
                str(self.port),
            ],
            cwd=self.workdir,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        until(lambda: httpx.get(f"{self.url}/chat/meta", timeout=1).status_code == 200, 60, "the server")
        return self

    def kill(self):
        os.kill(self.proc.pid, signal.SIGKILL)
        self.proc.wait(10)

    def stop(self):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(15)
            except subprocess.TimeoutExpired:
                self.proc.kill()


@pytest.fixture
def workdir(tmp_path):
    (tmp_path / "agent.py").write_text(AGENT)
    return tmp_path


def test_a_session_survives_kill_9_mid_tool(workdir):
    server = Server(workdir).start()
    try:
        assert httpx.get(f"{server.url}/chat/meta").json()["durable"] is True
        session_id = httpx.post(f"{server.url}/chat/sessions", json={}).json()["session_id"]
        (workdir / "hang").touch()
        assert httpx.post(f"{server.url}/chat/sessions/{session_id}/messages", json={"text": "pay"}).status_code == 202
        until(lambda: (workdir / "charges.log").exists(), what="the charge to run")
        server.kill()
    finally:
        server.stop()
    (workdir / "hang").unlink()

    server.start()  # same port, a new process
    try:
        base = f"{server.url}/chat/sessions/{session_id}"
        # Restored at startup (it had a turn in flight), without any client asking.
        state = until(
            lambda: (s := httpx.get(base).json()) and s.get("turns") == 2 and not s["busy"] and s, what="recovery"
        )
        assert state["last_status"]["code"] == "success"
        notice = (workdir / "notice.txt").read_text()
        assert "Started but not confirmed" in notice and "charge(" in notice
        assert "can only read" in notice
        assert (workdir / "charges.log").read_text().count("charged") == 1  # never re-run

        log = httpx.get(f"{base}/log", params={"after": 0}).json()
        assert log["gapped"]  # the old process' events are not held; the turns are, in the traces
        kinds = [e["data"].get("kind") for e in log["events"] if isinstance(e["data"], dict)]
        assert "restored" in kinds
        started = [
            e["data"] for e in log["events"] if isinstance(e["data"], dict) and e["data"].get("kind") == "turn_started"
        ]
        assert started[0]["trigger"]["kind"] == "recovery" and started[0]["trigger"]["cut"] == "crash"

        # And the conversation goes on.
        assert httpx.post(f"{base}/messages", json={"text": "hello"}).status_code == 202
        until(lambda: httpx.get(base).json().get("turns") == 3 and not httpx.get(base).json()["busy"], what="turn 3")
    finally:
        server.stop()


def test_a_graceful_restart_keeps_queued_messages(workdir):
    server = Server(workdir).start()
    try:
        session_id = httpx.post(f"{server.url}/chat/sessions", json={}).json()["session_id"]
        (workdir / "hang").touch()
        httpx.post(f"{server.url}/chat/sessions/{session_id}/messages", json={"text": "pay"})
        until(lambda: (workdir / "charges.log").exists(), what="the charge to run")
        queued = httpx.post(
            f"{server.url}/chat/sessions/{session_id}/messages", json={"text": "later", "mode": "followup"}
        ).json()["message"]["event_id"]
    finally:
        server.stop()  # SIGTERM: lifespan shutdown keeps the session, cuts the turn
    (workdir / "hang").unlink()

    server.start()
    try:
        base = f"{server.url}/chat/sessions/{session_id}"
        until(lambda: httpx.get(base).json().get("turns") == 3 and not httpx.get(base).json()["busy"], what="turns")
        log = httpx.get(f"{base}/log", params={"after": 0}).json()
        settled = [
            e["data"]
            for e in log["events"]
            if isinstance(e["data"], dict) and e["data"].get("kind") == "message_settled"
        ]
        assert any(s["event_id"] == queued and s["state"] == "delivered" for s in settled)
        assert (workdir / "charges.log").read_text().count("charged") == 1
    finally:
        server.stop()
