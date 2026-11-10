"""A real server killed with SIGKILL mid-tool, then started again: the session comes back on its own,
the dead turn is recovered by a read-only turn told what may have run, and nothing runs twice."""

import os
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
    def __init__(self, workdir: Path, *, workers: int = 1, agent_file: str = "agent.py"):
        self.workdir = workdir
        self.port = free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.proc: subprocess.Popen | None = None
        self.workers = workers
        self.agent_file = agent_file

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
                f"{self.workdir / self.agent_file}::agent",
                "--port",
                str(self.port),
                "--workers",
                str(self.workers),
            ],
            cwd=self.workdir,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        until(lambda: httpx.get(f"{self.url}/chat/meta", timeout=1).status_code == 200, 60, "the server")
        return self

    def kill(self):
        self.proc.kill()  # SIGKILL; TerminateProcess on Windows: no shutdown either way
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


@pytest.mark.skipif(sys.platform == "win32", reason="terminate() is TerminateProcess on Windows: no graceful shutdown")
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


def test_two_workers_on_one_file_never_run_a_session_twice(workdir):
    """A request that reaches the worker not serving the session gets a 409 (another live process holds it),
    never a second copy of the session: one turn per message, whichever worker took it."""
    server = Server(workdir, workers=2).start()
    try:
        session_id = httpx.post(f"{server.url}/chat/sessions", json={}).json()["session_id"]
        codes = []

        def send():
            response = httpx.post(f"{server.url}/chat/sessions/{session_id}/messages", json={"text": f"m{len(codes)}"})
            codes.append(response.status_code)
            return response.status_code

        for _ in range(12):
            send()
        # Which worker takes a connection is the OS's call, and not an even one (Windows can hand a run of them
        # to the same worker): keep going until one reaches the session's worker, not a fixed number of tries.
        until(lambda: 202 in codes or send() == 202, timeout=30, what="a message to reach the session's worker")
        assert set(codes) <= {202, 409}
        accepted = codes.count(202)

        def settled():
            for _ in range(10):
                response = httpx.get(f"{server.url}/chat/sessions/{session_id}")
                if response.status_code == 200:
                    state = response.json()
                    return state if state["turns"] == accepted and not state["busy"] and not state["pending"] else None
            return None

        state = until(settled, what="every accepted message to run once")
        assert state["turns"] == accepted
    finally:
        server.stop()


REAL_AGENT = textwrap.dedent(
    """
    import asyncio, json, os
    from pathlib import Path
    from timbal import Agent, Tool
    from timbal.state.tracing.providers import SqliteTracingProvider

    DIR = Path(os.environ["E2E_DIR"])

    async def charge(amount_eur: int) -> str:
        \"\"\"Charge the customer's card.\"\"\"
        with open(DIR / "charges.log", "a") as f:
            f.write(json.dumps({"amount_eur": amount_eur}) + "\\n")
        while (DIR / "hang").exists():
            await asyncio.sleep(0.05)
        return f"Charged {amount_eur} EUR."

    def list_charges() -> str:
        \"\"\"List the charges made to the customer's card so far.\"\"\"
        (DIR / "checked").touch()
        path = DIR / "charges.log"
        return path.read_text() if path.exists() else "No charges."

    agent = Agent(
        name="agent",
        model=os.environ["E2E_MODEL"],
        max_tokens=2048,
        system_prompt="You are a billing assistant. Charge the customer when they ask you to. Be brief.",
        tools=[Tool(name="charge", handler=charge), Tool(name="list_charges", handler=list_charges, read_only=True)],
        tracing_provider=SqliteTracingProvider.configured(_path=DIR / "timbal.db"),
    )
    """
)


def turn_texts(server, session_id):
    log = httpx.get(f"{server.url}/chat/sessions/{session_id}/log", params={"after": 0, "limit": 2000}).json()
    texts = []
    for entry in log["events"]:
        data = entry["data"]
        if isinstance(data, dict) and data.get("type") == "OUTPUT" and data.get("path") == "agent":
            output = data.get("output") or {}
            texts.append(" ".join(c.get("text", "") for c in output.get("content", []) if c.get("type") == "text"))
    return texts


@pytest.mark.integration
@pytest.mark.parametrize("model", ["anthropic/claude-haiku-5-5", "openai/gpt-6-luna"])
def test_a_real_model_recovers_a_charge_cut_by_kill_9(workdir, model, monkeypatch):
    (workdir / "real_agent.py").write_text(REAL_AGENT)
    monkeypatch.setenv("E2E_MODEL", model)
    server = Server(workdir, agent_file="real_agent.py").start()
    try:
        session_id = httpx.post(f"{server.url}/chat/sessions", json={}).json()["session_id"]
        (workdir / "hang").touch()
        httpx.post(f"{server.url}/chat/sessions/{session_id}/messages", json={"text": "Please charge me 5 euros."})
        until(lambda: (workdir / "charges.log").exists(), 60, "the model to call charge")
        server.kill()
    finally:
        server.stop()
    (workdir / "hang").unlink()

    server.start()
    try:
        base = f"{server.url}/chat/sessions/{session_id}"
        until(lambda: (s := httpx.get(base).json()) and s.get("turns") == 2 and not s["busy"], 90, "recovery")
        recovery = turn_texts(server, session_id)[-1]
        httpx.post(f"{base}/messages", json={"text": "Did my payment go through? Answer yes, no or not sure."})
        until(lambda: (s := httpx.get(base).json()) and s.get("turns") == 3 and not s["busy"], 90, "the answer")
        answer = turn_texts(server, session_id)[-1]
        print(  # noqa: T201 - the transcript is what a reviewer reads
            f"\n[{model}] checked charges: {(workdir / 'checked').exists()}\n  recovery: {recovery}\n  answer: {answer}"
        )
        assert (workdir / "charges.log").read_text().count("amount_eur") == 1  # never charged twice
        assert recovery.strip() and answer.strip()
    finally:
        server.stop()
