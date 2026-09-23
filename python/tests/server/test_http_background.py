"""`/runs/{run_id}/background/...` — polling detached children over HTTP."""

import asyncio
import json
from collections.abc import AsyncGenerator
from pathlib import Path

import httpx
import pytest
from timbal import Agent, Tool
from timbal.core.test_model import TestModel
from timbal.server.http import create_app
from timbal.server.jobs import JobStore
from timbal.types.content import ToolUseContent
from timbal.types.events.delta import TextDelta
from timbal.types.message import Message


async def _builder(prompt: str) -> AsyncGenerator[TextDelta, None]:
    for i in range(4):
        yield TextDelta(id="b", text_delta=f"[{prompt}] chunk {i} ")
        await asyncio.sleep(0.05)
    yield TextDelta(id="b", text_delta=f"[{prompt}] done")


async def _forever(prompt: str) -> AsyncGenerator[TextDelta, None]:
    while True:
        yield TextDelta(id="f", text_delta=prompt)
        await asyncio.sleep(0.05)


def _parent(handler) -> Agent:
    return Agent(
        name="composer",
        model=TestModel(
            responses=[
                Message(
                    role="assistant",
                    content=[
                        ToolUseContent(id="c1", name="builder", input={"prompt": "go", "run_in_background": True})
                    ],
                    stop_reason="tool_use",
                ),
                "Started.",
            ]
        ),
        tools=[Tool(name="builder", handler=handler, background_mode="auto")],
    )


@pytest.fixture
def make_client(monkeypatch):
    fixture = Path(__file__).parent / "fixtures" / "agent_fixture.py"
    monkeypatch.setenv("TIMBAL_RUNNABLE", f"{fixture}::agent_fixture")

    def _make(runnable) -> httpx.AsyncClient:
        app = create_app()
        app.state.runnable = runnable
        app.state.job_store = JobStore()
        # In-loop transport: children spawned by one request must still be
        # running when the next request polls them.
        return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")

    return _make


async def _start(client: httpx.AsyncClient, run_id: str) -> str:
    """Run the parent to completion over `/run` and return the child's task id."""
    response = await client.post("/run", json={"prompt": "start", "context": {"id": run_id}})
    assert response.status_code == 200
    assert json.dumps(response.json())
    tasks = (await client.get(f"/runs/{run_id}/background")).json()["tasks"]
    assert len(tasks) == 1
    return tasks[0]["task_id"]


class TestBackgroundPolling:
    async def test_list_then_long_poll_until_terminal(self, make_client):
        async with make_client(_parent(_builder)) as client:
            task_id = await _start(client, "poll-me")

            running = (await client.get(f"/runs/poll-me/background/{task_id}")).json()
            assert running["status"] == "running"
            assert running["name"] == "builder"

            done = (await client.get(f"/runs/poll-me/background/{task_id}", params={"wait_ms": 5000})).json()
            assert done["status"] == "completed"
            assert "[go] done" in done["summary"]["text"]

    async def test_wait_with_a_cursor_returns_on_progress(self, make_client):
        async with make_client(_parent(_forever)) as client:
            task_id = await _start(client, "progress")
            url = f"/runs/progress/background/{task_id}"

            first = (await client.get(url, params={"after": 0, "wait_ms": 5000})).json()
            assert first["status"] == "running"
            cursor = first["transcript_cursor"]
            assert cursor > 0

            later = (await client.get(url, params={"after": cursor, "wait_ms": 5000})).json()
            assert later["transcript_cursor"] > cursor

            await client.post(f"{url}/cancel")

    async def test_wait_that_lapses_returns_the_running_snapshot(self, make_client):
        async with make_client(_parent(_forever)) as client:
            task_id = await _start(client, "lapse")
            url = f"/runs/lapse/background/{task_id}"

            snap = (await client.get(url, params={"wait_ms": 50})).json()
            assert snap["status"] == "running"

            await client.post(f"{url}/cancel")

    async def test_events_page_to_done(self, make_client):
        async with make_client(_parent(_builder)) as client:
            task_id = await _start(client, "events")
            url = f"/runs/events/background/{task_id}/events"

            first = (await client.get(url, params={"limit": 1, "wait_ms": 5000})).json()
            assert len(first["events"]) == 1
            assert first["next_cursor"] == 1
            assert first["done"] is False

            await client.get(f"/runs/events/background/{task_id}", params={"wait_ms": 5000})
            rest = (await client.get(url, params={"after": first["next_cursor"]})).json()
            assert rest["status"] == "completed"
            assert rest["done"] is True
            assert rest["gapped"] is False
            assert rest["next_cursor"] == 1 + len(rest["events"])

    async def test_cancel(self, make_client):
        async with make_client(_parent(_forever)) as client:
            task_id = await _start(client, "stop-me")

            cancelled = (await client.post(f"/runs/stop-me/background/{task_id}/cancel")).json()
            assert cancelled["status"] == "cancelled"

            final = (await client.get(f"/runs/stop-me/background/{task_id}", params={"wait_ms": 5000})).json()
            assert final["status"] == "cancelled"

    async def test_unknown_run_or_task(self, make_client):
        async with make_client(_parent(_builder)) as client:
            listed = await client.get("/runs/never-ran/background")
            assert listed.status_code == 200
            assert listed.json() == {"run_id": "never-ran", "tasks": []}

            assert (await client.get("/runs/never-ran/background/abc")).status_code == 404
            assert (await client.get("/runs/never-ran/background/abc/events")).status_code == 404
            assert (await client.post("/runs/never-ran/background/abc/cancel")).status_code == 404

            task_id = await _start(client, "known")
            assert (await client.get("/runs/known/background/not-a-task")).status_code == 404
            await client.get(f"/runs/known/background/{task_id}", params={"wait_ms": 5000})
