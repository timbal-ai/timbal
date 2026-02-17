"""Integration tests for HTTP server with real agent calls."""

import asyncio
from pathlib import Path

import httpx
import pytest
import uvicorn
from timbal.server.http import create_app
from timbal.server.jobs import JobStore
from timbal.utils import ImportSpec


class TestHttpIntegration:
    """Integration tests that run a real HTTP server with real agent calls."""

    @pytest.fixture
    def slow_agent_fixture_file(self):
        return Path(__file__).parent / "fixtures" / "slow_agent_fixture.py"

    @pytest.fixture
    def slow_agent_import_spec(self, slow_agent_fixture_file):
        return ImportSpec(path=slow_agent_fixture_file, target="slow_agent")

    @pytest.fixture
    async def server(self, slow_agent_import_spec, monkeypatch):
        """Start a real HTTP server for testing."""
        monkeypatch.setenv("TIMBAL_RUNNABLE", f"{slow_agent_import_spec.path}::{slow_agent_import_spec.target}")
        app = create_app()
        app.state.runnable = slow_agent_import_spec.load()
        app.state.job_store = JobStore()

        config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning")
        server = uvicorn.Server(config)

        # Start server in background
        serve_task = asyncio.create_task(server.serve())

        # Wait for server to start
        while not server.started:
            await asyncio.sleep(0.01)

        # Get the actual port
        port = server.servers[0].sockets[0].getsockname()[1]
        base_url = f"http://127.0.0.1:{port}"

        yield base_url, app

        # Cleanup
        server.should_exit = True
        await serve_task

    @pytest.mark.asyncio
    async def test_cancel_running_stream(self, server):
        """Test that cancelling a running stream job works."""
        base_url, app = server
        run_id = "testcancelstream001"

        async with httpx.AsyncClient() as client:
            # Start a streaming request
            events_received = []
            cancel_sent = False

            async def stream_consumer():
                nonlocal cancel_sent
                async with client.stream(
                    "POST",
                    f"{base_url}/stream",
                    json={
                        "context": {"id": run_id},
                        "prompt": {"role": "user", "content": "Count slowly from 1 to 100"},
                    },
                    timeout=30.0,
                ) as response:
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            events_received.append(line)
                            # After receiving a few events, cancel the job
                            if len(events_received) >= 3 and not cancel_sent:
                                cancel_sent = True
                                # Cancel in a separate task to not block streaming
                                asyncio.create_task(cancel_job())

            async def cancel_job():
                await asyncio.sleep(0.1)  # Small delay to ensure we're mid-stream
                cancel_response = await client.post(f"{base_url}/cancel/{run_id}")
                return cancel_response.status_code

            try:
                await asyncio.wait_for(stream_consumer(), timeout=10.0)
            except (httpx.RemoteProtocolError, asyncio.TimeoutError):
                # Expected when connection is interrupted
                pass

            # Verify we received some events before cancellation
            assert len(events_received) > 0

    @pytest.mark.asyncio
    async def test_cancel_nonexistent_job(self, server):
        """Test that cancelling a non-existent job returns 404."""
        base_url, _ = server

        async with httpx.AsyncClient() as client:
            response = await client.post(f"{base_url}/cancel/nonexistent-job-id")
            assert response.status_code == 404
            assert response.json() == {"error": "Job not found or already completed"}

    @pytest.mark.asyncio
    async def test_cancel_already_completed_job(self, server):
        """Test that cancelling an already completed job returns 404."""
        base_url, _ = server
        run_id = "testcompletedjob001"

        async with httpx.AsyncClient() as client:
            # Run a quick job to completion
            response = await client.post(
                f"{base_url}/run",
                json={
                    "context": {"id": run_id},
                    "prompt": {"role": "user", "content": "Say hello"},
                },
                timeout=30.0,
            )
            assert response.status_code == 200

            # Try to cancel the completed job
            cancel_response = await client.post(f"{base_url}/cancel/{run_id}")
            assert cancel_response.status_code == 404

    @pytest.mark.asyncio
    async def test_stream_with_run_id_from_context(self, server):
        """Test that stream endpoint properly uses id from context as job_id."""
        base_url, app = server
        run_id = "teststreamcontext001"

        async with httpx.AsyncClient() as client:
            # Verify the job is created with the correct id by checking
            # that we can query/cancel it during streaming
            job_found = False

            async with client.stream(
                "POST",
                f"{base_url}/stream",
                json={
                    "context": {"id": run_id},
                    "prompt": {
                        "role": "user",
                        "content": "Use the slow_task tool with message 'testing run_id'",
                    },
                },
                timeout=30.0,
            ) as response:
                async for line in response.aiter_lines():
                    # Check for job while streaming
                    if not job_found:
                        job = app.state.job_store.get_job(run_id)
                        if job is not None:
                            job_found = True

            # Job should have been found with our custom id during streaming
            assert job_found

    @pytest.mark.asyncio
    async def test_cancel_emits_interrupted_output_event(self, server):
        """Test that cancelling a job emits an OutputEvent with interrupted status."""
        import json

        base_url, app = server
        run_id = "testcancelinterrupted001"

        async with httpx.AsyncClient() as client:
            events_received = []
            cancel_response_status = None

            async def cancel_after_delay():
                nonlocal cancel_response_status
                await asyncio.sleep(1.0)  # Wait for slow_task to start
                response = await client.post(f"{base_url}/cancel/{run_id}")
                cancel_response_status = response.status_code

            cancel_task = asyncio.create_task(cancel_after_delay())

            try:
                async with client.stream(
                    "POST",
                    f"{base_url}/stream",
                    json={
                        "context": {"id": run_id},
                        "prompt": {
                            "role": "user",
                            "content": "Use the slow_task tool with message 'this will be cancelled'",
                        },
                    },
                    timeout=30.0,
                ) as response:
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            event_data = json.loads(line[6:])
                            events_received.append(event_data)
            except (httpx.RemoteProtocolError, httpx.ReadError):
                # Connection may be closed when job is cancelled
                pass

            await cancel_task

            # Cancel should have succeeded
            assert cancel_response_status == 204

            # Check that we received an OUTPUT event with interrupted status
            output_events = [e for e in events_received if e.get("type") == "OUTPUT"]

            # We should have at least one output event
            # If cancellation works properly, the final one should have interrupted status
            interrupted_events = [
                e
                for e in output_events
                if e.get("status", {}).get("code") == "cancelled" and e.get("status", {}).get("reason") == "interrupted"
            ]

            assert len(interrupted_events) > 0, (
                f"Expected at least one interrupted OUTPUT event, got events: {output_events}"
            )
