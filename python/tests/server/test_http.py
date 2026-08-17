import os
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from timbal import __version__
from timbal.server.http import create_app
from timbal.server.jobs import JobStore
from timbal.utils import ImportSpec


class TestHttpScript:
    @pytest.fixture
    def tool_fixture_file(self):
        return Path(__file__).parent / "fixtures" / "tool_fixture.py"

    @pytest.fixture
    def agent_fixture_file(self):
        return Path(__file__).parent / "fixtures" / "agent_fixture.py"

    def test_version_flag(self):
        """Test the --version flag outputs correct version."""
        result = subprocess.run(
            [sys.executable, "-m", "timbal.server.http", "--version"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,
        )

        assert result.returncode == 0
        assert f"timbal.server.http {__version__}" in result.stdout

    def test_no_import_spec_error(self):
        """Test error when no import spec is provided."""
        env = os.environ.copy()
        env.pop("TIMBAL_RUNNABLE", None)
        env.pop("TIMBAL_FLOW", None)
        result = subprocess.run(
            [sys.executable, "-m", "timbal.server.http"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,
            env=env,
        )

        assert result.returncode == 1
        assert "No import spec provided" in result.stderr

    def test_invalid_import_spec_format(self):
        """Test error with invalid import spec format."""
        result = subprocess.run(
            [sys.executable, "-m", "timbal.server.http", "--import_spec", "invalid_format"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,
        )

        assert result.returncode == 1
        assert "Invalid import spec format" in result.stderr

    def test_empty_import_spec_parts(self):
        """Test error with malformed import spec (missing ::)."""
        result = subprocess.run(
            [sys.executable, "-m", "timbal.server.http", "--import_spec", "just_a_path"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,
        )

        assert result.returncode == 1
        assert "Invalid import spec format" in result.stderr

    def test_too_many_import_spec_parts(self):
        """Test error with malformed import spec (too many :: parts)."""
        result = subprocess.run(
            [sys.executable, "-m", "timbal.server.http", "--import_spec", "path::target::extra"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,
        )

        assert result.returncode == 1
        assert "Invalid import spec format" in result.stderr


class TestFastAPIApp:
    @pytest.fixture
    def tool_fixture_file(self):
        return Path(__file__).parent / "fixtures" / "tool_fixture.py"

    @pytest.fixture
    def agent_fixture_file(self):
        return Path(__file__).parent / "fixtures" / "agent_fixture.py"

    @pytest.fixture
    def tool_import_spec(self, tool_fixture_file):
        """Create a real ImportSpec for the tool fixture."""
        return ImportSpec(path=tool_fixture_file, target="tool_fixture")

    @pytest.fixture
    def agent_import_spec(self, agent_fixture_file):
        """Create a real ImportSpec for the agent fixture."""
        return ImportSpec(path=agent_fixture_file, target="agent_fixture")

    @pytest.fixture
    def tool_app(self, tool_import_spec, monkeypatch):
        """Create a FastAPI app with real tool fixture."""
        monkeypatch.setenv("TIMBAL_RUNNABLE", f"{tool_import_spec.path}::{tool_import_spec.target}")
        app = create_app()
        # Manually load the runnable since TestClient doesn't trigger lifespan
        app.state.runnable = tool_import_spec.load()
        app.state.job_store = JobStore()
        return app

    @pytest.fixture
    def agent_app(self, agent_import_spec, monkeypatch):
        """Create a FastAPI app with real agent fixture."""
        monkeypatch.setenv("TIMBAL_RUNNABLE", f"{agent_import_spec.path}::{agent_import_spec.target}")
        app = create_app()
        # Manually load the runnable since TestClient doesn't trigger lifespan
        app.state.runnable = agent_import_spec.load()
        app.state.job_store = JobStore()
        return app

    @pytest.fixture
    def tool_client(self, tool_app):
        """Create a test client with tool fixture."""
        return TestClient(tool_app)

    @pytest.fixture
    def agent_client(self, agent_app):
        """Create a test client with agent fixture."""
        return TestClient(agent_app)

    def test_healthcheck_endpoint(self, tool_client):
        """Test /healthcheck endpoint returns 204."""
        response = tool_client.get("/healthcheck")
        assert response.status_code == 204

    def test_params_model_schema_endpoint_tool(self, tool_client):
        """Test /params_model_schema endpoint returns correct schema for tool."""
        response = tool_client.get("/params_model_schema")
        assert response.status_code == 200

        data = response.json()
        assert "properties" in data
        assert "x" in data["properties"]
        assert data["properties"]["x"]["type"] == "string"

    def test_params_model_schema_endpoint_agent(self, agent_client):
        """Test /params_model_schema endpoint returns correct schema for agent."""
        response = agent_client.get("/params_model_schema")
        assert response.status_code == 200

        data = response.json()
        assert "properties" in data

    def test_return_model_schema_endpoint_tool(self, tool_client):
        """Test /return_model_schema endpoint returns correct schema for tool."""
        response = tool_client.get("/return_model_schema")
        assert response.status_code == 200

        data = response.json()
        assert isinstance(data, dict)

    def test_return_model_schema_endpoint_agent(self, agent_client):
        """Test /return_model_schema endpoint returns correct schema for agent."""
        response = agent_client.get("/return_model_schema")
        assert response.status_code == 200

        data = response.json()
        assert isinstance(data, dict)

    def test_run_endpoint_tool_without_context(self, tool_client):
        """Test /run endpoint with tool fixture without run context."""
        test_data = {"x": "test input"}

        response = tool_client.post("/run", json=test_data)
        assert response.status_code == 200

        data = response.json()
        assert "result: test input" in str(data)

    def test_run_endpoint_tool_with_context(self, tool_client):
        """Test /run endpoint with tool fixture with run context."""
        run_context = {"trace_id": "test-trace-id", "span_id": "test-span-id"}
        test_data = {"x": "test input", "context": run_context}

        response = tool_client.post("/run", json=test_data)
        assert response.status_code == 200

        data = response.json()
        assert "result: test input" in str(data)

    def test_run_endpoint_tool_with_legacy_run_context(self, tool_client):
        """Test /run endpoint with tool fixture with legacy run_context key."""
        run_context = {"trace_id": "test-trace-id", "span_id": "test-span-id"}
        test_data = {"x": "test input", "run_context": run_context}

        response = tool_client.post("/run", json=test_data)
        assert response.status_code == 200

        data = response.json()
        assert "result: test input" in str(data)

    def test_stream_endpoint_tool(self, tool_client):
        """Test /stream endpoint with tool fixture."""
        test_data = {"x": "test input"}

        response = tool_client.post("/stream", json=test_data)
        # Stream endpoint should return 200 with event-stream content type
        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")

    def test_cancel_endpoint_nonexistent_job(self, tool_client):
        """Test /cancel endpoint returns 404 for non-existent job."""
        response = tool_client.post("/cancel/nonexistent-job-id")
        assert response.status_code == 404
        assert response.json() == {"error": "Job not found or already completed"}

    def test_run_endpoint_uses_run_id_from_context(self, tool_app):
        """Test that /run endpoint uses run_id from context as job_id."""
        client = TestClient(tool_app)
        run_id = "custom-run-id-123"
        test_data = {
            "x": "test input",
            "context": {"run_id": run_id},
        }

        # The job will complete immediately for tool fixture,
        # but we can verify the job was created with the correct ID
        # by checking cancel returns 404 after completion
        response = client.post("/run", json=test_data)
        assert response.status_code == 200

        # A finished job is still addressable (it is retained so a late reader
        # can replay it), but it is no longer cancellable.
        cancel_response = client.post(f"/cancel/{run_id}")
        assert cancel_response.status_code == 404

    def test_stream_endpoint_uses_run_id_from_context(self, tool_app):
        """Test that /stream endpoint uses run_id from context as job_id."""
        client = TestClient(tool_app)
        run_id = "custom-stream-run-id-456"
        test_data = {
            "x": "test input",
            "context": {"run_id": run_id},
        }

        response = client.post("/stream", json=test_data)
        assert response.status_code == 200

        # A finished job is still addressable (it is retained so a late reader
        # can replay it), but it is no longer cancellable.
        cancel_response = client.post(f"/cancel/{run_id}")
        assert cancel_response.status_code == 404


class TestRunEventsEndpoint:
    """`GET /runs/{run_id}/events` — the reconnect path for a dropped stream."""

    @pytest.fixture
    def tool_fixture_file(self):
        return Path(__file__).parent / "fixtures" / "tool_fixture.py"

    @pytest.fixture
    def tool_app(self, tool_fixture_file, monkeypatch):
        import_spec = ImportSpec(path=tool_fixture_file, target="tool_fixture")
        monkeypatch.setenv("TIMBAL_RUNNABLE", f"{import_spec.path}::{import_spec.target}")
        app = create_app()
        app.state.runnable = import_spec.load()
        app.state.job_store = JobStore()
        return app

    @pytest.fixture
    def client(self, tool_app):
        return TestClient(tool_app)

    def test_a_finished_run_replays_from_the_start(self, client):
        run_id = "replay-me"
        client.post("/stream", json={"x": "test input", "context": {"id": run_id}})

        body = client.get(f"/runs/{run_id}/events").json()

        assert body["run_id"] == run_id
        assert body["expired"] is False
        assert body["done"] is True
        assert [event["seq"] for event in body["events"]] == list(
            range(1, len(body["events"]) + 1)
        )
        assert body["next_cursor"] == body["events"][-1]["seq"]

    def test_a_cursor_returns_only_what_came_after_it(self, client):
        run_id = "resume-me"
        client.post("/stream", json={"x": "test input", "context": {"id": run_id}})

        everything = client.get(f"/runs/{run_id}/events").json()["events"]
        tail = client.get(f"/runs/{run_id}/events", params={"after": 1}).json()["events"]

        assert [event["seq"] for event in tail] == [
            event["seq"] for event in everything if event["seq"] > 1
        ]

    def test_paging_stops_short_of_done(self, client):
        run_id = "page-me"
        client.post("/stream", json={"x": "test input", "context": {"id": run_id}})

        first = client.get(f"/runs/{run_id}/events", params={"limit": 1}).json()

        assert len(first["events"]) == 1
        assert first["next_cursor"] == 1
        # More events are buffered, so the run being over must not read as done.
        assert first["done"] is False

        rest = client.get(
            f"/runs/{run_id}/events", params={"after": first["next_cursor"]}
        ).json()

        assert rest["done"] is True

    def test_an_unknown_run_is_expired_not_a_clean_ending(self, client):
        """The whole point of the flag: this must not look like a drained stream."""
        body = client.get("/runs/never-existed/events", params={"after": 7}).json()

        assert body["expired"] is True
        assert body["done"] is True
        assert body["events"] == []
        assert body["next_cursor"] == 7

    def test_a_reaped_run_is_expired_too(self, tool_app):
        tool_app.state.job_store = JobStore(retention_secs=0)
        client = TestClient(tool_app)
        run_id = "reap-me"
        client.post("/stream", json={"x": "test input", "context": {"id": run_id}})

        tool_app.state.job_store.reap()

        assert client.get(f"/runs/{run_id}/events").json()["expired"] is True

    def test_waiting_on_a_finished_run_returns_immediately(self, client):
        run_id = "no-wait"
        client.post("/stream", json={"x": "test input", "context": {"id": run_id}})

        body = client.get(
            f"/runs/{run_id}/events",
            params={"after": 9999, "wait_ms": 30000},
        ).json()

        assert body["done"] is True
        assert body["events"] == []


class TestServerLifecycle:
    @pytest.fixture
    def tool_fixture_file(self):
        return Path(__file__).parent / "fixtures" / "tool_fixture.py"

    @pytest.fixture
    def tool_import_spec(self, tool_fixture_file):
        """Create a real ImportSpec for the tool fixture."""
        return ImportSpec(path=tool_fixture_file, target="tool_fixture")

    def test_create_app_basic(self, tool_import_spec, monkeypatch):
        """Test basic FastAPI app creation with real import spec."""
        monkeypatch.setenv("TIMBAL_RUNNABLE", f"{tool_import_spec.path}::{tool_import_spec.target}")
        app = create_app()

        # Verify app was created successfully
        assert app is not None
        assert hasattr(app, "state")

    def test_import_spec_loading(self, tool_import_spec):
        """Test that ImportSpec can load the real runnable."""
        runnable = tool_import_spec.load()

        # Verify the loaded runnable has expected attributes
        assert hasattr(runnable, "params_model_schema")
        assert hasattr(runnable, "return_model_schema")
        assert callable(runnable)

    def test_create_app_raises_without_env_var(self, monkeypatch):
        """Cover line 47: RuntimeError when TIMBAL_RUNNABLE is not set."""
        monkeypatch.delenv("TIMBAL_RUNNABLE", raising=False)
        with pytest.raises(RuntimeError, match="TIMBAL_RUNNABLE"):
            create_app()

    def test_ngrok_env_var_detection(self):
        """Test ngrok integration environment variable detection."""
        # Test that env var detection works correctly
        original_value = os.environ.get("TIMBAL_ENABLE_NGROK")

        try:
            os.environ["TIMBAL_ENABLE_NGROK"] = "true"
            assert os.getenv("TIMBAL_ENABLE_NGROK", "false").lower() == "true"

            os.environ["TIMBAL_ENABLE_NGROK"] = "false"
            assert os.getenv("TIMBAL_ENABLE_NGROK", "false").lower() == "false"
        finally:
            if original_value is not None:
                os.environ["TIMBAL_ENABLE_NGROK"] = original_value
            elif "TIMBAL_ENABLE_NGROK" in os.environ:
                del os.environ["TIMBAL_ENABLE_NGROK"]
