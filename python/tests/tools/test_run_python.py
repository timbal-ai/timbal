"""Runner semantics and Modal lifecycle tests; no remote compute in unit tests."""

import asyncio
import base64
import json
import os
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from pydantic import ValidationError
from timbal.state.tracing.providers.in_memory import InMemoryTracingProvider
from timbal.tools import RunPython
from timbal.tools._run_python_files import export_files
from timbal.tools._run_python_transfer import read_file
from timbal.tools.run_python import _RUNNER, _read_artifacts, _read_output, _resolve_dependencies
from timbal.types import File, Message


def run_code(code):
    # Only fixed test fixtures execute locally. Production always sends the runner to Modal.
    source = _RUNNER.read_text().split('if __name__ == "__main__":')[0]
    harness = source + "\nprint(json.dumps(execute(sys.stdin.read())))\n"
    result = subprocess.run([sys.executable, "-c", harness], input=code, capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.splitlines()[-1])


@pytest.mark.parametrize(
    ("code", "value"),
    [
        ("1 + 2", 3),
        ("x = 3\ndef f(): return x\nf()", 3),
        ("x = 1", None),
        ("", None),
        ("import asyncio\nx = await asyncio.sleep(0, result=4)\ndef f(): return x\nf()", 4),
        ("import asyncio\nawait asyncio.sleep(0, result=42)", 42),
        ("{1, 2}", "{1, 2}"),
        ("float('nan')", "nan"),
        ("print('hello')\n{'a': [1, 2]}", {"a": [1, 2]}),
    ],
)
def test_python_semantics(code, value):
    assert run_code(code) == {"return_value": value, "error": None}


@pytest.mark.parametrize(
    ("code", "error"),
    [
        ("1/0", "ZeroDivisionError"),
        ("if", "SyntaxError"),
        ("raise SystemExit(2)", "SystemExit"),
        ("import missing_timbal_test_module", "ModuleNotFoundError"),
    ],
)
def test_python_errors(code, error):
    result = run_code(code)
    assert result["return_value"] is None
    assert result["error"]["type"] == error
    assert "<run_python>" in result["error"]["traceback"]


async def chunks(*values):
    for value in values:
        yield value


@pytest.mark.parametrize("size", [0, 1, 39, 80, 100, 101, 10_000])
async def test_bounded_output(size):
    value = "0123456789" * (size // 10) + "x" * (size % 10)
    result = await _read_output(chunks(*(value[i : i + 17] for i in range(0, size, 17))), 100)
    assert len(result) <= 100
    if size <= 100:
        assert result == value
    else:
        assert "[truncated]" in result
        assert result.startswith(value[:20])
        assert result.endswith(value[-20:])


def aio(**kwargs):
    return SimpleNamespace(aio=AsyncMock(**kwargs))


def process(stdout="", stderr="", returncode=0):
    return SimpleNamespace(
        stdout=chunks(stdout),
        stderr=chunks(stderr),
        wait=aio(return_value=returncode),
        stdin=SimpleNamespace(write=Mock(), write_eof=Mock(), drain=aio()),
    )


@pytest.fixture
def modal_mock(monkeypatch):
    execution = process("hello\n", "warning\n")
    result = process(json.dumps({"return_value": 42, "error": None}))
    sandbox = SimpleNamespace(
        exec=aio(side_effect=[execution, result]),
        terminate=aio(),
        filesystem=SimpleNamespace(make_directory=aio(), write_bytes=aio()),
    )
    artifacts = AsyncMock(return_value=[])
    monkeypatch.setattr("timbal.tools.run_python._read_artifacts", artifacts)
    image = Mock()
    image.apt_install.return_value = image
    image.pip_install.return_value = image
    image.run_commands.return_value = image
    module = SimpleNamespace(
        App=SimpleNamespace(lookup=aio(return_value="app")),
        Sandbox=SimpleNamespace(create=aio(return_value=sandbox)),
        Image=SimpleNamespace(debian_slim=Mock(return_value=image)),
    )
    monkeypatch.setitem(sys.modules, "modal", module)
    return SimpleNamespace(module=module, sandbox=sandbox, image=image, execution=execution, artifacts=artifacts)


async def test_tool_contract_and_isolation(modal_mock, monkeypatch):
    monkeypatch.setenv("HOST_SECRET", "must-not-forward")
    tool = RunPython(dependencies=["numpy==2.2.6"], timeout=10)
    event = await tool(code="21 * 2", dependencies=["numpy==2.2.6", "pandas"]).collect()
    assert event.error is None
    assert event.output == {
        "stdout": "hello\n",
        "stderr": "warning\n",
        "returncode": 0,
        "return_value": 42,
        "error": None,
        "status": "success",
        "artifacts": [],
        "artifact_error": None,
    }
    modal_mock.image.pip_install.assert_called_once_with("numpy==2.2.6", "pandas")
    config = modal_mock.module.Sandbox.create.aio.call_args.kwargs
    assert config["block_network"] is True
    assert config["memory"] == (512, 512)
    assert config["cpu"] == (1, 1)
    assert config["timeout"] == 190
    assert "env" not in config and "secrets" not in config and "volumes" not in config
    assert "must-not-forward" not in str(modal_mock.execution.stdin.write.call_args)
    request = json.loads(modal_mock.execution.stdin.write.call_args.args[0])
    assert request["code"] == "21 * 2"
    modal_mock.sandbox.terminate.aio.assert_awaited_once()
    assert set(tool.params_model.model_fields) == {"code", "dependencies", "files"}
    assert tool.get_config()["timeout"]["value"] == 10


async def test_image_build_order_and_agent_schema(modal_mock):
    tool = RunPython(
        system_dependencies=["ffmpeg"],
        dependencies=["playwright==1.58.0"],
        setup_commands=["playwright install --with-deps chromium"],
        block_network=False,
        description="Run Python with Playwright and preinstalled Chromium.",
    )
    await tool.handler("1")
    assert [call[0] for call in modal_mock.image.mock_calls] == ["apt_install", "pip_install", "run_commands"]
    modal_mock.image.apt_install.assert_called_once_with("ffmpeg")
    modal_mock.image.run_commands.assert_called_once_with("playwright install --with-deps chromium")
    schema = tool.params_model.model_json_schema()["properties"]
    assert set(schema) == {"code", "dependencies", "files"}
    assert "Markdown" in schema["code"]["description"]
    assert "cannot be overridden" in schema["dependencies"]["description"]
    assert "playwright==1.58.0" in tool.description
    assert "preinstalled Chromium" in tool.description
    assert "network access: enabled" in tool.description
    assert "playwright install" not in tool.description
    config = tool.get_config()
    assert config["system_dependencies"]["value"] == ["ffmpeg"]
    assert config["setup_commands"]["value"] == ["playwright install --with-deps chromium"]


@pytest.mark.parametrize(
    ("configured", "requested", "expected"),
    [
        (["numpy==2.2.6"], ["numpy"], ["numpy==2.2.6"]),
        (["Some_Package==1.0"], ["some-package==1.0"], ["Some_Package==1.0"]),
        (["httpx==0.28.1"], ["httpx[http2]"], ["httpx[http2]==0.28.1"]),
        ([], ["numpy", "numpy"], ["numpy"]),
        (["playwright==1.58.0"], ["numpy==2.2.6"], ["playwright==1.58.0", "numpy==2.2.6"]),
    ],
)
def test_dependency_resolution(configured, requested, expected):
    assert _resolve_dependencies(configured, requested) == expected


@pytest.mark.parametrize(
    "requested", ["playwright==1.51.0", "playwright>=1.58", "playwright @ https://example.com/p.whl"]
)
async def test_configured_requirements_cannot_be_replaced(modal_mock, requested):
    with pytest.raises(ValueError, match="cannot replace"):
        await RunPython(dependencies=["playwright==1.58.0"]).handler("1", [requested])
    modal_mock.module.Image.debian_slim.assert_not_called()
    modal_mock.module.Sandbox.create.aio.assert_not_awaited()


@pytest.mark.parametrize("setting", ["dependencies", "system_dependencies", "setup_commands"])
def test_empty_build_configuration(setting):
    with pytest.raises(ValidationError, match="must not be empty"):
        RunPython(**{setting: [" "]})


def test_sandbox_lifetime_leaves_cleanup_budget():
    with pytest.raises(ValidationError):
        RunPython(timeout=86400)


async def test_execution_failure_cleanup(modal_mock):
    modal_mock.sandbox.exec.aio.side_effect = RuntimeError("Modal unavailable")
    with pytest.raises(RuntimeError, match="Modal unavailable"):
        await RunPython().handler("1")
    modal_mock.sandbox.terminate.aio.assert_awaited_once()


async def test_timeout_cleanup(modal_mock):
    async def hang(*_args, **_kwargs):
        await asyncio.Event().wait()

    modal_mock.sandbox.exec.aio.side_effect = hang
    result = await RunPython(timeout=0.01).handler("while True: pass")
    assert result["error"]["type"] == "TimeoutError"
    modal_mock.sandbox.terminate.aio.assert_awaited_once()


async def test_host_timeout_preserves_bounded_partial_output(modal_mock):
    stdout_value = "start:" + "x" * 200 + ":end"

    async def output_then_hang(value):
        yield value
        await asyncio.Event().wait()

    async def wait_forever():
        await asyncio.Event().wait()

    execution = SimpleNamespace(
        stdout=output_then_hang(stdout_value),
        stderr=output_then_hang("warning before timeout\n"),
        wait=SimpleNamespace(aio=wait_forever),
        stdin=SimpleNamespace(write=Mock(), write_eof=Mock(), drain=aio()),
    )
    modal_mock.sandbox.exec.aio.side_effect = [execution]

    result = await RunPython(timeout=0.01, max_output_chars=100).handler("while True: pass")

    assert result["error"]["type"] == "TimeoutError"
    assert result["returncode"] is None
    assert len(result["stdout"]) == 100
    assert result["stdout"].startswith("start:")
    assert result["stdout"].endswith(":end")
    assert "[truncated]" in result["stdout"]
    assert result["stderr"] == "warning before timeout\n"
    modal_mock.sandbox.exec.aio.assert_awaited_once()
    modal_mock.sandbox.terminate.aio.assert_awaited_once()


async def test_provider_timeout_preserves_logs(modal_mock):
    modal_mock.sandbox.exec.aio.side_effect = [process("before timeout\n", returncode=-1)]
    result = await RunPython().handler("while True: pass")
    assert result["error"]["type"] == "TimeoutError"
    assert result["stdout"] == "before timeout\n"
    modal_mock.sandbox.exec.aio.assert_awaited_once()
    modal_mock.sandbox.terminate.aio.assert_awaited_once()


async def test_cancellation_cleanup(modal_mock):
    started = asyncio.Event()

    async def hang(*_args, **_kwargs):
        started.set()
        await asyncio.Event().wait()

    modal_mock.sandbox.exec.aio.side_effect = hang
    task = asyncio.create_task(RunPython().handler("1"))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    modal_mock.sandbox.terminate.aio.assert_awaited_once()


@pytest.mark.parametrize("raw", ["[]", '{"error": null}', '{"return_value": 1, "error": 123}', "not json", "x" * 1025])
async def test_invalid_remote_result_cleanup(modal_mock, raw):
    modal_mock.sandbox.exec.aio.side_effect = [process("captured output"), process(raw)]
    result = await RunPython(max_result_bytes=1024).handler("1")
    assert result["error"]["type"] == ("OutputLimitError" if len(raw) > 1024 else "ResultError")
    assert result["stdout"] == "captured output"
    modal_mock.sandbox.terminate.aio.assert_awaited_once()


@pytest.mark.parametrize("returncode", [7, 137])
async def test_crash_preserves_execution_details(modal_mock, returncode):
    modal_mock.sandbox.exec.aio.side_effect = [process("before crash\n", "native failure\n", returncode)]
    event = await RunPython()(code="import os; os._exit(7)").collect()
    assert event.error is None
    assert event.output["status"] == "error"
    assert event.output["error"]["type"] == "ExecutionError"
    assert event.output["returncode"] == returncode
    assert event.output["stdout"] == "before crash\n"
    assert event.output["stderr"] == "native failure\n"
    modal_mock.sandbox.exec.aio.assert_awaited_once()
    modal_mock.sandbox.terminate.aio.assert_awaited_once()


async def test_missing_result_is_not_an_output_limit_error(modal_mock):
    modal_mock.sandbox.exec.aio.side_effect = [process("before exit\n"), process(returncode=1)]
    result = await RunPython().handler("import os; os._exit(0)")
    assert result["error"]["type"] == "ExecutionError"
    assert result["stdout"] == "before exit\n"
    assert result["returncode"] == 0


async def test_result_read_timeout_is_reported_as_timeout(modal_mock):
    modal_mock.sandbox.exec.aio.side_effect = [process("execution complete\n"), process(returncode=-1)]
    result = await RunPython().handler("42")
    assert result["status"] == "error"
    assert result["error"]["type"] == "TimeoutError"
    assert result["stdout"] == "execution complete\n"
    assert result["returncode"] == 0
    modal_mock.sandbox.terminate.aio.assert_awaited_once()


@pytest.mark.parametrize("cancel_execution", [False, True])
async def test_cancellation_during_cleanup_waits_for_termination(modal_mock, cancel_execution):
    execution_started = asyncio.Event()
    cleanup_started = asyncio.Event()
    release_cleanup = asyncio.Event()
    cleanup_finished = asyncio.Event()

    async def hang(*_args, **_kwargs):
        execution_started.set()
        await asyncio.Event().wait()

    async def terminate():
        cleanup_started.set()
        await release_cleanup.wait()
        cleanup_finished.set()

    modal_mock.sandbox.terminate.aio.side_effect = terminate
    if cancel_execution:
        modal_mock.sandbox.exec.aio.side_effect = hang
    task = asyncio.create_task(RunPython().handler("1"))
    if cancel_execution:
        await execution_started.wait()
        task.cancel()
    await cleanup_started.wait()
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    release_cleanup.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cleanup_finished.is_set()
    modal_mock.sandbox.terminate.aio.assert_awaited_once()


async def test_reject_large_code_before_remote_creation(modal_mock):
    with pytest.raises(ValueError, match="Code exceeds"):
        await RunPython().handler("é" * 50_001)
    modal_mock.module.Sandbox.create.aio.assert_not_awaited()


@pytest.mark.parametrize("config", [{"timeout": 0}, {"timeout": float("nan")}, {"cpu": 0}, {"memory": 0}])
def test_invalid_configuration(config):
    with pytest.raises(ValidationError):
        RunPython(**config)


async def test_missing_optional_dependency(monkeypatch):
    monkeypatch.setitem(sys.modules, "modal", None)
    with pytest.raises(ImportError, match=r"timbal\[modal\]"):
        await RunPython().handler("1")


def test_runner_main_result_limit(tmp_path):
    result_path = tmp_path / "result.json"
    runner = _RUNNER.read_text().replace('"/tmp/timbal-result.json"', repr(str(result_path)))
    completed = subprocess.run(
        [sys.executable, "-c", runner],
        input=json.dumps({"code": "'x' * 10000", "max_result_bytes": 1024}),
        text=True,
        capture_output=True,
        timeout=5,
    )
    assert completed.returncode == 0, completed.stderr
    payload = result_path.read_bytes()
    assert len(payload) <= 1024
    assert json.loads(payload)["error"]["type"] == "OutputLimitError"


async def test_cleanup_failure_preserves_original_error(modal_mock):
    modal_mock.sandbox.exec.aio.side_effect = RuntimeError("execution failed")
    modal_mock.sandbox.terminate.aio.side_effect = RuntimeError("cleanup failed")
    with pytest.raises(RuntimeError, match="execution failed"):
        await RunPython().handler("1")


async def test_cleanup_failure_preserves_success(modal_mock, caplog):
    modal_mock.sandbox.terminate.aio.side_effect = RuntimeError("cleanup failed")
    result = await RunPython().handler("1")
    assert result["status"] == "success"
    assert result["return_value"] == 42
    assert result["stdout"] == "hello\n"
    assert result["stderr"] == "warning\n"
    assert "cleanup failed" in caplog.text


async def test_cleanup_deadline_preserves_success(modal_mock, monkeypatch, caplog):
    monkeypatch.setattr("timbal.tools.run_python._CLEANUP_TIMEOUT", 0.01)

    async def hang():
        await asyncio.Event().wait()

    modal_mock.sandbox.terminate.aio.side_effect = hang
    result = await asyncio.wait_for(RunPython().handler("1"), timeout=1)
    assert result["status"] == "success"
    assert result["return_value"] == 42
    assert "cleanup failed" in caplog.text


async def test_python_error_envelope(modal_mock):
    error = {"type": "ZeroDivisionError", "message": "division by zero", "traceback": "example"}
    modal_mock.sandbox.exec.aio.side_effect = [
        process("before error\n"),
        process(json.dumps({"return_value": None, "error": error})),
    ]
    result = await RunPython().handler("print('before error')\n1/0")
    assert result["status"] == "error"
    assert result["error"] == error
    assert result["stdout"] == "before error\n"


async def test_setup_failure_does_not_execute(modal_mock):
    modal_mock.module.Sandbox.create.aio.side_effect = RuntimeError("image build failed")
    with pytest.raises(RuntimeError, match="image build failed"):
        await RunPython().handler("1")
    modal_mock.sandbox.exec.aio.assert_not_awaited()
    modal_mock.sandbox.terminate.aio.assert_not_awaited()


async def test_file_inputs_use_modal_transfer_without_runtime_network(modal_mock):
    source = File(b"a,b\n1,2\n", name="data.csv")
    source.seek(2)
    tool = RunPython(max_file_bytes=100, tracing_provider=InMemoryTracingProvider)
    event = await tool(code="42", files={"nested/data.csv": source}).collect()
    assert event.error is None, event.error
    assert event.output["status"] == "success"
    modal_mock.sandbox.filesystem.write_bytes.aio.assert_awaited_once_with(
        b"a,b\n1,2\n", "/workspace/inputs/nested/data.csv"
    )
    assert modal_mock.sandbox.exec.aio.call_args_list[0].kwargs["workdir"] == "/workspace"
    assert modal_mock.module.Sandbox.create.aio.call_args.kwargs["block_network"] is True


@pytest.mark.parametrize("name", ["../secret", "/tmp/file", "a/../../b", "a\\b", "a//b", "a/./b", "", "C:/x", "a\x00b"])
async def test_reject_input_path_escape_before_creation(modal_mock, name):
    with pytest.raises(ValueError, match="relative paths"):
        await RunPython().handler("1", files={name: File(b"x")})
    modal_mock.module.Sandbox.create.aio.assert_not_awaited()


async def test_reject_ambiguous_input_paths(modal_mock):
    with pytest.raises(ValueError, match="directory"):
        await RunPython().handler("1", files={"a/b": File(b"x"), "a/b/c": File(b"y")})
    modal_mock.module.Sandbox.create.aio.assert_not_awaited()


async def test_agent_cannot_supply_host_path(modal_mock, tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("host-only")
    tool = RunPython()
    with pytest.raises(ValidationError, match="file URL"):
        tool.params_model(code="1", files={"data.txt": str(secret)})
    with pytest.raises(ValueError, match="file URL"):
        await tool.handler("1", files={"data.txt": str(secret)})
    # Explicit File objects supplied by application code are supported.
    assert await read_file(File(secret), 20) == b"host-only"
    modal_mock.module.Sandbox.create.aio.assert_not_awaited()


async def test_input_file_count_and_aggregate_size_limits(modal_mock):
    files = {"a": File(b"12"), "b": File(b"34")}
    with pytest.raises(ValueError, match="max_files"):
        await RunPython(max_files=1).handler("1", files=files)
    with pytest.raises(ValueError, match="max_file_bytes"):
        await RunPython(max_file_bytes=3).handler("1", files=files)
    modal_mock.module.Sandbox.create.aio.assert_not_awaited()


async def test_bounded_data_url_and_stream_position():
    source = File(b"hello")
    source.seek(2)
    assert await read_file(source, 5) == b"hello"
    assert source.tell() == 2
    with pytest.raises(ValueError, match="max_file_bytes"):
        await read_file(File("data:text/plain;base64,aGVsbG8="), 4)


async def test_http_input_limit_without_content_length(monkeypatch):
    import httpx

    monkeypatch.setattr(
        asyncio.get_running_loop(), "getaddrinfo", AsyncMock(return_value=[(2, 1, 6, "", ("93.184.215.14", 443))])
    )
    consumed = 0

    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            nonlocal consumed
            for _ in range(100):
                consumed += 1
                yield b"x" * 65536

    client_type = httpx.AsyncClient
    transport = httpx.MockTransport(lambda _request: httpx.Response(200, stream=Stream()))
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: client_type(transport=transport, **kwargs))
    with pytest.raises(ValueError, match="max_file_bytes"):
        await read_file(File("https://example.com/input.bin"), 100)
    assert consumed == 1


async def test_url_input_stays_a_reference_until_bounded_download(modal_mock, monkeypatch):
    import httpx

    monkeypatch.setattr(
        asyncio.get_running_loop(), "getaddrinfo", AsyncMock(return_value=[(2, 1, 6, "", ("93.184.215.14", 443))])
    )
    client_type = httpx.AsyncClient
    transport = httpx.MockTransport(lambda _request: httpx.Response(200, content=b"hello"))
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: client_type(transport=transport, **kwargs))
    persist = AsyncMock(side_effect=AssertionError("Input URL should not be persisted or fetched by tracing"))
    monkeypatch.setattr(File, "persist", persist)
    tool = RunPython(tracing_provider=InMemoryTracingProvider)
    params = tool.params_model(code="1", files={"data.txt": "https://example.com/data.txt"})
    assert isinstance(params.files["data.txt"], str)
    event = await tool(**params.model_dump()).collect()
    assert event.error is None, event.error
    assert event.output["status"] == "success"
    persist.assert_not_awaited()
    modal_mock.sandbox.filesystem.write_bytes.aio.assert_awaited_once_with(b"hello", "/workspace/inputs/data.txt")


async def test_upload_failure_terminates_sandbox(modal_mock):
    modal_mock.sandbox.filesystem.write_bytes.aio.side_effect = RuntimeError("upload failed")
    with pytest.raises(RuntimeError, match="upload failed"):
        await RunPython().handler("1", files={"data": File(b"x")})
    modal_mock.sandbox.exec.aio.assert_not_awaited()
    modal_mock.sandbox.terminate.aio.assert_awaited_once()


@pytest.mark.parametrize("phase", ["upload", "download"])
async def test_transfer_deadline_terminates_sandbox(modal_mock, phase):
    async def hang(*_args):
        await asyncio.Event().wait()

    if phase == "upload":
        modal_mock.sandbox.filesystem.write_bytes.aio.side_effect = hang
    else:
        modal_mock.artifacts.side_effect = hang
    result = await RunPython(transfer_timeout=0.01).handler("1", files={"data": File(b"x")})
    assert result["error"]["type"] == ("TimeoutError" if phase == "upload" else "ArtifactError")
    modal_mock.sandbox.terminate.aio.assert_awaited_once()


@pytest.mark.parametrize("python_error", [False, True])
async def test_artifact_failure_preserves_execution_result(modal_mock, python_error):
    if python_error:
        modal_mock.sandbox.exec.aio.side_effect = [process("before crash", returncode=7)]
    modal_mock.artifacts.side_effect = ValueError("Outputs exceed max_file_bytes.")
    result = await RunPython().handler("1")
    assert result["status"] == "error"
    assert result["stdout"] == ("before crash" if python_error else "hello\n")
    assert result["return_value"] == (None if python_error else 42)
    assert result["error"]["type"] == ("ExecutionError" if python_error else "ArtifactError")
    assert result["artifact_error"]["type"] == "ArtifactError"
    assert result["artifacts"] == []
    modal_mock.sandbox.terminate.aio.assert_awaited_once()


async def test_artifact_persistence_and_agent_visible_reference(modal_mock, monkeypatch):
    payload = b"\x00\xff\x01"
    manifest = {"artifacts": [{"name": "nested/data.bin", "data": base64.b64encode(payload).decode()}], "error": None}
    modal_mock.sandbox.exec.aio.side_effect = [process(json.dumps(manifest))]
    persisted = []

    async def persist(file):
        persisted.append(file.read())
        return "https://example.com/data.bin"

    monkeypatch.setattr(File, "persist", persist)
    artifacts = await _read_artifacts(modal_mock.sandbox, 20, 100)
    assert persisted == [payload]
    assert artifacts == [
        {
            "name": "nested/data.bin",
            "size": 3,
            "content_type": "application/octet-stream",
            "file": "https://example.com/data.bin",
        }
    ]
    message = Message.validate(
        {"role": "tool", "content": [{"type": "tool_result", "id": "test", "content": {"artifacts": artifacts}}]}
    )
    assert "https://example.com/data.bin" in message.content[0].content[0].text


@pytest.mark.parametrize(
    "entries",
    [
        [{"name": "../escape", "data": "eA=="}],
        [{"name": "a", "data": "not base64"}],
        [{"name": "a", "data": "eHh4eA=="}],
        [{"name": "a", "data": "eA=="}, {"name": "a", "data": "eA=="}],
    ],
)
async def test_artifact_manifest_is_validated_on_host(modal_mock, entries):
    modal_mock.sandbox.exec.aio.side_effect = [process(json.dumps({"artifacts": entries, "error": None}))]
    with pytest.raises(ValueError):
        await _read_artifacts(modal_mock.sandbox, 2, 3)


@pytest.mark.skipif(os.name == "nt", reason="Exporter runs inside Linux Modal sandboxes.")
def test_exporter_nested_binary_and_empty_files(tmp_path):
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "data.bin").write_bytes(b"\x00\xff")
    (tmp_path / "empty.txt").write_bytes(b"")
    result = export_files(str(tmp_path), 2, 2)
    assert {item["name"]: base64.b64decode(item["data"]) for item in result} == {
        "nested/data.bin": b"\x00\xff",
        "empty.txt": b"",
    }
    with pytest.raises(ValueError, match="max_files"):
        export_files(str(tmp_path), 1, 2)
    with pytest.raises(ValueError, match="max_file_bytes"):
        export_files(str(tmp_path), 2, 1)


@pytest.mark.skipif(os.name == "nt", reason="Exporter runs inside Linux Modal sandboxes.")
@pytest.mark.parametrize("kind", ["file-link", "directory-link", "fifo", "root-link"])
def test_exporter_rejects_links_and_special_files(tmp_path, kind):
    root = tmp_path / "outputs"
    root.mkdir()
    if kind == "file-link":
        (tmp_path / "secret").write_bytes(b"secret")
        (root / "link").symlink_to(tmp_path / "secret")
    elif kind == "directory-link":
        (root / "link").symlink_to(tmp_path, target_is_directory=True)
    elif kind == "fifo":
        os.mkfifo(root / "pipe")
    else:
        link = tmp_path / "root-link"
        link.symlink_to(root, target_is_directory=True)
        root = link
    with pytest.raises((ValueError, OSError)):
        export_files(str(root), 20, 100)


@pytest.mark.integration
@pytest.mark.parametrize(
    ("code", "dependencies", "timeout", "expected_value", "expected_error"),
    [
        ("import asyncio\nprint('hello')\nawait asyncio.sleep(0, result=42)", None, 60, 42, None),
        (
            "import numpy as np\nimport os\n"
            "assert 'MODAL_TOKEN_SECRET' not in os.environ\n"
            "print('hello')\nfloat(np.mean([1, 2, 3]))",
            ["numpy==2.2.6"],
            60,
            2.0,
            None,
        ),
        ("print('hello')\n1 / 0", None, 60, None, "ZeroDivisionError"),
        ("import os\nprint('hello')\nos._exit(7)", None, 60, None, "ExecutionError"),
        ("import time\ntime.sleep(30)", None, 5, None, "TimeoutError"),
    ],
    ids=["await-and-output", "dependency-and-secret-isolation", "python-error", "hard-exit", "timeout"],
)
async def test_live_modal(monkeypatch, code, dependencies, timeout, expected_value, expected_error):
    if os.environ.get("TIMBAL_TEST_MODAL") != "1":
        pytest.skip("Set TIMBAL_TEST_MODAL=1 with Modal credentials to run paid remote compute.")
    import modal

    sandboxes = []
    create = modal.Sandbox.create.aio

    async def track_creation(*args, **kwargs):
        sandbox = await create(*args, **kwargs)
        sandboxes.append(sandbox)
        return sandbox

    monkeypatch.setattr(modal.Sandbox.create, "aio", track_creation)
    tool = RunPython(timeout=timeout, tracing_provider=InMemoryTracingProvider)
    try:
        event = await tool(code=code, dependencies=dependencies).collect()
        assert event.error is None, event.error
        result = event.output
        assert result["return_value"] == expected_value, result
        if expected_error:
            assert result["status"] == "error", result
            assert result["error"]["type"] == expected_error, result
        else:
            assert result["status"] == "success", result
            assert result["error"] is None
        if expected_error != "TimeoutError":
            assert result["stdout"] == "hello\n"
        if expected_error == "ExecutionError":
            assert result["returncode"] == 7
        assert len(sandboxes) == 1
        async with asyncio.timeout(15):
            while await sandboxes[0].poll.aio() is None:
                await asyncio.sleep(0.25)
    finally:
        # Also clean up if an assertion fails, without hiding whether the tool
        # itself terminated its sandbox (checked above).
        for sandbox in sandboxes:
            await sandbox.terminate.aio()


@pytest.mark.integration
async def test_live_modal_chromium(monkeypatch):
    if os.environ.get("TIMBAL_TEST_MODAL") != "1":
        pytest.skip("Set TIMBAL_TEST_MODAL=1 with Modal credentials to run paid remote compute.")
    import modal

    sandboxes = []
    create = modal.Sandbox.create.aio

    async def track_creation(*args, **kwargs):
        sandbox = await create(*args, **kwargs)
        sandboxes.append(sandbox)
        return sandbox

    monkeypatch.setattr(modal.Sandbox.create, "aio", track_creation)
    tool = RunPython(
        dependencies=["playwright==1.58.0"],
        system_dependencies=["fonts-dejavu-core"],
        setup_commands=["playwright install --with-deps chromium"],
        description="Run Python with Playwright and preinstalled Chromium. Use async Playwright APIs.",
        setup_timeout=600,
        timeout=60,
        memory=1024,
        tracing_provider=InMemoryTracingProvider,
    )
    try:
        event = await tool(
            code="""
from playwright.async_api import async_playwright
async with async_playwright() as p:
    browser = await p.chromium.launch()
    try:
        page = await browser.new_page()
        await page.set_content("<title>Timbal Chromium</title><h1>42</h1>")
        await page.screenshot(path="/workspace/outputs/page.png")
        result = {"title": await page.title(), "value": await page.locator("h1").inner_text()}
    finally:
        await browser.close()
result
"""
        ).collect()
        assert event.error is None, event.error
        assert event.output["status"] == "success", event.output
        assert event.output["return_value"] == {"title": "Timbal Chromium", "value": "42"}
        assert event.output["artifacts"][0]["name"] == "page.png"
        screenshot = File(event.output["artifacts"][0]["file"])
        await screenshot.load()
        assert screenshot.read(8) == b"\x89PNG\r\n\x1a\n"
        assert len(sandboxes) == 1
        async with asyncio.timeout(15):
            while await sandboxes[0].poll.aio() is None:
                await asyncio.sleep(0.25)
    finally:
        for sandbox in sandboxes:
            await sandbox.terminate.aio()


@pytest.mark.integration
async def test_live_modal_file_roundtrip(monkeypatch):
    if os.environ.get("TIMBAL_TEST_MODAL") != "1":
        pytest.skip("Set TIMBAL_TEST_MODAL=1 with Modal credentials to run paid remote compute.")
    import modal

    sandboxes = []
    create = modal.Sandbox.create.aio

    async def track_creation(*args, **kwargs):
        sandbox = await create(*args, **kwargs)
        sandboxes.append(sandbox)
        return sandbox

    monkeypatch.setattr(modal.Sandbox.create, "aio", track_creation)
    tool = RunPython(max_result_bytes=1024, tracing_provider=InMemoryTracingProvider)
    binary_payload = bytes(range(256)) * 4096
    try:
        event = await tool(
            files={
                "data.csv": File(b"value\n2\n3\n", name="data.csv"),
                "nested/blob.bin": File(binary_payload, name="blob.bin"),
            },
            code="""
import csv
from pathlib import Path
assert Path.cwd() == Path('/workspace')
total = sum(int(row['value']) for row in csv.DictReader(Path('inputs/data.csv').open()))
Path('outputs/result.txt').write_text(str(total))
Path('outputs/nested').mkdir()
Path('outputs/nested/blob.bin').write_bytes(Path('inputs/nested/blob.bin').read_bytes())
total
""",
        ).collect()
        assert event.error is None, event.error
        assert event.output["status"] == "success", event.output
        assert event.output["return_value"] == 5
        artifacts = {item["name"]: item for item in event.output["artifacts"]}
        binary = File(artifacts["nested/blob.bin"]["file"])
        await binary.load()
        assert binary.read() == binary_payload
        # A second fresh sandbox can consume a persisted output from the first.
        followup = await tool(
            files={"previous.txt": File(artifacts["result.txt"]["file"])},
            code="from pathlib import Path\nassert not Path('inputs/data.csv').exists()\n"
            "Path('outputs/partial.txt').write_text(Path('inputs/previous.txt').read_text())\n1/0",
        ).collect()
        assert followup.error is None, followup.error
        assert followup.output["error"]["type"] == "ZeroDivisionError"
        assert followup.output["artifacts"][0]["name"] == "partial.txt"
        partial = File(followup.output["artifacts"][0]["file"])
        await partial.load()
        assert partial.read() == b"5"
        assert len(sandboxes) == 2
        for sandbox in sandboxes:
            async with asyncio.timeout(15):
                while await sandbox.poll.aio() is None:
                    await asyncio.sleep(0.25)
    finally:
        for sandbox in sandboxes:
            await sandbox.terminate.aio()
