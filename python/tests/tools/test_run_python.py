"""Runner semantics and Modal lifecycle tests; no remote compute in unit tests."""

import asyncio
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
from timbal.tools.run_python import _RUNNER, _read_output, _resolve_dependencies


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
    sandbox = SimpleNamespace(exec=aio(side_effect=[execution, result]), terminate=aio())
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
    return SimpleNamespace(module=module, sandbox=sandbox, image=image, execution=execution)


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
    }
    modal_mock.image.pip_install.assert_called_once_with("numpy==2.2.6", "pandas")
    config = modal_mock.module.Sandbox.create.aio.call_args.kwargs
    assert config["block_network"] is True
    assert config["memory"] == (512, 512)
    assert config["cpu"] == (1, 1)
    assert config["timeout"] == 70
    assert "env" not in config and "secrets" not in config and "volumes" not in config
    assert "must-not-forward" not in str(modal_mock.execution.stdin.write.call_args)
    request = json.loads(modal_mock.execution.stdin.write.call_args.args[0])
    assert request["code"] == "21 * 2"
    modal_mock.sandbox.terminate.aio.assert_awaited_once()
    assert set(tool.params_model.model_fields) == {"code", "dependencies"}
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
    assert set(schema) == {"code", "dependencies"}
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
    modal_mock.sandbox.exec.aio.side_effect = [process(), process(raw)]
    with pytest.raises(ValueError):
        await RunPython(max_result_bytes=1024).handler("1")
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
        ("import time\ntime.sleep(30)", None, 5, None, "TimeoutError"),
    ],
    ids=["await-and-output", "dependency-and-secret-isolation", "python-error", "timeout"],
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
        result = {"title": await page.title(), "value": await page.locator("h1").inner_text()}
    finally:
        await browser.close()
result
"""
        ).collect()
        assert event.error is None, event.error
        assert event.output["status"] == "success", event.output
        assert event.output["return_value"] == {"title": "Timbal Chromium", "value": "42"}
        assert len(sandboxes) == 1
        async with asyncio.timeout(15):
            while await sandboxes[0].poll.aio() is None:
                await asyncio.sleep(0.25)
    finally:
        for sandbox in sandboxes:
            await sandbox.terminate.aio()
