"""
RunPython tool for executing Python scripts in isolated environments.

Supports:
- Real CPython execution via uv (--with dependencies) with venv+pip fallback
- Auto-detection of dependencies from import statements
- Code mode: scripts can call back into other Timbal tools via a unix-socket RPC bridge
- Structured stdout/stderr/return_value/error capture
"""

from __future__ import annotations

import ast
import asyncio
import contextlib
import inspect
import json
import os
import re
import shutil
import signal
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, PrivateAttr

from ..core.tool import Tool
from ..state import get_run_context

# Common import name -> PyPI package name mismatches.
_IMPORT_TO_PACKAGE: dict[str, str] = {
    "PIL": "pillow",
    "bs4": "beautifulsoup4",
    "cv2": "opencv-python",
    "dotenv": "python-dotenv",
    "googleapiclient": "google-api-python-client",
    "jwt": "pyjwt",
    "sklearn": "scikit-learn",
    "yaml": "pyyaml",
}

_RESULT_FILENAME = ".timbal_run_result.json"
_USER_CODE_FILENAME = "user_code.py"
_RPC_ENV_VAR = "TIMBAL_RPC_SOCKET"
_USER_CODE_PATH_ENV = "TIMBAL_USER_CODE_PATH"
_EXPOSED_TOOLS_ENV = "TIMBAL_EXPOSED_TOOLS"
_RUNNER_PATH = Path(__file__).parent / "_run_python_runner.py"

# Hardening limits.
_MAX_CODE_CHARS = 200_000
_DEFAULT_MAX_OUTPUT_CHARS = 10_000
_MAX_RESULT_BYTES = 1_000_000

# Minimal set of environment variables forwarded to the child process so the
# interpreter/uv can run. Everything else (secrets like API keys, tokens) is
# stripped by default so executed code cannot exfiltrate host credentials.
# Mirrors the top-performer guidance (E2B/Monty): the sandbox receives inputs
# and returns outputs; it should never receive your secrets.
_SAFE_ENV_VARS: frozenset[str] = frozenset({
    "PATH",
    "HOME",
    "LANG",
    "LANGUAGE",
    "LC_ALL",
    "LC_CTYPE",
    "TZ",
    "TERM",
    "TMPDIR",
    "TEMP",
    "TMP",
    # Windows essentials
    "SYSTEMROOT",
    "SystemRoot",
    "PATHEXT",
    "USERPROFILE",
    "COMSPEC",
    "WINDIR",
    "NUMBER_OF_PROCESSORS",
    "PROCESSOR_ARCHITECTURE",
    # uv / Python cache + config so isolated runs stay fast without leaking secrets
    "UV_CACHE_DIR",
    "UV_PYTHON_INSTALL_DIR",
    "XDG_CACHE_HOME",
    "XDG_DATA_HOME",
    "XDG_CONFIG_HOME",
    "PYTHONPATH",
})

def _build_child_env(passthrough: list[str] | None) -> dict[str, str]:
    """Build a minimal environment for the child process.

    Only safe, non-secret variables are forwarded by default. Additional names
    can be explicitly allowlisted via ``env_passthrough``.
    """
    allowed = set(_SAFE_ENV_VARS)
    if passthrough:
        allowed.update(passthrough)
    return {key: value for key, value in os.environ.items() if key in allowed}


def _truncate_output(text: str, limit: int) -> str:
    """Cap output to ``limit`` chars, keeping head and tail with a drop marker."""
    if limit <= 0 or len(text) <= limit:
        return text
    dropped = len(text) - limit
    head_len = limit // 2
    tail_len = limit - head_len
    head = text[:head_len]
    tail = text[-tail_len:]
    return f"{head}\n... [{dropped} characters truncated] ...\n{tail}"


def _ensure_tool(obj: Any) -> Tool:
    if isinstance(obj, Tool):
        return obj
    if callable(obj):
        return Tool(handler=obj)
    raise TypeError(f"Expected Tool or callable, got {type(obj)!r}")


def _detect_imports(code: str) -> list[str]:
    """Extract top-level import module names from Python source."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module.split(".")[0])
    return sorted(modules)


def _stdlib_modules() -> set[str]:
    return set(sys.stdlib_module_names)


def _dep_base_name(spec: str) -> str:
    """Extract the normalized distribution name from a requirement spec.

    Handles forms like ``timbal``, ``timbal>=1.0``, ``timbal[extra]`` and the
    PEP 508 URL form ``timbal @ file:///path``.
    """
    s = spec.split(";", 1)[0].strip()
    if "@" in s:
        s = s.split("@", 1)[0].strip()
    s = re.split(r"[\s\[<>=!~(]", s, maxsplit=1)[0].strip()
    return s.replace("_", "-").lower()


def _resolve_dependencies(
    code: str,
    explicit: list[str] | None,
    default: list[str] | None,
) -> list[str]:
    """Merge explicit/default dependencies with auto-detected imports."""
    resolved: list[str] = []
    seen: set[str] = set()

    def add(dep: str) -> None:
        dep = dep.strip()
        if dep and dep not in seen:
            seen.add(dep)
            resolved.append(dep)

    pinned_bases: set[str] = set()
    for source in (default or [], explicit or []):
        for dep in source:
            pinned_bases.add(_dep_base_name(dep))
            add(dep)

    stdlib = _stdlib_modules()
    for module in _detect_imports(code):
        if module in stdlib or module.startswith("_"):
            continue
        package = _IMPORT_TO_PACKAGE.get(module, module)
        # Skip auto-detected packages already pinned explicitly (e.g. a local
        # `timbal @ file://...` must not be shadowed by a bare `timbal`).
        if _dep_base_name(package) in pinned_bases:
            continue
        add(package)

    return resolved


def _serialize_output(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return {"__timbal_serialized__": repr(value), "__type__": type(value).__name__}


class ToolDispatchError(Exception):
    """Structured tool error from the code-mode RPC bridge."""

    def __init__(self, error: dict[str, Any]) -> None:
        self.error = error
        super().__init__(error.get("message", str(error)))


async def _dispatch_exposed_tool(tool: Tool, args: list[Any], kwargs: dict[str, Any]) -> Any:
    """Invoke an exposed tool from the code-mode RPC bridge."""
    sig = inspect.signature(tool.handler)
    bound = sig.bind(*args, **kwargs)
    var_positional = next(
        (param.name for param in sig.parameters.values() if param.kind == inspect.Parameter.VAR_POSITIONAL),
        None,
    )
    var_keyword = next(
        (param.name for param in sig.parameters.values() if param.kind == inspect.Parameter.VAR_KEYWORD),
        None,
    )
    positional_params = [
        param.name
        for param in sig.parameters.values()
        if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]

    async def _call_handler(**call_kwargs: Any) -> Any:
        if inspect.iscoroutinefunction(tool.handler):
            return await tool.handler(**call_kwargs)
        return await asyncio.to_thread(tool.handler, **call_kwargs)

    if var_keyword and not positional_params and not var_positional:
        return await _call_handler(**bound.kwargs)
    if var_positional and not positional_params and not var_keyword:
        handler_args = bound.arguments.get(var_positional, ())
        if inspect.iscoroutinefunction(tool.handler):
            return await tool.handler(*handler_args)
        return await asyncio.to_thread(tool.handler, *handler_args)
    if var_keyword and positional_params:
        pos_values = {param_name: bound.arguments[param_name] for param_name in positional_params}
        extra_kw = bound.arguments.get(var_keyword, {})
        return await _call_handler(**pos_values, **extra_kw)

    result = await tool(**bound.arguments).collect()
    if result.error:
        raise ToolDispatchError(result.error)
    return result.output


class _ToolRpcServer:
    """Async unix-socket RPC server dispatching tool calls from child process."""

    def __init__(self, tools: dict[str, Tool]) -> None:
        self._tools = tools
        self._socket_path: str | None = None
        self._server: asyncio.AbstractServer | None = None

    @property
    def socket_path(self) -> str | None:
        return self._socket_path

    async def start(self) -> None:
        if not self._tools:
            return
        tmp = tempfile.NamedTemporaryFile(prefix="timbal-rpc-", suffix=".sock", delete=False)
        tmp.close()
        self._socket_path = tmp.name
        Path(self._socket_path).unlink(missing_ok=True)
        self._server = await asyncio.start_unix_server(self._handle_client, path=self._socket_path)

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        if self._socket_path:
            Path(self._socket_path).unlink(missing_ok=True)
            self._socket_path = None

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            data = await reader.readline()
            if not data:
                return
            request = json.loads(data.decode())
            name = request.get("name")
            args = request.get("args") or []
            kwargs = request.get("kwargs") or {}
            tool = self._tools.get(name)
            if tool is None:
                response = {"output": None, "error": {"type": "KeyError", "message": f"Unknown tool: {name}"}}
            else:
                try:
                    output = await _dispatch_exposed_tool(tool, args, kwargs)
                    response = {"output": _serialize_output(output), "error": None}
                except ToolDispatchError as exc:
                    response = {"output": None, "error": exc.error}
                except TypeError as exc:
                    response = {
                        "output": None,
                        "error": {"type": "TypeError", "message": str(exc), "traceback": traceback.format_exc()},
                    }
                except Exception as exc:
                    response = {
                        "output": None,
                        "error": {
                            "type": type(exc).__name__,
                            "message": str(exc),
                            "traceback": traceback.format_exc(),
                        },
                    }
            writer.write(json.dumps(response).encode() + b"\n")
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()


def _kill_process_tree(process: asyncio.subprocess.Process) -> None:
    """Best-effort terminate the child and any subprocesses it spawned.

    On POSIX the child is started in its own session (process group) so a
    runaway ``uv run`` that forks Python is killed entirely, not just the
    top-level wrapper.
    """
    if os.name == "posix":
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            return
    with contextlib.suppress(ProcessLookupError):
        process.kill()


async def _run_subprocess(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout: float,
) -> tuple[int, str, str]:
    popen_kwargs: dict[str, Any] = {}
    if os.name == "posix":
        popen_kwargs["start_new_session"] = True

    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(cwd),
        env=env,
        **popen_kwargs,
    )
    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(process.communicate(), timeout=timeout)
    except TimeoutError:
        _kill_process_tree(process)
        with contextlib.suppress(Exception):
            await process.wait()
        raise
    stdout = stdout_bytes.decode("utf-8", errors="replace") if stdout_bytes else ""
    stderr = stderr_bytes.decode("utf-8", errors="replace") if stderr_bytes else ""
    return process.returncode or 0, stdout, stderr


def _build_command(
    dependencies: list[str],
    executor: Literal["uv", "venv"],
    venv_python: Path | None,
) -> list[str]:
    runner = str(_RUNNER_PATH)
    if executor == "uv":
        uv = shutil.which("uv")
        if uv:
            cmd = [uv, "run", "--no-project", *(f"--with={dep}" for dep in dependencies), runner]
            return cmd
    if venv_python is not None:
        return [str(venv_python), runner]
    return [sys.executable, runner]


async def _ensure_venv_deps(venv_dir: Path, dependencies: list[str]) -> Path:
    python_path = venv_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not python_path.exists():
        create = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "venv",
            str(venv_dir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await create.wait()
        if create.returncode != 0:
            _, stderr = await create.communicate()
            raise RuntimeError(f"Failed to create venv: {stderr.decode()}")

    if dependencies:
        pip_cmd = [str(python_path), "-m", "pip", "install", "--quiet", *dependencies]
        pip_proc = await asyncio.create_subprocess_exec(
            *pip_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await pip_proc.wait()
        if pip_proc.returncode != 0:
            _, stderr = await pip_proc.communicate()
            raise RuntimeError(f"Failed to install dependencies: {stderr.decode()}")

    return python_path


def _tools_description(tools: dict[str, Tool]) -> str:
    if not tools:
        return ""
    lines = ["Available functions in the script environment:"]
    for name, tool in tools.items():
        try:
            sig = inspect.signature(tool.handler)
            lines.append(f"- {name}{sig}")
        except (TypeError, ValueError):
            lines.append(f"- {name}(...)")
    return "\n".join(lines)


class RunPython(Tool):
    """Execute Python code in an isolated environment with optional code-mode tool callbacks."""

    default_dependencies: list[str] | None = Field(
        default=None,
        description="Default pip dependencies merged with auto-detected imports and per-call overrides.",
    )
    timeout: float = Field(default=60.0, description="Maximum execution time in seconds.")
    executor: Literal["uv", "venv", "auto"] = Field(
        default="auto",
        description="Execution backend: uv (--with deps), venv+pip fallback, or auto (prefer uv).",
    )
    exposed_tools: list[Any] = Field(
        default_factory=list,
        description="Tools/callables exposed to the script as functions (code mode).",
    )
    max_output_chars: int = Field(
        default=_DEFAULT_MAX_OUTPUT_CHARS,
        description="Max chars kept per stdout/stderr channel (head+tail) to protect the context window.",
    )
    env_passthrough: list[str] | None = Field(
        default=None,
        description=(
            "Allowlist of host environment variable names forwarded to the executed code. "
            "By default secrets (API keys, tokens) are NOT exposed to the sandbox."
        ),
    )

    _tool_map: dict[str, Tool] = PrivateAttr(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        self._tool_map = {}
        for item in self.exposed_tools:
            tool = _ensure_tool(item)
            self._tool_map[tool.name] = tool

    def __init__(
        self,
        tools: list[Any] | None = None,
        dependencies: list[str] | None = None,
        timeout: float = 60.0,
        executor: Literal["uv", "venv", "auto"] = "auto",
        max_output_chars: int = _DEFAULT_MAX_OUTPUT_CHARS,
        env_passthrough: list[str] | None = None,
        **kwargs: Any,
    ):
        tool_map: dict[str, Tool] = {}
        for item in tools or []:
            t = _ensure_tool(item)
            tool_map[t.name] = t

        default_deps = dependencies

        desc = (
            "Execute Python code in an isolated environment. "
            "Captures stdout, stderr, and the value of the last expression as return_value. "
            "Dependencies are auto-detected from imports or can be specified explicitly. "
            "Exposed tools are callable from the script (code mode)."
        )
        tools_desc = _tools_description(tool_map)
        if tools_desc:
            desc = f"{desc}\n\n{tools_desc}"

        async def _run_python(code: str, dependencies: list[str] | None = None) -> dict[str, Any]:
            if len(code) > _MAX_CODE_CHARS:
                return {
                    "stdout": "",
                    "stderr": "",
                    "return_value": None,
                    "status": "error",
                    "error": {
                        "type": "ValueError",
                        "message": f"Code exceeds maximum size of {_MAX_CODE_CHARS} characters",
                    },
                }

            run_context = get_run_context()
            base_dir = run_context.resolve_cwd() if run_context else Path.cwd()
            run_dir = Path(tempfile.mkdtemp(prefix=".timbal_run_", dir=base_dir))

            resolved_deps = _resolve_dependencies(code, dependencies, default_deps)
            user_code_path = run_dir / _USER_CODE_FILENAME
            result_path = run_dir / _RESULT_FILENAME
            user_code_path.write_text(code, encoding="utf-8")

            rpc = _ToolRpcServer(tool_map)
            await rpc.start()

            # Strip host secrets: only forward a safe baseline + explicit allowlist.
            env = _build_child_env(env_passthrough)
            env["TIMBAL_RESULT_PATH"] = str(result_path)
            env[_USER_CODE_PATH_ENV] = str(user_code_path)
            env[_EXPOSED_TOOLS_ENV] = json.dumps(list(tool_map.keys()))
            if rpc.socket_path:
                env[_RPC_ENV_VAR] = rpc.socket_path

            venv_dir = base_dir / ".timbal_run_venv"
            venv_python: Path | None = None
            chosen_executor: Literal["uv", "venv"]
            if executor == "auto":
                chosen_executor = "uv" if shutil.which("uv") else "venv"
            else:
                chosen_executor = executor

            try:
                if chosen_executor == "venv" or (chosen_executor == "uv" and not shutil.which("uv")):
                    venv_python = await _ensure_venv_deps(venv_dir, resolved_deps)
                    chosen_executor = "venv"

                cmd = _build_command(resolved_deps, chosen_executor, venv_python)
                try:
                    returncode, stdout, stderr = await _run_subprocess(
                        cmd,
                        cwd=run_dir,
                        env=env,
                        timeout=timeout,
                    )
                except TimeoutError:
                    return {
                        "stdout": "",
                        "stderr": "",
                        "return_value": None,
                        "status": "error",
                        "error": {
                            "type": "TimeoutError",
                            "message": f"Execution exceeded timeout of {timeout}s",
                        },
                    }

                stdout = _truncate_output(stdout, max_output_chars)
                stderr = _truncate_output(stderr, max_output_chars)

                payload: dict[str, Any] | None = None
                if result_path.exists():
                    if result_path.stat().st_size > _MAX_RESULT_BYTES:
                        return {
                            "stdout": stdout,
                            "stderr": stderr,
                            "return_value": None,
                            "status": "error",
                            "error": {
                                "type": "ValueError",
                                "message": (
                                    f"return_value exceeds maximum size of {_MAX_RESULT_BYTES} bytes; "
                                    "write large outputs to a file instead"
                                ),
                            },
                        }
                    try:
                        payload = json.loads(result_path.read_text(encoding="utf-8"))
                    except json.JSONDecodeError:
                        payload = None

                if payload and payload.get("error"):
                    return {
                        "stdout": stdout,
                        "stderr": stderr,
                        "return_value": None,
                        "status": "error",
                        "error": payload["error"],
                    }

                return_value = payload.get("return_value") if payload else None
                status = "success" if returncode == 0 else "error"
                error = None
                if returncode != 0 and not payload:
                    error = {
                        "type": "ProcessError",
                        "message": f"Process exited with code {returncode}",
                    }
                if returncode != 0 and stderr and error is None:
                    error = {"type": "ProcessError", "message": stderr.strip()}

                return {
                    "stdout": stdout,
                    "stderr": stderr,
                    "return_value": return_value,
                    "status": status,
                    "error": error,
                }
            finally:
                await rpc.stop()
                shutil.rmtree(run_dir, ignore_errors=True)

        super().__init__(
            name="run_python",
            description=desc,
            handler=_run_python,
            default_dependencies=dependencies,
            timeout=timeout,
            executor=executor,
            exposed_tools=tools or [],
            max_output_chars=max_output_chars,
            env_passthrough=env_passthrough,
            background_mode="auto",
            **kwargs,
        )
        self._tool_map = tool_map

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {
                    "default_dependencies": self.default_dependencies,
                    "timeout": self.timeout,
                    "executor": self.executor,
                    "exposed_tools": [t.name for t in self._tool_map.values()],
                    "max_output_chars": self.max_output_chars,
                    "env_passthrough": self.env_passthrough,
                }
            ),
        }
