"""Python execution in a fresh Modal Sandbox per call (install ``timbal[modal]``)."""

import asyncio
import base64
import json
import logging
import math
import mimetypes
from pathlib import Path, PurePosixPath
from typing import Annotated, Any

from pydantic import BeforeValidator, Field, WithJsonSchema, field_validator, model_validator

from ..core.tool import Tool
from ..types import File
from ._run_python_transfer import read_file, validate_file, validate_path

_RUNNER = Path(__file__).with_name("_run_python_runner.py")
_EXPORTER = Path(__file__).with_name("_run_python_files.py")
_MAX_CODE_BYTES = 100_000
_TRUNCATED = "\n... [truncated] ...\n"
_logger = logging.getLogger(__name__)
_CLEANUP_TIMEOUT = 30


async def _terminate_sandbox(sandbox: Any) -> None:
    """Finish bounded teardown before propagating caller cancellation."""

    async def terminate() -> None:
        try:
            async with asyncio.timeout(_CLEANUP_TIMEOUT):
                await sandbox.terminate.aio()
        except Exception:
            # Teardown must not discard completed output or the original error.
            _logger.warning("Modal sandbox cleanup failed; provider TTL remains active.", exc_info=True)

    task = asyncio.create_task(terminate())
    cancellation = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as exc:
            cancellation = exc
    await task
    if cancellation is not None:
        raise cancellation


def _error_result(kind: str, message: str) -> dict[str, Any]:
    return {"return_value": None, "error": {"type": kind, "message": message}}


async def _read_result(sandbox: Any, limit: int) -> dict[str, Any]:
    """Keep missing, invalid, and oversized results distinct without losing logs."""
    try:
        async with asyncio.timeout(30):
            # Bound the read inside Modal even if code left a writer running.
            reader = await sandbox.exec.aio(
                "python",
                "-c",
                "import sys; sys.stdout.buffer.write(open('/tmp/timbal-result.json', 'rb').read(int(sys.argv[1])))",
                str(limit + 1),
                timeout=30,
            )
            raw, _, returncode = await _collect(reader, limit + 1)
        # Modal's ContainerProcess.wait returns -1 when the exec deadline fires.
        if returncode == -1:
            raise TimeoutError
        if returncode != 0:
            return _error_result("ExecutionError", "Python exited without a readable execution result.")
        if len(raw.encode()) > limit:
            return _error_result("OutputLimitError", "Result exceeds max_result_bytes.")
        result = json.loads(raw)
        if not isinstance(result, dict) or "return_value" not in result or "error" not in result:
            raise ValueError("Python produced an invalid result envelope.")
        if result["error"] is not None and not isinstance(result["error"], dict):
            raise ValueError("Python produced an invalid error envelope.")
        return result
    except TimeoutError:
        raise
    except Exception:
        _logger.warning("Could not read the Python execution result.", exc_info=True)
        return _error_result("ResultError", "Could not retrieve a valid Python execution result.")


def _resolve_dependencies(configured: list[str], requested: list[str]) -> list[str]:
    """Keep developer requirements authoritative; accept extras and new packages."""
    from packaging.requirements import InvalidRequirement, Requirement
    from packaging.utils import canonicalize_name

    resolved = {}
    for value in configured:
        requirement = Requirement(value)
        name = canonicalize_name(requirement.name)
        if name in resolved and str(resolved[name]) != str(requirement):
            raise ValueError(f"Multiple configured requirements for {name}; combine them into one requirement.")
        resolved[name] = requirement
    protected = set(resolved)
    for value in requested:
        try:
            requirement = Requirement(value)
        except InvalidRequirement as exc:
            raise ValueError(
                f"Invalid pip requirement: {value!r}. Supply a package name or PEP 508 requirement."
            ) from exc
        name = canonicalize_name(requirement.name)
        existing = resolved.get(name)
        if existing is not None:
            if (
                (requirement.specifier and requirement.specifier != existing.specifier)
                or (requirement.url and requirement.url != existing.url)
                or (requirement.marker and requirement.marker != existing.marker)
            ):
                origin = "configured" if name in protected else "previously requested"
                raise ValueError(
                    f"Dependency {name} is {origin} as {str(existing)!r}; cannot replace it with {value!r}. "
                    "Omit that dependency or repeat its existing requirement."
                )
            existing.extras |= requirement.extras
        else:
            resolved[name] = requirement
    return [str(requirement) for requirement in resolved.values()]


class _OutputBuffer:
    """Retain a bounded output head and tail while a stream is being read."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.head_size = (limit - len(_TRUNCATED)) // 2
        self.tail_size = limit - len(_TRUNCATED) - self.head_size
        self.head = self.tail = ""
        self.total = 0

    def append(self, chunk: str) -> None:
        self.total += len(chunk)
        take = min(len(chunk), self.head_size - len(self.head))
        self.head += chunk[:take]
        self.tail = (self.tail + chunk[take:])[-(self.limit - self.head_size) :]

    def value(self) -> str:
        if self.total <= self.limit:
            return self.head + self.tail
        return self.head + _TRUNCATED + self.tail[-self.tail_size :]


async def _read_output(stream: Any, limit: int, buffer: _OutputBuffer | None = None) -> str:
    """Drain the stream into a buffer that remains readable after cancellation."""
    buffer = buffer or _OutputBuffer(limit)
    async for chunk in stream:
        buffer.append(chunk)
    return buffer.value()


async def _collect(
    process: Any,
    limit: int,
    stdout_buffer: _OutputBuffer | None = None,
    stderr_buffer: _OutputBuffer | None = None,
) -> tuple[str, str, int]:
    # TaskGroup cancels sibling readers on failure/cancellation; no orphaned reads.
    stdout_buffer = stdout_buffer or _OutputBuffer(limit)
    stderr_buffer = stderr_buffer or _OutputBuffer(limit)
    async with asyncio.TaskGroup() as group:
        stdout = group.create_task(_read_output(process.stdout, limit, stdout_buffer))
        stderr = group.create_task(_read_output(process.stderr, limit, stderr_buffer))
        wait = group.create_task(process.wait.aio())
    return stdout.result(), stderr.result(), wait.result()


async def _read_artifacts(sandbox: Any, max_files: int, max_bytes: int, timeout: float = 60) -> list[dict[str, Any]]:
    # Both the exporter and the host enforce limits. Do not trust a stat followed
    # by an unbounded filesystem.read_bytes: background writers can grow a file.
    reader = await sandbox.exec.aio(
        "python",
        "-I",
        "-c",
        _EXPORTER.read_text(),
        "/workspace/outputs",
        str(max_files),
        str(max_bytes),
        timeout=math.ceil(timeout),
    )
    wire_limit = 4 * ((max_bytes + 2) // 3) + max_files * 2048 + 4096
    raw, _, returncode = await _collect(reader, wire_limit)
    if returncode == -1:
        raise TimeoutError("Artifact retrieval timed out.")
    if returncode != 0:
        raise ValueError("Could not retrieve output files.")
    result = json.loads(raw)
    if result["error"]:
        raise ValueError(result["error"]["message"])
    entries = result["artifacts"]
    if not isinstance(entries, list) or len(entries) > max_files:
        raise ValueError("Outputs exceed max_files or contain an invalid manifest.")
    artifacts = []
    names = set()
    total = 0
    for entry in entries:
        name = validate_path(entry["name"])
        if name in names:
            raise ValueError("Duplicate output file name.")
        names.add(name)
        data = base64.b64decode(entry["data"], validate=True)
        total += len(data)
        if total > max_bytes:
            raise ValueError("Outputs exceed max_file_bytes.")
        file = File(data, name=Path(name).name, extension=Path(name).suffix)
        artifacts.append(
            {
                "name": name,
                "size": len(data),
                "content_type": mimetypes.guess_type(name)[0] or "application/octet-stream",
                "file": file,
            }
        )
    # Persist only after validating the entire manifest. File references survive
    # sandbox teardown and appear as usable URLs/paths in agent tool results.
    for artifact in artifacts:
        artifact["file"] = await artifact["file"].persist()
    return artifacts


class RunPython(Tool):
    """Execute Python using Modal's configured credentials; no local fallback.

    Dependencies are explicit pip requirements, baked into Modal's cached image.
    Calls are independent; explicitly supplied files are staged for each call.
    Files written to /workspace/outputs are exported before sandbox teardown.
    Setup, execution, file transfer, and result retrieval have separate deadlines.
    """

    name: str = "run_python"
    description: str | None = (
        "Execute Python 3.11 in a fresh, isolated Modal sandbox. Returns stdout, stderr, "
        "the final expression as return_value, returncode, status, artifacts, and a structured error. "
        "Supports top-level await. Specify third-party pip dependencies explicitly. "
        "Input files are available under /workspace/inputs. Write files to /workspace/outputs to return them as artifacts. "
        "The working directory is /workspace. Other files and variables do not persist between calls."
    )
    dependencies: list[str] = Field(default_factory=list, description="Default pip requirements for every call.")
    system_dependencies: list[str] = Field(
        default_factory=list, description="Debian apt packages installed before Python dependencies."
    )
    setup_commands: list[str] = Field(
        default_factory=list,
        description="Trusted shell commands run remotely during image build, after pip installation.",
    )
    app_name: str = Field(default="timbal-run-python", min_length=1)
    timeout: float = Field(default=60, gt=0, le=86340, allow_inf_nan=False)
    setup_timeout: float = Field(default=300, gt=0, le=3600, allow_inf_nan=False)
    memory: int = Field(default=512, ge=128, description="Memory request and hard limit in MiB.")
    cpu: float = Field(default=1, gt=0, allow_inf_nan=False, description="CPU request and hard limit in cores.")
    block_network: bool = True
    max_output_chars: int = Field(default=20_000, ge=100)
    max_result_bytes: int = Field(default=1_000_000, ge=1024)
    max_files: int = Field(default=20, ge=1, le=1000, description="Maximum file count in each transfer direction.")
    max_file_bytes: int = Field(
        default=10_000_000, ge=1, le=100_000_000, description="Total input or output file bytes per call, separately."
    )
    transfer_timeout: float = Field(default=60, gt=0, le=3600, allow_inf_nan=False)

    @model_validator(mode="after")
    def _validate_lifetime(self) -> "RunPython":
        if self.timeout + 2 * self.transfer_timeout + 60 > 86400:
            raise ValueError("Execution and transfer timeouts plus cleanup must fit Modal's 24-hour sandbox lifetime.")
        return self

    @field_validator("dependencies", "system_dependencies", "setup_commands")
    @classmethod
    def _validate_build_items(cls, values: list[str]) -> list[str]:
        if any(not value.strip() for value in values):
            raise ValueError("Build requirements and commands must not be empty.")
        return [value.strip() for value in values]

    def __init__(self, **kwargs: Any) -> None:
        async def _run_python(
            code: Annotated[
                str,
                Field(
                    description=(
                        "Runnable Python source without Markdown fences. The final expression is returned; "
                        "use print for logs. Top-level await is supported. Each call starts with fresh variables and files."
                    )
                ),
            ],
            dependencies: Annotated[
                list[str] | None,
                Field(
                    description=(
                        "Additional pip requirements, e.g. ['pillow', 'numpy==2.2.6']. "
                        "Standard-library imports need no dependency. Configured requirements cannot be overridden; "
                        "omit packages already configured. System packages and image setup are developer-configured."
                    )
                ),
            ] = None,
            files: Annotated[
                dict[
                    str,
                    Annotated[
                        str | File,
                        BeforeValidator(validate_file),
                        WithJsonSchema(
                            {"type": "string", "format": "uri", "description": "Supplied HTTP(S) file URL or data URL."}
                        ),
                    ],
                ]
                | None,
                Field(
                    description=(
                        "Input files: map relative filenames to supplied file URLs or data URLs, "
                        "e.g. {'sales.csv': 'https://.../sales.csv'}. Read them at /workspace/inputs/<filename>. "
                        "Use actual file references from the conversation; do not invent URLs or host paths. "
                        "Write deliverables to /workspace/outputs; they are returned as artifacts automatically."
                    )
                ),
            ] = None,
        ) -> dict[str, Any]:
            if len(code.encode()) > _MAX_CODE_BYTES:
                raise ValueError(f"Code exceeds {_MAX_CODE_BYTES} bytes.")
            try:
                import modal
            except ImportError as exc:
                raise ImportError("RunPython requires Modal. Install it with: pip install 'timbal[modal]'") from exc

            requirements = _resolve_dependencies(self.dependencies, dependencies or [])
            inputs = {}
            if files:
                if len(files) > self.max_files:
                    raise ValueError("Inputs exceed max_files.")
                for name, file in files.items():
                    inputs[validate_path(name)] = File(validate_file(file))
                for name in inputs:
                    if any(str(parent) in inputs for parent in PurePosixPath(name).parents):
                        raise ValueError("An input file cannot also be another input's directory.")
            image = modal.Image.debian_slim(python_version="3.11")
            if self.system_dependencies:
                image = image.apt_install(*self.system_dependencies)
            if requirements:
                image = image.pip_install(*requirements)
            if self.setup_commands:
                image = image.run_commands(*self.setup_commands)

            sandbox = None
            stdout = stderr = ""
            returncode = None
            stdout_buffer = _OutputBuffer(self.max_output_chars)
            stderr_buffer = _OutputBuffer(self.max_output_chars)
            try:
                input_data = {}
                async with asyncio.timeout(self.transfer_timeout):
                    total = 0
                    for name, file in inputs.items():
                        data = await read_file(file, self.max_file_bytes - total)
                        total += len(data)
                        input_data[name] = data
                async with asyncio.timeout(self.setup_timeout):
                    app = await modal.App.lookup.aio(self.app_name, create_if_missing=True)
                    sandbox = await modal.Sandbox.create.aio(
                        app=app,
                        image=image,
                        # Provider TTL also bounds orphan lifetime if the client disappears.
                        timeout=math.ceil(self.timeout + 2 * self.transfer_timeout + 60),
                        cpu=(self.cpu, self.cpu),
                        memory=(self.memory, self.memory),
                        block_network=self.block_network,
                    )
                async with asyncio.timeout(self.transfer_timeout):
                    await sandbox.filesystem.make_directory.aio("/workspace/inputs")
                    await sandbox.filesystem.make_directory.aio("/workspace/outputs")
                    for name, data in input_data.items():
                        await sandbox.filesystem.write_bytes.aio(data, f"/workspace/inputs/{name}")
                async with asyncio.timeout(self.timeout):
                    process = await sandbox.exec.aio(
                        "python",
                        "-u",
                        "-c",
                        _RUNNER.read_text(),
                        timeout=math.ceil(self.timeout),
                        workdir="/workspace",
                    )
                    process.stdin.write(json.dumps({"code": code, "max_result_bytes": self.max_result_bytes}))
                    process.stdin.write_eof()
                    await process.stdin.drain.aio()
                    stdout, stderr, returncode = await _collect(
                        process,
                        self.max_output_chars,
                        stdout_buffer,
                        stderr_buffer,
                    )
                    # Modal's ContainerProcess.wait currently returns -1 on exec timeout.
                    if returncode == -1:
                        raise TimeoutError

                if returncode != 0:
                    result = _error_result("ExecutionError", f"Python exited with code {returncode}.")
                else:
                    result = await _read_result(sandbox, self.max_result_bytes)
                artifacts = []
                artifact_error = None
                try:
                    async with asyncio.timeout(self.transfer_timeout):
                        artifacts = await _read_artifacts(
                            sandbox, self.max_files, self.max_file_bytes, self.transfer_timeout
                        )
                except Exception as exc:
                    artifact_error = {"type": "ArtifactError", "message": str(exc) or "Artifact retrieval timed out."}
                    _logger.warning("Could not export Python artifacts.", exc_info=True)
                error = result["error"]
                return {
                    "stdout": stdout,
                    "stderr": stderr,
                    "returncode": returncode,
                    "return_value": result["return_value"],
                    "artifacts": artifacts,
                    "artifact_error": artifact_error,
                    "error": error or artifact_error,
                    "status": "error" if error or artifact_error else "success",
                }
            except TimeoutError:
                return {
                    "stdout": stdout_buffer.value(),
                    "stderr": stderr_buffer.value(),
                    "returncode": returncode,
                    "return_value": None,
                    "artifacts": [],
                    "artifact_error": None,
                    "status": "error",
                    "error": {
                        "type": "TimeoutError",
                        "message": "Modal setup, execution, input transfer, or result retrieval timed out.",
                    },
                }
            finally:
                if sandbox is not None:
                    # The finally block also runs on caller cancellation. Server TTL is
                    # the fallback if termination cannot reach Modal.
                    await _terminate_sandbox(sandbox)

        super().__init__(handler=_run_python, **kwargs)
        # Do not expose shell commands: they can contain developer-only build details.
        environment = (
            f" Execution timeout: {self.timeout:g}s. Runtime network access: "
            f"{'blocked' if self.block_network else 'enabled'}. "
            f"Configured Python requirements: {', '.join(self.dependencies) or 'none'}. "
            f"Configured system packages: {', '.join(self.system_dependencies) or 'none'}."
            f" File limits: {self.max_files} files and {self.max_file_bytes} bytes per direction."
        )
        self.description = (self.description or "") + environment

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config(
                {
                    key: getattr(self, key)
                    for key in (
                        "dependencies",
                        "system_dependencies",
                        "setup_commands",
                        "app_name",
                        "timeout",
                        "setup_timeout",
                        "memory",
                        "cpu",
                        "block_network",
                        "max_output_chars",
                        "max_result_bytes",
                        "max_files",
                        "max_file_bytes",
                        "transfer_timeout",
                    )
                }
            ),
        }
